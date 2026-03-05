import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

try:
    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5Config as Qwen35Config,
        Qwen3_5TextConfig as Qwen35TextConfig,
        Qwen3_5VisionConfig as Qwen35VisionConfig,
    )
    _QWEN35_NATIVE = True
except Exception:
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config as Qwen35TextConfig
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLConfig as Qwen35Config,
        Qwen3VLTextConfig as Qwen35TextConfigVLCompat,
        Qwen3VLVisionConfig as Qwen35VisionConfig,
    )
    _QWEN35_NATIVE = False

from ssd.layers.activation import SiluAndMul
from ssd.layers.attention import Attention
from ssd.layers.layernorm import RMSHeadNorm, RMSDNorm
from ssd.layers.linear import ColumnParallelLinear, QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from ssd.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from ssd.utils.context import get_context


def _get_rope_parameters(config) -> dict:
    rope_parameters = getattr(config, "rope_parameters", None)
    if rope_parameters is not None:
        rope_parameters = dict(rope_parameters)
    else:
        rope_parameters = {}
        rope_parameters["rope_theta"] = getattr(config, "rope_theta", 1000000.0)
        rope_scaling = getattr(config, "rope_scaling", None)
        if isinstance(rope_scaling, dict):
            rope_parameters.update(rope_scaling)
    rope_parameters.setdefault("rope_theta", 1000000.0)
    rope_parameters.setdefault("rope_type", "default")
    if "partial_rotary_factor" not in rope_parameters:
        rope_parameters["partial_rotary_factor"] = getattr(config, "partial_rotary_factor", 1.0)
    return rope_parameters


def _get_layer_types(config) -> list[str]:
    layer_types = getattr(config, "layer_types", None)
    if layer_types is None:
        return ["full_attention"] * int(config.num_hidden_layers)
    return list(layer_types)


def _build_default_mrope_section(rotary_dim: int) -> list[int]:
    # Match HF Qwen3.5 default for common rotary_dim=64.
    if rotary_dim == 64:
        return [11, 11, 10]
    half = rotary_dim // 2
    a = half // 3
    b = half // 3
    c = half - a - b
    return [a, b, c]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb_flat(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = q.shape[0]
    q_shape, k_shape = q.shape, k.shape
    q = q.view(num_tokens, -1, head_dim)
    k = k.view(num_tokens, -1, head_dim)

    q_dtype = q.dtype
    k_dtype = k.dtype
    q = q.float()
    k = k.float()
    cos = cos.unsqueeze(1).float()
    sin = sin.unsqueeze(1).float()

    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q.to(q_dtype).reshape(q_shape), k.to(k_dtype).reshape(k_shape)


def _apply_rotary_pos_emb_vision(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_dtype = q.dtype
    k_dtype = k.dtype
    q = q.float()
    k = k.float()
    cos = cos.unsqueeze(1).float()
    sin = sin.unsqueeze(1).float()
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q.to(q_dtype), k.to(k_dtype)


def _segmented_vision_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    training: bool = False,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    # query/key/value: [seq_len, num_heads, head_dim]
    outputs = []
    for i in range(cu_seqlens.numel() - 1):
        s = int(cu_seqlens[i].item())
        e = int(cu_seqlens[i + 1].item())
        if e <= s:
            continue
        q = query[s:e].transpose(0, 1)  # [H, L, D]
        k = key[s:e].transpose(0, 1)
        v = value[s:e].transpose(0, 1)
        attn = (q @ k.transpose(-1, -2)) * scale
        attn = torch.softmax(attn.float(), dim=-1).to(q.dtype)
        if training and dropout_p > 0.0:
            attn = F.dropout(attn, p=dropout_p)
        o = attn @ v
        outputs.append(o.transpose(0, 1))
    if len(outputs) == 0:
        return torch.empty_like(query)
    return torch.cat(outputs, dim=0)


def _normalize_mrope_positions(positions: torch.Tensor) -> torch.Tensor:
    # Returns [3, N].
    if positions.ndim == 1:
        return positions.view(1, -1).expand(3, -1)
    if positions.ndim == 2:
        if positions.size(0) == 3:
            return positions
        if positions.size(1) == 3:
            return positions.transpose(0, 1)
        # Fallback to standard 1D positions duplicated for mrope.
        return positions.reshape(1, -1).expand(3, -1)
    if positions.ndim == 3:
        # [3, B, S] or [B, 3, S]
        if positions.size(0) == 3:
            return positions.reshape(3, -1)
        if positions.size(1) == 3:
            return positions.transpose(0, 1).reshape(3, -1)
        return positions.reshape(1, -1).expand(3, -1)
    raise ValueError(f"Unsupported positions shape for mrope: {tuple(positions.shape)}")


class Qwen35RMSNormGated(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hs = hidden_states.to(torch.float32)
        variance = hs.pow(2).mean(-1, keepdim=True)
        hs = hs * torch.rsqrt(variance + self.variance_epsilon)
        hs = self.weight * hs.to(input_dtype)
        if gate is not None:
            hs = hs * F.silu(gate.to(torch.float32))
        return hs.to(input_dtype)


def _torch_causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    # hidden_states: [B, C, L], conv_state: [B, C, K-1], weight: [C, K]
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:])
    out = out.to(hidden_states.dtype)
    return out


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def _torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    # query/key/value: [B, S, H, D]
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]  # [B, H, S, ...]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    if initial_state is None:
        last_recurrent_state = torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, device=value.device, dtype=value.dtype)
    else:
        last_recurrent_state = initial_state.to(value)

    core_attn_out = torch.zeros_like(value)
    mask_upper = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask_upper, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def _torch_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]  # [B, H, S, ...]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim, device=value.device, dtype=value.dtype)
    if initial_state is None:
        last_recurrent_state = torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, device=value.device, dtype=value.dtype)
    else:
        last_recurrent_state = initial_state.to(value)

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)
        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


class Qwen35LinearAttention(nn.Module):

    def __init__(
        self,
        config,
        layer_idx: int,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int = 1,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.conv_kernel_size = int(config.linear_conv_kernel_dim)
        self.head_k_dim = int(config.linear_key_head_dim)
        self.head_v_dim = int(config.linear_value_head_dim)
        self.total_num_k_heads = int(config.linear_num_key_heads)
        self.total_num_v_heads = int(config.linear_num_value_heads)
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.tp_rank = dist.get_rank(group=tp_group) if (tp_size > 1 and tp_group is not None) else 0

        if self.total_num_k_heads % self.tp_size != 0 or self.total_num_v_heads % self.tp_size != 0:
            raise ValueError(
                f"linear heads not divisible by tp_size: "
                f"k={self.total_num_k_heads}, v={self.total_num_v_heads}, tp={self.tp_size}"
            )

        self.num_k_heads = self.total_num_k_heads // self.tp_size
        self.num_v_heads = self.total_num_v_heads // self.tp_size
        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError(
                f"local linear head ratio must be integer, got v={self.num_v_heads}, k={self.num_k_heads}"
            )
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.total_key_dim = self.total_num_k_heads * self.head_k_dim
        self.total_value_dim = self.total_num_v_heads * self.head_v_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.total_conv_dim = self.total_key_dim * 2 + self.total_value_dim

        self.in_proj_qkv = ColumnParallelLinear(
            self.hidden_size, self.total_conv_dim, bias=False, tp_group=tp_group, tp_size=tp_size
        )
        self.in_proj_z = ColumnParallelLinear(
            self.hidden_size, self.total_value_dim, bias=False, tp_group=tp_group, tp_size=tp_size
        )
        self.in_proj_b = ColumnParallelLinear(
            self.hidden_size, self.total_num_v_heads, bias=False, tp_group=tp_group, tp_size=tp_size
        )
        self.in_proj_a = ColumnParallelLinear(
            self.hidden_size, self.total_num_v_heads, bias=False, tp_group=tp_group, tp_size=tp_size
        )

        self.conv1d_weight = nn.Parameter(torch.empty(self.conv_dim, 1, self.conv_kernel_size))
        self.conv1d_weight.weight_loader = self._conv1d_weight_loader

        self.dt_bias = nn.Parameter(torch.zeros(self.num_v_heads))
        self.dt_bias.weight_loader = self._head_shard_loader
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))
        self.A_log.weight_loader = self._head_shard_loader

        self.norm = Qwen35RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = RowParallelLinear(
            self.total_value_dim, self.hidden_size, bias=False, tp_group=tp_group, tp_size=tp_size
        )

        self._committed: dict[int, tuple[int, torch.Tensor, torch.Tensor]] = {}
        self._pending: dict[int, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}

    def _head_shard_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        shard_size = param.data.size(0)
        start_idx = self.tp_rank * shard_size
        param.data.copy_(loaded_weight.narrow(0, start_idx, shard_size))

    def _conv1d_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        shard_size = param.data.size(0)
        start_idx = self.tp_rank * shard_size
        param.data.copy_(loaded_weight.narrow(0, start_idx, shard_size))

    def reset_state_cache(self):
        self._committed.clear()
        self._pending.clear()

    def _get_prev_slot(self, row_idx: int, start_pos: int, context) -> int:
        prev_pos = start_pos - 1
        if prev_pos < 0:
            return -1
        if context.block_tables is None or context.block_size <= 0:
            return -1
        block_idx = prev_pos // context.block_size
        pos_in_block = prev_pos % context.block_size
        block_id = int(context.block_tables[row_idx, block_idx].item())
        if block_id < 0:
            return -1
        return block_id * context.block_size + pos_in_block

    def _get_initial_states(self, seq_id: int, row_idx: int, start_pos: int, context):
        prev_slot = self._get_prev_slot(row_idx, start_pos, context)
        if prev_slot < 0:
            return None, None

        committed = self._committed.get(seq_id, None)
        if committed is not None and committed[0] == prev_slot:
            return committed[1].clone(), committed[2].clone()

        pending = self._pending.get(seq_id, {})
        if prev_slot in pending:
            conv_state, recurrent_state = pending[prev_slot]
            self._committed[seq_id] = (prev_slot, conv_state.clone(), recurrent_state.clone())
            return conv_state.clone(), recurrent_state.clone()

        if committed is not None:
            # Best-effort fallback to latest committed state.
            return committed[1].clone(), committed[2].clone()
        return None, None

    def _run_segment(
        self,
        hidden_states: torch.Tensor,  # [L, D]
        conv_state: torch.Tensor | None,
        recurrent_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden_states.numel() == 0:
            empty_conv = torch.zeros(self.conv_dim, self.conv_kernel_size - 1, device=hidden_states.device, dtype=hidden_states.dtype)
            empty_rec = torch.zeros(self.num_v_heads, self.head_k_dim, self.head_v_dim, device=hidden_states.device, dtype=torch.float32)
            return hidden_states, empty_conv, empty_rec

        seq_len = hidden_states.size(0)
        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(0, 1).unsqueeze(0)  # [1, C, L]
        z = self.in_proj_z(hidden_states).view(1, seq_len, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(hidden_states).view(1, seq_len, self.num_v_heads)
        a = self.in_proj_a(hidden_states).view(1, seq_len, self.num_v_heads)

        if conv_state is None:
            conv_state = torch.zeros(
                1, self.conv_dim, self.conv_kernel_size - 1, device=hidden_states.device, dtype=mixed_qkv.dtype
            )
        else:
            conv_state = conv_state.unsqueeze(0).to(device=hidden_states.device, dtype=mixed_qkv.dtype)

        mixed_qkv = _torch_causal_conv1d_update(
            mixed_qkv,
            conv_state,
            self.conv1d_weight.squeeze(1),
            bias=None,
        )
        mixed_qkv = mixed_qkv.transpose(1, 2)  # [1, L, C]
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(1, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.reshape(1, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(1, seq_len, self.num_v_heads, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp()[None, None, :] * F.softplus(a.float() + self.dt_bias[None, None, :])
        if self.num_v_heads // self.num_k_heads > 1:
            rep = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)

        if recurrent_state is not None and seq_len == 1:
            core_attn_out, last_recurrent_state = _torch_recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state.unsqueeze(0),
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out, last_recurrent_state = _torch_chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state.unsqueeze(0) if recurrent_state is not None else None,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z_flat = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z_flat)
        core_attn_out = core_attn_out.reshape(seq_len, self.value_dim)
        output = self.out_proj(core_attn_out)

        new_conv_state = conv_state.squeeze(0).detach()
        new_recurrent_state = last_recurrent_state.squeeze(0).detach()
        return output, new_conv_state, new_recurrent_state

    def _record_pending(self, seq_id: int, slot: int, conv_state: torch.Tensor, recurrent_state: torch.Tensor):
        if slot < 0:
            return
        pending = self._pending.setdefault(seq_id, {})
        pending[slot] = (conv_state.detach().clone(), recurrent_state.detach().clone())
        if len(pending) > 64:
            first_key = next(iter(pending))
            pending.pop(first_key)

    def _commit(self, seq_id: int, slot: int, conv_state: torch.Tensor, recurrent_state: torch.Tensor):
        if slot < 0:
            return
        self._committed[seq_id] = (slot, conv_state.detach().clone(), recurrent_state.detach().clone())
        self._pending[seq_id] = {slot: (conv_state.detach().clone(), recurrent_state.detach().clone())}

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        del positions  # linear attention path does not use RoPE in the custom kernel path.
        context = get_context()
        if context.seq_ids is None:
            raise RuntimeError("Qwen35LinearAttention requires seq_ids in context.")

        seq_ids = context.seq_ids
        if context.cu_seqlens_q is not None:
            cu = context.cu_seqlens_q
            batch_size = cu.numel() - 1
        else:
            batch_size = seq_ids.numel()
            cu = torch.arange(batch_size + 1, dtype=torch.int32, device=hidden_states.device)

        verify_mode = (not context.is_prefill) and (context.cu_seqlens_q is not None)
        outputs: list[torch.Tensor] = []
        for i in range(batch_size):
            s = int(cu[i].item())
            e = int(cu[i + 1].item())
            if e <= s:
                continue
            seq_id = int(seq_ids[i].item())
            seq_hidden = hidden_states[s:e]
            start_pos = 0
            if context.context_lens is not None and context.cu_seqlens_q is not None:
                seqlen_q = e - s
                start_pos = int(context.context_lens[i].item()) - seqlen_q
            elif context.cu_seqlens_q is None and context.context_lens is not None:
                start_pos = int(context.context_lens[i].item()) - 1
            conv_state, recurrent_state = self._get_initial_states(seq_id, i, start_pos, context)

            if verify_mode:
                seq_out = []
                for t in range(seq_hidden.size(0)):
                    out_t, conv_state, recurrent_state = self._run_segment(seq_hidden[t : t + 1], conv_state, recurrent_state)
                    seq_out.append(out_t)
                    slot = int(context.slot_mapping[s + t].item()) if context.slot_mapping is not None else -1
                    self._record_pending(seq_id, slot, conv_state, recurrent_state)
                outputs.append(torch.cat(seq_out, dim=0))
            else:
                seq_out, conv_state, recurrent_state = self._run_segment(seq_hidden, conv_state, recurrent_state)
                outputs.append(seq_out)
                last_slot = int(context.slot_mapping[e - 1].item()) if context.slot_mapping is not None else -1
                self._commit(seq_id, last_slot, conv_state, recurrent_state)

        if len(outputs) == 0:
            return torch.empty_like(hidden_states)
        return torch.cat(outputs, dim=0)


class Qwen35TextMRotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_dim: int,
        max_position: int,
        rope_parameters: dict | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_position = max_position
        self.rope_parameters = rope_parameters or {}
        self.base = float(self.rope_parameters.get("rope_theta", 1000000.0))
        self.partial_rotary_factor = float(self.rope_parameters.get("partial_rotary_factor", 1.0))
        self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        if self.rotary_dim <= 0:
            raise ValueError(f"Invalid rotary_dim={self.rotary_dim} from head_dim={self.head_dim}")
        if self.rotary_dim % 2 != 0:
            # Keep rotary dimension even.
            self.rotary_dim -= 1

        if self.rotary_dim % 2 != 0:
            raise ValueError(f"rotary_dim must be even for RoPE, got {self.rotary_dim}")

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_position, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        self.register_buffer("cos_cache", freqs.cos(), persistent=False)
        self.register_buffer("sin_cache", freqs.sin(), persistent=False)

        default_mrope_section = _build_default_mrope_section(self.rotary_dim)

        self.mrope_section = self.rope_parameters.get("mrope_section", default_mrope_section)
        if not isinstance(self.mrope_section, (list, tuple)) or len(self.mrope_section) != 3:
            raise ValueError(f"mrope_section must be a length-3 list/tuple, got {self.mrope_section}")

    def _expand_standard_cache(self, max_pos: int, device: torch.device) -> None:
        if max_pos < self.cos_cache.size(0):
            return
        new_len = max_pos + 1
        t = torch.arange(new_len, dtype=torch.float32, device=device)
        inv_freq = self.inv_freq.to(device)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        self.cos_cache = freqs.cos()
        self.sin_cache = freqs.sin()

    def _standard_rope(self, positions: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        positions = positions.reshape(-1).long()
        max_pos = int(positions.max().item()) if positions.numel() > 0 else 0
        self._expand_standard_cache(max_pos, positions.device)
        cos = self.cos_cache[positions]
        sin = self.sin_cache[positions]
        emb_cos = torch.cat((cos, cos), dim=-1).to(dtype=dtype)
        emb_sin = torch.cat((sin, sin), dim=-1).to(dtype=dtype)
        return emb_cos, emb_sin

    def _mrope(self, positions: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        # positions: [3, N]
        positions = _normalize_mrope_positions(positions).float()
        inv_freq = self.inv_freq.to(device=positions.device, dtype=torch.float32)
        freqs = positions[..., None] * inv_freq[None, None, :]  # [3, N, D/2]

        freqs_t = freqs[0].clone()
        half = self.rotary_dim // 2
        for dim, offset in enumerate((1, 2), start=1):
            length = min(int(self.mrope_section[dim]) * 3, half)
            if length <= offset:
                continue
            idx = torch.arange(offset, length, 3, device=positions.device)
            freqs_t[..., idx] = freqs[dim, ..., idx]

        emb = torch.cat((freqs_t, freqs_t), dim=-1)
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)

    def forward(self, positions: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if positions.ndim == 1:
            return self._standard_rope(positions, dtype=dtype)

        norm_positions = _normalize_mrope_positions(positions)
        # If all 3 streams are equal, standard RoPE is equivalent and faster.
        if torch.equal(norm_positions[0], norm_positions[1]) and torch.equal(norm_positions[0], norm_positions[2]):
            return self._standard_rope(norm_positions[0], dtype=dtype)
        return self._mrope(norm_positions, dtype=dtype)


class Qwen35TextAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_parameters: dict | None = None,
        use_mrope: bool = False,
        # speculation args
        draft: bool = False,
        speculate: bool = False,
        spec_k: int = 1,
        async_fan_out: int = 1,
        draft_async: bool = False,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int = 1,
    ) -> None:
        super().__init__()
        self.draft = draft
        self.draft_async = draft_async
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.use_mrope = use_mrope

        self.total_num_heads = num_heads
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_heads % tp_size == 0
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_parameters = rope_parameters or {"rope_theta": 1000000.0, "rope_type": "default", "partial_rotary_factor": 1.0}

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            tp_group=self.tp_group,
            tp_size=self.tp_size,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            tp_group=self.tp_group,
            tp_size=self.tp_size,
        )

        self.mrotary_emb = Qwen35TextMRotaryEmbedding(
            head_dim=self.head_dim,
            max_position=max_position,
            rope_parameters=self.rope_parameters,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            draft=draft,
            speculate=speculate,
            draft_async=draft_async,
            F=async_fan_out,
            K=spec_k,
        )
        self.q_norm = RMSHeadNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSHeadNorm(self.head_dim, eps=rms_norm_eps)

    def _apply_rope(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mrotary_emb is None:
            raise RuntimeError("Qwen35TextAttention: mrotary_emb is None")

        rotary_dim = self.mrotary_emb.rotary_dim
        cos, sin = self.mrotary_emb(positions, dtype=q.dtype)

        num_tokens = q.shape[0]
        q_heads = q.view(num_tokens, -1, self.head_dim)
        k_heads = k.view(num_tokens, -1, self.head_dim)

        q_rot = q_heads[..., :rotary_dim].reshape(num_tokens, -1)
        k_rot = k_heads[..., :rotary_dim].reshape(num_tokens, -1)
        q_rot, k_rot = _apply_rotary_pos_emb_flat(q_rot, k_rot, cos, sin, rotary_dim)
        q_rot = q_rot.view(num_tokens, -1, rotary_dim)
        k_rot = k_rot.view(num_tokens, -1, rotary_dim)

        if rotary_dim < self.head_dim:
            q_pass = q_heads[..., rotary_dim:]
            k_pass = k_heads[..., rotary_dim:]
            q_heads = torch.cat((q_rot, q_pass), dim=-1)
            k_heads = torch.cat((k_rot, k_pass), dim=-1)
        else:
            q_heads = q_rot
            k_heads = k_rot

        return q_heads.reshape(q.shape), k_heads.reshape(k.shape)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q_by_head = q.reshape(-1, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.reshape(q.shape)

        k_by_head = k.reshape(-1, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.reshape(k.shape)

        q, k = self._apply_rope(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o)
        return output


class Qwen35MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int = 1,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            tp_group=tp_group,
            tp_size=tp_size,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            tp_group=tp_group,
            tp_size=tp_size,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen35DecoderLayer(nn.Module):

    def __init__(
        self,
        config,
        layer_idx: int,
        layer_type: str,
        draft: bool,
        speculate: bool,
        spec_k: int,
        async_fan_out: int,
        draft_async: bool,
        use_mrope: bool = False,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int = 1,
    ) -> None:
        super().__init__()
        self.layer_type = layer_type
        if self.layer_type == "full_attention":
            rope_parameters = _get_rope_parameters(config)
            self.self_attn = Qwen35TextAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                max_position=config.max_position_embeddings,
                rms_norm_eps=config.rms_norm_eps,
                qkv_bias=getattr(config, "attention_bias", False),
                head_dim=getattr(config, "head_dim", None),
                rope_parameters=rope_parameters,
                use_mrope=use_mrope,
                draft=draft,
                speculate=speculate,
                spec_k=spec_k,
                async_fan_out=async_fan_out,
                draft_async=draft_async,
                tp_group=tp_group,
                tp_size=tp_size,
            )
            self.linear_attn = None
        elif self.layer_type == "linear_attention":
            self.self_attn = None
            self.linear_attn = Qwen35LinearAttention(
                config=config,
                layer_idx=layer_idx,
                tp_group=tp_group,
                tp_size=tp_size,
            )
        else:
            raise ValueError(f"Unsupported layer_type '{self.layer_type}' for Qwen35DecoderLayer")

        self.mlp = Qwen35MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            tp_group=tp_group,
            tp_size=tp_size,
        )
        self.input_layernorm = RMSDNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSDNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        if self.layer_type == "linear_attention":
            assert self.linear_attn is not None
            hidden_states = self.linear_attn(positions, hidden_states)
        else:
            assert self.self_attn is not None
            hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen35TextModel(nn.Module):

    def __init__(
        self,
        config,
        draft: bool = False,
        speculate: bool = False,
        spec_k: int = 1,
        async_fan_out: int = 1,
        draft_async: bool = False,
        use_mrope: bool = False,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int = 1,
    ) -> None:
        super().__init__()
        self.config = config
        self.use_mrope = use_mrope
        self.layer_types = _get_layer_types(config)
        if len(self.layer_types) != config.num_hidden_layers:
            raise ValueError(
                f"Qwen35TextModel: layer_types length={len(self.layer_types)} "
                f"does not match num_hidden_layers={config.num_hidden_layers}"
            )
        self.has_linear_attention = any(layer_type == "linear_attention" for layer_type in self.layer_types)

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            draft_async=draft_async,
            tp_group=tp_group,
            tp_size=tp_size,
        )
        self.layers = nn.ModuleList(
            [
                Qwen35DecoderLayer(
                    config,
                    layer_idx=layer_idx,
                    layer_type=self.layer_types[layer_idx],
                    draft=draft,
                    speculate=speculate,
                    spec_k=spec_k,
                    async_fan_out=async_fan_out,
                    draft_async=draft_async,
                    use_mrope=use_mrope,
                    tp_group=tp_group,
                    tp_size=tp_size,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSDNorm(config.hidden_size, eps=config.rms_norm_eps)

    def reset_state_cache(self):
        for layer in self.layers:
            if getattr(layer, "linear_attn", None) is not None:
                layer.linear_attn.reset_state_cache()

    def _apply_deepstack(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        visual_pos_mask: torch.Tensor,
        visual_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        mask = visual_pos_mask.to(device=hidden_states.device, dtype=torch.bool).reshape(-1)
        if mask.numel() != hidden_states.size(0):
            raise ValueError(
                f"Qwen35TextModel._apply_deepstack: mask size {mask.numel()} "
                f"does not match token count {hidden_states.size(0)}"
            )
        num_visual = int(mask.sum().item())
        if num_visual != visual_embeds.size(0):
            raise ValueError(
                f"Qwen35TextModel._apply_deepstack: mask true count {num_visual} "
                f"!= visual_embeds length {visual_embeds.size(0)}"
            )
        visual_embeds = visual_embeds.to(device=hidden_states.device, dtype=hidden_states.dtype)
        if residual is None:
            hidden_states[mask] = hidden_states[mask] + visual_embeds
        else:
            residual[mask] = residual[mask] + visual_embeds
        return hidden_states, residual

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
        visual_pos_mask: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            if input_ids is None:
                raise ValueError("Qwen35TextModel.forward requires input_ids when input_embeds is None")
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        residual = None
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(positions, hidden_states, residual)
            if (
                deepstack_visual_embeds is not None
                and visual_pos_mask is not None
                and layer_idx < len(deepstack_visual_embeds)
            ):
                hidden_states, residual = self._apply_deepstack(
                    hidden_states,
                    residual,
                    visual_pos_mask,
                    deepstack_visual_embeds[layer_idx],
                )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen35ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen35TextConfig,
        draft: bool = False,
        speculate: bool = False,
        use_eagle: bool = False,
        spec_k: int = 1,
        async_fan_out: int = 1,
        draft_async: bool = False,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int = 1,
    ) -> None:
        super().__init__()
        self.draft = draft
        self.draft_async = draft_async
        self.tp_group = tp_group
        self.tp_size = tp_size

        assert not use_eagle, "ERROR in Qwen35ForCausalLM: use_eagle not supported for Qwen3.5 text path"
        assert not (tp_group is None and self.tp_size > 1), "ERROR in Qwen35ForCausalLM: tp_group is None and tp_size > 1"

        print(f"Starting Qwen35ForCausalLM init, draft={draft}, speculate={speculate}, spec_k={spec_k}")
        self.model = Qwen35TextModel(
            config=config,
            draft=draft,
            speculate=speculate,
            spec_k=spec_k,
            async_fan_out=async_fan_out,
            draft_async=draft_async,
            use_mrope=False,
            tp_group=tp_group,
            tp_size=self.tp_size,
        )
        self.requires_eager = self.model.has_linear_attention
        self.async_fan_out = async_fan_out
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            draft_async=draft_async,
            tp_group=tp_group,
            tp_size=self.tp_size,
        )
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data
        print(f"Finishing Qwen35ForCausalLM init, draft={draft}, speculate={speculate}, spec_k={spec_k}")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids=input_ids, positions=positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        last_only: bool = True,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states, last_only=last_only)

    def reset_state_cache(self):
        self.model.reset_state_cache()

class Qwen35VLVisionMLP(nn.Module):

    def __init__(self, config: Qwen35VisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        if config.hidden_act == "gelu_pytorch_tanh":
            self.act_fn = nn.GELU(approximate="tanh")
        elif config.hidden_act == "gelu":
            self.act_fn = nn.GELU()
        elif config.hidden_act == "silu":
            self.act_fn = nn.SiLU()
        else:
            raise ValueError(f"Unsupported vision hidden_act: {config.hidden_act}")

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class Qwen35VLVisionPatchEmbed(nn.Module):

    def __init__(self, config: Qwen35VisionConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class Qwen35VLVisionRotaryEmbedding(nn.Module):

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class Qwen35VLVisionPatchMerger(nn.Module):

    def __init__(self, config: Qwen35VisionConfig, use_postshuffle_norm: bool = False) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(
            self.hidden_size if use_postshuffle_norm else config.hidden_size,
            eps=1e-6,
        )
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.hidden_size)).view(-1, self.hidden_size)
        else:
            x = self.norm(x).view(-1, self.hidden_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


class Qwen35VLVisionAttention(nn.Module):

    def __init__(self, config: Qwen35VisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim, bias=True)
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        qkv = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, self.head_dim).permute(1, 0, 2, 3)
        query_states, key_states, value_states = qkv.unbind(0)
        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)
        attn_output = _segmented_vision_attention(
            query_states,
            key_states,
            value_states,
            cu_seqlens=cu_seqlens,
            scale=self.scaling,
            training=self.training,
            dropout_p=self.attention_dropout,
        )
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)


class Qwen35VLVisionBlock(nn.Module):

    def __init__(self, config: Qwen35VisionConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen35VLVisionAttention(config=config)
        self.mlp = Qwen35VLVisionMLP(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen35VLVisionModel(nn.Module):

    def __init__(self, config: Qwen35VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen35VLVisionPatchEmbed(config=config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen35VLVisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([Qwen35VLVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen35VLVisionPatchMerger(config=config, use_postshuffle_norm=False)

        self.deepstack_visual_indexes = list(getattr(config, "deepstack_visual_indexes", []))
        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen35VLVisionPatchMerger(
                    config=config,
                    use_postshuffle_norm=True,
                )
                for _ in range(len(self.deepstack_visual_indexes))
            ]
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size
        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.device

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)
        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)

            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]
            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            coords = torch.stack((row_idx, col_idx), dim=-1)
            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]
        return embeddings.flatten(1)

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, int(h.item()), device=self.pos_embed.weight.device)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, int(w.item()), device=self.pos_embed.weight.device)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs_floor + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs_floor + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=self.pos_embed.weight.device)
        weight_tensor = torch.tensor(
            weight_list,
            dtype=self.pos_embed.weight.dtype,
            device=self.pos_embed.weight.device,
        )
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
        patch_pos_embeds = patch_pos_embeds.split([int(h.item() * w.item()) for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            t_i = int(t.item())
            h_i = int(h.item())
            w_i = int(w.item())
            pos_embed = pos_embed.repeat(t_i, 1)
            pos_embed = (
                pos_embed.view(
                    t_i,
                    h_i // merge_size,
                    merge_size,
                    w_i // merge_size,
                    merge_size,
                    -1,
                )
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        return torch.cat(patch_pos_embeds_permute, dim=0)

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        seq_len, _ = hidden_states.size()
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
            if layer_num in self.deepstack_visual_indexes:
                ds_idx = self.deepstack_visual_indexes.index(layer_num)
                deepstack_feature = self.deepstack_merger_list[ds_idx](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)

        hidden_states = self.merger(hidden_states)
        return hidden_states, deepstack_feature_lists

class Qwen35VLModel(nn.Module):

    def __init__(
        self,
        config: Qwen35Config,
        draft: bool = False,
        speculate: bool = False,
        spec_k: int = 1,
        async_fan_out: int = 1,
        draft_async: bool = False,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int = 1,
    ) -> None:
        super().__init__()
        self.config = config
        self.visual = Qwen35VLVisionModel(config.vision_config)
        self.language_model = Qwen35TextModel(
            config=config.text_config,
            draft=draft,
            speculate=speculate,
            spec_k=spec_k,
            async_fan_out=async_fan_out,
            draft_async=draft_async,
            use_mrope=True,
            tp_group=tp_group,
            tp_size=tp_size,
        )
        self.has_linear_attention = self.language_model.has_linear_attention
        self.image_token_id = config.image_token_id
        self.video_token_id = config.video_token_id
        self.vision_start_token_id = config.vision_start_token_id
        self.vision_end_token_id = config.vision_end_token_id

    def get_input_embeddings(self) -> VocabParallelEmbedding:
        return self.language_model.embed_tokens

    def set_input_embeddings(self, value: VocabParallelEmbedding) -> None:
        self.language_model.embed_tokens = value

    def reset_state_cache(self):
        self.language_model.reset_state_cache()

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        return self.get_image_features(pixel_values_videos, video_grid_thw)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if image_grid_thw is None:
            raise ValueError("Qwen35VLModel.get_image_features requires image_grid_thw")
        pixel_values = pixel_values.to(dtype=self.visual.dtype)
        image_embeds, deepstack_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = list(torch.split(image_embeds, split_sizes))
        return image_embeds, deepstack_image_embeds

    def get_placeholder_mask(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        image_features: torch.Tensor | None = None,
        video_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids is None:
            raise ValueError("Qwen35VLModel.get_placeholder_mask requires input_ids")

        special_image_mask = input_ids == self.image_token_id
        special_video_mask = input_ids == self.video_token_id

        n_image_tokens = int(special_image_mask.sum().item())
        n_video_tokens = int(special_video_mask.sum().item())

        special_image_mask_2d = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        special_video_mask_2d = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)

        if image_features is not None and inputs_embeds[special_image_mask_2d].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens={n_image_tokens}, "
                f"features={image_features.shape[0]}"
            )
        if video_features is not None and inputs_embeds[special_video_mask_2d].numel() != video_features.numel():
            raise ValueError(
                f"Video features and video tokens do not match: tokens={n_video_tokens}, "
                f"features={video_features.shape[0]}"
            )

        return special_image_mask_2d, special_video_mask_2d

    def get_rope_index_flat(
        self,
        input_ids: torch.Tensor,
        mm_token_type_ids: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 1:
            raise ValueError("Qwen35VLModel.get_rope_index_flat expects flattened 1D input_ids")

        seq_len = input_ids.numel()
        default_positions = torch.arange(seq_len, device=input_ids.device, dtype=torch.long).view(1, -1).expand(3, -1)
        if image_grid_thw is None and video_grid_thw is None:
            return default_positions

        # Qwen3.5 processor can supply token type ids: text=0 image=1 video=2.
        if mm_token_type_ids is not None:
            if mm_token_type_ids.ndim != 1 or mm_token_type_ids.numel() != seq_len:
                raise ValueError("mm_token_type_ids must be flattened and aligned with input_ids")
            if video_grid_thw is not None:
                video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0).clone()
                video_grid_thw[:, 0] = 1

            spatial_merge_size = self.config.vision_config.spatial_merge_size
            image_iter = iter(image_grid_thw) if image_grid_thw is not None else None
            video_iter = iter(video_grid_thw) if video_grid_thw is not None else None

            groups: list[tuple[int, int, int]] = []
            token_types = mm_token_type_ids.tolist()
            start = 0
            while start < len(token_types):
                t = int(token_types[start])
                end = start + 1
                while end < len(token_types) and int(token_types[end]) == t:
                    end += 1
                groups.append((t, start, end))
                start = end

            llm_pos_ids_list: list[torch.Tensor] = []
            current_pos = 0
            for modality_type, start_idx, end_idx in groups:
                if modality_type == 0:
                    text_len = end_idx - start_idx
                    if text_len > 0:
                        llm_pos_ids_list.append(
                            torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + current_pos
                        )
                        current_pos += text_len
                elif modality_type == 1:
                    if image_iter is None:
                        return default_positions
                    grid_thw = next(image_iter, None)
                    if grid_thw is None:
                        return default_positions
                    t, h, w = grid_thw
                    llm_grid_t = int(t.item())
                    llm_grid_h = int(h.item()) // spatial_merge_size
                    llm_grid_w = int(w.item()) // spatial_merge_size
                    vision_len = llm_grid_t * llm_grid_h * llm_grid_w
                    if vision_len != (end_idx - start_idx):
                        return default_positions
                    t_index = torch.full((vision_len,), current_pos, device=input_ids.device, dtype=torch.long)
                    h_index = (
                        torch.arange(current_pos, current_pos + llm_grid_h, device=input_ids.device, dtype=torch.long)
                        .repeat_interleave(llm_grid_w * llm_grid_t)
                    )
                    w_index = (
                        torch.arange(current_pos, current_pos + llm_grid_w, device=input_ids.device, dtype=torch.long)
                        .repeat(llm_grid_h * llm_grid_t)
                    )
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index], dim=0))
                    current_pos += max(llm_grid_h, llm_grid_w)
                elif modality_type == 2:
                    if video_iter is None:
                        return default_positions
                    grid_thw = next(video_iter, None)
                    if grid_thw is None:
                        return default_positions
                    t, h, w = grid_thw
                    llm_grid_t = int(t.item())
                    llm_grid_h = int(h.item()) // spatial_merge_size
                    llm_grid_w = int(w.item()) // spatial_merge_size
                    vision_len = llm_grid_t * llm_grid_h * llm_grid_w
                    if vision_len != (end_idx - start_idx):
                        return default_positions
                    t_index = torch.full((vision_len,), current_pos, device=input_ids.device, dtype=torch.long)
                    h_index = (
                        torch.arange(current_pos, current_pos + llm_grid_h, device=input_ids.device, dtype=torch.long)
                        .repeat_interleave(llm_grid_w * llm_grid_t)
                    )
                    w_index = (
                        torch.arange(current_pos, current_pos + llm_grid_w, device=input_ids.device, dtype=torch.long)
                        .repeat(llm_grid_h * llm_grid_t)
                    )
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index], dim=0))
                    current_pos += max(llm_grid_h, llm_grid_w)
                else:
                    return default_positions

            if len(llm_pos_ids_list) == 0:
                return default_positions
            llm_positions = torch.cat(llm_pos_ids_list, dim=1)
            if llm_positions.size(1) != seq_len:
                return default_positions
            return llm_positions

        # Backward-compatible token-id-driven fallback.
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.image_token_id
        video_token_id = self.video_token_id
        input_tokens = input_ids.tolist()
        image_count = sum(1 for t in input_tokens if t == image_token_id)
        video_count = sum(1 for t in input_tokens if t == video_token_id)
        if image_count == 0 and video_count == 0:
            return default_positions
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0).clone()
            video_grid_thw[:, 0] = 1
        image_index = 0
        video_index = 0
        remain_images = image_count
        remain_videos = video_count
        llm_pos_ids_list: list[torch.Tensor] = []
        st = 0
        for _ in range(image_count + video_count):
            if remain_images > 0:
                try:
                    ed_image = input_tokens.index(image_token_id, st)
                except ValueError:
                    ed_image = len(input_tokens) + 1
            else:
                ed_image = len(input_tokens) + 1
            if remain_videos > 0:
                try:
                    ed_video = input_tokens.index(video_token_id, st)
                except ValueError:
                    ed_video = len(input_tokens) + 1
            else:
                ed_video = len(input_tokens) + 1
            if ed_image > len(input_tokens) and ed_video > len(input_tokens):
                break
            if ed_image < ed_video:
                if image_grid_thw is None or image_index >= image_grid_thw.size(0):
                    return default_positions
                t, h, w = image_grid_thw[image_index]
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                if video_grid_thw is None or video_index >= video_grid_thw.size(0):
                    return default_positions
                t, h, w = video_grid_thw[video_index]
                video_index += 1
                remain_videos -= 1
                ed = ed_video
            llm_grid_t = int(t.item())
            llm_grid_h = int(h.item()) // spatial_merge_size
            llm_grid_w = int(w.item()) // spatial_merge_size
            text_len = ed - st
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            if text_len > 0:
                llm_pos_ids_list.append(
                    torch.arange(text_len, device=input_ids.device, dtype=torch.long).view(1, -1).expand(3, -1) + st_idx
                )
            t_index = torch.arange(llm_grid_t, device=input_ids.device).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
            h_index = torch.arange(llm_grid_h, device=input_ids.device).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
            w_index = torch.arange(llm_grid_w, device=input_ids.device).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
            llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w
        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                torch.arange(text_len, device=input_ids.device, dtype=torch.long).view(1, -1).expand(3, -1) + st_idx
            )
        if len(llm_pos_ids_list) == 0:
            return default_positions
        llm_positions = torch.cat(llm_pos_ids_list, dim=1)
        if llm_positions.size(1) != seq_len:
            return default_positions
        return llm_positions

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            if input_ids is None:
                raise ValueError("Qwen35VLModel.forward requires input_ids when input_embeds is None")
            input_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None
        deepstack_image_embeds = None
        deepstack_video_embeds = None

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(input_embeds.device, input_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds=input_embeds, image_features=image_embeds)
            input_embeds = input_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(input_embeds.device, input_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(input_ids, inputs_embeds=input_embeds, video_features=video_embeds)
            input_embeds = input_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            image_mask_1d = image_mask[..., 0]
            video_mask_1d = video_mask[..., 0]
            visual_pos_masks = image_mask_1d | video_mask_1d
            deepstack_visual_embeds = []
            image_mask_joint = image_mask_1d[visual_pos_masks]
            video_mask_joint = video_mask_1d[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            visual_pos_masks = image_mask[..., 0]
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            visual_pos_masks = video_mask[..., 0]
            deepstack_visual_embeds = deepstack_video_embeds

        if positions is None:
            if input_ids is None:
                positions = torch.arange(input_embeds.size(0), device=input_embeds.device, dtype=torch.long)
            else:
                positions = self.get_rope_index_flat(
                    input_ids,
                    mm_token_type_ids=mm_token_type_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                )

        outputs = self.language_model(
            input_ids=input_ids,
            positions=positions,
            input_embeds=input_embeds,
            visual_pos_mask=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        return outputs


class Qwen35VLForConditionalGeneration(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen35Config,
        draft: bool = False,
        speculate: bool = False,
        use_eagle: bool = False,
        spec_k: int = 1,
        async_fan_out: int = 1,
        draft_async: bool = False,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int = 1,
    ) -> None:
        super().__init__()
        self.draft = draft
        self.draft_async = draft_async
        self.tp_group = tp_group
        self.tp_size = tp_size

        assert not use_eagle, "ERROR in Qwen35VLForConditionalGeneration: use_eagle is not supported"
        assert not (tp_group is None and self.tp_size > 1), "ERROR in Qwen35VLForConditionalGeneration: tp_group is None and tp_size > 1"

        print(
            f"Starting Qwen35VLForConditionalGeneration init, draft={draft}, "
            f"speculate={speculate}, spec_k={spec_k}"
        )
        self.model = Qwen35VLModel(
            config=config,
            draft=draft,
            speculate=speculate,
            spec_k=spec_k,
            async_fan_out=async_fan_out,
            draft_async=draft_async,
            tp_group=tp_group,
            tp_size=self.tp_size,
        )
        self.requires_eager = self.model.has_linear_attention
        self.lm_head = ParallelLMHead(
            config.text_config.vocab_size,
            config.text_config.hidden_size,
            draft_async=draft_async,
            tp_group=tp_group,
            tp_size=self.tp_size,
        )
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.language_model.embed_tokens.weight.data
        print(
            f"Finishing Qwen35VLForConditionalGeneration init, draft={draft}, "
            f"speculate={speculate}, spec_k={spec_k}"
        )

    @property
    def language_model(self) -> Qwen35TextModel:
        return self.model.language_model

    @property
    def visual(self) -> Qwen35VLVisionModel:
        return self.model.visual

    def get_input_embeddings(self) -> VocabParallelEmbedding:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value: VocabParallelEmbedding) -> None:
        self.model.set_input_embeddings(value)

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        return self.model.get_image_features(pixel_values, image_grid_thw)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            input_embeds=input_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        last_only: bool = True,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states, last_only=last_only)

    def reset_state_cache(self):
        self.model.reset_state_cache()


__all__ = [
    "Qwen35ForCausalLM",
    "Qwen35TextModel",
    "Qwen35VLForConditionalGeneration",
    "Qwen35VLModel",
    "Qwen35VLVisionModel",
]
