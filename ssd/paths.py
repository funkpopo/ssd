import os

# cuda arch for flashinfer kernel compilation. set this to match your gpu:
# "9.0" for H100/H200, "8.0" for A100, "8.9" for L40/4090, etc.
CUDA_ARCH = os.environ.get("SSD_CUDA_ARCH", "9.0")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", CUDA_ARCH)


def _optional_env(var_name: str) -> str | None:
    value = os.environ.get(var_name)
    return value if value else None


def _default_from_hf_cache(explicit_env_var: str, relative_hf_path: str) -> str:
    explicit = os.environ.get(explicit_env_var)
    if explicit:
        return explicit
    if HF_CACHE_DIR:
        return f"{HF_CACHE_DIR}/{relative_hf_path}"
    return ""


# root directory where huggingface model snapshots are stored. each model
# lives under this as models--org--name/snapshots/<hash>/. if you downloaded
# models with `huggingface-cli download`, this is your HF_HOME/hub directory.
HF_CACHE_DIR = _optional_env("SSD_HF_CACHE")

# default target and draft model snapshot paths. these are full paths to the
# snapshot directory containing config.json. override if your models live
# somewhere else or you want to use different model sizes.
DEFAULT_TARGET = _default_from_hf_cache(
    "SSD_TARGET_MODEL",
    "models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
)
DEFAULT_DRAFT = _default_from_hf_cache(
    "SSD_DRAFT_MODEL",
    "models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6",
)

# eagle3 draft model paths. override via env vars if your models live elsewhere.
EAGLE3_SPECFORGE_70B = _default_from_hf_cache(
    "SSD_EAGLE3_SPECFORGE_70B",
    "models--lmsys--SGLang-EAGLE3-Llama-3.3-70B-Instruct-SpecForge",
)
EAGLE3_YUHUILI_8B = _default_from_hf_cache(
    "SSD_EAGLE3_8B",
    "models--yuhuili--EAGLE3-LLaMA3.1-Instruct-8B",
)
EAGLE3_QWEN_32B = _default_from_hf_cache(
    "SSD_EAGLE3_QWEN_32B",
    "models--RedHatAI--Qwen3-32B-speculator.eagle3",
)

# directory containing preprocessed benchmark datasets (jsonl files).
# each dataset is a subdirectory with a file like humaneval_data_10000.jsonl.
# you can generate these with scripts/get_data_from_hf.py.
DATASET_DIR = _optional_env("SSD_DATASET_DIR")
DATASET_PATHS = (
    {
        "humaneval":     f"{DATASET_DIR}/humaneval/humaneval_data_10000.jsonl",
        "alpaca":        f"{DATASET_DIR}/alpaca/alpaca_data_10000.jsonl",
        "c4":            f"{DATASET_DIR}/c4/c4_data_10000.jsonl",
        "gsm":           f"{DATASET_DIR}/gsm8k/gsm8k_data_10000.jsonl",
        "ultrafeedback": f"{DATASET_DIR}/ultrafeedback/ultrafeedback_data_10000.jsonl",
    }
    if DATASET_DIR
    else {}
)
