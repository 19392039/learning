"""ATK executor for MinghuasLab/flash-attention-npu v3.

NPU node: calls the real custom Ascend extension.
CPU node: computes a float reference for ATK's single-benchmark comparison.
"""

import importlib
import math
import os
import sys
from pathlib import Path

import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


def _load_flash_attn_func():
    """Load the v3 API and report which installation layer is missing.

    ``flash_attn_npu_v3.__init__`` in the 0.2.0b1 source distribution does
    not re-export ``flash_attn_func``.  The stable source-level entry for
    this snapshot is ``flash_attn_npu_v3.flash_attn_interface``.
    """

    repo_root = os.environ.get("FLASH_ATTN_NPU_REPO")
    if repo_root:
        candidate = Path(repo_root).expanduser().resolve()
        if not candidate.is_dir():
            raise RuntimeError(
                f"FLASH_ATTN_NPU_REPO does not point to a directory: {candidate}"
            )
        candidate_text = str(candidate)
        if candidate_text not in sys.path:
            sys.path.insert(0, candidate_text)

    try:
        interface = importlib.import_module(
            "flash_attn_npu_v3.flash_attn_interface"
        )
    except ModuleNotFoundError as exc:
        if exc.name == "flash_attn_npu_v3":
            raise RuntimeError(
                "ATK's Python environment cannot find the flash_attn_npu_v3 "
                "package. Build/install v3 in the same Python environment as "
                "ATK, or export FLASH_ATTN_NPU_REPO=/absolute/path/to/"
                "flash-attention-npu and add it to PYTHONPATH. "
                f"ATK worker Python: {sys.executable}"
            ) from exc
        if exc.name == "flash_attn_npu_3":
            raise RuntimeError(
                "The Python package is visible, but its compiled NPU extension "
                "flash_attn_npu_3 is missing. Rebuild with "
                "FLASH_ATTN_BUILD_VERSION=v3 using the same torch/torch_npu/CANN "
                f"environment. ATK worker Python: {sys.executable}"
            ) from exc
        raise

    try:
        return interface.flash_attn_func
    except AttributeError as exc:
        raise RuntimeError(
            "flash_attn_npu_v3.flash_attn_interface was imported, but it has no "
            "flash_attn_func. Check the installed package version and source tree."
        ) from exc


def _repeat_kv_for_gqa(x: torch.Tensor, query_heads: int) -> torch.Tensor:
    """Expand [B, Hkv, S, D] to [B, Hq, S, D] for MQA/GQA."""
    kv_heads = x.shape[1]
    if query_heads == kv_heads:
        return x
    if kv_heads <= 0 or query_heads % kv_heads != 0:
        raise ValueError(
            f"Q heads ({query_heads}) must be divisible by KV heads ({kv_heads})."
        )
    return x.repeat_interleave(query_heads // kv_heads, dim=1)


def _cpu_flash_attention_reference(q, k, v, **kwargs):
    """Mathematical reference matching flash_attn_func's BSHD interface."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k and v must all be rank-4 BSHD tensors.")
    if q.shape[0] != k.shape[0] or k.shape[0] != v.shape[0]:
        raise ValueError("q, k and v must have the same batch size.")
    if k.shape[1] != v.shape[1] or k.shape[2] != v.shape[2]:
        raise ValueError("k and v must have the same sequence and head counts.")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must have the same head dimension.")

    unsupported = {
        "qv": kwargs.get("qv"),
        "q_descale": kwargs.get("q_descale"),
        "k_descale": kwargs.get("k_descale"),
        "v_descale": kwargs.get("v_descale"),
    }
    used_unsupported = [name for name, value in unsupported.items() if value is not None]
    if used_unsupported:
        raise NotImplementedError(
            "CPU benchmark does not implement: " + ", ".join(used_unsupported)
        )
    if int(kwargs.get("attention_chunk", 0)) != 0:
        raise NotImplementedError("CPU benchmark does not implement attention_chunk.")

    softmax_scale = kwargs.get("softmax_scale")
    causal = bool(kwargs.get("causal", False))
    window_size = kwargs.get("window_size", (-1, -1))
    softcap = float(kwargs.get("softcap", 0.0))
    return_attn_probs = bool(kwargs.get("return_attn_probs", False))

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    else:
        softmax_scale = float(softmax_scale)

    # BSHD -> BHSD. Compute the reference in fp32 for fp16/bf16 inputs.
    qh = q.transpose(1, 2).float()
    kh = k.transpose(1, 2).float()
    vh = v.transpose(1, 2).float()
    query_heads = qh.shape[1]
    kh = _repeat_kv_for_gqa(kh, query_heads)
    vh = _repeat_kv_for_gqa(vh, query_heads)

    scores = torch.matmul(qh, kh.transpose(-2, -1)) * softmax_scale
    if softcap > 0.0:
        scores = softcap * torch.tanh(scores / softcap)

    seqlen_q = q.shape[1]
    seqlen_k = k.shape[1]
    offset = seqlen_k - seqlen_q
    q_index = torch.arange(seqlen_q, device=q.device).view(-1, 1)
    k_index = torch.arange(seqlen_k, device=q.device).view(1, -1)
    keep = torch.ones((seqlen_q, seqlen_k), dtype=torch.bool, device=q.device)

    if causal:
        # flash-attn aligns unequal-length causal masks to the bottom-right.
        keep &= k_index <= q_index + offset

    if window_size is None:
        window_size = (-1, -1)
    left, right = tuple(window_size)
    if left >= 0:
        keep &= k_index >= q_index + offset - int(left)
    if right >= 0:
        keep &= k_index <= q_index + offset + int(right)

    scores = scores.masked_fill(~keep.view(1, 1, seqlen_q, seqlen_k), -torch.inf)
    softmax_lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1)
    # A completely masked row must produce zero, rather than NaN.
    probs = torch.nan_to_num(probs, nan=0.0)
    out = torch.matmul(probs, vh).transpose(1, 2).contiguous().to(q.dtype)

    if return_attn_probs:
        return out, softmax_lse
    return out


@register("flash_attn_npu_v3")
class FlashAttnNpuV3Api(BaseApi):
    """Select the real NPU operator or the CPU mathematical benchmark."""

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        args = list(input_data.args)
        kwargs = dict(input_data.kwargs)
        backend = str(self.device).lower()

        if backend == "npu":
            # Import only in the NPU worker: the compiled extension is NPU-only.
            import torch_npu  # noqa: F401  # Registers the torch "npu" device.
            flash_attn_func = _load_flash_attn_func()
            output = flash_attn_func(*args, **kwargs)
        elif backend == "cpu":
            if len(args) < 3:
                raise ValueError("flash_attn_func requires positional q, k and v inputs.")
            q, k, v = args[:3]
            output = _cpu_flash_attention_reference(q, k, v, **kwargs)
        else:
            raise NotImplementedError(
                f"Backend {self.device!r} is not supported by this ATK executor."
            )

        return output if with_output else None
