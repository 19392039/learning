# Copyright (c) Tianjin University, Ltd. 2025. All rights reserved.

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig


KEY_DIM = 128
MAX_RUNTIME_INPUT_ELEMENTS = 25_000_000

# This is a bounded pairwise coverage matrix, not an unconstrained Cartesian
# product.  Case 0 reproduces the checked-in ATK JSON, case 1 reproduces the
# repository's case_fast_128 shape, and case 2 exercises the case_fast varlen
# mode with bounded random sequence lengths.  The remaining profiles cover
# every compiled kernel axis while keeping CPU golden generation and device
# memory practical for ATK.
CASE_PROFILES = (
    # name, qkv, mixed-g, fixed, B, HK, HV, T(metadata for varlen), BT, V,
    # scale, sequence-count, min-sequence-length, max-sequence-length
    dict(name="checked_fp16_fixed", qkv="fp16", mix=True, fixed=True,
         B=1, HK=4, HV=4, T=128, BT=64, V=128, scale=0.088),
    dict(name="repo_bf16_fixed_tail", qkv="bf16", mix=False, fixed=True,
         B=1, HK=2, HV=2, T=200, BT=128, V=128, scale=0.088),
    dict(name="repo_bf16_varlen_gqa", qkv="bf16", mix=True, fixed=False,
         B=1, HK=2, HV=4, T=512, BT=128, V=256, scale=0.088,
         seq_count=3, seq_min=16, seq_max=256),
    dict(name="fp16_fixed_long", qkv="fp16", mix=False, fixed=True,
         B=1, HK=4, HV=4, T=1024, BT=64, V=128, scale=0.088),
    dict(name="bf16_fixed_mixed", qkv="bf16", mix=True, fixed=True,
         B=2, HK=4, HV=4, T=1024, BT=64, V=128, scale=0.088),
    dict(name="fp16_fixed_v256_gqa", qkv="fp16", mix=True, fixed=True,
         B=1, HK=2, HV=4, T=512, BT=64, V=256, scale=0.0625),
    dict(name="fp16_varlen", qkv="fp16", mix=True, fixed=False,
         B=1, HK=4, HV=4, T=512, BT=64, V=128, scale=0.0625,
         seq_count=4, seq_min=16, seq_max=192),
    dict(name="bf16_varlen_same_dtype", qkv="bf16", mix=False, fixed=False,
         B=1, HK=8, HV=8, T=1024, BT=64, V=128, scale=0.0442,
         seq_count=4, seq_min=64, seq_max=256),
    dict(name="fp16_fixed_bt128_v256", qkv="fp16", mix=False, fixed=True,
         B=1, HK=2, HV=2, T=384, BT=128, V=256, scale=0.0625),
    dict(name="bf16_fixed_v256_gqa", qkv="bf16", mix=True, fixed=True,
         B=1, HK=2, HV=4, T=1024, BT=128, V=256, scale=0.03125),
    dict(name="fp16_fixed_ratio4", qkv="fp16", mix=True, fixed=True,
         B=2, HK=2, HV=8, T=256, BT=128, V=128, scale=0.0442),
    dict(name="fp16_varlen_bt128_v256", qkv="fp16", mix=False, fixed=False,
         B=1, HK=2, HV=4, T=512, BT=128, V=256, scale=0.03125,
         seq_count=3, seq_min=32, seq_max=192),
)


def _ceil_div(a, b):
    return (a + b - 1) // b


def _runtime_upper_numel(profile):
    """Upper bound for tensors materialized by executor.init_by_input_data."""
    if profile["fixed"]:
        tokens = profile["T"]
        chunks = _ceil_div(tokens, profile["BT"])
    else:
        tokens = profile["seq_count"] * profile["seq_max"]
        chunks = profile["seq_count"] * _ceil_div(
            profile["seq_max"], profile["BT"]
        )

    batch = profile["B"]
    hk = profile["HK"]
    hv = profile["HV"]
    value_dim = profile["V"]

    q_and_k = 2 * batch * hk * tokens * KEY_DIM
    v_do_dv = 3 * batch * hv * tokens * value_dim
    gate = batch * hv * tokens
    states = 2 * batch * hv * chunks * KEY_DIM * value_dim
    # ATK materializes w from JSON before the executor replaces the ABI input
    # with None. Count it so the bound covers generation and execution.
    w_seed = batch * hv * tokens * KEY_DIM
    return q_and_k + v_do_dv + gate + states + w_seed


def _validate_profile(profile):
    if profile["qkv"] not in ("fp16", "bf16"):
        raise ValueError("qkv must be fp16 or bf16")
    if profile["BT"] not in (64, 128):
        raise ValueError("chunk_size must be 64 or 128")
    if profile["V"] not in (128, 256):
        raise ValueError("V must be 128 or 256")
    if profile["HK"] <= 0 or profile["HV"] % profile["HK"] != 0:
        raise ValueError("HV must be a positive multiple of HK")
    if profile["B"] <= 0 or profile["T"] <= 0:
        raise ValueError("B and T must be positive")
    if not profile["fixed"]:
        if profile["B"] != 1:
            raise ValueError("variable-length mode requires B == 1")
        if profile["seq_count"] <= 0:
            raise ValueError("seq_count must be positive")
        if not (0 < profile["seq_min"] <= profile["seq_max"]):
            raise ValueError("invalid variable-length bounds")

    elements = _runtime_upper_numel(profile)
    if elements > MAX_RUNTIME_INPUT_ELEMENTS:
        raise ValueError(
            "profile %s expands to %d input elements (limit %d)"
            % (profile["name"], elements, MAX_RUNTIME_INPUT_ELEMENTS)
        )


for _profile in CASE_PROFILES:
    _validate_profile(_profile)


@GENERATOR_REGISTRY.register("generator_chunk_bwd_dqkwg")
class ChunkBwdDqkwgGenerator(CaseGenerator):
    def __init__(self, config):
        super().__init__(config)
        if len(self) != len(CASE_PROFILES):
            raise ValueError(
                "YAML dtype_numbers must equal CASE_PROFILES length: %d"
                % len(CASE_PROFILES)
            )

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        inputs = {input_config.name: input_config for input_config in case_config.inputs}
        profile = CASE_PROFILES[self.index - 1]

        qkv_type = profile["qkv"]
        g_type = "fp32" if profile["mix"] else qkv_type
        batch = profile["B"]
        qk_heads = profile["HK"]
        value_heads = profile["HV"]
        tokens = profile["T"]
        chunk_size = profile["BT"]
        value_dim = profile["V"]
        num_chunks = _ceil_div(tokens, chunk_size)

        inputs["scale"].range_values = profile["scale"]
        inputs["chunk_size"].range_values = chunk_size
        inputs["is_mix"].range_values = profile["mix"]
        inputs["is_fix"].range_values = profile["fixed"]
        inputs["use_exp2"].range_values = False
        inputs["transpose_state_layout"].range_values = False
        inputs["qkv_type"].range_values = qkv_type

        for name in ("q", "k", "v", "h", "do", "dh", "dv", "w"):
            inputs[name].dtype = qkv_type
        inputs["g"].dtype = g_type

        if profile["fixed"]:
            # The executor discards these two tensors in fixed mode. Preserve
            # the checked-in JSON metadata for direct contract comparison.
            inputs["cu_seqlens"].shape = [5]
            inputs["cu_seqlens"].range_values = [1, 100]
            inputs["chunk_indices"].shape = [2, 2]
            inputs["chunk_indices"].range_values = [1, 100]
        else:
            # These are positive sequence lengths, not cumulative endpoints.
            # The unchanged executor computes cumsum and chunk_indices.
            inputs["cu_seqlens"].shape = [profile["seq_count"]]
            inputs["cu_seqlens"].range_values = [
                profile["seq_min"],
                profile["seq_max"],
            ]
            inputs["chunk_indices"].shape = [1, 2]
            inputs["chunk_indices"].range_values = [0, 1]

        inputs["q"].shape = [batch, qk_heads, tokens, KEY_DIM]
        inputs["k"].shape = [batch, qk_heads, tokens, KEY_DIM]
        inputs["v"].shape = [batch, value_heads, tokens, value_dim]
        inputs["g"].shape = [batch, value_heads, tokens]
        inputs["h"].shape = [batch, value_heads, num_chunks, KEY_DIM, value_dim]
        inputs["do"].shape = [batch, value_heads, tokens, value_dim]
        inputs["dh"].shape = [batch, value_heads, num_chunks, KEY_DIM, value_dim]
        inputs["dv"].shape = [batch, value_heads, tokens, value_dim]
        inputs["w"].shape = [batch, value_heads, tokens, KEY_DIM]
        inputs["g_gamma"].shape = [1, 1]

        return case_config
