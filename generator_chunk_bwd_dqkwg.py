# Copyright (c) Tianjin University, Ltd. 2025. All rights reserved.

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig


QKV_DTYPES = ("bf16", "fp16")
CHUNK_SIZES = (64, 128)
VALUE_DIMS = (128, 256)
HEAD_RATIOS = (1, 2)

FIXED_BATCH = 1
QK_HEADS = 4
FIXED_TOKENS = 128
KEY_DIM = 128
VARLEN_SEQUENCE_COUNT = 4
VARLEN_MIN_LENGTH = 16
VARLEN_MAX_LENGTH = 64


@GENERATOR_REGISTRY.register("generator_chunk_bwd_dqkwg")
class ChunkBwdDqkwgGenerator(CaseGenerator):
    def __init__(self, config):
        super().__init__(config)

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        inputs = {input_config.name: input_config for input_config in case_config.inputs}
        case_index = self.index - 1

        # 64 generated cases cover the complete Cartesian product of the six
        # binary dimensions below.  Keeping this deterministic also makes a
        # regenerated JSON reviewable and prevents random oversized cases.
        qkv_type = QKV_DTYPES[case_index & 1]
        chunk_size = CHUNK_SIZES[(case_index >> 1) & 1]
        is_mix = (True, False)[(case_index >> 2) & 1]
        is_fix = (True, False)[(case_index >> 3) & 1]
        value_dim = VALUE_DIMS[(case_index >> 4) & 1]
        head_ratio = HEAD_RATIOS[(case_index >> 5) & 1]

        inputs["chunk_size"].range_values = chunk_size
        inputs["is_mix"].range_values = is_mix
        inputs["is_fix"].range_values = is_fix
        inputs["qkv_type"].range_values = qkv_type

        for name in ("q", "k", "v", "h", "do", "dh", "dv", "w"):
            inputs[name].dtype = qkv_type
        if not is_mix:
            inputs["g"].dtype = qkv_type

        batch = FIXED_BATCH
        qk_heads = QK_HEADS
        value_heads = qk_heads * head_ratio
        tokens = FIXED_TOKENS
        num_chunks = (tokens + chunk_size - 1) // chunk_size

        # The executor discards both tensors for fixed-length cases.  Retain the
        # checked-in smoke case metadata there, while bounding variable-length
        # seed lengths so they cannot expand to tens of thousands of tokens.
        if is_fix:
            inputs["cu_seqlens"].shape = [5]
            inputs["cu_seqlens"].range_values = [1, 100]
            inputs["chunk_indices"].shape = [2, 2]
            inputs["chunk_indices"].range_values = [1, 100]
        else:
            inputs["cu_seqlens"].shape = [VARLEN_SEQUENCE_COUNT]
            inputs["cu_seqlens"].range_values = [
                VARLEN_MIN_LENGTH,
                VARLEN_MAX_LENGTH,
            ]
            inputs["chunk_indices"].shape = [1, 2]
            inputs["chunk_indices"].range_values = [0, 1]

        # q, k: [B, HK, T, K]; v, dox, dv: [B, HV, T, V]
        # g: [B, HV, T]; h, dh: [B, HV, num_chunks, K, V]
        # outputs: dq, dk, dw: [B, HV, T, K]; dg: [B, HV, T]
        inputs["q"].shape = [batch, qk_heads, tokens, KEY_DIM]
        inputs["k"].shape = [batch, qk_heads, tokens, KEY_DIM]
        inputs["v"].shape = [batch, value_heads, tokens, value_dim]
        inputs["g"].shape = [batch, value_heads, tokens]
        inputs["h"].shape = [
            batch,
            value_heads,
            num_chunks,
            KEY_DIM,
            value_dim,
        ]
        inputs["do"].shape = [batch, value_heads, tokens, value_dim]
        inputs["dh"].shape = [
            batch,
            value_heads,
            num_chunks,
            KEY_DIM,
            value_dim,
        ]
        inputs["dv"].shape = [batch, value_heads, tokens, value_dim]
        inputs["w"].shape = [batch, value_heads, tokens, KEY_DIM]
        inputs["g_gamma"].shape = [1, 1]

        return case_config
