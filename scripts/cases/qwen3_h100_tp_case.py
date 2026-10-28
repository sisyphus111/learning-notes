#!/usr/bin/env python3
"""Recompute the note's ideal BF16 GEMM/TP costs; no GPU measurements."""
import json

K, N, TP = 5120, 25600, 4
HBM_BYTES_S, DENSE_FLOPS_S = 3.35e12, 989e12
NVLINK_ONEWAY_BYTES_S = 450e9
rows, ffn = [], []
for m in (1, 32, 128, 512, 1024):
    ops = 2 * m * K * N
    data = 2 * (m * K + K * N + m * N)
    shard_n = N // TP
    shard_ops = 2 * m * K * shard_n
    shard_data = 2 * (m * K + K * shard_n + m * shard_n)
    compute = ops / DENSE_FLOPS_S
    memory = data / HBM_BYTES_S
    local = max(shard_ops / DENSE_FLOPS_S, shard_data / HBM_BYTES_S)
    rows.append(dict(M=m, intensity=ops / data, compute_us=compute * 1e6,
                     hbm_us=memory * 1e6, model_us=max(compute, memory) * 1e6,
                     tp4_intensity=shard_ops / shard_data, tp4_model_us=local * 1e6))
    if m in (1, 128, 1024):
        payload = 2 * m * K
        ring_bytes = 2 * (TP - 1) / TP * payload
        ffn.append(dict(M=m, tp1_matmuls_us=3 * max(compute, memory) * 1e6,
                        tp4_matmuls_us=3 * local * 1e6,
                        ring_bandwidth_only_us=ring_bytes / NVLINK_ONEWAY_BYTES_S * 1e6))

print(json.dumps(dict(
    evidence="Calculated ideal model, not measured latency or throughput",
    ridge_flop_per_byte=DENSE_FLOPS_S / HBM_BYTES_S,
    weight_bytes=2 * K * N, projection=rows, ffn=ffn,
    tile=dict(BM=128, BN=128, BK=64, stage_input_bytes=32768,
              double_buffer_bytes=65536, accumulator_registers_per_consumer=128,
              example_registers_per_cta=128 * 32 + 128 * 160),
    kv_bytes_per_history_token=2 * 64 * 8 * 128 * 2,
), ensure_ascii=False, indent=2))
