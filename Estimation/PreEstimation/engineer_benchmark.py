# -*- coding: utf-8 -*-
"""
一些通用的模型工程侧性能测试工具：
- 统计参数量
- 推理延迟（latency）
- 吞吐量（throughput）
- 显存占用（GPU 上）

使用方式：
    from engineer_benchmark import (
        count_parameters,
        benchmark_latency,
        benchmark_throughput,
        benchmark_memory,
        benchmark_all,
    )

    model = ...   # 你的 Mamba / Transformer 模型
    x = torch.randn(1, 200, 12).to(device)  # 举例：batch=1, 序列长度=200, 通道=12

    benchmark_all(model, x)
"""

import time
import torch

from Model.EMGMambaAttentionAdapter import EMGMambaAdapter
from utils.sEMG_models.sEMG_LSTM import sEMG_LSTM
from utils.sEMG_models.sEMG_RoFormer import RoFormerEMG
from utils.sEMG_models.transformer import MAFN


# =========================
# 1. 参数量
# =========================

def count_parameters(model):
    """统计可训练参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =========================
# 2. 推理延迟（latency）
# =========================

@torch.no_grad()
def benchmark_latency(model, example_input, iters=200, warmup=50):
    """
    测试单次前向推理的平均延迟（ms）
    - model: 已经在对应 device 上的模型
    - example_input: 一个代表性的输入张量（建议 batch=1，用于在线场景）
    """
    model.eval()
    device = next(model.parameters()).device

    # 预热，让 cudnn/kernel 稳定
    for _ in range(warmup):
        _ = model(example_input)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.time()
    for _ in range(iters):
        _ = model(example_input)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    end = time.time()

    avg_latency = (end - start) / iters * 1000.0  # 转成 ms
    return avg_latency


# =========================
# 3. 吞吐量（throughput）
# =========================

@torch.no_grad()
def benchmark_throughput(model, example_input, iters=100, warmup=20):
    """
    测试吞吐量：samples / second
    - example_input: 可以用一个比较大的 batch，例如 [64, 200, 12]
    """
    model.eval()
    device = next(model.parameters()).device
    batch_size = example_input.size(0)

    # 预热
    for _ in range(warmup):
        _ = model(example_input)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.time()
    for _ in range(iters):
        _ = model(example_input)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    end = time.time()

    total_time = end - start
    samples = batch_size * iters
    throughput = samples / total_time
    return throughput


# =========================
# 4. 显存占用（peak memory）
# =========================

@torch.no_grad()
def benchmark_memory(model, example_input):
    """
    测试单次前向过程中的最大显存占用（MB）
    仅在 CUDA 下有效
    """
    device = next(model.parameters()).device
    if device.type != "cuda":
        print("⚠ benchmark_memory 仅在 GPU 上有意义（当前不是 cuda）")
        return 0.0

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    model.eval()
    _ = model(example_input)

    peak_mem = torch.cuda.max_memory_allocated(device) / 1024 ** 2  # MB
    return peak_mem


# =========================
# 5. 一键综合测试
# =========================

def benchmark_all(model, example_input, example_input_large=None):
    """
    一次性打印：
    - 参数量
    - latency (batch=1)
    - throughput (可选，大 batch)
    - peak memory
    """
    device = next(model.parameters()).device
    params = count_parameters(model)
    print(f"Device: {device}")
    print(f"#Params: {params} ({params / 1e6:.3f} M)")

    # latency：一般用 batch=1 的 example_input
    latency = benchmark_latency(model, example_input)
    print(f"Latency: {latency:.3f} ms / forward (batch={example_input.size(0)})")

    # throughput：如果没有传 large input，就用 example_input
    if example_input_large is None:
        example_input_large = example_input

    throughput = benchmark_throughput(model, example_input_large)
    print(f"Throughput: {throughput:.1f} samples/s (batch={example_input_large.size(0)})")

    # peak memory（只有 cuda 有用）
    peak_mem = benchmark_memory(model, example_input)
    if peak_mem > 0:
        print(f"Peak memory (forward): {peak_mem:.1f} MB")

    print("-" * 60)
    return {
        "params": params,
        "latency_ms": latency,
        "throughput_sps": throughput,
        "peak_mem_mb": peak_mem,
    }


# =========================
# 6. 简单示例（可选）
# =========================

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1 = MAFN(200, patch_size=1, in_c=1, num_classes=10, depth=4, num_heads=4, embed_dim=12,
                         attn_drop_ratio=0, drop_ratio=0.3).to(device)

    # 举例：表面肌电窗口 [B, 200, 12]
    x_small = torch.randn(1, 200, 12).to(device)   # 用于 latency/memory（在线）
    x_large = torch.randn(64, 200, 12).to(device)  # 用于 throughput（批量）

    benchmark_all(model1, x_small, x_large)
    model2 = EMGMambaAdapter(input_dim=12, output_dim=10).to(device)
    benchmark_all(model2, x_small, x_large)
    model3 = RoFormerEMG(
        input_dim=12, output_dim=10,
        d_model=120, num_layers=2, num_heads=5,
        use_mu_law=False
    ).to(device)
    benchmark_all(model3, x_small, x_large)
    model4 = sEMG_LSTM(vocab_size=200, hidden=128, n_layers=4).to(device)