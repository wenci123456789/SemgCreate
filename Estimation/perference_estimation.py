import torch
import time
import pandas as pd
from ptflops import get_model_complexity_info

# -----------------------------
# 配置
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
seq_len = 200
input_dim = 12
n_iters = 50

# 构造统一输入
dummy_input = torch.randn(batch_size, seq_len, input_dim, device=device)
dummy_subject = torch.zeros(batch_size, dtype=torch.long, device=device)

# -----------------------------
# 导入模型
# -----------------------------
from Model.EMGMambaAttention import EMGMambaAttenRegressor

models_to_eval = {
    "EMGMambaAttnAdapt": EMGMambaAttenRegressor(num_subjects=2),
    "OtherModel1": OtherModel1(),
    "OtherModel2": OtherModel2()
}


# -----------------------------
# 评估函数
# -----------------------------
def evaluate_model(model, model_name):
    model = model.to(device)
    model.eval()

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 热身
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input, dummy_subject)

    # 测时间
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    with torch.no_grad():
        for _ in range(n_iters):
            _ = model(dummy_input, dummy_subject)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    avg_time_per_sample = (end_time - start_time) / n_iters / batch_size * 1000  # ms

    # FLOPs
    try:
        flops, _ = get_model_complexity_info(
            model, (seq_len, input_dim),
            custom_args={'subject_id': dummy_subject},
            as_strings=True, print_per_layer_stat=False, verbose=False
        )
    except Exception as e:
        flops = "N/A"

    print(f"[{model_name}] Params: {total_params:,}, Trainable: {trainable_params:,}, "
          f"FLOPs: {flops}, Avg Time/sample: {avg_time_per_sample:.3f} ms")

    return {
        "Model": model_name,
        "Total Params": total_params,
        "Trainable Params": trainable_params,
        "FLOPs": flops,
        "Avg Time per Sample (ms)": round(avg_time_per_sample, 3)
    }


# -----------------------------
# 批量评估
# -----------------------------
results = []
for name, model in models_to_eval.items():
    result = evaluate_model(model, name)
    results.append(result)

# -----------------------------
# 输出表格
# -----------------------------
df = pd.DataFrame(results)
print("\n=== Model Comparison ===")
print(df)

# 保存到 CSV
df.to_csv("model_comparison.csv", index=False)
