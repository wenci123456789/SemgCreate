import os
import sys

import matplotlib
import numpy as np
import pandas as pd
from DataProcess import NinaPro

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from utils.BaseModels.BERT.Bert import BERT
from utils.Methods.methods import pearson_CC, draw_graph_2c, savitzky_golay_smoothing
import time
from skimage import metrics
import sklearn.metrics as skmetrics
from utils.Methods.methods import avg_smoothing_np, get_smooth_curve
from utils.sEMG_models.sEMG_LSTM import sEMG_LSTM
from utils.sEMG_models.sEMG_RoFormer import RoFormerEMG
from utils.sEMG_models.sEMG_TCN import sEMG_TCN
from utils.sEMG_models.transformer import MAFN

matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.append("..")
from Model.EMGMambaAttentionAdapter import EMGMambaAdapter
normalization = "miu"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = "Roformer"  # sEMGMamba Roformer LSTM Transformer
trans_method = "noft" #cdanrpp atl ft noft
test_subjects = [f"S{i}" for i in range(31, 41)]  # which subjects to test
# 'S1','S3','S5','S9','S10','S11','S13','S14','S21','S23','S24','S27','S29','S30','S33'
if "S0" in test_subjects:
    test_subjects.remove("S0")
    test_subjects = ['S1','S3','S5','S9','S10','S11','S13','S14','S21','S23','S24','S27','S29','S30','S33'] + test_subjects


def data_wrapper_leconv(data):
    return data.unsqueeze(3)


def load_state_flex(ckpt_path: str):
    """
    从 cdanrpp_best.pth 这类 checkpoint 中取出 state_dict。
    兼容 {'model_state': ...} / 直接 state_dict / 整模型三种情况。
    """
    obj = torch.load(ckpt_path, map_location='cpu')
    if isinstance(obj, dict) and 'model_state' in obj:
        return obj['model_state']
    if hasattr(obj, 'state_dict'):  # 保存的是整模型
        return obj.state_dict()
    return obj  # 兜底：已经是 state_dict


def estimation(test_subject):
    print("=" * 49 + test_subject + "=" * 49)
    emgtest_dir = f"../../../feature/ninapro_db2_trans/{test_subject}_E2_A1_rms_test.h5"
    glovetest_dir = f"../../../feature/ninapro_db2_trans/{test_subject}_E2_A1_glove_test.h5"

    data_read_test = NinaPro.NinaPro(emgtest_dir, glovetest_dir, window_size=200, subframe=200,
                                     normalization=normalization, mu=2 ** 20, dummy_label=0, class_num=1, )
    # dummy_tsk=model.task_num - 1, tsk_num=model.task_num)
    loader_test = DataLoader(dataset=data_read_test, batch_size=32, shuffle=False, drop_last=True)
    output_predict = torch.Tensor([])
    output_target = torch.Tensor([])
    x_produce = torch.Tensor([])
    x_true = torch.Tensor([])
    model.eval()
    hidden = None
    print(len(loader_test))
    for step, batch_tr in tqdm(enumerate(loader_test), total=len(loader_test)):
        start_time = time.time()
        # x_true = torch.cat([x_true, batch_tr[0].permute(0,2,1).squeeze().detach().cpu()])
        x_true = torch.cat([x_true, batch_tr[0].squeeze().detach().cpu()])
        data = batch_tr[0].squeeze(3).to(device)
        target = batch_tr[1].to(device)
        output_test= model(data)  # Convit_MDFA
        if isinstance(output_test, (tuple, list)):
            output_test = output_test[0]
        if output_test.dim() == 3:
            output_test = output_test.mean(dim=1)
        output_test = output_test.detach().cpu()
        target = target.view(target.shape[0],
                             target.shape[2]).detach().cpu()

        output_predict = torch.cat([output_predict, output_test])
        output_target = torch.cat([output_target, target])

    output_predict = output_predict.detach().cpu().numpy()
    output_target = output_target.detach().cpu().numpy()

    # 如果需要平滑就保留
    if trans_method == "cdanrpp":
        output_predict = savitzky_golay_smoothing(9, 2, output_predict)

    # ========= ① 每个被试“整体”指标（10 个关节一起算） =========
    # 这里直接用全矩阵 output_target.shape = [N, 10], output_predict.shape = [N, 10]

    # 整体 NRMSE（对 10 维一起算）
    NRMSE = metrics.normalized_root_mse(
        output_target,
        output_predict,
        normalization="min-max"
    )

    # 整体 CC（把所有维度 flatten 之后算一个 Pearson）
    CC_pearson = pearson_CC(output_target, output_predict)

    # 整体 R²（对 10 维同时算，variance_weighted）
    r2 = skmetrics.r2_score(
        output_target,
        output_predict,
        multioutput="variance_weighted"
    )

    # ========= ② 如果你还想要“关节间的标准差”可以在内部算，不再输出每关节 =========
    nrmses = []
    ccs = []
    r2s = []
    for i in range(10):
        nrmses.append(
            metrics.normalized_root_mse(
                output_target[:, i],
                output_predict[:, i],
                normalization="min-max"
            )
        )
        ccs.append(pearson_CC(output_target[:, i], output_predict[:, i]))
        r2s.append(
            skmetrics.r2_score(
                output_target[:, i],
                output_predict[:, i],
                multioutput="variance_weighted"
            )
        )
    std_nrmse = np.std(nrmses, ddof=1)
    std_cc = np.std(ccs, ddof=1)
    std_r2 = np.std(r2s, ddof=1)

    # ========= ③ 不再保存每个关节的 excel，不再打印每个关节 =========
    # 直接打印这个被试的整体结果
    rec = -1
    smooth = 0
    for i in range(10):
        smooth += get_smooth_curve(output_predict[:, i])[0]
    smooth /= 10

    print(f"[*]{test_subject} CC:{CC_pearson}, NRMSE:{NRMSE}, R2:{r2}, Smooth:{smooth}, Recovery:{rec}")
    print(f"[*]{test_subject} CCstd:{std_cc}, NRMSEstd:{std_nrmse}, R2std:{std_r2}")
    print("-" * 100 + "\n")

    fig = draw_graph_2c(output_predict, output_target)
    if not os.path.exists(f"/mnt/data_nvme/zwc/semg-code/resultFinal/ninapro/{model_name}/{trans_method}"):
        os.makedirs(f"/mnt/data_nvme/zwc/semg-code/resultFinal/ninapro/{model_name}/{trans_method}")
    plt.savefig(f"/mnt/data_nvme/zwc/semg-code/resultFinal/ninapro/{model_name}/{trans_method}/{test_subject}.pdf")

    return CC_pearson, NRMSE, r2, std_cc, std_nrmse, std_r2


if __name__ == "__main__":
    cclist = []
    mselist = []
    r2list = []
    stdcclist = []
    stdmselist = []
    stdr2list = []
    for subject in test_subjects:
        try:
            if trans_method == "noft":
                ckpt = f'../result/ninapro/checkpoints_pretrain/{model_name}/model_best.pth'
            else:
                if model_name == "Roformer":
                    ckpt = f'../result/ninapro/Estimation_result/{model_name}/checkpoints_{trans_method}/{trans_method}_roformer_{subject}/{trans_method}_best.pth'
                else:
                    ckpt = f'../result/ninapro/Estimation_result/{model_name}/checkpoints_{trans_method}/{trans_method}_{subject}/{trans_method}_best.pth'
            state = load_state_flex(ckpt)
            if model_name== "sEMGMamba":
                model = EMGMambaAdapter(input_dim=12, output_dim=10).to(device)
            elif model_name =="BERT":
                model = BERT(vocab_size=200, hidden=128, feature_dim=1, n_layers=4, attn_heads=8).to(device)
            elif model_name == "Roformer":
                model = RoFormerEMG(input_dim=12, output_dim=10, d_model=120, num_layers=2, num_heads=5, use_mu_law=False).to(device)
            elif model_name == "LSTM":
                model = sEMG_LSTM(vocab_size=200, hidden=128, n_layers=4).to(device)
            elif model_name == "TCN":
                model = sEMG_TCN(12, [128, 128, 128, 128, 10], 3, 0.7).to(device)
            elif model_name == "Transformer":
                model = MAFN(200, patch_size=1, in_c=1, num_classes=10, depth=4, num_heads=4, embed_dim=12,
                         attn_drop_ratio=0, drop_ratio=0.3).to(device)
            else:
                print("模型名称错误!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            res = model.load_state_dict(state, strict=False)
            model.eval()
        except Exception as e:
            raise e
        cc, nrmse, r2 ,std_cc,std_nrmse,std_r2= estimation(subject)
        cclist.append(cc)
        mselist.append(nrmse)
        r2list.append(r2)
        stdcclist.append(std_cc)
        stdmselist.append(std_nrmse)
        stdr2list.append(std_r2)
    df = pd.DataFrame({
        'Subject': test_subjects,
        'NRMSE': mselist,
        'Pearson CC': cclist,
        'R²': r2list,
        'stdNRMSE': stdmselist,
        'stdPearson CC': stdcclist,
        'stdR²': stdr2list

    })
    output_file = f'/mnt/data_nvme/zwc/semg-code/resultFinal/ninapro/{model_name}/{trans_method}/{model_name}_{trans_method}_results.xlsx'
    df.to_excel(output_file, index=False)
    print("=" * 49 + "==" + "=" * 49)

    # print(
    #     f"[*]TaskCC:{sum(cclist) / len(cclist)},TaskNRMSE:{sum(mselist) / len(mselist)}, TaskstdC:{sum(stdc) / len(stdc)}, TaskstdC:{sum(stdn) / len(stdn)}")
