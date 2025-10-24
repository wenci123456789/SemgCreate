import os
import sys

import matplotlib
import numpy as np
import pandas as pd
from DataProcess import NinaPro

from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from utils.Methods.methods import pearson_CC, draw_graph_2c, savitzky_golay_smoothing
import time
from skimage import metrics
import sklearn.metrics as skmetrics
from utils.Methods.methods import avg_smoothing_np, get_smooth_curve
from scipy.signal import savgol_filter
# 检查是否在无头环境中运行，如果是则使用Agg后端
import os
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
sys.path.append("..")

normalization = "miu"

test_name = "S1"  # trained model name
model_name = "mamba"  # trained model type

test_subjects = ['S1']  # which subjects to test
# 'S1','S3','S5','S9','S10','S11','S13','S14','S21','S23','S24','S27','S29','S30','S33'
if "S0" in test_subjects:
    test_subjects.remove("S0")
    test_subjects = ['S1','S3','S5','S9','S10','S11','S13','S14','S21','S23','S24','S27','S29','S30','S33'] + test_subjects


def data_wrapper_leconv(data):
    return data.unsqueeze(3)





def estimation(test_subject):
    print("=" * 49 + test_subject + "=" * 49)

    emgtest_dir = f"../../semg-wenci/Data/featureset/{test_subject}_E2_A1_rms_test.h5"
    glovetest_dir = f"../../semg-wenci/Data/featureset/{test_subject}_E2_A1_glove_test.h5"

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
        data = batch_tr[0].squeeze(3).cuda()
        target = batch_tr[1].cuda()

        # output_test = model(data)[0]  # BERT-based/TCN
        # output_test, y_c, y_s, x_hat, = model(data)[0:4]  # BERT-based/TCN
        output_test= model(data)  # Convit_MDFA
        # output_test = model.inference(data)  # LCSN
        # output_test, hidden = model(data_wrapper_leconv(batch_tr[0].squeeze(3)).cuda(), hidden)
        output_test = output_test.view(output_test.shape[0],
                                       output_test.shape[2]).detach().cpu()#Convit_MDFA
        # output_test = output_test.view(output_test.shape[0],
        #                                output_test.shape[2]).detach().cpu()#TCN
        target = target.view(target.shape[0],
                             target.shape[2]).detach().cpu()

        output_predict = torch.cat([output_predict, output_test])
        output_target = torch.cat([output_target, target])

    output_target = output_target.numpy()
    output_predict = output_predict.numpy()

    # if model_name[0] == "s":
    # output_predict = avg_smoothing_np(5, output_predict)
    # 修复：由于 output_predict 已经是 numpy 数组，不能调用 .device 属性
    # 使用 savgol_filter 直接处理 numpy 数组

    output_predict = savgol_filter(output_predict, 9, 2, axis=0)
    # output_predict = savitzky_golay_smoothing(9,2,output_predict)
    nrmses = list()
    ccs = list()
    r2s = list()
    for i in range(10):
        NRMSE = metrics.normalized_root_mse( output_target[:, i], output_predict[:, i],normalization="min-max")
        CC = pearson_CC( output_target[:, i],output_predict[:, i])

        print("第{}个关节的cc{}".format(i, CC))
        r2 = skmetrics.r2_score( output_target[:, i],output_predict[:, i], multioutput="variance_weighted")
        nrmses.append(NRMSE)
        ccs.append(CC)
        r2s.append(r2)

    dp = pd.DataFrame({
        'Joint': [f'Joint {i + 1}' for i in range(10)],  # 关节名称
        'CC': ccs,
        'NRMSE': nrmses,
        'R²': r2s
    })
    if not os.path.exists(f'./result/s{model_name}_joint/'):
        os.makedirs(f'./result/s{model_name}_joint/')
    dp.to_excel(f'./result/s{model_name}_joint/{test_subject}_joint.xlsx', index=False)
    NRMSE = np.mean(nrmses)
    CC_pearson = np.mean(ccs)
    r2 = np.mean(r2s)
    std_nrmse = np.std(nrmses, ddof=1)
    std_cc = np.std(ccs)
    std_r2 = np.std(r2s, ddof=1)
    # CC = pearson_CC(output_predict, output_target)
    # NRMSE = metrics.normalized_root_mse(output_target, output_predict, normalization="min-max")
    # R2 = skmetrics.r2_score(output_target, output_predict, multioutput="variance_weighted")
    # rec = pearson_CC(x_true, x_produce)
    rec = -1
    smooth = 0
    for i in range(10):
        smooth += get_smooth_curve(output_predict[:, i])[0]
    smooth /= 10
    # R2 = skmetrics.r2_score(output_target, output_predict,)

    print(f"[*]CC:{CC_pearson},NRMSE:{NRMSE},R2:{r2}, Smooth:{smooth}, Recovery:{rec}")
    print(f"[*]CCstd:{std_cc}, NRMSEstd:{std_nrmse}, R2std:{std_r2}")
    print("-" * 100 + "\n")

    # if test_subject == "S1":
    fig = draw_graph_2c(output_predict, output_target)
    # fig = draw_graph(x_produce, x_true,12)
    # 创建 estimation 目录（如果不存在）
    if not os.path.exists(f"../result/estimation/{model_name}"):
        os.makedirs(f"../result/estimation/{model_name}")
    # 保存图表到 estimation 目录
    plt.savefig(f"../result/estimation/{model_name}/{subject}.pdf")

    return CC_pearson, NRMSE, r2,std_cc,std_nrmse,std_r2


if __name__ == "__main__":
    cclist = []
    mselist = []
    r2list = []
    stdcclist = []
    stdmselist = []
    stdr2list = []
    for subject in test_subjects:
        try:
            # 修复模型路径，使用正斜杠和正确的目录结构
            model = torch.load(f'../result/demo/miu/{subject}/{model_name}/model_best.pth')
            # model = torch.load(f'D:\ZWC\semg-wenci\models\\new\miu\{subject}\ssConvit_MDFA_FN\model_best.pth')
        except Exception as e:
            print(f"Error loading model for {subject}: {e}")
            print(f"Attempted to load model from: ../result/demo/miu/{subject}/{model_name}/model_best.pth")
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
    output_file = f'{model_name}_results.xlsx'
    df.to_excel(output_file, index=False)
    print("=" * 49 + "==" + "=" * 49)

    # print(
    #     f"[*]TaskCC:{sum(cclist) / len(cclist)},TaskNRMSE:{sum(mselist) / len(mselist)}, TaskstdC:{sum(stdc) / len(stdc)}, TaskstdC:{sum(stdn) / len(stdn)}")
