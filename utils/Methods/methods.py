import argparse
import math

import numpy as np
import sklearn.metrics
import torch
from scipy.signal import savgol_filter
from torch import nn
import matplotlib.pyplot as plt

import torch.nn.functional as F

"""=========================Feature Extraction============================="""


def rms(data):
    return np.sqrt((np.sum(data ** 2)) / data.shape[0])


def m0(data):
    return np.log(np.sqrt(np.sum(data ** 2)))


def m2(data):
    i_array = np.array([i + 1 for i in range(data.shape[0])])
    m2_temp = np.sum((i_array * data) ** 2)
    # for i in range(data.shape[0]):
    #     m2_temp = m2_temp + ((i + 1) * data[i]) ** 2
    return np.sqrt(m2_temp / data.shape[0])


def m4(data):
    i_array = np.array([i + 1 for i in range(data.shape[0])])
    m4_temp = np.sum(((i_array ** 2) * data) ** 2)
    # for i in range(data.shape[0]):
    #     m4_temp = m4_temp + (data[i] * ((i + 1) ** 2)) ** 2
    return np.sqrt(m4_temp / data.shape[0])


def PS(data):
    """Peak Stress"""
    return m4(data) / (m2(data) * m0(data))


def SE(data):
    """Shake Expectation"""
    i_array = np.array([i + 1 for i in range(data.shape[0])])
    se_temp = np.sum(np.abs((i_array ** 2) * data))
    return se_temp / data.shape[0]


def USTD(data):
    """Unbiased Standard Deviation"""
    return np.std(data, ddof=1)


"""=======================Criteria========================="""


def js_div(p_output, q_output, get_softmax=True):
    """
    Function that measures JS divergence between target and output logits:
    """
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    if get_softmax:
        p_output = F.softmax(p_output)
        q_output = F.softmax(q_output)
    # log_mean_output = ((p_output + q_output )/2).log()
    log_mean_output = ((p_output + q_output) / 2)
    return (KLDivLoss(p_output, log_mean_output) + KLDivLoss(q_output, log_mean_output)) / 2


def pearson_CC(x, y):
    """
    :param x: A tensor
    :param y: A tensor
    :return: Pearson CC of X & Y
    """

    x = np.array(x)
    y = np.array(y)
    assert x.shape == y.shape
    stdx = x.std()
    stdy = y.std()
    covxy = np.mean((x - x.mean()) * (y - y.mean()))
    return covxy / (stdx * stdy)


def curvature(x, y):
    import numpy.linalg as LA
    """
    input  : the coordinate of the three point
    output : the curvature and norm direction
    """

    t_a = LA.norm([x[1] - x[0], y[1] - y[0]])
    t_b = LA.norm([x[2] - x[1], y[2] - y[1]])

    M = np.array([
        [1, -t_a, t_a ** 2],
        [1, 0, 0],
        [1, t_b, t_b ** 2]
    ])

    a = np.matmul(LA.inv(M), x)
    b = np.matmul(LA.inv(M), y)

    kappa = 2 * (a[2] * b[1] - b[2] * a[1]) / (a[1] ** 2. + b[1] ** 2.) ** (1.5)
    return kappa, [b[1], -a[1]] / np.sqrt(a[1] ** 2. + b[1] ** 2.)


def get_smooth_curve(curve):
    ka = []
    no = []
    pos = []
    for idx, theta in enumerate(curve[1:-2]):
        x = list(range(idx, idx + 3))
        y = curve[idx: idx + 3]
        # print(x,y)
        kappa, norm = curvature(x, y)
        ka.append(np.abs(kappa))
        no.append(norm)
        pos.append((x[1], y[1]))

    return np.average(ka), no, pos


"""================Normalization========================="""


def Mu_Normalization(data, Mu=256):
    # print(data.shape)
    result = np.sign(data) * np.log((1 + Mu * np.abs(data))) / np.log((1 + Mu))
    return result


def kaiming_normal_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')


"""==================Common Module======================="""


class SEBlock(nn.Module):
    def __init__(self, input_dim, reduction):
        super().__init__()
        mid = int(input_dim / reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, reduction),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class NoSEBlock(nn.Module):
    def __init__(self, input_dim, reduction):
        super().__init__()

    def forward(self, x):
        return x


"""============Post-process============="""


def avg_smoothing(window_size, s):
    """
    For Pytorch
    :param window_size: processing window size (int)
    :param s: serial data to handle(numpy array)
    :return: smoothened data(numpy array)
    """
    for j in range(s.shape[0]):
        for i in range(s.shape[1]):
            s[j, i] = torch.mean(s[j:j + window_size, i])
    return s


def avg_smoothing_np(window_size, s):
    """
    For numpy
    :param window_size: processing window size (int)
    :param s: serial data to handle(numpy array)
    :return: smoothened data(numpy array)
    """
    for j in range(s.shape[0]):
        for i in range(s.shape[1]):
            s[j, i] = np.mean(s[j:j + window_size, i])
    return s


"""=========================Misc=============================="""


def draw_graph(output, target, channel_num=10):
    fig = plt.figure(figsize=(20, channel_num))
    for i in range(channel_num):
        plt.subplot(math.ceil(channel_num / 2), 2, i + 1)
        # plt.plot(total_output_test[:, i].detach().cpu().numpy(), label="predict", color="b")
        # plt.plot(total_target_test[:, i].detach().cpu().numpy(), label="truth", color="r")
        plt.plot(output[:, i], label="predict", color="b")
        plt.plot(target[:, i], label="truth", color="r")
        # print(total_output_test[:, i].detach().cpu().numpy().shape)
    return fig

def draw_graph_2c(output, target):
    fig = plt.figure(figsize=(10, 4))

    plt.subplot(2, 1, 1)  # 第一个子图在上方
    plt.plot(output[:, 3], label="predict", color="b")
    plt.plot(target[:, 3], label="truth", color="r")

    plt.subplot(2, 1, 2)  # 第二个子图在下方
    plt.plot(output[:, 5], label="predict", color="b")
    plt.plot(target[:, 5], label="truth", color="r")
    fig.tight_layout()
    return fig

def str2bool(v):
    if isinstance(v, bool):
        return v
    if str(v).strip().lower() in ['yes', 'true', 't', 'y', '1']:
        return True
    elif str(v).strip().lower() in ['no', 'false', 'f', 'n', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected. Your input:{v} and input type:{type(v)}')

def compute_metrics_numpy(y_true, y_pred):
    """
    计算与预训练阶段一致的三项指标：NRMSE（min-max 归一化）、CC(皮尔逊)、R2（variance_weighted）
    输入形状：y_true, y_pred -> [N, 10] 或 [*, 10]，内部会 reshape
    返回：NRMSE(float), CC(float), R2(float)
    """
    from skimage import metrics as skimetrics
    from sklearn.metrics import r2_score

    y_true = np.asarray(y_true).reshape(-1, 10)
    y_pred = np.asarray(y_pred).reshape(-1, 10)

    # NRMSE：skimage >= 0.21 提供 normalized_root_mse
    NRMSE = float(skimetrics.normalized_root_mse(y_true, y_pred, normalization="min-max"))

    # Pearson CC：按列求相关并聚合（与预训练保持一致）
    CC = float(pearson_CC(y_true, y_pred))

    # R2：对每个输出通道求R2，再按方差加权
    R2 = float(r2_score(y_true.T, y_pred.T, multioutput="variance_weighted"))
    return NRMSE, CC, R2
import numpy as np
import torch
from scipy.signal import savgol_filter

def savitzky_golay_smoothing(window_length: int, polyorder: int, data):
    """
    对 1D/2D 数据做 Savitzky–Golay 平滑。
    - 支持 torch.Tensor 或 numpy.ndarray。
    - 若是 Tensor：返回值为 Tensor，且保持原 device 与 dtype。
    - 期望形状为 [N] 或 [N, D]，沿 axis=0（时间/样本轴）平滑。
    """

    # ---- 0) 统一成 numpy，并记录是否需要还原为 torch ----
    is_tensor = torch.is_tensor(data)
    if is_tensor:
        device = data.device
        dtype  = data.dtype
        data_np = data.detach().cpu().numpy()
    else:
        data_np = np.asarray(data)

    # ---- 1) 形状规范化：支持 [N] 或 [N, D] ----
    if data_np.ndim == 1:
        data_np = data_np.reshape(-1, 1)   # 临时变成 [N,1]
        squeeze_back = True
    elif data_np.ndim == 2:
        squeeze_back = False
    else:
        raise ValueError(f"data 的维度应为 1 或 2，当前是 {data_np.ndim} 维。")

    N = data_np.shape[0]
    if N < 2:
        # 太短无法平滑，直接原样返回
        out_np = data_np
    else:
        # ---- 2) 窗口与阶的鲁棒修正 ----
        # 窗口必须为奇数且 <= N；最小取 3（若 N<3，则取 N 或 N-1 中最大的奇数）
        if window_length <= 0:
            window_length = 3
        # 先限制到 N 以内
        window_length = min(window_length, N)
        # 变成奇数
        if window_length % 2 == 0:
            window_length = max(1, window_length - 1)
            if window_length < 3 and N >= 3:
                window_length = 3
        # 若 N 仍然很小，保证 window_length >= 1 且为奇数
        if window_length < 1:
            window_length = 1 if N >= 1 else 1

        # polyorder 必须 < window_length，且 >= 0
        polyorder = max(0, min(polyorder, window_length - 1))

        # 当窗口太小（比如 1）时，savgol 等价于不变
        if window_length == 1:
            out_np = data_np
        else:
            out_np = savgol_filter(
                x=data_np,
                window_length=window_length,
                polyorder=polyorder,
                axis=0,
                mode="interp"
            )

    if squeeze_back:
        out_np = out_np.reshape(-1)

    # ---- 3) 还原类型：需要 torch 的话保持原 device/dtype ----
    if is_tensor:
        return torch.from_numpy(out_np).to(device=device, dtype=dtype)
    else:
        return out_np


if __name__ == "__main__":
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([6, 7, 8, 9, 100])
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    f = m * x + c
    print(f)
    print(pearson_CC(x, y))
    print(pearson_CC(x, y) ** 2)
    print(sklearn.metrics.r2_score(x, y))
    print(sklearn.metrics.r2_score(y, f))

