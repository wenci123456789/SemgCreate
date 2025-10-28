# export_by_repetition_S1_S30.py
import os, numpy as np
from FeatureExtract_Ninapro import Extractor  # 复用你现有类
from scipy.io import loadmat

SAVE_DIR = "../../../feature/ninapro_db2_trans"          # 你项目中默认的特征目录
RAW_DIR  = "../../../db2"             # .mat 所在目录
STIM = [18, 19, 22, 25, 27, 37]         # 6 种抓握动作（与你代码一致）
REPS_CALIB = {1,2,3,4,5}                # 前 5 次 → 训练
REP_VAL = 6                             # 第 6 次 → 验证

def export_one(mat_path, save_dir=SAVE_DIR):
    ex = Extractor(mat_path, save_dir, windowsize=200, step=1)  # 100ms/0.5ms
    data_all  = ex.getdata()
    stim      = ex.getdata(data_all, "restimulus").reshape(-1)
    # 兼容不同字段名
    rep_key = "repetition"
    if rep_key not in data_all:
        for k in data_all.keys():
            if "repet" in k.lower(): rep_key = k; break
    reps     = ex.getdata(data_all, rep_key).reshape(-1)

    emg_raw   = ex.getdata(data_all, "emg")
    glove_raw = ex.getdata(data_all, "glove")

    def slice_by_rep(rep_set):
        emgs, gloves = [], []
        for lab in STIM:
            idx = np.where((stim == lab) & np.isin(reps, list(rep_set)))[0]
            if idx.size == 0: continue
            head, tail = idx[0], idx[-1] + 1
            emgs.append(emg_raw[head:tail]); gloves.append(glove_raw[head:tail])
        if len(emgs)==0:
            return np.zeros((0, emg_raw.shape[1])), np.zeros((0, glove_raw.shape[1]))
        return np.concatenate(emgs, axis=0), np.concatenate(gloves, axis=0)

    # 前5次→train_ninapro，第6次→val
    emg_tr, glove_tr = slice_by_rep(REPS_CALIB)
    emg_va, glove_va = slice_by_rep({REP_VAL})

    # 调用你现成的 RMS 抽取
    feat_tr = ex.extract(emg_tr, method="rms")
    feat_va = ex.extract(emg_va, method="rms")

    # 计算窗口对齐差值（推回 windowsize），对齐 glove 长度
    # feature_len = raw_len - windowsize + 1  ⇒ windowsize = raw_len - feature_len + 1
    if emg_tr.shape[0] > 0:
        w_tr = emg_tr.shape[0] - feat_tr.shape[0] + 1
        glove_tr = glove_tr[:-w_tr+1] if glove_tr.shape[0] >= w_tr else glove_tr[:0]
    if emg_va.shape[0] > 0:
        w_va = emg_va.shape[0] - feat_va.shape[0] + 1
        glove_va = glove_va[:-w_va+1] if glove_va.shape[0] >= w_va else glove_va[:0]

    # 按你现有保存格式输出 *_rms_{train_ninapro|test}.h5 和 *_glove_{train_ninapro|test}.h5
    ex.save(feat_tr, savedir=save_dir, suffix="rms_train")
    ex.save(feat_va, savedir=save_dir, suffix="rms_test")   # 这里把“验证”写在 _test，方便直接被 Trainer 使用
    ex.save(glove_tr, savedir=save_dir, suffix="glove_train")
    ex.save(glove_va, savedir=save_dir, suffix="glove_test")

if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    # 批量导出 S1–S30（E1_A1，你也可以换成 E2_A1）
    for sid in range(31, 41):
        mat = os.path.join(RAW_DIR, f"DB2_s{sid}/S{sid}_E2_A1.mat")
        print("=> exporting", mat)
        export_one(mat, save_dir=SAVE_DIR)
    print("All done.")
