import os
import numpy as np
from scipy.io import loadmat, savemat
import re
from scipy.signal import resample


# 同步信号
def resample_glove(data, orig_fs, new_fs, final_num_points):
    """

    :param data: channels x samples 2D array
    :param orig_fs: sampling frequency of the data
    :param final_num_points: how many points the resampled array should have
    :return: resampled data to the new sampling frequency
    """
    t = np.linspace(0, data.shape[-1]/orig_fs, data.shape[-1], endpoint=True)
    upsampled_data = resample(data, num=int(new_fs/orig_fs)*data.shape[-1], t=t, axis=1)[0]
    return upsampled_data[:, :final_num_points]

def process_and_concat(folder_path,subject_id, session_id,save_path):
    """
    拼接文件夹中的所有 .mat 文件，并保留未拼接的其他键值数据。

    1. 将 emg 和 glove 转置为 timestep * channel，然后按行拼接。
    2. 将 movement 和 speed 扩展为 timestep * 1，然后按行拼接。
    3. 其他键值数据保留一份未修改。

    :param folder_path: 包含 .mat 文件的输入文件夹路径。
    :param save_path: 保存拼接后结果的文件路径。
    """
    all_emg = []
    all_glove = []
    pattern = rf'detop_exp\d+_subj{subject_id}_Sess{session_id}_.*\.mat$'
    for file in os.listdir(folder_path):
        if re.match(pattern, file):  # 筛选 .mat 文件
            file_path = os.path.join(folder_path, file)
            mat_data = loadmat(file_path)

            # # 检查所需的键是否存在
            # if emg is None or glove is None or movement is None or speed is None:
            #     print(f"文件 {file} 缺少必要的数据键，跳过。")
            #     continue

            # 提取所需数据
            emg = mat_data.get("emg")[:126, :]
            glove = mat_data.get("glove")
            # 数据处理
            emg = emg.T  # 转置为 timestep * channel

            upsampled_glove = resample_glove(glove, float(256), float(2048),
                                             emg.shape[0])
            glove = upsampled_glove.T  # 转置为 timestep * channel

            trimmed_emg = emg[2000:-1000, :]
            trimmed_angle = glove[2000:-1000, :]

            # 分别拼接
            all_emg.append(trimmed_emg)
            all_glove.append(trimmed_angle)


    # 拼接所有数据
    concatenated_data = {
        "emg": np.vstack(all_emg) if all_emg else None,
        "glove": np.vstack(all_glove) if all_glove else None,

    }


    # 保存拼接结果到新的 .mat 文件
    savemat(save_path, concatenated_data)
    print(f"拼接后的数据已保存到 {save_path}")


# session_id=2
for j in range(3):
    session_id=j+1
    for i in range(16):
        subject_id = i + 10
        folder_path="D:/SEEDS/alldata/rawdata/train/"
        save_path = r'D:/SEEDS/alldata/rawdata/S'+str(subject_id)+'_S' + str(session_id) + '_EA_train.mat' # 替换为保存路径

        process_and_concat(folder_path, subject_id, session_id,save_path)