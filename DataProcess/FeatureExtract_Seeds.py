import sys

sys.path.append("../..")
import os
import h5py
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

from utils.Methods.methods import *

class Extractor:
    def __init__(self, DataDir1, DataDir2, SaveDir, windowsize=200, step=1):
        """
        初始化 Extractor 实例
        :param DataDir1: 第一个 .mat 文件路径
        :param DataDir2: 第二个 .mat 文件路径
        :param SaveDir: 保存的目录路径
        :param windowsize: 滑动窗口大小
        :param step: 滑动窗口步长
        """
        self.DataDir1 = DataDir1
        self.DataDir2 = DataDir2

        self.SaveDir = SaveDir
        self.windowsize = windowsize
        self.step = step
        self.data1 = self.__loadmat(self.DataDir1)
        self.data2 = self.__loadmat(self.DataDir2)

    def __loadmat(self, filepath):
        """
        加载 .mat 文件
        :param filepath: .mat 文件路径
        :return: mat 文件中的数据
        """
        try:
            return loadmat(filepath)
        except Exception as e:
            raise Exception(f"Error loading mat file {filepath}: {e}")

    def extract(self, data=None, method=None):
        if method is None or method == "rms":
            method = [rms]
        if method=="mav":
            method=[mav]
        if method=="su":
            method=[SE,USTD]
        if method == "TDDLF":
            method = [m0, m2, m4, PS, SE, USTD]
        assert type(method) == list, "Plz input a list or TDDLF"

        # data = data if (data is not None) else self.getdata("emg")
        featuremap = []
        print("[.]Now feature is being extracted...")
        for eachmethod in method:
            feature_k = np.zeros([(data.shape[0] - self.windowsize + 1) // self.step
                                     , data.shape[1]])
            j = 0
            for i in tqdm(range(0, data.shape[0], self.step)):
                if i + self.windowsize > data.shape[0]:
                    break
                for eachchannel in range(data.shape[1]):
                    feature_k[j, eachchannel] = eachmethod(data[i:i + self.windowsize, eachchannel])
                j += 1
            featuremap.append(feature_k)
        featuremap = np.transpose(np.array(featuremap), [1, 2, 0])
        # if featuremap.shape[2] == 1:
        #     featuremap = featuremap[:, :, 0].reshape([featuremap.shape[0], featuremap.shape[1]])
        print("[*]Extract complete!")
        return featuremap

    # def extract(self, data=None, method=None):
    #     if method is None or method == "rms":
    #         method = [rms]
    #     if method=="mav":
    #         method=[mav]
    #     if method == "TDDLF":
    #         method = [m0, m2, m4, PS, SE, USTD]
    #     assert type(method) == list, "Plz input a list or TDDLF"
    #
    #     # data = data if (data is not None) else self.getdata("emg")
    #     featuremap = []
    #     print("[.]Now feature is being extracted...")
    #     for eachmethod in method:
    #         # 计算窗口数
    #         num_windows = max(0, (data.shape[0] - self.windowsize) // self.step + 1)
    #         feature_k = np.zeros([num_windows, data.shape[1]])
    #
    #         j = 0
    #         for i in range(0, data.shape[0], self.step):
    #             # 保持最简单的终止条件
    #             if i + self.windowsize > data.shape[0]:
    #                 break
    #             for eachchannel in range(data.shape[1]):
    #                 feature_k[j, eachchannel] = eachmethod(data[i:i + self.windowsize, eachchannel])
    #             j += 1
    #         featuremap.append(feature_k)
    #
    #     featuremap = np.transpose(np.array(featuremap), [1, 2, 0])
    #     # if featuremap.shape[2] == 1:
    #     #     featuremap = featuremap[:, :, 0].reshape([featuremap.shape[0], featuremap.shape[1]])
    #     print("[*]Extract complete!")
    #     return featuremap

    def extract_glove_features(self, data):
        """
        从 glove 数据中提取特征，每个窗口选择最后一个关节角度值。

        参数:
            glove_data (numpy.ndarray): 原始 glove 数据 (N, num_joints)。
            windowsize (int): 滑动窗口大小。
            step (int): 滑动步长。

        返回:
            numpy.ndarray: 提取的特征序列，每个窗口对应最后一个关节角度值。
        """
        num_samples, num_joints = data.shape
        # 计算滑动窗口提取的特征数量
        num_features = (num_samples - self.windowsize) // self.step + 1
        # 初始化存储特征的数组
        features = np.zeros((num_features, num_joints))

        # 滑动窗口提取
        idx = 0
        for i in range(0, num_samples - self.windowsize + 1, self.step):
            # 每个窗口取最后一个时间点的关节角度
            features[idx, :] = data[i + self.windowsize - 1, :]
            idx += 1

        return features


    def save(self, data, datadir,savedir=None, suffix=None):
        # data = pd.DataFrame(data)
        tosavedir = savedir if savedir else self.SaveDir
        if not os.path.exists(tosavedir):
            os.makedirs(tosavedir)
        filename = datadir.split("/")[-1][:-10]
        filepath = tosavedir \
                   + "/" + filename + \
                   (("_" + suffix) if suffix else "") + ".h5"
        # print(filepath)
        # exit()
        # data.to_csv(filepath, index=False, header=False)
        with h5py.File(filepath, "w") as f:
            f.create_dataset("featureset", data=data, compression="gzip", compression_opts=5)

    def process(self, savedir="",extract_method="rms"):
        """
        处理数据集并存储为 H5 文件
        :param train_ratio: 训练集比例
        """
        emg1 = self.data1.get("filter")
        glove1 = self.data1.get("angle")
        emg2 = self.data2.get("filter")
        glove2 = self.data2.get("angle")

        assert emg1 is not None and glove1 is not None, "File 1 must contain 'emg' and 'glove'."
        assert emg2 is not None and glove2 is not None, "File 2 must contain 'emg' and 'glove'."


        # 划分训练集和测试集
        emg_train, emg_test, glove_train, glove_test = emg1,emg2,glove1,glove2

        # 提取特征
        feature_train = self.extract(emg_train, method=extract_method)
        feature_test = self.extract(emg_test, method=extract_method)

        # glove_train = self.extract_glove_features(glove_train)
        # glove_test = self.extract_glove_features(glove_test)

        # glove_train = np.array(glove_train)[:-self.windowsize + 1]
        # glove_test = np.array(glove_test)[:-self.windowsize + 1]

        print("[.]Now writing files……")
        self.save(feature_train, datadir=self.DataDir1,savedir=savedir, suffix=f"{extract_method}_train_ff")
        self.save(feature_test, datadir=self.DataDir1,savedir=savedir, suffix=f"{extract_method}_test_ff")

        self.save(glove_train, datadir=self.DataDir1,savedir=savedir, suffix="glove_train_ff")
        self.save(glove_test,datadir=self.DataDir1, savedir=savedir, suffix="glove_test_ff")

        print("[*]Written!")

        # # 保存数据到 H5 文件
        # self.save_h5(
        #     {
        #         "emg_train": feature_train,
        #         "emg_test": feature_test,
        #         "glove_train": glove_train,
        #         "glove_test": glove_test,
        #     },
        #     os.path.join(self.SaveDir, "dataset.h5"),
        # )


# 示例用法
if __name__ == "__main__":
    SaveDir = "D:/SEEDS/alldata/rawdata/featureset/"  # 保存目录
    # sess=2
    for j in range(3):
        session=j+1
        # for i in [1, 2, 3, 5, 9, 11, 14, 16, 18, 23]:
        for i in [6,8,10,12,13,15,17,19,20,22,24,25]:
            sub = i
            DataDir1 = "D:/SEEDS/alldata/rawdata/S" + str(sub) + "_S" + str(
                session) + "_EA_train.mat"  # 第一个 .mat 文件路径
            DataDir2 = "D:/SEEDS/alldata/rawdata/S" + str(sub) + "_S" + str(
                session) + "_EA_test.mat"  # 第二个 .mat 文件路径
            extractor = Extractor(DataDir1, DataDir2, SaveDir, windowsize=200, step=1)
            extractor.process()