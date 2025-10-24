import numpy as np
import torch.nn as nn

from model.MDFA import MDFA


class LE_MDFA(nn.Module):
    def __init__(self, ):
        super(LE_MDFA, self).__init__()
        self.name = 'LE_MDFA'
        self.MDFA = MDFA(dim_in=1, dim_out=1)  # 实例化模块

    def forward(self, input_tensor):
        b, w, c = input_tensor.size()
        input_tensor = input_tensor.reshape([b, 1, w, c])

        output = self.MDFA(input_tensor)
        output = np.squeeze(output)
        return output
