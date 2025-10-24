from functools import partial

import torch.nn as nn
from torch.nn.modules.module import T

from model.VisionTransformer import VisionTransformer


class LE_ConVit(nn.Module):
    def __init__(self, winsize=200, patch_size=12, in_chans=1, num_classes=20, embed_dim=32, depth=3,
                 num_heads=8, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, global_pool=None,
                 local_up_to_layer=10, locality_strength=1., use_pos_embed=True):
        super(LE_ConVit, self).__init__()

        self.name = 'LE_ConVit'
        self.ConVit = VisionTransformer(200, patch_size=12, in_chans=1, num_classes=10, depth=4, num_heads=4, embed_dim=32, drop_rate=0.3,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))

    def forward(self, input_tensor):
        b,w, c = input_tensor.size()#16,200,12
        input_tensor = input_tensor.reshape([b, 1, w,c ])

        output= self.ConVit(input_tensor)

        return output
