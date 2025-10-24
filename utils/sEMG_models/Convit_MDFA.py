from functools import partial

import torch

import torch.nn as nn
from torch.nn.modules.module import T

from model.MDFA import MDFA
from model.VisionTransformer import VisionTransformer


class Convit_MDFA(nn.Module):
    def __init__(self, winsize=200, patch_size=12, in_chans=1, num_classes=20, embed_dim=32, depth=3,
                 num_heads=8, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, global_pool=None,
                 local_up_to_layer=10, locality_strength=1., use_pos_embed=True):
        super(Convit_MDFA, self).__init__()

        self.name = 'maxmin'
        self.ConVit = VisionTransformer(200, patch_size=12, in_chans=1, num_classes=10, depth=4, num_heads=4, embed_dim=32, drop_rate=0.3,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.MDFA = MDFA(dim_in=1,dim_out=1)
    def forward(self, input_tensor):
        b,w, c = input_tensor.size()
        input_tensor = input_tensor.reshape([b, 1, w,c ])
        output = self.MDFA(input_tensor)
        output= self.ConVit(output)
        # b,w = output.size()
        # output = output.reshape([b,1,1, w])

        output = torch.squeeze(output)
        return output
if __name__ == '__main__':
    input = torch.randn(16, 200, 12)
    model = Convit_MDFA(
        200, patch_size=12, in_chans=1, num_classes=10, depth=4, num_heads=4, embed_dim=32, drop_rate=0.3,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )

    # summary(model, (1,200,12))
    output = model(input)
    print(output.shape)