import torch.nn as nn
import torch

from utils.BaseModels.BERT.Bert import BERT


# import torchsnooper
# @torchsnooper.snoop()
class sEMG_BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, input_dim=12, hidden=24, feature_dim=6, n_layers=12, attn_heads=12, dropout=0.1,
                 use_se=False):
        """
        :param vocab_size: vocab_size of total words
        :param input_dim: sEMG channels
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.name = "BERT"
        self.bert = BERT(vocab_size, input_dim, hidden, feature_dim, n_layers, attn_heads, dropout, use_se=use_se)
        self.fc = nn.Linear(vocab_size * hidden * 1, 1 * 10)
        self.vocab_size = vocab_size

    def forward(self, x):
        x1 = self.bert(x)
        distr = x1
        x1_flatten = x1.view(x1.size(0), -1)
        output = self.fc(x1_flatten)
        output = output.view(x1.size(0), 1, 10)
        return output, distr


if __name__ == "__main__":
    # 初始化时使用正确的输入维度和词汇表大小
    sim_batch = torch.rand([1, 200, 12]).float().cuda()  # 使用 float 类型
    vocab_size = 2400  # 这是你预训练模型的 vocab_size
    input_dim = 12  # 这是你预训练模型的输入维度

    # 确保模型初始化时，vocab_size 和 input_dim 正确
    model = sEMG_BERT(vocab_size=vocab_size, hidden=12, input_dim=input_dim).cuda()
    print(sim_batch.shape)
    output = model(sim_batch)
    print(output.shape)