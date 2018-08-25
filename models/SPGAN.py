import torch
import torch.nn as nn
import functools
import sys
import SPGAN
class Discrimnator(nn.Module):
    def __init__(self, df_dim=64):
        super(Discrimnator, self).__init__()
        #instance_norm = functools.partial(nn.InstanceNorm2d, eps=1e-5, momentum=0.9)
        self.net = nn.Sequential(
            # output_size: n * 64 * 128, 128
            Conv2d_padding_same(3, df_dim, 4, 2, "SAME"),
            nn.LeakyReLU(0.2, inplace=True),
            # output_size: n * 128 * 64 * 64
            Conv2d_padding_same(df_dim, df_dim*2, 4, 2, "SAME"),
            nn.InstanceNorm2d(df_dim*2),
            nn.LeakyReLU(0.2, inplace=True),
            # output_size: n * 256 * 32 * 32
            Conv2d_padding_same(df_dim*2, df_dim*4, 4, 2, "SAME"),
            nn.InstanceNorm2d(df_dim*4),
            nn.LeakyReLU(0.2, inplace=True),
            # output_size: n * 512 * 32 * 32
            Conv2d_padding_same(df_dim*4, df_dim*8, 4, 1, "SAME"),
            nn.InstanceNorm2d(df_dim*8),
            nn.LeakyReLU(0.2, inplace=True),
            # output_size: n * 1 * 32 * 32
            Conv2d_padding_same(df_dim*8, 1 , 4, 1, "SAME")
        )
    
    def forward(self, x):
        return self.net(x)

x = torch.randn(2, 3, 256, 256)
D = Discrimnator()
output = D(x)
print(output.size())
