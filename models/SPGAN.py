import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import models.conv2d_same_padding as conv

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = x.size()
        flatten_shape = shape[1] * shape[2] * shape[3]
        return x.view(-1, flatten_shape)

class Residule_block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Residule_block, self).__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            conv.Conv2d_padding_same(in_dim, out_dim, 3, 1, "VALID"),
            nn.InstanceNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            conv.Conv2d_padding_same(out_dim, out_dim, 3, 1, "VALID"),
            nn.InstanceNorm2d(out_dim)
        )

    def forward(self, x):
        y = self.net(x)
        return y + x

class discrimnator(nn.Module):
    def __init__(self, df_dim=64):
        super(discrimnator, self).__init__()
        #instance_norm = functools.partial(nn.InstanceNorm2d, eps=1e-5, momentum=0.9)
        self.net = nn.Sequential(
            # output_size: n * 64 * 128, 128
            conv.Conv2d_padding_same(3, df_dim, 4, 2, "SAME"),
            nn.LeakyReLU(0.2, inplace=True),
            # output_size: n * 128 * 64 * 64
            conv.Conv2d_padding_same(df_dim, df_dim*2, 4, 2, "SAME"),
            nn.InstanceNorm2d(df_dim*2),
            nn.LeakyReLU(0.2, inplace=True),
            # output_size: n * 256 * 32 * 32
            conv.Conv2d_padding_same(df_dim*2, df_dim*4, 4, 2, "SAME"),
            nn.InstanceNorm2d(df_dim*4),
            nn.LeakyReLU(0.2, inplace=True),
            # output_size: n * 512 * 32 * 32
            conv.Conv2d_padding_same(df_dim*4, df_dim*8, 4, 1, "SAME"),
            nn.InstanceNorm2d(df_dim*8),
            nn.LeakyReLU(0.2, inplace=True),
            # output_size: n * 1 * 32 * 32
            conv.Conv2d_padding_same(df_dim*8, 1 , 4, 1, "SAME")
        )
    
    def forward(self, x):
        return self.net(x)

class metric_net(nn.Module):
    def __init__(self, df_dim=64):
        super(metric_net, self).__init__()
        self.net = nn.Sequential(
            conv.Conv2d_padding_same(3, df_dim, 4, 2, "SAME"), # output_size : n * 64 * 128 * 128
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2), # output_size : n * 64 * 64 * 64
            conv.Conv2d_padding_same(df_dim, df_dim*2, 4, 2, "SAME"), # output_size: n * 128 * 32 * 32
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2), # output_size: n * 128 * 16 * 16
            conv.Conv2d_padding_same(df_dim*2, df_dim*4, 4, 2, "SAME"), # output_size : n * 256 * 8 * 8
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2), # output_size : n * 256 * 4 * 4
            conv.Conv2d_padding_same(df_dim*4, df_dim*8, 4, 2, "SAME"), # output_size : n * 512 * 2 * 2
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2), # output_size : n * 512 * 1 * 1
            Flatten(), # output_size : n * 512
            nn.Linear(df_dim*8, df_dim*2), # output_size : n * 128
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=True),
            nn.Linear(df_dim*2, df_dim*1) # output_size : n * 64
        )
    
    def forward(self, x):
        return self.net(x)


class generator(nn.Module):
    def __init__(self, gf_dim=64):
        super(generator, self).__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d((3, 3, 3, 3)), # output_size : n * 3 * 262 * 262
            Conv2d_padding_same(3, gf_dim, 7, 1, "VALID"), # output_size : n * 64 * 256 * 256
            nn.InstanceNorm2d(gf_dim),
            nn.ReLU(inplace=True),
            Conv2d_padding_same(gf_dim, gf_dim*2, 3, 2, "SAME"), # output_size : n * 128 * 128 * 128
            nn.InstanceNorm2d(gf_dim*2),
            nn.ReLU(inplace=True),
            Conv2d_padding_same(gf_dim*2, gf_dim*4, 3, 2, "SAME"), # output_size : n * 256 * 64 * 64
            nn.InstanceNorm2d(gf_dim*4),
            nn.ReLU(inplace=True),

            ######################
            # The Res_block part #
            ######################

            Residule_block(gf_dim*4, gf_dim*4),
            Residule_block(gf_dim*4, gf_dim*4),
            Residule_block(gf_dim*4, gf_dim*4),
            Residule_block(gf_dim*4, gf_dim*4),
            Residule_block(gf_dim*4, gf_dim*4),
            Residule_block(gf_dim*4, gf_dim*4),
            Residule_block(gf_dim*4, gf_dim*4),
            Residule_block(gf_dim*4, gf_dim*4),
            Residule_block(gf_dim*4, gf_dim*4),

            ########################
            # Transposed Conv part #
            ########################

            nn.ConvTranspose2d(gf_dim*4, gf_dim*2, 3, 2, padding=1, output_padding=1),
            nn.InstanceNorm2d(gf_dim*2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(gf_dim*2, gf_dim, 3, 2, padding=1, output_padding=1),
            nn.InstanceNorm2d(gf_dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((3, 3, 3, 3)),
            nn.Conv2d(gf_dim, 3, 7, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return(self.net(x))

#input = torch.randn(1, 3, 256, 256)
#net = generator()
#ouput = net(input)
#print(ouput.size())
