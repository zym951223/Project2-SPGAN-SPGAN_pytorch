import torch
import torch.nn as nn
import torch.nn.functional as F

# the Conv2d mudule supports padding=="same" mode
def conv2d_same_padding(input, weight, bias=None, stride=1, padding="VALID", dilation=1, groups=1):
    print(padding)
    if padding == 'SAME':
        padding = 0

        input_rows = input.size(2)
        filter_rows = weight.size(2)
        out_rows = (input_rows + stride[0] - 1) // stride[0]
        padding_rows = max(0, (out_rows - 1) * stride[0] +
                            (filter_rows - 1) * dilation[0] + 1 - input_rows)
        rows_odd = padding_rows % 2

        input_cols = input.size(3)
        filter_cols = weight.size(3)
        out_cols = (input_cols + stride[1] - 1) // stride[1]
        padding_cols = max(0, (out_cols - 1) * stride[1] +
                            (filter_cols - 1) * dilation[1] + 1 - input_cols)
        cols_odd = padding_cols % 2

        input = F.pad(input, [padding_cols // 2, padding_cols // 2 + int(cols_odd),
                            padding_rows // 2, padding_rows // 2 + int(rows_odd)])
    
    elif padding == 'VALID':
        padding = 0
    
    elif type(padding) != tuple:
        print(type(padding), padding)        
        raise ValueError('Padding should be SAME, VALID or specific integer, but not {}.'.format(padding))

    return F.conv2d(input, weight, bias, stride, padding=0,
                    dilation=dilation, groups=groups)



class Conv2d_padding_same(nn.Conv2d):

    def forward(self, input):
        print(self.stride, self.padding)
        return conv2d_same_padding(input, self.weight, self.bias, self.stride, \
                                   self.padding, self.dilation, self.groups)

#x = torch.randn(3, 3, 256, 256)
#layer = Conv2d_padding_same(3, 3, 4, 2, "SAME")
#layer = nn.Conv2d(3, 3, 4, 2)
#output = layer(x)
#print(output.size())