import numpy as np
from .module import Module
from ...tensor import Tensor, concatenate
from functools import reduce

__all__ = ['Conv2d']


# Сверточный слой 2d
class Conv2d(Module):

    @staticmethod
    def get_image_section(layer, row_from, row_to, col_from, col_to):
        section = layer[:,:,row_from:row_to,col_from:col_to]

        return section.reshape(-1,1,row_to-row_from, col_to-col_from)


    def __init__(self, image_size, in_channels, out_channels, kernel_size=(3, 3), stride=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_rows = kernel_size[0]
        self.kernel_cols = kernel_size[1]
        self.image_rows = image_size[0]
        self.image_cols = image_size[1]
        self.stride = stride

        self.num_kernels = self.out_channels #* self.in_channels

        kernels = [0.02 * np.random.random((self.kernel_rows * self.kernel_cols, self.num_kernels)) - 0.01 for _ in range(self.in_channels)]
        self.kernels = [Tensor(krnl, autograd=True) for krnl in kernels]

        self.parameters.extend(self.kernels)

    # input [N, C, ROWS, COLS]
    def forward(self, images):

        output = list()

        for channel in range(self.in_channels):
            layer_0 = images[:,channel:channel+1,:,:]

            sects = list()

            # print(layer_0.shape)
            for row_start in range(0, layer_0.shape[2] - self.kernel_rows + 1, self.stride):
                for col_start in range(0, layer_0.shape[3] - self.kernel_cols + 1, self.stride):
                    # print(row_start, col_start)
                    sect = self.get_image_section(layer_0,
                                             row_start,
                                             row_start+self.kernel_rows,
                                             col_start,
                                             col_start+self.kernel_cols)
                    sects.append(sect)
                    # print(type(sect), sect.shape)

            expanded_input = concatenate(sects, axis=1)
            # print(expanded_input.shape)
            es = expanded_input.shape
            flattened_input = expanded_input.reshape(es[0]*es[1],-1)

            # print(flattened_input.shape)

            kernel_output = flattened_input.mm(self.kernels[channel])
            # print(kernel_output.shape)


            h_out = int(np.floor((self.image_rows - self.kernel_rows) / self.stride ) + 1)
            w_out = int(np.floor((self.image_cols - self.kernel_cols) / self.stride ) + 1)
            # print(h_out)
            layer_1 = kernel_output.reshape(es[0], self.out_channels, h_out, w_out)
            # print(layer_1.shape)
            output.append(layer_1)
        # print(all(x.autograd for x in output))
        result = reduce(lambda x, y: x + y, output) * Tensor([1 / self.in_channels], autograd=True)
        # print(result.autograd)
        return result