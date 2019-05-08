
import numpy as np

''' assuming dilation = 1 and no padding for pooling or output padding for transpose conv'''

#Conv2d:
def conv_h_out(h_in, k, stride, pad):
    return ((h_in + 2*pad - k) / stride) + 1   

#ConvTranspose2d:
def conv_tr_h_out(h_in, k, stride, pad):
    return (h_in-1)*stride - 2*pad + k - 1 + 1

def maxpool_out(h_in, k, stride):
    return np.floor( ((h_in - k) / stride) + 1 )


#Examples: 

#h_out = conv_h_out(28, 3, 3, 1)
#print(maxpool_out(6, 2, 2))
#print(conv_tr_h_out(2, 3, 2, 0))
#print(maxpool_out(6,3,2))


