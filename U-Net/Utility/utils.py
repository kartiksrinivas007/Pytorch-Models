import torch
import torch.nn as nn
from .model import *

def test(n, c, h, w):
    x = torch.randn((n, c, h, w))
    mod = UNET(in_channels = c , out_channels = 1)
    preds = mod(x)
    assert preds.shape == x.shape , "Segmentation model failed!"

    