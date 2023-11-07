import torch.nn as nn
import torch
import torchvision.transforms as tfms
import sys
import torch.nn.functional as F
# sys.path.append('/home/mayug/projects/FastSal')
# import model1.fastSal as fastsal
# from utils import load_weight


# coco_c = 'weights/coco_C.pth'  # coco_C
# coco_a = '/home/mayug/projects/FastSal/weights/coco_A.pth'  # coco_A
# salicon_c = 'weights/salicon_C.pth'  # salicon_C
# salicon_a = 'weights/salicon_A.pth'  # coco_A

# model = fastsal.fastsal(pretrain_mode=False, model_type='A')
# model = model.eval()
# model = model.cuda()
# state_dict, opt_state = load_weight(coco_a, remove_decoder=False)
# model.load_state_dict(state_dict)



# def get_sal(data, out_size=(84, 84)):
#     # print(['data', data.min(), data.max(), data.shape])
#     res = tfms.Resize(size=(192,256))
#     y= res(data)
#     # print(['y', y.min(), y.max(), y.shape])
#     with torch.no_grad():
#         z = model(y)
#     # print(['z', z.min(), z.max(), z.shape])

#     z = minmax(z)

#     # z= F.sigmoid(z)

#     res = tfms.Resize(size=out_size)
#     y= res(z)
#     return y


def minmax(img):
    img = (img- img.min())/(img.max()-img.min())
    return img  

# Basic ConvNet with Pooling layer
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class ConvNet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
    def forward(self, x):
        x = self.encoder(x)
        return x


