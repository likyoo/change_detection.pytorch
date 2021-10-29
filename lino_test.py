import torch
from torch.utils.data import DataLoader, Dataset

import change_detection_pytorch as cdp
from change_detection_pytorch.datasets import LEVIR_CD_Dataset, SVCD_Dataset
from change_detection_pytorch.utils.lr_scheduler import GradualWarmupScheduler

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model = cdp.STANet(
    encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    return_distance_map=False
).to(DEVICE)

sampel1 = torch.ones([1, 3, 256, 256]).to(DEVICE)
sampel2 = torch.ones([1, 3, 256, 256]).to(DEVICE)
preds = model(sampel1, sampel2)
print(preds.size())
