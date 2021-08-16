import torch
from change_detection_pytorch.stanet import STANet

if __name__ == '__main__':

    samples = torch.ones([1, 3, 256, 256])
    model = STANet(
        encoder_name='vgg16',
        in_channels=3
    )
    dist = model(samples, samples)
    print(dist.size())
