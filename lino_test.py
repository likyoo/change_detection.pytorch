import torch
from change_detection_pytorch.encoders import get_encoder

if __name__ == '__main__':
    sample = torch.randn(1, 3, 256, 256)
    model = get_encoder('mit-b0', img_size=256)
    res = model(sample)
    for x in res:
        print(x.size())
