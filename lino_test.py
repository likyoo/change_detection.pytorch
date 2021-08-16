import torch



if __name__ == '__main__':
    from change_detection_pytorch.encoders import get_encoder

    samples = torch.ones([1, 3, 256, 256])
    encoder = get_encoder('resnet34')
    features0 = encoder(samples)
    for fc in features0:
        print(fc.size())
    features1 = encoder(samples)
    model = STANetDecoder(encoder_out_channels=encoder.out_channels)
    features0, features1 = model(features0, features1)
    print(features0, features1)