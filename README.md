<h1 align="center">
  <b>Change Detection Models</b><br>
</h1>
<p align="center">
      <b>Python library with Neural Networks for Change Detection based on PyTorch.</b>
</p>


<img src="resources/model architecture.png" alt="model architecture" style="zoom:80%;" />


This project is inspired by **[segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)** and built based on it. 😄

### 🌱 How to use <a name="use"></a>

Please refer to local_test.py temporarily.



### 🔭 Models <a name="models"></a>

#### Architectures <a name="architectures"></a>
- [x] Unet [[paper](https://arxiv.org/abs/1505.04597)]

- [x] Unet++ [[paper](https://arxiv.org/pdf/1807.10165.pdf)]

- [x] MAnet [[paper](https://ieeexplore.ieee.org/abstract/document/9201310)]

- [x] Linknet [[paper](https://arxiv.org/abs/1707.03718)]

- [x] FPN [[paper](http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf)]

- [x] PSPNet [[paper](https://arxiv.org/abs/1612.01105)]

- [x] PAN [[paper](https://arxiv.org/abs/1805.10180)]

- [x] DeepLabV3 [[paper](https://arxiv.org/abs/1706.05587)]

- [x] DeepLabV3+ [[paper](https://arxiv.org/abs/1802.02611)]

- [x] UPerNet [[paper](https://arxiv.org/abs/1807.10221)]

- [x] STANet [[paper](https://www.mdpi.com/2072-4292/12/10/1662)]

#### Encoders <a name="encoders"></a>

The following is a list of supported encoders in the CDP. Select the appropriate family of encoders and click to expand the table and select a specific encoder and its pre-trained weights (`encoder_name` and `encoder_weights` parameters).

<details>
<summary style="margin-left: 25px;">ResNet</summary>
<div style="margin-left: 25px;">

| Encoder   |        Weights        | Params, M |
| --------- | :-------------------: | :-------: |
| resnet18  | imagenet / ssl / swsl |    11M    |
| resnet34  |       imagenet        |    21M    |
| resnet50  | imagenet / ssl / swsl |    23M    |
| resnet101 |       imagenet        |    42M    |
| resnet152 |       imagenet        |    58M    |

</div>
</details>

<details>
<summary style="margin-left: 25px;">ResNeXt</summary>
<div style="margin-left: 25px;">

| Encoder           |              Weights              | Params, M |
| ----------------- | :-------------------------------: | :-------: |
| resnext50_32x4d   |       imagenet / ssl / swsl       |    22M    |
| resnext101_32x4d  |            ssl / swsl             |    42M    |
| resnext101_32x8d  | imagenet / instagram / ssl / swsl |    86M    |
| resnext101_32x16d |      instagram / ssl / swsl       |   191M    |
| resnext101_32x32d |             instagram             |   466M    |
| resnext101_32x48d |             instagram             |   826M    |

</div>
</details>

<details>
<summary style="margin-left: 25px;">ResNeSt</summary>
<div style="margin-left: 25px;">

| Encoder                 | Weights  | Params, M |
| ----------------------- | :------: | :-------: |
| timm-resnest14d         | imagenet |    8M     |
| timm-resnest26d         | imagenet |    15M    |
| timm-resnest50d         | imagenet |    25M    |
| timm-resnest101e        | imagenet |    46M    |
| timm-resnest200e        | imagenet |    68M    |
| timm-resnest269e        | imagenet |   108M    |
| timm-resnest50d_4s2x40d | imagenet |    28M    |
| timm-resnest50d_1s4x24d | imagenet |    23M    |

</div>
</details>

<details>
<summary style="margin-left: 25px;">Res2Ne(X)t</summary>
<div style="margin-left: 25px;">

| Encoder                | Weights  | Params, M |
| ---------------------- | :------: | :-------: |
| timm-res2net50_26w_4s  | imagenet |    23M    |
| timm-res2net101_26w_4s | imagenet |    43M    |
| timm-res2net50_26w_6s  | imagenet |    35M    |
| timm-res2net50_26w_8s  | imagenet |    46M    |
| timm-res2net50_48w_2s  | imagenet |    23M    |
| timm-res2net50_14w_8s  | imagenet |    23M    |
| timm-res2next50        | imagenet |    22M    |

</div>
</details>

<details>
<summary style="margin-left: 25px;">RegNet(x/y)</summary>
<div style="margin-left: 25px;">

| Encoder          | Weights  | Params, M |
| ---------------- | :------: | :-------: |
| timm-regnetx_002 | imagenet |    2M     |
| timm-regnetx_004 | imagenet |    4M     |
| timm-regnetx_006 | imagenet |    5M     |
| timm-regnetx_008 | imagenet |    6M     |
| timm-regnetx_016 | imagenet |    8M     |
| timm-regnetx_032 | imagenet |    14M    |
| timm-regnetx_040 | imagenet |    20M    |
| timm-regnetx_064 | imagenet |    24M    |
| timm-regnetx_080 | imagenet |    37M    |
| timm-regnetx_120 | imagenet |    43M    |
| timm-regnetx_160 | imagenet |    52M    |
| timm-regnetx_320 | imagenet |   105M    |
| timm-regnety_002 | imagenet |    2M     |
| timm-regnety_004 | imagenet |    3M     |
| timm-regnety_006 | imagenet |    5M     |
| timm-regnety_008 | imagenet |    5M     |
| timm-regnety_016 | imagenet |    10M    |
| timm-regnety_032 | imagenet |    17M    |
| timm-regnety_040 | imagenet |    19M    |
| timm-regnety_064 | imagenet |    29M    |
| timm-regnety_080 | imagenet |    37M    |
| timm-regnety_120 | imagenet |    49M    |
| timm-regnety_160 | imagenet |    80M    |
| timm-regnety_320 | imagenet |   141M    |

</div>
</details>

<details>
<summary style="margin-left: 25px;">GERNet</summary>
<div style="margin-left: 25px;">

| Encoder       | Weights  | Params, M |
| ------------- | :------: | :-------: |
| timm-gernet_s | imagenet |    6M     |
| timm-gernet_m | imagenet |    18M    |
| timm-gernet_l | imagenet |    28M    |

</div>
</details>

<details>
<summary style="margin-left: 25px;">SE-Net</summary>
<div style="margin-left: 25px;">

| Encoder             | Weights  | Params, M |
| ------------------- | :------: | :-------: |
| senet154            | imagenet |   113M    |
| se_resnet50         | imagenet |    26M    |
| se_resnet101        | imagenet |    47M    |
| se_resnet152        | imagenet |    64M    |
| se_resnext50_32x4d  | imagenet |    25M    |
| se_resnext101_32x4d | imagenet |    46M    |

</div>
</details>

<details>
<summary style="margin-left: 25px;">SK-ResNe(X)t</summary>
<div style="margin-left: 25px;">

| Encoder                | Weights  | Params, M |
| ---------------------- | :------: | :-------: |
| timm-skresnet18        | imagenet |    11M    |
| timm-skresnet34        | imagenet |    21M    |
| timm-skresnext50_32x4d | imagenet |    25M    |

</div>
</details>

<details>
<summary style="margin-left: 25px;">DenseNet</summary>
<div style="margin-left: 25px;">

| Encoder     | Weights  | Params, M |
| ----------- | :------: | :-------: |
| densenet121 | imagenet |    6M     |
| densenet169 | imagenet |    12M    |
| densenet201 | imagenet |    18M    |
| densenet161 | imagenet |    26M    |

</div>
</details>

<details>
<summary style="margin-left: 25px;">Inception</summary>
<div style="margin-left: 25px;">

| Encoder           |             Weights             | Params, M |
| ----------------- | :-----------------------------: | :-------: |
| inceptionresnetv2 | imagenet /  imagenet+background |    54M    |
| inceptionv4       | imagenet /  imagenet+background |    41M    |
| xception          |            imagenet             |    22M    |

</div>
</details>

<details>
<summary style="margin-left: 25px;">EfficientNet</summary>
<div style="margin-left: 25px;">

| Encoder                 |              Weights               | Params, M |
| ----------------------- | :--------------------------------: | :-------: |
| efficientnet-b0         |              imagenet              |    4M     |
| efficientnet-b1         |              imagenet              |    6M     |
| efficientnet-b2         |              imagenet              |    7M     |
| efficientnet-b3         |              imagenet              |    10M    |
| efficientnet-b4         |              imagenet              |    17M    |
| efficientnet-b5         |              imagenet              |    28M    |
| efficientnet-b6         |              imagenet              |    40M    |
| efficientnet-b7         |              imagenet              |    63M    |
| timm-efficientnet-b0    | imagenet / advprop / noisy-student |    4M     |
| timm-efficientnet-b1    | imagenet / advprop / noisy-student |    6M     |
| timm-efficientnet-b2    | imagenet / advprop / noisy-student |    7M     |
| timm-efficientnet-b3    | imagenet / advprop / noisy-student |    10M    |
| timm-efficientnet-b4    | imagenet / advprop / noisy-student |    17M    |
| timm-efficientnet-b5    | imagenet / advprop / noisy-student |    28M    |
| timm-efficientnet-b6    | imagenet / advprop / noisy-student |    40M    |
| timm-efficientnet-b7    | imagenet / advprop / noisy-student |    63M    |
| timm-efficientnet-b8    |         imagenet / advprop         |    84M    |
| timm-efficientnet-l2    |           noisy-student            |   474M    |
| timm-efficientnet-lite0 |              imagenet              |    4M     |
| timm-efficientnet-lite1 |              imagenet              |    5M     |
| timm-efficientnet-lite2 |              imagenet              |    6M     |
| timm-efficientnet-lite3 |              imagenet              |    8M     |
| timm-efficientnet-lite4 |              imagenet              |    13M    |

</div>
</details>

<details>
<summary style="margin-left: 25px;">MobileNet</summary>
<div style="margin-left: 25px;">

| Encoder                            | Weights  | Params, M |
| ---------------------------------- | :------: | :-------: |
| mobilenet_v2                       | imagenet |    2M     |
| timm-mobilenetv3_large_075         | imagenet |   1.78M   |
| timm-mobilenetv3_large_100         | imagenet |   2.97M   |
| timm-mobilenetv3_large_minimal_100 | imagenet |   1.41M   |
| timm-mobilenetv3_small_075         | imagenet |   0.57M   |
| timm-mobilenetv3_small_100         | imagenet |   0.93M   |
| timm-mobilenetv3_small_minimal_100 | imagenet |   0.43M   |

</div>
</details>

<details>
<summary style="margin-left: 25px;">DPN</summary>
<div style="margin-left: 25px;">

| Encoder |   Weights   | Params, M |
| ------- | :---------: | :-------: |
| dpn68   |  imagenet   |    11M    |
| dpn68b  | imagenet+5k |    11M    |
| dpn92   | imagenet+5k |    34M    |
| dpn98   |  imagenet   |    58M    |
| dpn107  | imagenet+5k |    84M    |
| dpn131  |  imagenet   |    76M    |

</div>
</details>

<details>
<summary style="margin-left: 25px;">VGG</summary>
<div style="margin-left: 25px;">

| Encoder  | Weights  | Params, M |
| -------- | :------: | :-------: |
| vgg11    | imagenet |    9M     |
| vgg11_bn | imagenet |    9M     |
| vgg13    | imagenet |    9M     |
| vgg13_bn | imagenet |    9M     |
| vgg16    | imagenet |    14M    |
| vgg16_bn | imagenet |    14M    |
| vgg19    | imagenet |    20M    |
| vgg19_bn | imagenet |    20M    |

</div>
</details>



### :truck: Dataset <a name="dataset"></a>

- [x] [LEVIR-CD](https://justchenhao.github.io/LEVIR/)
- [x] [SVCD](https://www.researchgate.net/publication/325470033_CHANGE_DETECTION_IN_REMOTE_SENSING_IMAGES_USING_CONDITIONAL_ADVERSARIAL_NETWORKS) [[google drive](https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9/edit) | [baidu disk](https://pan.baidu.com/s/1bU9bSRxQnlfw7OkOw7hqjA) (x8gi)] 
- [ ] ...



### 🏆 Competitions won with the library

`change_detection.pytorch` has competitiveness and potential in the change detection competitions.
[Here](https://github.com/likyoo/change_detection.pytorch/blob/main/COMPETITIONS.md) you can find competitions, names of the winners and links to their solutions.



### :page_with_curl: Citing <a name="citing"></a>

```
@misc{likyoocdp:2021,
  Author = {Kaiyu Li, Fulin Sun, Xudong Liu},
  Title = {Change Detection Pytorch},
  Year = {2021},
  Publisher = {GitHub},
  Journal = {GitHub repository},
  Howpublished = {\url{https://github.com/likyoo/change_detection.pytorch}}
}
```



### :books: Reference <a name="reference"></a>

- [qubvel/segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)
- [albumentations-team/albumentations](https://github.com/albumentations-team/albumentations)
- [open-mmlab/mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
- [wenhwu/awesome-remote-sensing-change-detection](https://github.com/wenhwu/awesome-remote-sensing-change-detection)



### :mailbox: Contact<a name="contact"></a>

⚡⚡⚡ I am trying to build this project, if you are interested, don't hesitate to join us! 

👯👯👯 Contact me at likyoo@sdust.edu.cn or pull a request directly or join our WeChat group.
<div align=center><img src="resources/wechat.jpg" alt="wechat group" width="38%" height="38%"  style="zoom:80%;" /></div>
