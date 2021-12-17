from change_detection_pytorch import *

archs = [Unet, UnetPlusPlus, MAnet, Linknet, FPN, PSPNet, DeepLabV3, DeepLabV3Plus, PAN, STANet, UPerNet]
archs_dict = {a.__name__.lower(): a for a in archs}
print(archs_dict.keys())