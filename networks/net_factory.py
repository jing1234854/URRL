try:
    from .unet import UNet
except ImportError:
    from networks.unet import UNet


def net_factory(net_type="unet", in_chns=1, class_num=3):
    if net_type != "unet":
        raise ValueError(f"Unsupported model: {net_type}. This release includes U-Net only.")
    return UNet(in_chns=in_chns, class_num=class_num)
