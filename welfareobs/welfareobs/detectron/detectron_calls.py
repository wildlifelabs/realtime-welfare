import os
import torch
import torchvision
from PIL import Image


def image_tensor(image, size):
    """tx image, returns cuda tensor"""
    transform = torchvision.transforms.Compose([
        torchvision.transforms.PILToTensor(),
        torchvision.transforms.Resize(
            size=(size,size),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            max_size=None,
            antialias=True
        ),
        # torchvision.transforms.Resize(
        #     size=(size, size)
        # ),
    ])
    image = transform(image)
    return image.cuda()


def image_loader(image_name, size):
    """load image, returns cuda tensor"""
    return image_tensor(Image.open(image_name), size)


def predict(img, model):
    with torch.no_grad():
        outputs = model([{"image": img}])
    return outputs[0]  # usually returns list of results, one per image

