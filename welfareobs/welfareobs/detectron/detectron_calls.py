import os
import torch
import torchvision
from PIL import Image
from welfareobs.utils.performance_monitor import PerformanceMonitor
import numpy as np
from welfareobs.utils.bgr_transform import BGRTransform 


def image_tensor(image, size):
    """tx image, returns cuda tensor"""
    transform = torchvision.transforms.Compose([
        torchvision.transforms.PILToTensor(), # Convert a PIL Image or ndarray to tensor and scale the values 0->255 to 0.0->1.0
        torchvision.transforms.Resize(
            size=(size,size),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            max_size=None,
            antialias=True
        ),
        # note that we do NOT normalise the image because that is happening inside Detectron2
        BGRTransform()  # since our data source is RGB
    ])
    image = transform(image)
    return image.cuda()


def image_loader(image_name, size):
    """load image, returns cuda tensor"""
    return image_tensor(Image.open(image_name), size)


def predict(img_list, model):
    with torch.no_grad():
        outputs = model([{"image": img, "height": 384, "width": 384} for img in img_list])
    return outputs  # usually returns list of results, one per image





        # with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
        #     # Apply pre-processing to image.
        #     if self.input_format == "RGB":
        #         # whether the model expects BGR inputs or RGB
        #         original_image = original_image[:, :, ::-1]
        #     height, width = original_image.shape[:2]
        #     image = self.aug.get_transform(original_image).apply_image(original_image)
        #     image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        #     image.to(self.cfg.MODEL.DEVICE)

        #     inputs = {"image": image, "height": height, "width": width}

        #     predictions = self.model([inputs])[0]
        #     return predictions