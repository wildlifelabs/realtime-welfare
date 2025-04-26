import os
import torch
import torchvision
from PIL import Image
from welfareobs.utils.performance_monitor import PerformanceMonitor


def image_tensor(image, size):
    """tx image, returns cuda tensor"""
    performance_monitor: PerformanceMonitor = PerformanceMonitor(
        label="detectron:image_tensor",
        history_size=1
    )
    performance_monitor.track_start()
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
    performance_monitor.track_end()
    print(str(performance_monitor))
    return image.cuda()


def image_loader(image_name, size):
    """load image, returns cuda tensor"""
    return image_tensor(Image.open(image_name), size)


def predict(img, model):
    performance_monitor: PerformanceMonitor = PerformanceMonitor(
        label="detectron:predict",
        history_size=1
    )
    performance_monitor.track_start()
    with torch.no_grad():
        outputs = model([{"image": img}])
    performance_monitor.track_end()
    print(str(performance_monitor))
    return outputs[0]  # usually returns list of results, one per image

