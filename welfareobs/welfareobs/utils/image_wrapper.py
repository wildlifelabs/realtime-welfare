import cv2


class ImageWrapper(object):
    def __init__(self, filename: str, w: int = 1920, h: int = 1080):
        self.filename: str = filename
        self.__w = w
        self.__h = h

    @property
    def width(self) -> int:
        return self.__w

    @property
    def height(self) -> int:
        return self.__h

    def image(self, output_width=None, output_height=None, overriding_image = None):
        w: int = output_width if output_width is not None else self.__w
        h: int = output_height if output_height is not None else self.__h
        img = overriding_image
        if overriding_image is None:
            img = cv2.imread(self.filename)
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
