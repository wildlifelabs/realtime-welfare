class BGRTransform:
    def __call__(self, img):
        return img[[2, 1, 0], :, :]
        