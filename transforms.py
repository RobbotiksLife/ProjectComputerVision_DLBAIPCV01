import torchvision.transforms as T

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image

class ToTensor:
    def __call__(self, image):
        return T.ToTensor()(image)

class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        return T.Resize(self.size)(image)
