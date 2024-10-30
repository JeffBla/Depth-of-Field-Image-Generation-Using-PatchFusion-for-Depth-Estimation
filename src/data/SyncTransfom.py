import random
from torchvision import transforms
import torchvision.transforms.functional as TF

class SyncRandomFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, images):
        if random.random() < self.p:
            return [TF.hflip(img) for img in images]
        return images

class SyncRandomRotation:
    def __init__(self, degrees=10):
        self.degrees = degrees

    def __call__(self, images):
        angle = random.uniform(-self.degrees, self.degrees)
        return [TF.rotate(img, angle) for img in images]

class SyncColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.color_jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )

    def __call__(self, images):
        # Apply color jitter only to RGB images (not depth)
        transform = self.color_jitter.get_params(
            self.color_jitter.brightness,
            self.color_jitter.contrast,
            self.color_jitter.saturation,
            self.color_jitter.hue
        )
        
        results = []
        for i, img in enumerate(images):
            if i != 2:  # Skip depth image (assuming it's the third image)
                img = transform(img)
            results.append(img)
        return results