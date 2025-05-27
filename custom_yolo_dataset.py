import os
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset

class YoloFormatDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.labels = list(sorted(os.listdir(os.path.join(root, "labels"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        label_path = os.path.join(self.root, "labels", self.labels[idx])

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        boxes = []
        labels = []
        with open(label_path) as f:
            for line in f:
                cls, x_center, y_center, width, height = map(float, line.strip().split())
                xmin = (x_center - width / 2) * w
                xmax = (x_center + width / 2) * w
                ymin = (y_center - height / 2) * h
                ymax = (y_center + height / 2) * h
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(int(cls) + 1)  # Background class is 0

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Use a just tensor instead of [tensor] for engine.py to work
        image_id = torch.tensor(idx)
        # image_id = torch.tensor([idx])

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)
