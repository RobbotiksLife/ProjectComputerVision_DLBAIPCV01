from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSDClassificationHead

import utils
from custom_yolo_dataset import YoloFormatDataset
from torchvision.transforms import v2 as T

from engine.engine import train_one_epoch
from engine.utils import collate_fn
from evaluate_utils import evaluate_custom


def get_model(num_classes):
    model = ssd300_vgg16(weights="DEFAULT")

    classification_head = model.head.classification_head

    layers = [m for m in classification_head.modules() if isinstance(m, torch.nn.Conv2d)]
    in_channels = [layer.in_channels for layer in layers if isinstance(layer, torch.nn.Conv2d)]
    num_anchors = [layer.out_channels // 91 for layer in layers if isinstance(layer, torch.nn.Conv2d)]

    model.head.classification_head = SSDClassificationHead(in_channels, num_anchors, num_classes)

    return model

def get_transform():
    return T.Compose([
        T.Resize((300, 300)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True)
    ]) # Looks strange but fine


def main(num_epochs = 10):
    dataset = YoloFormatDataset(root="OxfordPets_v2_by_species/train", transforms=get_transform())
    # dataset_valid = YoloFormatDataset(root="OxfordPets_v2_by_species/valid", transforms=get_transform()) # <- Uncoment if desided to use valid imagies during training

    data_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    # data_loader_valid = DataLoader(dataset_valid, batch_size=8, shuffle=False, collate_fn=lambda b: tuple(zip(*b))) # <- Uncoment if desided to use valid imagies during training

    model = get_model(num_classes=3) # 2 classes + 1 something = 3
    model.to('cuda')

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=1e-3, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device='cuda', epoch=epoch, print_freq=10)
        lr_scheduler.step()

        # evaluate(model, data_loader_valid, device='cuda') # <- Uncoment if desided to use valid imagies during training

    torch.save(model.state_dict(), 'ssd/ssd_cat_dog.pth')


if __name__ == '__main__':
    # # Train
    # Path('ssd').mkdir(parents=True, exist_ok=True)
    # utils.print_execution_time(lambda: main(num_epochs=15))
    # print(10510.54/60) # 175.1756 min -> 15 epochs

    # -------------------------------------------------------
    # Evaluate
    model = get_model(3)
    model.load_state_dict(torch.load('ssd/ssd_cat_dog.pth'))
    model.to('cuda')

    # # Test prediction
    # model.eval()
    # image_paths = [
    #     f'OxfordPets_v2_by_species\\test\\images\\{n}' for n in [
    #         'american_pit_bull_terrier_134_jpg.rf.a54d3d1580baca55216d2e195d12c515.jpg',
    #         'Abyssinian_15_jpg.rf.0e12ac0df99238e4f77a9eb02877b769.jpg'
    #     ]
    # ]
    # predictions = utils.predict(image_paths, model, get_transform_func=get_transform, device='cuda')
    # predictions = utils.filter_predictions(predictions, score_threshold=0.5)
    # for image_path, prediction in zip(image_paths, predictions):
    #     utils.show_prediction(image_path, prediction)

    # Test Performace
    test_dataset = YoloFormatDataset("OxfordPets_v2_by_species/test", transforms=get_transform())
    evaluate_custom(model, test_dataset, get_transform_func=get_transform)


# 📊 Evaluation Results (per class):
#
# Class 1:
#   TP: 72, FP: 32, FN: 53, TN: 242
#   Precision: 0.6923
#   Recall:    0.5760
#   F1 Score:  0.6288
#   Accuracy:  0.7870
#
# Class 2:
#   TP: 171, FP: 59, FN: 72, TN: 123
#   Precision: 0.7435
#   Recall:    0.7037
#   F1 Score:  0.7230
#   Accuracy:  0.6918
#
# Class all:
#   TP: 243, FP: 91, FN: 125, TN: 365
#   Precision: 0.7275
#   Recall:    0.6603
#   F1 Score:  0.6923
#   Accuracy:  0.7379