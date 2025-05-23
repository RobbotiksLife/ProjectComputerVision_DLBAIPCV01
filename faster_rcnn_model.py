import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from custom_yolo_dataset import YoloFormatDataset
from engine.engine import train_one_epoch
import utils
import transforms as T

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

from evaluate_utils import evaluate_custom

weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights)

import torch.backends.cudnn
torch.backends.cudnn.benchmark = False

def get_model(num_classes):
    # model = fasterrcnn_resnet50_fpn(pretrained=True)
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_transform():
    return T.Compose([T.ToTensor()])


def main(num_epochs=10):
    dataset = YoloFormatDataset('OxfordPets_v2_by_species/train', transforms=get_transform())
    # dataset_valid = YoloFormatDataset('OxfordPets_v2_by_species/valid', transforms=get_transform())

    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=utils.collate_fn)
    # data_loader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=False, collate_fn=utils.collate_fn)

    model = get_model(num_classes=3)  # 2 classes + background
    model.to('cuda')
    # device = torch.device('cpu')
    # model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device='cuda', epoch=epoch, print_freq=10)
        lr_scheduler.step()
        # evaluate(model, data_loader_valid, device='cuda')

    torch.save(model.state_dict(), 'faster_rcnn/faster_rcnn_cat_dog.pth')


if __name__ == '__main__':
    # Train
    # Path('faster_rcnn').mkdir(parents=True, exist_ok=True)
    # utils.print_execution_time(lambda: main(num_epochs=2))
    # print(7379.60/60) -> 12.99 min

    # -------------------------------------------------------
    # Evaluate
    model = get_model(3)
    model.load_state_dict(torch.load('faster_rcnn/faster_rcnn_cat_dog.pth'))
    model.to('cuda')

    # Test prediction
    # model.eval()
    # image_paths= [
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
#   TP: 120, FP: 39, FN: 5, TN: 211
#   Precision: 0.7547
#   Recall:    0.9600
#   F1 Score:  0.8451
#   Accuracy:  0.8827
#
# Class 2:
#   TP: 231, FP: 22, FN: 12, TN: 124
#   Precision: 0.9130
#   Recall:    0.9506
#   F1 Score:  0.9315
#   Accuracy:  0.9126
#
# Class all:
#   TP: 351, FP: 61, FN: 17, TN: 335
#   Precision: 0.8519
#   Recall:    0.9538
#   F1 Score:  0.9000
#   Accuracy:  0.8979