from typing import List
from ultralytics import YOLO
import os

YOLO_MODEL = "yolov8s"
YOLO_PROJECT_NAME = "yolov8_training"
YOLO_OXFORD_PETS_NAME = f"{YOLO_MODEL}_oxfordpets"

def train_yolov8():
    data_yaml = os.path.abspath('OxfordPets_v2_by_species/data.yaml')

    model = YOLO('yolov8s.pt')

    model.train(
        data=data_yaml,
        epochs=10,
        imgsz=320,
        batch=16,
        project=YOLO_PROJECT_NAME,
        name=YOLO_OXFORD_PETS_NAME,
        save=True
    )

def test_predictions_yolov8(test_image_list: List[str] = None):
    if test_image_list is None:
        test_image_list = [
            f'OxfordPets_v2_by_species\\test\\images\\{n}' for n in [
                'american_pit_bull_terrier_134_jpg.rf.a54d3d1580baca55216d2e195d12c515.jpg',
                'Abyssinian_15_jpg.rf.0e12ac0df99238e4f77a9eb02877b769.jpg'
            ]
        ]
    model = YOLO(f'{YOLO_PROJECT_NAME}/{YOLO_OXFORD_PETS_NAME}/weights/best.pt')
    results = model(
        test_image_list
    )
    for result in results:
        result.show()
        # print(result.boxes)

def test_yolov8():
    model_path = 'yolov8_training/yolov8s_oxfordpets/weights/best.pt'
    data_yaml = os.path.abspath('OxfordPets_v2_by_species/data.yaml')

    model = YOLO(model_path)

    results = model.val(
        data=data_yaml,
        split='test'
    )

    print("Metrics:")
    print(results)

if __name__ == '__main__':
    # train_yolov8()
    # print(6151.08/60) -> 102.518 min

    # test_predictions_yolov8()

    test_yolov8()

# Class     Images  Instances      Box(P          R      mAP50  mAP50-95):
#   all        368        368      0.973      0.964       0.99      0.833
#   cat        125        125      0.981       0.96      0.989      0.869
#   dog        243        243      0.964      0.967      0.991      0.798