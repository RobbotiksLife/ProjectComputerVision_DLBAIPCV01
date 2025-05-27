from random import random
import time
import torch

from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def print_execution_time(task):
    start_time = time.time()
    task()
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"execution_time: {execution_time:.2f}")


@torch.no_grad()
def predict(image_paths, model, get_transform_func, device='cuda'):
    images = []
    for image_path in image_paths:
        image = get_transform_func()(Image.open(image_path).convert("RGB")).to(device)
        images.append(image)

    model.eval()
    with torch.no_grad():
        outputs = model(images)

    return outputs

def random_color_rgb():
    return (random(), random(), random())

def show_prediction(image_path, prediction):
    image = Image.open(image_path).convert("RGB")
    plt.imshow(image)
    ax = plt.gca()

    boxes = prediction['boxes'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()

    for box, score, label in zip(boxes, scores, labels):
        xmin, ymin, xmax, ymax = box
        color = random_color_rgb()
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin - 5, f'Class: {label}, Score: {score:.2f}', color=color, fontsize=8)

    plt.axis('off')
    plt.show()


def filter_predictions(predictions, score_threshold=0.5):
    filtered = []
    for prediction in predictions:
        keep = prediction['scores'] >= score_threshold
        filtered_prediction = {
            'boxes': prediction['boxes'][keep],
            'scores': prediction['scores'][keep],
            'labels': prediction['labels'][keep]
        }
        filtered.append(filtered_prediction)
    return filtered