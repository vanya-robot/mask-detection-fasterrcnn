import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os
import random
import string
import torch
from object_detection.utils import visualization_utils as vis_utils


def get_transform(train):
    transform = [transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float)]
    if train:
        transform.append(transforms.RandomHorizontalFlip(0.5))
    return transforms.Compose(transform)


def label(label_num=int):
    if label_num == 1:
        return "with_mask"
    elif label_num == 2:
        return "without_mask"
    elif label_num == 3:
        return "mask_worn_incorrect"
    else:
        return "Unknown label"


def visualize(img_path, output_path, model):
    """
    Visualizes image with boundary boxes, defined by FasterRCNN.

    :param output_path: Path to save image with boundary boxes
    :param img_path: Path to image
    :param model: custom pytorch model FasterRCNN, created in mask_detection.ipynb (4 outputs)
    :return: Opens image with boundary boxes
    """
    transform = get_transform(train=False)
    img = Image.open(img_path).convert("RGB")
    img_model = transform(img)
    result = model((img_model,))

    min_score = 0.5
    visualize_objects = 0
    while result[0]["scores"][visualize_objects].item() >= min_score:
        visualize_objects += 1
    for i in range(visualize_objects):
        box_tensor = result[0]["boxes"][i]
        caption = label(result[0]["labels"][i].item())
        score = str(round(result[0]["scores"][i].item(), 2))
        vis_utils.draw_bounding_box_on_image(image=img, xmin=box_tensor[0].item(), ymin=box_tensor[1].item(),
                                             xmax=box_tensor[2].item(),
                                             ymax=box_tensor[3].item(), use_normalized_coordinates=False,
                                             display_str_list=[caption, score], thickness=3)

    img_name = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits))
    img.save(os.path.join(output_path, img_name))
    return "Done."
