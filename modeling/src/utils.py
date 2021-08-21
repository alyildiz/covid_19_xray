import glob
import os

from src.constants import DATASET_PATH, class_dic
from torchvision import transforms


def load_dataset(data_part):
    list_file, list_class = [], []
    for case in ["Covid", "Normal", "Viral Pneumonia"]:
        path = f"{DATASET_PATH}/{data_part}/{case}"
        os.chdir(path)
        for file in glob.glob("*"):
            list_file.append(path + "/" + file)
            list_class.append(class_dic[case])

    return list_file, list_class


transform_augmentation = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

transform_inference = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
