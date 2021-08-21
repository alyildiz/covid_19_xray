from sklearn.model_selection import train_test_split
from src.utils import load_dataset


def get_data():
    train_list_file, train_list_class = load_dataset(data_part="train")
    test_x, test_y = load_dataset(data_part="test")

    train_x, val_x, train_y, val_y = train_test_split(
        train_list_file, train_list_class, stratify=train_list_class, test_size=0.3
    )

    return train_x, val_x, test_x, train_y, val_y, test_y
