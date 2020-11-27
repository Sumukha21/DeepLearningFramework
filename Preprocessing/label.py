import numpy as np


def one_hot_encoder(label, no_classes):
    encoder = np.eye(no_classes)
    label_encoded = encoder[label]
    return label_encoded


def generate_label_name_cat_dog(image_name, startswith_key):
    if image_name.startswith(startswith_key):
        return 1
    else:
        return 0
