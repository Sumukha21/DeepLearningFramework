import cv2
import numpy as np
import matplotlib.pyplot as plt


def overlay_classifier(image, prediction, class_mapping, flip=False):
    predicted_label = class_mapping[prediction[0]]
    label_array = np.zeros((100, image.shape[1], 3), dtype=np.uint8)
    label_array = cv2.putText(label_array, "Predicted as: %s" % predicted_label, (0, 25),
                              cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (255, 255, 255))
    final_array = np.vstack([image, label_array])
    if flip:
        final_array = np.flip(final_array, 2)
    return final_array
