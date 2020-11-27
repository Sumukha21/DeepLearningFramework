from Utils.utils import file_writer_normal_list, file_reader, yaml_reader, folder_list_generator, random_list_generator
from Preprocessing.image import image_reader


# folder_path = "C:/Users/Sumukha/Desktop/Projects/Data/dogs-vs-cats/train/train/"
# save_path = "C:/Users/Sumukha/Desktop/Projects/Data/dogs-vs-cats/Lists/sample_30.txt"
# limit = None
# folder_list_generator(folder_path, save_path, limit)

# all_train_list_path = "C:/Users/Sumukha/Desktop/Projects/Data/dogs-vs-cats/Lists/all_train.txt"
# save_path1 = "C:/Users/Sumukha/Desktop/Projects/Data/dogs-vs-cats/Lists/train_list_from_all_train.txt"
# save_path2 = "C:/Users/Sumukha/Desktop/Projects/Data/dogs-vs-cats/Lists/valid_list_from_all_train.txt"
# all_train_list = file_reader(all_train_list_path)
# valid_list = random_list_generator(all_train_list, 5000)
# train_list = [i for i in all_train_list if i not in valid_list]
# file_writer_normal_list(save_path1, train_list)
# file_writer_normal_list(save_path2, valid_list)

# from Models.vgg16 import vgg16_classifier
# import numpy as np
# import os
# from Preprocessing.image import image_reader, image_normalize, image_resize
# os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
# img_path = "C:/Users/Sumukha/Desktop/Projects/Data/dogs-vs-cats/test1/test1/1.jpg"
# image_folder = "C:/Users/Sumukha/Desktop/Projects/Data/dogs-vs-cats/train/train/"
# no_classes = 2
# img_size = [256, 256]
# weights = "C:/Users/Sumukha/Desktop/Projects/Results/cat_vs_dog/classifier_vgg16/20201121-144642/Model_Weights/model_weights.001-0.693165.hdf5"
# img = image_reader(img_path, image_folder, flip=True)
# img = image_resize(img, img_size)
# img = image_normalize(img)
# img = img[np.newaxis, :, :, :]
# model = vgg16_classifier((*img_size, 3), no_classes, weights)
# prediction = np.argmax(model.predict(img), -1)
# print("")

from Orchestrator.orchestrator_v3 import Orchestrator
import matplotlib.pyplot as plt
img_pah = "C:/Users/Sumukha/Desktop/Projects/Data/dogs-vs-cats/test1/test1/2.jpg"
config_path = "C:/Users/Sumukha/Desktop/Projects/Deep_Learning_Framework/Applications/cat_vs_dog/process_input.yaml"
config = yaml_reader(config_path)

img = image_reader(img_pah)
generator = Orchestrator(config)
result = generator({"image": img})
print("")
