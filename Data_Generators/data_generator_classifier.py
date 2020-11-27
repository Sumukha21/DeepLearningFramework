import numpy as np
import copy
from tensorflow.keras.utils import Sequence
import matplotlib.pyplot as plt
from Utils.utils import file_reader, instance_generator
from Orchestrator.orchestrator_v2 import Orchestrator


class DataGenerator(Sequence):
    """
    Generates batches of images and corresponding labels for training and validation
    """
    def __init__(self, batch_size, img_list_path, processing_graph):
        super(DataGenerator, self).__init__()
        self.batch_size = batch_size
        self.data_list = self.generate_data_list(img_list_path)
        self.data_processing_graph = processing_graph
        self.orchestrator = Orchestrator(copy.deepcopy(self.data_processing_graph))

    @staticmethod
    def generate_callable(function_dict):
        """
        Instantiates the factory/function defined in function_Def
        :param function_dict:
        :return:
        """
        if function_dict.get("factory") is not None and function_dict.get("function") is not None:
            raise Exception("Either factory or function path should be provided, not both")
        elif function_dict.get("factory") is None and function_dict.get("function") is None:
            raise Exception("Either factory or function path should be provided")
        elif function_dict.get("factory") is not None:
            raise instance_generator(factory=function_dict["factory"], params=function_dict["params"])
        elif function_dict.get("function") is not None:
            instance_generator(factory=function_dict["function"])

    @staticmethod
    def generate_data_list(image_list_path):
        """
        Create a data list from the given image list path
        :param image_list_path:
        :return:
        """
        img_list = file_reader(image_list_path)
        data_list = dict()
        data_list["img"] = img_list
        return data_list

    def load_data(self, idx):
        """
        Generate batches of data
        :param idx:
        :return:
        """
        batch_info = dict()
        batch_info["img"] = self.data_list["img"][idx * self.batch_size: (idx + 1) * self.batch_size]
        return batch_info

    def process_data(self, item):
        batch_values = self.load_data(item)
        refined_images = []
        refined_labels = []
        external_inputs = dict()
        for current_img_path in batch_values["img"]:
            external_inputs["img"] = current_img_path
            # print("Processing: ", current_img_path)
            outputs = self.orchestrator(external_inputs)
            refined_images.append(outputs["img"])
            refined_labels.append(outputs["lbl"])
        return np.asarray(refined_images), np.asarray(refined_labels)

    def __len__(self):
        """
        Computes the length of the data samples
        :return:
        """
        return int(np.ceil(len(self.data_list["img"]) / self.batch_size))

    def __getitem__(self, item):
        """
        Returns a batch of images and labels
        :param item:
        :return:
        """
        x, y = self.process_data(item)
        return x, y
