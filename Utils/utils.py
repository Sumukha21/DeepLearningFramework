import os
import yaml
import importlib
import numpy as np
import random


def check_path(path_str):
    if not os.path.exists(path_str):
        path_str = r"%s" % path_str
        os.mkdir(path_str)


def yaml_reader(file_path):
    with open(file_path, "r") as f:
        contents = yaml.safe_load(f)
    return contents


def file_reader(file_path):
    try:
        if not file_path.endswith(".txt"):
            raise ValueError("The provided path %s does not correspond to a text document" % file_path)
        with open(file_path, "r") as f:
            contents = f.readlines()
        contents = [i.split()[0] for i in contents]
        return contents
    except FileNotFoundError as e:
        print("Provided path %s does not exist" % file_path)


def file_writer_normal_list(save_path, contents):
    with open(save_path, "w") as f:
        for line in contents:
            f.write(str(line) + "\n")


def yaml_writer(path, data):
    try:
        if not (path.endswith(".yml")) or (path.endswith(".yaml")):
            raise AttributeError("The path provided does not correspond to a yaml file")
        with open(path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
    except MemoryError as e:
        print("Out of Memory when writing to %s" % path, e)
    except FileExistsError as e:
        print("The file %s you are writing to already exists " % path)


def random_list_generator(distribution, sample_size):
    sample_list = random.sample(distribution, sample_size)
    return sample_list


def object_instantiate(module_name, package=None):
    """
    Imports the module and returns the object of the attribute required
    :param module_name:
        Type: str
        The path to the factory/function
    :param package:
        Type: str
        The path to the project directory which can be used as root
    :return:
    """
    try:
        module_name, attribute_name = module_name.rsplit(".", 1)
        module_init = importlib.import_module(module_name, package=package)
        return getattr(module_init, attribute_name)
    except ModuleNotFoundError as e:
        print("Provided module path %s does not correspond to a valid module" % module_name)


def instance_generator(function=None, factory=None, params=None, package=None):
    """
    Returns an instantiation of the function/factory
    :param function:
        Type: str
        Relative path to the function
    :param factory:
        Type: str
        Relative path to the factory
    :param params:
        Type: dict
        Parameters of the class/function
    :param package:
        Type: str
        The path to the project folder which is used as root when instanntiating
    :return: Instance of the factory/function
    """
    if factory is not None:
        if function is not None:
            raise AttributeError("Either factory path or function path have to be provided, not both")
        if params is None:
            params = dict()
        factory_obj = object_instantiate(module_name=factory, package=package)
        return factory_obj(**params)
    elif function is not None:
        return object_instantiate(module_name=function, package=package)


def folder_list_generator(folder_path, save_path, limit=None):
    contents = os.listdir(folder_path)
    if limit is not None:
        contents = contents[: limit]
    file_writer_normal_list(save_path, contents)


def inference_generator(model, img):
    img = img[np.newaxis, :, :, :]
    prediction = model.predict(img)
    print("Scores: ", prediction)
    return np.argmax(prediction, -1)
