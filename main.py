import os
import re
import datetime
import argparse
from Utils.utils import yaml_reader, check_path, yaml_writer
from Orchestrator.orchestrator_v3 import Orchestrator


def main(experiment_config_path):
    """
    Invokes the experiment defined in the experiment_config_path
    :param experiment_config_path:
        Type: str
        Path to the config file containing the experiment definition
    :return:
    """
    if not(experiment_config_path.endswith(".yml")) and not(experiment_config_path.endswith(".yaml")):
        raise NotImplementedError("Current version only supports yml files for experiment definition")
    experiment_config = yaml_reader(experiment_config_path)
    check_path(experiment_config["variables"]["save_folder"])
    gpu = experiment_config.pop("gpu_device", "-1")
    if not isinstance(gpu, str):
        raise TypeError("Provided gpuDevice should be a string representing gpuID")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    experiment_config = config_file_loader(experiment_config)
    variables = experiment_config.get("variables", None)
    if variables and variables.get("logs_save_folder", None):
        yaml_writer(os.path.join(variables["logs_save_folder"], "experiment_config.yaml"), experiment_config)
    experiment = experiment_config.pop("experiment", None)
    experiment_obj = Orchestrator(experiment.get("control_flow"))
    experiment_obj()


def config_file_loader(config_file):
    """
    Used for replacing placeholders with corresponding variable values in the config file.
    Patern it searches for is "a part of a string starting with a $"
    :param config_file:
        Type: str
        config file containing the experiment configuration
    :return:
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    pattern = r'\$(\w+|([^}]*))'
    variables = config_file.get("variables", dict())
    variables["timestamp"] = timestamp
    replaceable_vars = re.compile(pattern)
    return var_finder(config_file, replaceable_vars, variables)


def var_finder(dictionary, pattern=None, replace_values=None):
    """
    Function to search for the provided pattern in the dictionary input and replace
    them with corresponding replace values
    :param dictionary:
        Type: dict
        Input dictionary which has to be searched at all levels for th provided pattern
    :param pattern:
        Type: regex pattern
        The pattern to be searched for within the input dictionary
    :param replace_values:
        Type: dict
        Dictionary containing the values to replace the found pattern
    :return: Modified dictionary with the patterns replaced
    """
    if isinstance(dictionary, dict):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                var_finder(value, pattern, replace_values)
            elif isinstance(value, list):
                for part_value in value:
                    if isinstance(part_value, dict):
                        var_finder(part_value, pattern, replace_values)
                    elif isinstance(part_value, list):
                        var_finder(part_value, pattern, replace_values)
                    elif isinstance(part_value, str):
                        to_replace1 = list(re.finditer(pattern, part_value))
                        for match_object in to_replace1:
                            match_value = match_object[0].split("$")[1]
                            try:
                                value = value.replace(match_object[0], replace_values[match_value])
                            except ValueError as e:
                                print("%s is not defined in the config file: " % match_value, e)
                        if len(to_replace1):
                            check_path(value)
                            dictionary[key] = value
            else:
                if isinstance(value, str):
                    to_replace1 = list(re.finditer(pattern, value))
                    for match_object in to_replace1:
                        match_value = match_object[0].split("$")[1]
                        try:
                            value = value.replace(match_object[0], replace_values[match_value])
                        except ValueError as e:
                            print("%s is not defined in the config file: " % match_value, e)
                    if len(to_replace1):
                        check_path(value)
                        dictionary[key] = value
    elif isinstance(dictionary, list):
        for key, value in enumerate(dictionary):
            if isinstance(value, dict):
                var_finder(value, pattern, replace_values)
            elif isinstance(value, list):
                for part_value in value:
                    if isinstance(part_value, dict):
                        var_finder(part_value, pattern, replace_values)
                    elif isinstance(part_value, list):
                        var_finder(part_value, pattern, replace_values)
                    elif isinstance(part_value, str):
                        to_replace1 = list(re.finditer(pattern, part_value))
                        for match_object in to_replace1:
                            match_value = match_object[0].split("$")[1]
                            try:
                                value = value.replace(match_object[0], replace_values[match_value])
                            except ValueError as e:
                                print("%s is not defined in the config file: " % match_value, e)
                        if len(to_replace1):
                            check_path(value)
                            dictionary[key] = value
            else:
                if isinstance(value, str):
                    to_replace1 = list(re.finditer(pattern, value))
                    for match_object in to_replace1:
                        match_value = match_object[0].split("$")[1]
                        try:
                            value = value.replace(match_object[0], replace_values[match_value])
                        except ValueError as e:
                            print("%s is not defined in the config file: " % match_value, e)
                    if len(to_replace1):
                        check_path(value)
                        dictionary[key] = value
    return dictionary


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("config_file_path", type=str, help="Path to an experiment config")
    parser = argument_parser.parse_args()
    main(parser.config_file_path)
