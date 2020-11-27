"""
Module to automatically execute system flow defined in configuration file
"""
__author__ = "Sumukha Manjunath"
__github_name__ = "Sumukha21"
__Version__ = "3.0"
__Date__ = "March 28 2020"


import multiprocessing as mp
import threading
import copy
from Utils.utils import instance_generator


# apache kafka


class AddGraphNode:
    def __init__(self, key):
        """
        Instantiates a node of a double linked graph
        :param key: The identifier for the node
        """
        self.name = key
        self.value = None
        self.predecessors = []
        self.successors = []
        self.outputs = None
        self.root = False
        self.leaf = False
        self.independent = False
        self.internal_graph = None


class Orchestrator:
    def __init__(self, config, sample_generator_def=None, sample_generator_inputs=None, multi_threading=False):
        """
        Given a config file path containing the project modules and the link between the modules,
        a double linked graph is created and the flow indicated by the graph is executed
        :param config:
            Type: dict
            Dictionary containing the project flow
        :param sample_generator_def:
            Type: Dict
            Dictionary containing path and parameters of the sample generator. Sample generation definition is provided
            when looping is involved
        :param sample_generator_inputs:
            Type: Dict/list/str
            Inputs to the sample generator call method
        :param multi_threading:
            Type: Boolean
            If set to True, multi-threading will be used
        """
        self.config = config
        if sample_generator_def is not None:
            self.sample_generator = self.instantiate_sample_generator(sample_generator_def, sample_generator_inputs)
            self.multi_threading = multi_threading
            self.input_length = self.sample_generator.input_length
        else:
            self.sample_generator = None

    @staticmethod
    def instantiate_sample_generator(sample_generator_def, sample_generator_inputs=None):
        """
        Instantiates sample generator
        :param sample_generator_def:
            Type: Dict
            Dictionary consisting of sample generator function/factory path and parameters
        :return: Instantiates sample generator
        """
        if sample_generator_inputs is not None:
            parameters = sample_generator_inputs["params"]
            if isinstance(parameters, dict):
                for parameter_name, parameter_value in parameters.items():
                    if isinstance(parameter_value, str) and parameter_value.startswith("sample_generator_inputs"):
                        required_param = parameter_value.split(".")[-1]
                        sample_generator_def["params"][parameter_name] = sample_generator_inputs[required_param]
            elif isinstance(parameters, list):
                for parameter_index, parameter_value in enumerate(parameters):
                    if isinstance(parameter_value, str) and parameter_value.startswith("sample_generator_inputs"):
                        required_param = parameter_value.split(".")[-1]
                        sample_generator_def["params"][parameter_index] = sample_generator_inputs[required_param]
            elif isinstance(parameters, str):
                if parameters.startswith("sample_generator_inputs"):
                    sample_generator_def["params"] = sample_generator_inputs[parameters.split(".")[-1]]
        return instance_generator(**sample_generator_def)

    def __call__(self, external_inputs=None):
        if self.sample_generator is not None and callable(self.sample_generator):
            sample_key = self.sample_generator.sample_name
            if sample_key is None:
                outputs = []
            else:
                outputs = dict()
            if self.multi_threading:
                pool = mp.pool()
                for i in range(self.input_length):
                    batch_sample = self.sample_generator(i)
                    batch_input = [tuple(copy.deepcopy(self.config), batch_sample_i)
                                   for batch_sample_i in batch_sample]
                    batch_results = pool.starmap(execute_flow, batch_input)
                    for sample, batch_result in zip(batch_sample, batch_results):
                        if sample_key is None:
                            outputs.append(batch_result)
                        else:
                            outputs[sample[sample_key]] = batch_result
                pool.close()
                pool.join()
            else:
                if self.sample_generator.batch_size > 1:
                    raise AttributeError("Batch size should be one if multi-threading is not being used")
                for i in range(self.input_length):
                    batch_sample = self.sample_generator(i)
                    batch_results = execute_flow(copy.deepcopy(self.config), *batch_sample)
                    if sample_key is None:
                        outputs.append(batch_results)
                    else:
                        outputs[batch_sample[0][sample_key]] = batch_results
        else:
            outputs = execute_flow(self.config, external_inputs)
        return outputs


def double_linked_graph_generator(dictionary, external_inputs=None):
    """
    Generates a double linked graph for the dictionary containing the project flow
    :param external_inputs:
        Type: Dict
        Dynamic inputs for the nodes when looping is involved
    :param dictionary:
        Type: dict
        Dictionary containing the project flow
    :return: double linked graph representation of the dictionary
    """
    double_linked_graph = dict()
    dictionary_keys = list(dictionary.keys())
    for key in dictionary_keys:
        node = AddGraphNode(key)
        if dictionary[key].get("internal_graph", None):
            node.internal_graph = dictionary[key]["internal_graph"]
        else:
            node.value = path_param_generator(dictionary[key], external_inputs)
        inputs = dictionary[key].get('inputs', None)
        if inputs:
            if isinstance(inputs, list):
                for input_value in inputs:
                    if input_value.split('.')[0] == key:
                        raise ValueError("Inputs to a module cannot be the module itself :"
                                         " ('key': %s, 'inputs: %s')" % (key, input_value.split('.')[0]))
                    elif input_value.split('.')[0] not in dictionary_keys:
                        raise ValueError("Error in Inputs '%s' for module '%s' " % (input_value.split('.')[0], key))
                    else:
                        node.predecessors.append(input_value)
            elif isinstance(inputs, dict):
                node.predecessors = dict()
                for input_key in list(inputs.keys()):
                    if inputs[input_key].split('.')[0] == key:
                        raise ValueError("Inputs to a module cannot be the module itself :"
                                         " ('key': %s, 'inputs: %s')" % (key, inputs[input_key].split('.')[0]))
                    elif inputs[input_key].split('.')[0] not in dictionary_keys:
                        raise ValueError("Error in Inputs '%s' for module '%s' " % (inputs[input_key], key))
                    else:
                        node.predecessors[input_key] = inputs[input_key]
            elif isinstance(inputs, str):
                if inputs.split('.')[0] == key:
                    raise ValueError("Inputs to a module cannot be the module itself :"
                                     " ('key': %s, 'inputs: %s')" % (key, inputs.split('.')[0]))
                elif inputs.split('.')[0] not in dictionary_keys:
                    raise ValueError("Error in Inputs '%s' for module '%s' " % (inputs.split('.')[0], key))
                else:
                    node.predecessors.append(inputs)
            else:
                raise ValueError("inputs should be either a list, dictionary or a string", str(inputs +
                                 "is of type %s" % type(inputs)))
        else:
            node.root = True
        successors = []
        for other_key in dictionary_keys:
            if not other_key == key:
                other_inputs = dictionary[other_key].get('inputs')
                if other_inputs:
                    if isinstance(other_inputs, list):
                        other_inputs = [i.split('.')[0] for i in other_inputs]
                        if key in other_inputs:
                            successors.append(other_key)
                    elif isinstance(other_inputs, dict):
                        other_inputs = [other_inputs[i].split('.')[0] for i in list(other_inputs.keys())]
                        if key in other_inputs:
                            successors.append(other_key)
                    elif isinstance(other_inputs, str):
                        other_inputs = other_inputs.split('.')[0]
                        if key == other_inputs:
                            successors.append(other_key)
                    else:
                        raise ValueError("inputs should be either a list, dictionary or a string", str(other_inputs)
                                         + "is of type %s" % type(other_inputs))
        if not(len(successors)):
            node.leaf = True
        else:
            node.successors = successors
        if node.leaf and node.root:
            node.independent = True
            node.leaf = False
            node.root = False
        double_linked_graph[key] = node
    return double_linked_graph


def path_param_generator(dictionary, external_inputs=None):
    instantiation_dict = dict()
    if dictionary.get('function', None) and dictionary.get('factory', None):
        raise AttributeError("Either factory path or function path has to be given not both")
    elif dictionary.get('function', None):
        instantiation_dict['function'] = dictionary.get('function')
    elif dictionary.get('factory', None):
        instantiation_dict['factory'] = dictionary.get('factory')
    else:
        raise AttributeError("Either factory path or function path has to be given")
    if dictionary.get('params', None):
        parameters = dictionary.get("params")
        if isinstance(parameters, dict):
            for parameter_name, parameter_value in parameters.items():
                if isinstance(parameter_value, str) and parameter_value.startswith("external"):
                    required_param = parameter_value.split(".")[-1]
                    dictionary["params"][parameter_name] = external_inputs[required_param]
        elif isinstance(parameters, list):
            for parameter_index, parameter_value in enumerate(parameters):
                if isinstance(parameter_value, str) and parameter_value.startswith("external"):
                    required_param = parameter_value.split(".")[-1]
                    dictionary["params"][parameter_index] = external_inputs[required_param]
        elif isinstance(parameters, str):
            if parameters.startswith("external"):
                dictionary["params"] = external_inputs[parameters.split(".")[-1]]
        instantiation_dict["params"] = dictionary["params"]
    return instantiation_dict


def flow_generator(double_linked_graph):
    """
    Generates all the parallel flows in the double linked graph
    :return: Parallel flows
    """
    graph_leaves = [i for i in list(double_linked_graph.keys()) if
                    double_linked_graph[i].leaf is True]
    independent_graph_nodes = [i for i in list(double_linked_graph.keys()) if
                               double_linked_graph[i].independent is True]
    parallel_flows = []
    for leaf in graph_leaves:
        flow = [leaf]
        parallel_flows.append(path_finder(flow, double_linked_graph))
    for node in independent_graph_nodes:
        parallel_flows.append([node])
    return parallel_flows


def path_finder(flow_list, double_linked_graph, node_name=None):
    """
    1) If node name is not given we start from the root and directly go to its predecessors.
    2) Start with one predecessor at a time by modifying the flow_list as [predecessor1, leaf]
    3) With node name as predecessor, repeat step 1 and 2
    4) The process is repeated till all the nodes in the flow are covered
    :param double_linked_graph:
        Type: Dict
        Graph Data Structure constructed using user config
    :param flow_list:
        Type: list
        List of node nameswhich are dependent on one another
    :param node_name:
        Type: str
        Name of the node which has to be added to the flow in the right order
    :return: Updated flow list (Ordered from predecessors to successors)
    """
    if node_name is None:
        node_name = flow_list[0]
        inputs = double_linked_graph[node_name].predecessors
        if isinstance(inputs, dict):
            inputs = [inputs[i].split('.')[0] for i in list(inputs.keys())]
            for input_value in inputs:
                node_index = flow_list.index(node_name)
                if input_value not in flow_list:
                    flow_list.insert(node_index, input_value)
                    path_finder(flow_list, double_linked_graph, input_value)
                elif input_value in flow_list and (node_index < flow_list.index(input_value)):
                    flow_list.remove(input_value)
                    flow_list.insert(node_index, input_value)
        elif isinstance(inputs, list):
            inputs = [i.split('.')[0] for i in inputs]
            for input_value in inputs:
                node_index = flow_list.index(node_name)
                if input_value not in flow_list:
                    flow_list.insert(node_index, input_value)
                    flow_list = path_finder(flow_list, double_linked_graph, input_value)
                elif input_value in flow_list and (node_index < flow_list.index(input_value)):
                    flow_list.remove(input_value)
                    flow_list.insert(node_index, input_value)
        return flow_list

    else:
        inputs = double_linked_graph[node_name].predecessors
        if len(inputs):
            if isinstance(inputs, dict):
                inputs = [inputs[i].split('.')[0] for i in list(inputs.keys())]
                for input_value in list(inputs):
                    node_index = flow_list.index(node_name)
                    if input_value not in flow_list:
                        flow_list.insert(node_index, input_value)
                        flow_list = path_finder(flow_list, double_linked_graph, input_value)
                    elif input_value in flow_list and (node_index < flow_list.index(input_value)):
                        flow_list.remove(input_value)
                        flow_list.insert(node_index, input_value)
            elif isinstance(inputs, list):
                inputs = [i.split('.')[0] for i in inputs]
                for input_value in inputs:
                    node_index = flow_list.index(node_name)
                    if input_value not in flow_list:
                        flow_list.insert(node_index, input_value)
                        flow_list = path_finder(flow_list, double_linked_graph, input_value)
                    elif input_value in flow_list and (node_index < flow_list.index(input_value)):
                        flow_list.remove(input_value)
                        flow_list.insert(node_index, input_value)
        return flow_list


def parallel_executor(flows, double_linked_graph):
    """
    Parallel execution of each flow in flows
    :param flows:
        Type: list
        List of flows to be executed (generated by flow_generator)
    :param double_linked_graph:
        Type: Dict
        Graph Data Structure constructed using user config
    :return:
    """
    flow_0 = flows[0]
    commonnality = dict()
    common_initial_flow = []
    for flow_i in flows[1:]:
        if not flows.index(flow_0) == flows.index(flow_i):
            commonnality[str(flows.index(flow_0)) + '_' + str(flows.index(flow_i))] = []
        for i, j in zip(flow_0, flow_i):
            if i == j:
                commonnality[str(flows.index(flow_0)) + '_' + str(flows.index(flow_i))].append(i)
        else:
            break

    if len(commonnality):
        if max(map(len, list(commonnality.values()))) > 1:
            common_initial_flow = [i for i in list(commonnality.values()) if len(i) ==
                                   max(map(len, list(commonnality.values())))][0]
        if len(common_initial_flow):
            for node in common_initial_flow:
                if double_linked_graph[node].outputs is not None:
                    continue
                else:
                    node_output = node_executor(node, double_linked_graph)
                    double_linked_graph[node].outputs = node_output

    threads = []
    for flow in flows:
        threads.append(threading.Thread(target=single_flow_executor, args=(flow, double_linked_graph)))

    for thread_i in threads:
        thread_i.start()

    for thread_i in threads:
        thread_i.join()

    return double_linked_graph


def sequential_flow_executor(flows, double_linked_graph):
    """
    Executes flow after flow in a sequential manner
    :param flows:
    :param double_linked_graph:
    :return:
    """
    flow_0 = flows[0]
    commonnality = dict()
    common_initial_flow = []
    for flow_i in flows[1:]:
        if not flows.index(flow_0) == flows.index(flow_i):
            commonnality[str(flows.index(flow_0)) + '_' + str(flows.index(flow_i))] = []
        for i, j in zip(flow_0, flow_i):
            if i == j:
                commonnality[str(flows.index(flow_0)) + '_' + str(flows.index(flow_i))].append(i)
        else:
            break

    if len(commonnality):
        if max(map(len, list(commonnality.values()))) > 1:
            common_initial_flow = [i for i in list(commonnality.values()) if len(i) ==
                                   max(map(len, list(commonnality.values())))][0]
        if len(common_initial_flow):
            for node in common_initial_flow:
                if double_linked_graph[node].outputs is not None:
                    continue
                else:
                    node_output = node_executor(node, double_linked_graph)
                    double_linked_graph[node].outputs = node_output
    for flow in flows:
        double_linked_graph = single_flow_executor(flow, double_linked_graph)
    return double_linked_graph


def single_flow_executor(flow, double_linked_graph):
    """
    Execution of provided flow node by node. Updates the double linked graph with each nodes output
    :param flow:
    :param double_linked_graph:
    :return:
    """
    for node in flow:
        if double_linked_graph[node].outputs is not None:
            continue
        else:
            node_output = node_executor(node, double_linked_graph)
            double_linked_graph[node].outputs = node_output
    return double_linked_graph


def node_executor(node_name, double_linked_graph):
    """
    Given a node name, the function/class corresponding to the node is executed. Before execution,
    we check if all the predecessors outputs are computed.
    Either the entire output of predecessor may be given as input or a part of the output can also
    be provided.
    :param double_linked_graph:
    :param node_name:
        Type: str
        Name of the node in the double linked graph whose output is required
    :return: Output of the function/class corresponding to node name
    """
    if double_linked_graph[node_name].root:
        if double_linked_graph[node_name].internal_graph is not None:
            internal_experiment = double_linked_graph[node_name].internal_graph
            multi_threading = internal_experiment.get("multi_threading", False)
            orchestrator = Orchestrator(config=internal_experiment.get("conrtol_flow"),
                                        sample_generator_def=internal_experiment.get("sample_generator"),
                                        multi_threading=multi_threading)
            return orchestrator()
        else:
            if double_linked_graph[node_name].value.get("factory", None):
                node_obj = instance_generator(**double_linked_graph[node_name].value)
                if callable(node_obj):
                    node_obj = node_obj()
                return node_obj
            elif double_linked_graph[node_name].value.get("function", None):
                node_obj = instance_generator(**double_linked_graph[node_name].value)
                if double_linked_graph[node_name].value.get("params", None):
                    node_obj = node_obj(**double_linked_graph[node_name].value['params'])
                return node_obj

    predecessors = double_linked_graph[node_name].predecessors
    predecessors_outputs = None
    if isinstance(predecessors, list):
        predecessors_outputs = []
        for predecessor in predecessors:
            if len(predecessor.split('.')) == 1:
                if double_linked_graph[predecessor].outputs is not None:
                    predecessors_outputs.append(double_linked_graph[predecessor].outputs)
                else:
                    predecessor_outputs = node_executor(predecessor, double_linked_graph)
                    if isinstance(predecessor_outputs, list):
                        predecessors_outputs.extend(predecessor_outputs)
                    elif isinstance(predecessor_outputs, dict):
                        predecessors_outputs.extend(list(predecessor_outputs.values()))
                    else:
                        predecessors_outputs.append(predecessor_outputs)
            elif len(predecessor.split('.')) == 3:
                predecessor_node_name, _, output_name = predecessor.split('.')
                if double_linked_graph[predecessor_node_name].outputs[output_name] is not None:
                    predecessors_outputs.append(double_linked_graph[predecessor_node_name].outputs[output_name])
                else:
                    predecessor_outputs = node_executor(predecessor_node_name, double_linked_graph)
                    if isinstance(predecessor_outputs, dict):
                        predecessor_output = predecessor_outputs[output_name]
                        predecessors_outputs.extend(predecessor_output)
                    else:
                        raise("Function/Factory expecting predecessor output to be of type "
                              "dictionary but got type : ", type(predecessor_outputs))
    elif isinstance(predecessors, dict):
        predecessors_outputs = dict()
        for predecessor_output_name, predecessor in predecessors.items():
            if len(predecessor.split('.')) == 1:
                if double_linked_graph[predecessor].outputs is not None:
                    predecessors_outputs[predecessor_output_name] =\
                        double_linked_graph[predecessor].outputs
                else:
                    predecessor_outputs = node_executor(predecessor, double_linked_graph)
                    if isinstance(predecessor_outputs, list):
                        predecessors_outputs[predecessor_output_name] = predecessor_outputs
                    elif isinstance(predecessor_outputs, dict):
                        predecessors_outputs[predecessor_output_name] = predecessor_outputs[predecessor_output_name]
                    else:
                        predecessors_outputs[predecessor_output_name] = predecessor_outputs
            elif len(predecessor.split('.')) == 3:
                predecessor_node_name, _, output_name = predecessor.split('.')
                if double_linked_graph[predecessor_node_name].outputs[output_name] is not None:
                    predecessors_outputs[predecessor_output_name] = \
                        double_linked_graph[predecessor_node_name].outputs[output_name]
                else:
                    predecessor_outputs = node_executor(predecessor_node_name, double_linked_graph)
                    if isinstance(predecessor_outputs, dict):
                        predecessor_output = predecessor_outputs[output_name]
                        predecessors_outputs[predecessor_output_name] = predecessor_output
                    else:
                        raise("Function/Factory expecting predecessor output to be of type "
                              "dictionary but got type : ", type(predecessor_outputs))

    if double_linked_graph[node_name].internal_graph is not None:
        internal_experiment = double_linked_graph[node_name].internal_graph
        multi_threading = internal_experiment.get("multi_threading", False)
        orchestrator = Orchestrator(config=internal_experiment.get("control_flow"),
                                    sample_generator_def=internal_experiment.get("sample_generator"),
                                    multi_threading=multi_threading)
        return orchestrator()

    else:
        if double_linked_graph[node_name].value.get('factory', None):
            node_obj = instance_generator(**double_linked_graph[node_name].value)
            if callable(node_obj):
                if isinstance(predecessors_outputs, list):
                    return node_obj(*predecessors_outputs)
                elif isinstance(predecessors_outputs, dict):
                    return node_obj(**predecessors_outputs)
            else:
                return node_obj
        elif double_linked_graph[node_name].value.get('function', None):
            node_obj = instance_generator(**double_linked_graph[node_name].value)
            if isinstance(predecessors_outputs, list):
                if double_linked_graph[node_name].value.get('params', None):
                    return node_obj(*predecessors_outputs, **double_linked_graph[node_name].value['params'])
                else:
                    return node_obj(*predecessors_outputs)
            elif isinstance(predecessors_outputs, dict):
                if double_linked_graph[node_name].value.get('params', None):
                    return node_obj(**predecessors_outputs, **double_linked_graph[node_name].value['params'])
                else:
                    return node_obj(**predecessors_outputs)


def return_outputs(double_linked_graph, required_outputs):
    """
    Extracts the outputs of the required nodes from the double linked graph
    :param double_linked_graph:
    :param required_outputs:
    :return:
    """
    outputs = None
    if isinstance(required_outputs, dict):
        outputs = dict()
        for output_name, required_node_name in required_outputs.items():
            if len(required_node_name.split(".")) > 1:
                node_name, _, specific_output_name = required_node_name.split(".")
                outputs[output_name] = double_linked_graph[node_name].outputs[specific_output_name]
            elif len(required_node_name.split(".")) == 1:
                outputs[output_name] = double_linked_graph[required_node_name].outputs
    elif isinstance(required_outputs, list):
        outputs = []
        for required_node_name in required_outputs:
            if len(required_node_name.split(".")) > 1:
                node_name, _, specific_output_name = required_node_name.split(".")
                outputs.append(double_linked_graph[node_name].outputs[specific_output_name])
            elif len(required_node_name.split(".")) == 1:
                outputs.append(double_linked_graph[required_node_name].outputs)
    elif isinstance(required_outputs, str):
        for required_node_name in required_outputs:
            if len(required_node_name.split(".")) > 1:
                node_name, _, specific_output_name = required_node_name.split(".")
                outputs = double_linked_graph[node_name].outputs[specific_output_name]
            elif len(required_node_name.split(".")) == 1:
                outputs = double_linked_graph[required_node_name].outputs
    else:
        raise AttributeError("Required outputs should be provided in one of the following formats: dict, list or str")
    return outputs


def execute_flow(config, external_inputs=None):
    """
    Construction of graph data structure from config file, generation of flows and its execution
    :param config:
    :param external_inputs:
    :return:
    """
    required_outputs = config.pop("outputs", None)
    double_linked_graph = double_linked_graph_generator(copy.deepcopy(config), external_inputs)
    parallel_flows = flow_generator(double_linked_graph)
    # double_linked_graph = parallel_executor(parallel_flows, double_linked_graph)
    double_linked_graph = sequential_flow_executor(parallel_flows, double_linked_graph)
    if required_outputs is not None:
        outputs = return_outputs(double_linked_graph, required_outputs)
        return outputs
    else:
        return None
