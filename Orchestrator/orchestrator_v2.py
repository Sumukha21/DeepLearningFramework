"""
Module to automatically execute system flow defined in configuration file
"""
__author__ = "Sumukha Manjunath"
__github_name__ = "Sumukha21"
__Version__ = "2.0"
__Date__ = "March 28 2020"


import copy
from Utils.utils import instance_generator


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
    def __init__(self, config):
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
        self.required_outputs = config.pop("outputs", None)

    def __call__(self, external_inputs_dict=None):
        self.external_inputs = external_inputs_dict
        self.double_linked_graph = self.double_linked_graph_generator(copy.deepcopy(self.config))
        self.graph_leaves = None
        self.graph_roots = None
        self.independent_graph_nodes = None
        self.flows = self.flow_generator()
        self.flow_executor()
        if self.required_outputs is not None:
            self.outputs = self.return_outputs()
        return self.outputs

    def reset(self):
        self.external_inputs = None
        self.double_linked_graph = None
        self.graph_leaves = None
        self.graph_roots = None
        self.independent_graph_nodes = None
        self.flows = None
        self.outputs = None

    def return_outputs(self):
        """
        Extracts the outputs of the required nodes from the double linked graph
        :param double_linked_graph:
        :param required_outputs:
        :return:
        """
        outputs = None
        if isinstance(self.required_outputs, dict):
            outputs = dict()
            for output_name, required_node_name in self.required_outputs.items():
                if len(required_node_name.split(".")) > 1:
                    node_name, _, specific_output_name = required_node_name.split(".")
                    outputs[output_name] = self.double_linked_graph[node_name].outputs[specific_output_name]
                elif len(required_node_name.split(".")) == 1:
                    outputs[output_name] = self.double_linked_graph[required_node_name].outputs
        elif isinstance(self.required_outputs, list):
            outputs = []
            for required_node_name in self.required_outputs:
                if len(required_node_name.split(".")) > 1:
                    node_name, _, specific_output_name = required_node_name.split(".")
                    outputs.append(self.double_linked_graph[node_name].outputs[specific_output_name])
                elif len(required_node_name.split(".")) == 1:
                    outputs.append(self.double_linked_graph[required_node_name].outputs)
        elif isinstance(self.required_outputs, str):
            for required_node_name in self.required_outputs:
                if len(required_node_name.split(".")) > 1:
                    node_name, _, specific_output_name = required_node_name.split(".")
                    outputs = self.double_linked_graph[node_name].outputs[specific_output_name]
                elif len(required_node_name.split(".")) == 1:
                    outputs = self.double_linked_graph[required_node_name].outputs
        else:
            raise AttributeError("Required outputs should be provided in one of the following formats: dict, list or str")
        return outputs

    def node_executor(self, node_name):
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
        if self.double_linked_graph[node_name].root:
            if self.double_linked_graph[node_name].value.get("factory", None):
                node_obj = instance_generator(**self.double_linked_graph[node_name].value)
                if callable(node_obj):
                    node_obj = node_obj()
                return node_obj
            elif self.double_linked_graph[node_name].value.get("function", None):
                node_obj = instance_generator(**self.double_linked_graph[node_name].value)
                if self.double_linked_graph[node_name].value.get("params", None):
                    node_obj = node_obj(**self.double_linked_graph[node_name].value['params'])
                return node_obj

        predecessors = self.double_linked_graph[node_name].predecessors
        predecessors_outputs = None
        if isinstance(predecessors, list):
            predecessors_outputs = []
            for predecessor in predecessors:
                if len(predecessor.split('.')) == 1:
                    if self.double_linked_graph[predecessor].outputs is not None:
                        predecessors_outputs.append(self.double_linked_graph[predecessor].outputs)
                    else:
                        predecessor_outputs = self.node_executor(predecessor)
                        if isinstance(predecessor_outputs, list):
                            predecessors_outputs.extend(predecessor_outputs)
                        elif isinstance(predecessor_outputs, dict):
                            predecessors_outputs.extend(list(predecessor_outputs.values()))
                        else:
                            predecessors_outputs.append(predecessor_outputs)
                elif len(predecessor.split('.')) == 3:
                    predecessor_node_name, _, output_name = predecessor.split('.')
                    if self.double_linked_graph[predecessor_node_name].outputs[output_name] is not None:
                        predecessors_outputs.append(self.double_linked_graph[predecessor_node_name].outputs[output_name])
                    else:
                        predecessor_outputs = self.node_executor(predecessor_node_name)
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
                    if self.double_linked_graph[predecessor].outputs is not None:
                        predecessors_outputs[predecessor_output_name] =\
                            self.double_linked_graph[predecessor].outputs
                    else:
                        predecessor_outputs = self.node_executor(predecessor)
                        if isinstance(predecessor_outputs, list):
                            predecessors_outputs[predecessor_output_name] = predecessor_outputs
                        elif isinstance(predecessor_outputs, dict):
                            predecessors_outputs[predecessor_output_name] = predecessor_outputs[predecessor_output_name]
                        else:
                            predecessors_outputs[predecessor_output_name] = predecessor_outputs
                elif len(predecessor.split('.')) == 3:
                    predecessor_node_name, _, output_name = predecessor.split('.')
                    if self.double_linked_graph[predecessor_node_name].outputs[output_name] is not None:
                        predecessors_outputs[predecessor_output_name] = \
                            self.double_linked_graph[predecessor_node_name].outputs[output_name]
                    else:
                        predecessor_outputs = self.node_executor(predecessor_node_name)
                        if isinstance(predecessor_outputs, dict):
                            predecessor_output = predecessor_outputs[output_name]
                            predecessors_outputs[predecessor_output_name] = predecessor_output
                        else:
                            raise("Function/Factory expecting predecessor output to be of type "
                                  "dictionary but got type : ", type(predecessor_outputs))

        if self.double_linked_graph[node_name].value.get('factory', None):
            node_obj = instance_generator(**self.double_linked_graph[node_name].value)
            if callable(node_obj):
                if isinstance(predecessors_outputs, list):
                    return node_obj(*predecessors_outputs)
                elif isinstance(predecessors_outputs, dict):
                    return node_obj(**predecessors_outputs)
            else:
                return node_obj
        elif self.double_linked_graph[node_name].value.get('function', None):
            node_obj = instance_generator(**self.double_linked_graph[node_name].value)
            if isinstance(predecessors_outputs, list):
                if self.double_linked_graph[node_name].value.get('params', None):
                    return node_obj(*predecessors_outputs, **self.double_linked_graph[node_name].value['params'])
                else:
                    return node_obj(*predecessors_outputs)
            elif isinstance(predecessors_outputs, dict):
                if self.double_linked_graph[node_name].value.get('params', None):
                    return node_obj(**predecessors_outputs, **self.double_linked_graph[node_name].value['params'])
                else:
                    return node_obj(**predecessors_outputs)

    def single_flow_executor(self, flow):
        """
        Execution of provided flow node by node. Updates the double linked graph with each nodes output
        :param flow:
        :param double_linked_graph:
        :return:
        """
        for node in flow:
            if self.double_linked_graph[node].outputs is not None:
                continue
            else:
                node_output = self.node_executor(node)
                self.double_linked_graph[node].outputs = node_output

    def flow_executor(self):
        """
        Executes flow after flow in a sequential manner
        :param flows:
        :param double_linked_graph:
        :return:
        """
        flow_0 = self.flows[0]
        commonnality = dict()
        common_initial_flow = []
        for flow_i in self.flows[1:]:
            if not self.flows.index(flow_0) == self.flows.index(flow_i):
                commonnality[str(self.flows.index(flow_0)) + '_' + str(self.flows.index(flow_i))] = []
            for i, j in zip(flow_0, flow_i):
                if i == j:
                    commonnality[str(self.flows.index(flow_0)) + '_' + str(self.flows.index(flow_i))].append(i)
            else:
                break

        if len(commonnality):
            if max(map(len, list(commonnality.values()))) > 1:
                common_initial_flow = [i for i in list(commonnality.values()) if len(i) ==
                                       max(map(len, list(commonnality.values())))][0]
            if len(common_initial_flow):
                for node in common_initial_flow:
                    if self.double_linked_graph[node].outputs is not None:
                        continue
                    else:
                        node_output = self.node_executor(node)
                        self.double_linked_graph[node].outputs = node_output
        for flow in self.flows:
            self.single_flow_executor(flow)

    def path_finder(self, flow_list, node_name=None):
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
            inputs = self.double_linked_graph[node_name].predecessors
            if isinstance(inputs, dict):
                inputs = [inputs[i].split('.')[0] for i in list(inputs.keys())]
                for input_value in inputs:
                    node_index = flow_list.index(node_name)
                    if input_value not in flow_list:
                        flow_list.insert(node_index, input_value)
                        self.path_finder(flow_list, input_value)
                    elif input_value in flow_list and (node_index < flow_list.index(input_value)):
                        flow_list.remove(input_value)
                        flow_list.insert(node_index, input_value)
            elif isinstance(inputs, list):
                inputs = [i.split('.')[0] for i in inputs]
                for input_value in inputs:
                    node_index = flow_list.index(node_name)
                    if input_value not in flow_list:
                        flow_list.insert(node_index, input_value)
                        flow_list = self.path_finder(flow_list, input_value)
                    elif input_value in flow_list and (node_index < flow_list.index(input_value)):
                        flow_list.remove(input_value)
                        flow_list.insert(node_index, input_value)
            return flow_list

        else:
            inputs = self.double_linked_graph[node_name].predecessors
            if len(inputs):
                if isinstance(inputs, dict):
                    inputs = [inputs[i].split('.')[0] for i in list(inputs.keys())]
                    for input_value in list(inputs):
                        node_index = flow_list.index(node_name)
                        if input_value not in flow_list:
                            flow_list.insert(node_index, input_value)
                            flow_list = self.path_finder(flow_list, input_value)
                        elif input_value in flow_list and (node_index < flow_list.index(input_value)):
                            flow_list.remove(input_value)
                            flow_list.insert(node_index, input_value)
                elif isinstance(inputs, list):
                    inputs = [i.split('.')[0] for i in inputs]
                    for input_value in inputs:
                        node_index = flow_list.index(node_name)
                        if input_value not in flow_list:
                            flow_list.insert(node_index, input_value)
                            flow_list = self.path_finder(flow_list, input_value)
                        elif input_value in flow_list and (node_index < flow_list.index(input_value)):
                            flow_list.remove(input_value)
                            flow_list.insert(node_index, input_value)
            return flow_list

    def flow_generator(self):
        """
        Generates all the parallel flows in the double linked graph
        :return: Parallel flows
        """
        self.graph_leaves = [i for i in list(self.double_linked_graph.keys()) if
                             self.double_linked_graph[i].leaf is True]
        self.graph_roots = [i for i in list(self.double_linked_graph.keys()) if
                            self.double_linked_graph[i].root is True]
        self.independent_graph_nodes = [i for i in list(self.double_linked_graph.keys()) if
                                        self.double_linked_graph[i].independent is True]
        parallel_flows = []
        for leaf in self.graph_leaves:
            flow = [leaf]
            parallel_flows.append(self.path_finder(flow))
        for node in self.independent_graph_nodes:
            parallel_flows.append([node])
        return parallel_flows

    def path_param_generator(self, dictionary):
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
                        dictionary["params"][parameter_name] = self.external_inputs[required_param]
            elif isinstance(parameters, list):
                for parameter_index, parameter_value in enumerate(parameters):
                    if isinstance(parameter_value, str) and parameter_value.startswith("external"):
                        required_param = parameter_value.split(".")[-1]
                        dictionary["params"][parameter_index] = self.external_inputs[required_param]
            elif isinstance(parameters, str):
                if parameters.startswith("external"):
                    dictionary["params"] = self.external_inputs[parameters.split(".")[-1]]
            instantiation_dict["params"] = dictionary["params"]
        return instantiation_dict

    def double_linked_graph_generator(self, dictionary):
        """
        Generates a double linked graph for the dictionary containing the project flow
        :param external_inputs:
            Type: Dict
            Dynamic inputs for the nodes when looping is involved
        :param dictionary:
            Type: dict
            Dictionary coontaining the project flow
        :return: double linked graph representation of the dictionary
        """
        double_linked_graph = dict()
        dictionary_keys = list(dictionary.keys())
        for key in dictionary_keys:
            node = AddGraphNode(key)
            node.value = self.path_param_generator(dictionary[key])
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
                                                                                                   "is of type %s" %
                                                                                                   type(inputs)))
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
            if not (len(successors)):
                node.leaf = True
            else:
                node.successors = successors
            if node.leaf and node.root:
                node.independent = True
                node.leaf = False
                node.root = False
            double_linked_graph[key] = node
        return double_linked_graph


