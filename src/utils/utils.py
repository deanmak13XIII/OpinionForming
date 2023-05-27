import logging
import os
import re
import traceback

import numpy
import pandas
import yaml
from yaml.loader import SafeLoader

from utils.constants import GRAPH_DATA_CSV_FILE_PATH, SEQUENCED_GRAPH_COUNT_FILE_PATH, SAMPLED_MATRICES_FILE_PATH, \
    DATA_DIR

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger("OpinionForming")


def bit_converter(decimal, bits):
    """
    Converts a decimal to a binary digit of given bit size
    :param decimal: The decimal number to be converted to binary
    :param bits: The bits the binary is meant to have
    :return: A *bits size binary digit of type string
    """
    binary = bin(decimal).replace("0b", "")
    if len(binary) <= bits:
        bit_diff = bits - len(binary)
        placeholder_bits = ""
        for bit in range(bit_diff):
            placeholder_bits = placeholder_bits + "0"
        binary = placeholder_bits + binary
    elif len(binary) > bits:
        _logger.warning(f"No {decimal} equivalent for {bits} bit binary number. Returning {len(binary)} bit binary instead!")
    return binary


def binary_to_adjacency_matrix(binary: str, vertices: int):
    """
    Creates an adjacency matrix
    :param binary: A string binary number that represents the connections of the vertices
    :param vertices: The number of vertices in the graph
    :return adj_matrix: An adjacency matrix representing the connections of the vertices in the graph
    """
    matrix = numpy.zeros(shape=(vertices, vertices), dtype=int)
    binary_index = 0
    for row in range(vertices):
        for col in range(row + 1, vertices):
            matrix[row][col] = int(binary[binary_index])
            binary_index += 1
    adj_matrix: numpy.array = matrix + matrix.T
    return adj_matrix


def decimal_matrix_dict_to_adjacency_matrix(decimal_matrix_dict: dict):
    """
    Converts decimal matrix dict to adjacency matrix
    :param decimal_matrix_dict: a dictionary of format {bits: decimal}, with each key-value pair representing
        a row in the adjacency matrix
    """
    vertices = len(decimal_matrix_dict) + 1
    matrix = numpy.zeros(shape=(vertices, vertices), dtype=int)
    adjacency_row_subtractor = 1
    for bits, decimal in decimal_matrix_dict.items():
        binary = bit_converter(decimal=decimal, bits=bits)
        matrix[len(decimal_matrix_dict) - adjacency_row_subtractor] = numpy.insert(
            numpy.array([int(digit) for digit in binary]), 0, numpy.zeros(vertices - adjacency_row_subtractor))
        adjacency_row_subtractor += 1
    matrix[len(decimal_matrix_dict)] = numpy.zeros(vertices)
    adj_matrix: numpy.array = matrix + matrix.T
    return adj_matrix


def colour_vector_to_decimal(colour_vector: numpy.ndarray):
    """
    Converts a colour vector, made of -1 and 1 (safeguarded for any number other than these too), to decimal
    """
    binary = ""
    for number in colour_vector:
        if number < 0:
            number = 0
        else:
            number = 1
        binary = binary + str(number)
    return int(binary, 2)


def decimal_to_colour_vector(decimal, bits):
    """
    Converts a decimal to a colour vector, made only of 1 and -1
    """
    binary = bit_converter(decimal, bits)
    vector = []
    for bin_char in binary:
        number = int(bin_char)
        if number == 0:
            number = -1
        vector.append(number)
    return numpy.array(vector)


def get_sequenced_graph_count(vertices: int, connected_graphs: bool = False):
    """
    Gets the sequenced graph count data for the vertices. This is because the sequences are computer generated/use
    complex math calculations.
    :returns: an int representing the number of possible graphs for graph type specified as true in bool
    """
    graph_type = "connected_non_isomorphic_graphs"
    if connected_graphs:
        graph_type = "connected_graphs"
    try:
        with open(SEQUENCED_GRAPH_COUNT_FILE_PATH()) as f:
            data = yaml.load(f, Loader=SafeLoader)
            if data:
                return [info[graph_type] for info in data if info['vertices'] == vertices][0]
    except IndexError as e:
        raise Exception(f"Sequenced Graph Counts YAML does not contain {vertices} vertices graph data.")


def write_sampled_matrices(sampled_graphs: list, vertices: int):
    """
    Writes the sampled decimal graphs to file
    """
    print(sampled_graphs)
    # Having to cast numpy array in sampled graphs to list as pyyaml cannot write them to file otherwise
    sample_graph_data = [{"vertices": vertices, "connected_noniso_matrices":
        [[' '.join([str(digit) for digit in list(matrow)]) for matrow in matrix] for matrix in sampled_graphs]}]
    if sample_graph_data:
        file_path = get_sampled_matrices_file_path(vertices)
        with open(file_path, 'w') as f:
            yaml.dump(sample_graph_data, f, sort_keys=False)
            _logger.info(f"Written Newly Sampled Matrices to .yaml file [{file_path}]")


def get_sampled_matrices_file_path(vertices: int):
    """
    Returns the name of the Sampled Matrices file path if it exists.
    :param vertices: The vertices to be included in the file name.
    :returns: A file path of the sampled matrices
    """
    file_path = SAMPLED_MATRICES_FILE_PATH()
    file_path_list = file_path.split('\\')
    file_name = file_path_list[-1]
    new_file_name = re.sub(r'\d{1,2}', f'{vertices}', file_name)
    file_path_list[-1] = new_file_name
    file_path = '\\'.join(file_path_list)
    return file_path


def load_written_adj_matrix_data(vertices: int):
    """
    Loads all written data from sample matrices yaml files
    """
    try:
        file_path = get_sampled_matrices_file_path(vertices)
        file_exists = os.path.isfile(file_path)
        _logger.info(f"Loading Sample Matrices [V: {vertices}, file_path: {file_path}]. Please wait.")
        if file_exists:
            with open(file_path) as f:
                data = yaml.load(f, Loader=SafeLoader)
                return data
        else:
            raise Exception(f"Sampled Matrices File Path for vertices[{vertices}] not found.")
    except Exception as e:
        return None


def get_yaml_connoniso_adj_matrices(loaded_data, vertices: int, already_sampled_matrices: list = None):
    """
    [DUPLICATE MATRIX SAFE] - Please know what you are doing before changing duplicate matrix safety
    Gets the sampled decimal graphs of 'vertices' size that has been written to file
    """
    if not loaded_data:
        if already_sampled_matrices:
            return already_sampled_matrices
        else:
            return []

    _logger.info(f"Getting {vertices}-vertex matrices from Loaded Sample Matrix Data.")
    connected_noniso_matrices = []
    if not already_sampled_matrices:
        already_sampled_matrices = []
    else:
        connected_noniso_matrices = already_sampled_matrices
    for info in loaded_data:
        if info['vertices'] == vertices:
            # Not interested in yaml if csv matrices are more than yaml. Yaml should be supplimentary
            if len(already_sampled_matrices) >= len(info['connected_noniso_matrices']):
                break
            for mat in info['connected_noniso_matrices']:
                adj_matrix = numpy.array([matrow.split(' ') for matrow in mat], dtype=int)
                # preventing duplicate adjacency matrices
                if tuple(adj_matrix.flatten()) in [tuple(matrix.flatten()) for matrix in already_sampled_matrices]:
                    continue
                connected_noniso_matrices.append(adj_matrix)
    return connected_noniso_matrices


def get_unique_or_latest_file_name(file_path: str, latest: bool = False):
    """
    Given the file path, returns either a unique file name if the file already exists, or the latest file name
    if the file has already been created and latest=TRUE.
    """
    # file_exists = os.path.isfile(file_path)
    file_path_split = file_path.split('\\')
    file_name = file_path_split[-1]
    file_name_ids = [int(re.search('\d+', file).group(0)) for file in os.listdir(DATA_DIR()) if
                     file.startswith(file_name[:7]) and re.search('\d+', file)]
    # If there are no files, file ids=[0]. If wanting latest, will return same filepath received as arg.
    # If wanting unique, will return file with id=1.
    if not file_name_ids:
        file_name_ids = [0]
    if latest:
        # return same file path if no other files and wanting latest.
        if len(file_name_ids) == 1 and file_name_ids[0] == 0:
            return file_path
        # return the latest of them if there are other files
        file_path_split[-1] = re.sub(r'\d+', f'{max(file_name_ids)}', file_name)
        return "\\".join(file_path_split)
    if not latest:
        # return new file_name by adding 1 to the max id
        file_path_split[-1] = re.sub(r'\d+', f'{max(file_name_ids) + 1}', file_name)
        return "\\".join(file_path_split)


def graph_data_to_csv_dataframe(graph_data_mapping: dict):
    """
    Generates csv data frame to be written to file. All graphdata to be included here.
    """
    header_row = ['vertices', 'matrix', 'graph_degree', 'max_degree', 'min_degree', 'node_degrees',
                  'graph_eccentricity', 'node_eccentricities', 'graph_diameter', 'graph_clique_number',
                  'node_closeness', 'average_closeness', 'min_closeness', 'max_closeness', 'node_betweenness',
                  'average_betweenness', 'min_betweenness', 'max_betweenness', 'average_path_length',
                  'initial_colour_decimal', 'colour_steps', 'colour_states', 'colour_loop_back_steps',
                  'colour_state_change_ratios']
    data_rows = []
    for vertex, graph_data_list in graph_data_mapping.items():
        for graph_data in graph_data_list:
            matrix = [list(row) for row in graph_data.adjacency_matrix]
            initial_state_dec_list = list(graph_data.all_decimal_colour_states.keys())
            colour_steps = list(graph_data.all_decimal_colour_steps.values())
            colour_states_list = list(graph_data.all_decimal_colour_states.values())
            colour_loop_back = list(graph_data.all_decimal_colour_steps_to_loop_back.values())
            colour_ratios = list(graph_data.colour_state_change_ratios.values())
            graph_data_row = [graph_data.vertices_count, matrix, graph_data.graph_degree, graph_data.max_degree,
                              graph_data.min_degree, graph_data.node_degrees, graph_data.graph_eccentricity,
                              graph_data.node_eccentricities, graph_data.graph_diameter,
                              graph_data.graph_clique_number,
                              graph_data.node_closeness, graph_data.average_closeness, graph_data.min_closeness,
                              graph_data.max_closeness, graph_data.node_betweenness,
                              graph_data.average_betweenness,
                              graph_data.min_betweenness, graph_data.max_betweenness,
                              graph_data.average_path_length,
                              initial_state_dec_list, colour_steps, colour_states_list, colour_loop_back,
                              colour_ratios]
            data_rows.append(graph_data_row)
    dataframe = pandas.DataFrame(columns=header_row, data=data_rows)
    return dataframe


def write_all_data_to_csv(graph_data_mapping: dict, csv_dataframe: pandas.DataFrame = None):
    """
    Writes graph data dataframes to csv.
    """
    if csv_dataframe is None:
        csv_dataframe = graph_data_to_csv_dataframe(graph_data_mapping=graph_data_mapping)
    try:
        _logger.info('Attempting to write GraphData to csv file')
        file_path = get_unique_or_latest_file_name(file_path=GRAPH_DATA_CSV_FILE_PATH())
        _logger.info("Writing to file...")
        csv_dataframe.to_csv(file_path, index=False, compression='gzip')
        _logger.info(f'Saved csv file: {file_path}')
    except Exception as e:
        traceback.print_exc()
