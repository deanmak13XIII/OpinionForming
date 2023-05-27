import ast
import logging
import traceback
from collections import OrderedDict
from dataclasses import dataclass

import igraph
import numpy
import pandas
from igraph import Graph

from utils.constants import GRAPH_DATA_CSV_FILE_PATH, MINIMUM_VERTICES_SAMPLE_SIZE, SAMPLE_GRAPH_COUNT
from utils.utils import bit_converter, colour_vector_to_decimal, decimal_to_colour_vector, binary_to_adjacency_matrix, \
    get_sequenced_graph_count, write_sampled_matrices, get_yaml_connoniso_adj_matrices, \
    decimal_matrix_dict_to_adjacency_matrix, load_written_adj_matrix_data, get_unique_or_latest_file_name, \
    write_all_data_to_csv

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger("OpinionForming")


@dataclass
class GraphData:
    """
    A dataclass that stores all the necessary data pertaining to a graph
    """
    adjacency_matrix: numpy.ndarray = None
    igraph: Graph = None
    vertices_count: int = None
    graph_degree: float = None
    max_degree: int = None
    min_degree: int = None
    node_degrees: list = None
    graph_eccentricity: float = None
    node_eccentricities: list = None
    graph_diameter: float = None
    graph_clique_number: int = None
    node_closeness: list = None
    average_closeness: float = None
    min_closeness: float = None
    max_closeness: float = None
    node_betweenness: list = None
    average_betweenness: float = None
    min_betweenness: float = None
    max_betweenness: float = None
    average_path_length: float = None
    all_decimal_colour_states: dict = None
    all_decimal_colour_steps: dict = None
    all_decimal_colour_steps_to_loop_back: dict = None
    colour_state_change_ratios: dict = None

    def fill_igraph(self):
        """
        Fills in the igraph variable if the adjacency matrix has been created
        """
        if self.adjacency_matrix is not None:
            self.igraph = Graph().Adjacency(self.adjacency_matrix, mode="undirected")
            return self
        else:
            raise Exception("Can't create igraph without adjacency matrix buddy")

    def fill_data(self):
        """
        Fills the dataclass given that the adjacency_matrix has been provided.
        """
        if self.adjacency_matrix is not None:
            self.igraph = Graph().Adjacency(self.adjacency_matrix, mode="undirected")
            self.vertices_count = int(self.igraph.vcount())
            self.graph_degree = float(round(numpy.mean(self.igraph.degree()), 3))
            self.max_degree = int(self.igraph.maxdegree())
            self.min_degree = int(numpy.min(self.igraph.degree()))
            self.node_degrees = list(self.igraph.degree())
            self.graph_eccentricity = float(round(numpy.mean(self.igraph.eccentricity()), 3))
            self.node_eccentricities = list(self.igraph.eccentricity())
            self.graph_diameter = int(self.igraph.diameter())
            self.graph_clique_number = int(self.igraph.clique_number())
            self.node_closeness = [float(round(cl, 3)) for cl in self.igraph.closeness()]
            self.average_closeness = float(round(numpy.mean(self.node_closeness), 3))
            self.min_closeness = float(round(numpy.min(self.node_closeness), 3))
            self.max_closeness = float(round(numpy.max(self.node_closeness), 3))
            self.node_betweenness = [float(round(bt, 3)) for bt in self.igraph.betweenness()]
            self.average_betweenness = float(round(numpy.mean(self.node_betweenness), 3))
            self.min_betweenness = float(round(numpy.min(self.node_betweenness), 3))
            self.max_betweenness = float(round(numpy.max(self.node_betweenness), 3))
            self.average_path_length = float(round(self.igraph.average_path_length(), 3))
            self.all_decimal_colour_states, self.all_decimal_colour_steps_to_loop_back = _create_colour_states(
                self.adjacency_matrix)
            self.all_decimal_colour_steps = {initial_col: len(col_states) for initial_col, col_states in
                                             self.all_decimal_colour_states.items()}
            self.colour_state_change_ratios = get_change_in_states(colour_states_dict=self.all_decimal_colour_states,
                                                                   vertex_count=self.vertices_count)
            return self
        else:
            raise Exception("Can't fill data without an adjacency matrix pal.")


def read_graph_data_from_csv(vertex_max_value: int = 0, vertex_min_value: int = 0, entire_dataframe: bool = False):
    """
    Reads GraphData written to csv and creates actual GraphData objects. Places the objects in a dictionary of
    {vertices: [GraphData list]}.
    :returns: A dictionary of vertices(key) against the list of GraphData's(value)
    """
    try:
        file_path = get_unique_or_latest_file_name(file_path=GRAPH_DATA_CSV_FILE_PATH(), latest=True)
        data = pandas.read_csv(file_path, index_col=False, compression='gzip')
        _logger.info(f"Completed reading saved GraphData from {file_path}")
        if entire_dataframe:
            return data
        _logger.info("Proceeding to create GraphData data classes from saved GraphData. Please Wait")
        graph_data_mapping = {}
        cutoff_vertex = vertex_max_value + 1
        for index, row in data.iterrows():
            vertices = row['vertices']
            if vertices == cutoff_vertex:
                # Breaking to avoid creating graph data that isn't wanted
                break
            if vertices < vertex_min_value:
                continue
            matrix = numpy.array(ast.literal_eval(row['matrix']))
            graph_degree = row['graph_degree']
            max_degree = row['max_degree']
            min_degree = row['min_degree']
            node_degrees = ast.literal_eval(row['node_degrees'])
            graph_eccentricity = row['graph_eccentricity']
            node_eccentricities = ast.literal_eval(row['node_eccentricities'])
            graph_diameter = row['graph_diameter']
            graph_clique_number = row['graph_clique_number']
            node_closeness = ast.literal_eval(row['node_closeness'])
            average_closeness = row['average_closeness']
            min_closeness = row['min_closeness']
            max_closeness = row['max_closeness']
            node_betweenness = ast.literal_eval(row['node_betweenness'])
            average_betweenness = row['average_betweenness']
            min_betweenness = row['min_betweenness']
            max_betweenness = row['max_betweenness']
            average_path_length = row['average_path_length']
            initial_colour_decimal = ast.literal_eval(row['initial_colour_decimal'])
            colour_steps = ast.literal_eval(row['colour_steps'])
            colour_states = ast.literal_eval(row['colour_states'])
            colour_loop_back_steps = ast.literal_eval(row['colour_loop_back_steps'])
            colour_ratios = ast.literal_eval(row['colour_state_change_ratios'])
            all_decimal_colour_states = {initial_col: colour_states[initial_colour_decimal.index(initial_col)]
                                         for initial_col in initial_colour_decimal}
            all_decimal_colour_states = OrderedDict(
                sorted(all_decimal_colour_states.items(), key=lambda x: len(x[1]), reverse=True))
            all_decimal_colour_steps = {initial_col: colour_steps[initial_colour_decimal.index(initial_col)]
                                        for initial_col in initial_colour_decimal}
            all_decimal_colour_steps_to_loop_back = {
                initial_col: colour_loop_back_steps[initial_colour_decimal.index(initial_col)]
                for initial_col in initial_colour_decimal}
            colour_state_change_ratios = {initial_col: round(colour_ratios[initial_colour_decimal.index(initial_col)], 3)
                                          for initial_col in initial_colour_decimal}
            if vertices not in graph_data_mapping:
                graph_data_mapping[vertices] = []
            if vertices > MINIMUM_VERTICES_SAMPLE_SIZE() and len(graph_data_mapping[vertices]) >= SAMPLE_GRAPH_COUNT():
                continue
            graph_data = GraphData(adjacency_matrix=matrix, vertices_count=vertices, graph_degree=graph_degree,
                                   max_degree=max_degree, min_degree=min_degree, node_degrees=node_degrees,
                                   graph_eccentricity=graph_eccentricity, node_eccentricities=node_eccentricities,
                                   graph_diameter=graph_diameter, graph_clique_number=graph_clique_number,
                                   node_closeness=node_closeness, average_closeness=average_closeness,
                                   min_closeness=min_closeness, max_closeness=max_closeness,
                                   node_betweenness=node_betweenness, average_betweenness=average_betweenness,
                                   min_betweenness=min_betweenness, max_betweenness=max_betweenness,
                                   average_path_length=average_path_length,
                                   all_decimal_colour_states=all_decimal_colour_states,
                                   all_decimal_colour_steps=all_decimal_colour_steps,
                                   all_decimal_colour_steps_to_loop_back=all_decimal_colour_steps_to_loop_back,
                                   colour_state_change_ratios=colour_state_change_ratios).fill_igraph()
            graph_data_mapping[vertices].append(graph_data)
        return graph_data_mapping
    except Exception as e:
        _logger.error(f"Encountered an error in reading csv: {e}")
        traceback.print_exc()
        return {}


def create_matrices(vertices: int):
    """
    Generates all the different possible unlabelled graphs possible from the number of vertices given
    :param vertices: The number of vertices the graph is to have
    """
    _logger.info(f"Beginning graph creations. Please wait...")
    matrix_gap_count = int(((vertices - 1) * vertices) / 2)
    binaries_count = numpy.power(2, matrix_gap_count) - 1
    connected_adj_matrices = []
    connected_noniso_adj_matrices = []
    for decimal in range(binaries_count):
        decimal += 1
        binary = bit_converter(decimal=decimal, bits=matrix_gap_count)
        adj_matrix = binary_to_adjacency_matrix(binary, vertices)
        connected_adj_matrices, connected_noniso_adj_matrices = _sort_adjacency_matrix(adj_matrix,
                                                                                       connected_adj_matrices,
                                                                                       connected_noniso_adj_matrices)
    _logger.info(
        f"Completed creating graphs. Created: {len(connected_adj_matrices)} connected graphs, of which {len(connected_noniso_adj_matrices)} are connected non-isomorphic graphs. [vertices: {vertices}]")
    return connected_adj_matrices, connected_noniso_adj_matrices


def _create_sample_matrices(vertices: int, already_sampled_connoniso_matrices: list = None, graph_count: int = SAMPLE_GRAPH_COUNT()):
    """
    [DUPLICATE MATRIX SAFE] - Please know what you are doing before changing duplicate matrix safety
    """
    # Getting the max possible connected graph count from YAML
    # If the graph count we want is greater than max, we make graph count 90% of max, as getting to 100% of max is
    # nearly based on luck due to the amount of randomisation we create. Also setting limit for larger max graph counts
    # so as to not spend too much time randomising
    max_possible_connected_graph_count = get_sequenced_graph_count(vertices=vertices)
    if max_possible_connected_graph_count < 10000 and graph_count >= max_possible_connected_graph_count:
        graph_count = round(max_possible_connected_graph_count * 0.9)
    elif max_possible_connected_graph_count > 10000 and graph_count > max_possible_connected_graph_count * 0.8:
        graph_count = round(max_possible_connected_graph_count * 0.8)

    connected_sampled_adj_matrices = []
    connected_noniso_sampled_adj_matrices = []
    if already_sampled_connoniso_matrices:
        connected_noniso_sampled_adj_matrices = already_sampled_connoniso_matrices

    while len(connected_noniso_sampled_adj_matrices) < graph_count:
        _logger.info(f"[VERTICES: {vertices}]: Current Sample Count: {len(connected_noniso_sampled_adj_matrices)}/{graph_count}")
        decimal_matrix_dict = {}
        for matrix_row_decimal_count in range(1, vertices):
            possible_decimals_count = numpy.power(2, matrix_row_decimal_count) - 1
            randomisation_upper_limit = possible_decimals_count + 1
            decimal_matrix_dict[matrix_row_decimal_count] = numpy.random.randint(1, randomisation_upper_limit)
        adj_matrix = decimal_matrix_dict_to_adjacency_matrix(decimal_matrix_dict)
        # preventing duplicate adjacency matrices
        if tuple(adj_matrix.flatten()) in [tuple(matrix.flatten()) for matrix in connected_sampled_adj_matrices]:
            continue
        connected_sampled_adj_matrices, connected_noniso_sampled_adj_matrices = \
            _sort_adjacency_matrix(adjacency_matrix=adj_matrix, connected_adj_matrices=connected_sampled_adj_matrices,
                                   connected_noniso_adj_matrices=connected_noniso_sampled_adj_matrices)
    return connected_noniso_sampled_adj_matrices


def get_sampled_matrices(vertices: int, graph_count: int = None, already_sampled_data=None,
                         already_sampled_matrices=None):
    """
    [DUPLICATE MATRIX SAFE] - Please know what you are doing before changing duplicate matrix safety
    Gets the sampled GraphData of 'vertices' size that has been written to file
    :returns: list of GraphData of sampled data from file
    """
    max_possible_connected_graph_count = get_sequenced_graph_count(vertices=vertices)
    if graph_count > max_possible_connected_graph_count:
        graph_count = round(max_possible_connected_graph_count)
    if not already_sampled_data:
        already_sampled_data = load_written_adj_matrix_data(vertices=vertices)

    connected_noniso_matrices = []
    if already_sampled_matrices:
        connected_noniso_matrices = already_sampled_matrices
    attempt = 1
    while len(connected_noniso_matrices) < graph_count:
        if attempt == 1:
            _logger.info(f"Retrieving Matrices Written to yaml. In addition to [{len(connected_noniso_matrices)}] "
                         f"matrices already sampled (possibly from csv)")
            connected_noniso_matrices = get_yaml_connoniso_adj_matrices(loaded_data=already_sampled_data,
                                                                        vertices=vertices,
                                                                        already_sampled_matrices=connected_noniso_matrices)
        elif attempt == 2:
            _logger.warning(f"Creating Sample Matrices. In addition to [{len(connected_noniso_matrices)}] "
                            f"matrices already sampled (possibly from csv)")
            connected_noniso_matrices = _create_sample_matrices(vertices=vertices,
                                                                already_sampled_connoniso_matrices=connected_noniso_matrices,
                                                                graph_count=graph_count)
        elif attempt > 2:
            break
        attempt += 1
    _logger.info(f"Retrieved sampled graphs. {len(connected_noniso_matrices)}. [vertices: {vertices}]")
    return connected_noniso_matrices[:graph_count]


def update_sample_graph_data(lower_vertices_count: int, upper_vertices_count: int, each_graph_count: int = SAMPLE_GRAPH_COUNT()):
    """
    Calls on the sample creation function and updates the data within the sample graph file
    Reducing each_graph_count below that in the file, the other sample data will be overwritten. Otherwise file is
    updated.
    """
    for v_count in range(lower_vertices_count, upper_vertices_count + 1):
        loaded_data = load_written_adj_matrix_data(vertices=v_count)
        sampled_con_noniso_mat = get_yaml_connoniso_adj_matrices(loaded_data=loaded_data, vertices=v_count)
        sample_data = _create_sample_matrices(v_count, graph_count=each_graph_count,
                                              already_sampled_connoniso_matrices=sampled_con_noniso_mat)
        _logger.info(f"Completed Creating Sample [Vertices: {v_count}]")
        write_sampled_matrices(sampled_graphs=sample_data, vertices=v_count)


def _sort_adjacency_matrix(adjacency_matrix: numpy.ndarray, connected_adj_matrices: list,
                           connected_noniso_adj_matrices: list):
    """
    Places the received graph into either/both the connected_graph_data list or the connected_nonisograph_data list
    based on whether the graph is connected and/or non-isomorphic.
    :param adjacency_matrix: The graph to be categorised, in the form of an adjacency matrix
    """
    current_graph: igraph.Graph = Graph().Adjacency(adjacency_matrix, mode="undirected")
    if current_graph.is_connected():
        connected_adj_matrices.append(adjacency_matrix)
        if len(connected_noniso_adj_matrices) == 0:
            connected_noniso_adj_matrices.append(adjacency_matrix)
        else:
            is_isomorphic = []
            # Looping through the connected non-isomorphic graph list and appending the new graph to list if it's
            # non-isomorphic in comparison to other graphs in that list
            for non_iso_adj_mat in connected_noniso_adj_matrices:
                connected_nonisograph: igraph.Graph = Graph().Adjacency(non_iso_adj_mat, mode="undirected")
                if connected_nonisograph.vcount() == current_graph.vcount() and current_graph.isomorphic(
                        connected_nonisograph):
                    is_isomorphic.append(True)
                else:
                    is_isomorphic.append(False)
            if True not in is_isomorphic:
                connected_noniso_adj_matrices.append(adjacency_matrix)
    return connected_adj_matrices, connected_noniso_adj_matrices


def _create_colour_states(adjacency_matrix):
    """
    Creates a dictionary containing the sequence of colour states {initial colour: [list of following colours]}.
    :returns colour_states_dict: A dictionary containing the decimal list of different colour states a graph goes
    through, key being initial state.
    :returns steps_to_loop_back: A dictionary containing the number of steps back to an earlier witnessed
    colour state, key being the duplicate/initial state
    """
    vertex_count = adjacency_matrix.shape[0]
    colour_states_dict = {}
    steps_to_loop_back = {}
    # Looping through every possible unique colour combination in decimal format, and appending to above dict
    for colour_decimal in range(0, pow(2, (vertex_count - 1))):
        colour_states = []
        next_colour_vector = decimal_to_colour_vector(colour_decimal, vertex_count)
        while colour_decimal not in colour_states:
            colour_states.append(colour_decimal)
            next_colour_vector = numpy.dot(adjacency_matrix, next_colour_vector.T)
            # Ensuring the dot product is of the -1, 1 format of colour vectors
            for i in range(len(next_colour_vector)):
                if next_colour_vector[i] < 0:
                    next_colour_vector[i] = -1
                elif next_colour_vector[i] > 0:
                    next_colour_vector[i] = 1
                else:
                    next_colour_vector[i] = \
                        decimal_to_colour_vector(colour_states[len(colour_states) - 1], vertex_count)[i]
            colour_decimal = colour_vector_to_decimal(next_colour_vector)
            # Adding duplicate to list once, as it means a colour state loop will have taken place
            if colour_decimal in colour_states:
                steps_to_loop_back[colour_states[0]] = len(colour_states) - colour_states.index(colour_decimal)
                colour_states.append(colour_decimal)
        colour_states_dict[colour_states[0]] = colour_states
    # Ordering from largest colour steps to lower for better visualisation
    colour_states_dict = OrderedDict(sorted(colour_states_dict.items(), key=lambda x: len(x[1]), reverse=True))
    return colour_states_dict, steps_to_loop_back


def get_change_in_states(colour_states_dict, vertex_count):
    """
    Calculates the Colour State Change Ratio
    """
    change_in_states = {}
    for initial_state, colour_states_list in colour_states_dict.items():
        initial_col_vect = decimal_to_colour_vector(initial_state, vertex_count)
        initial_col_ratio = float(round(numpy.sum(initial_col_vect)/vertex_count, 3))
        final_col_vect = decimal_to_colour_vector(colour_states_list[len(colour_states_list) - 1], vertex_count)
        final_col_ratio = float(round(numpy.sum(final_col_vect)/vertex_count, 3))
        difference = float(round((final_col_ratio - initial_col_ratio) * initial_col_ratio/(1 - initial_col_ratio), 3))
        change_in_states[initial_state] = abs(difference)
    return change_in_states
