import math

import igraph
import numpy
import pandas as pd
from scipy import stats

from utils import graph_generator
from utils.constants import SAMPLE_BATCH_SIZE


def calculate_statistical_measures(x: list, frequency: list):
    """
    Takes x data and its frequency and return the statistical measures for that data
    :param x: The data
    :param frequency: The frequency/weight of that data
    :return: a dictionary of the statistical measures calculated
    """
    if len(x) != len(frequency):
        return {}
    mean = round(numpy.average(x, weights=frequency), 3)
    median = numpy.median(numpy.repeat(x, frequency))
    mode = numpy.argmax(numpy.bincount(x, weights=frequency))
    variance = round(numpy.var(numpy.repeat(x, frequency), ddof=1), 3)
    skewness = round(stats.skew(numpy.repeat(x, frequency)), 3)
    return {'mean': mean, 'median': median, 'mode': mode, 'variance': variance, 'skewness': skewness}


# Collective graph data functions
def get_colour_loop_back_data(graph_data_list: list):
    """
    Calculates and returns a dictionary of the number of colour loop back steps,
    and the number of uniquely coloured graphs that has that many loop back steps during it's colour transitions
    :param graph_data_list: A list of GraphData
    :returns graph_loop_back_count: A dictionary - {(vertices, number of loop back steps): number of unique coloured graphs witnessing this number of loop back steps}
    :returns vertices_count_data: A list of the vertices of the graphs in the graph loop back count dict, in the same order
    """
    graph_loop_back_count = {}

    for graph_data in graph_data_list:
        all_decimal_colour_steps_to_loop_back = graph_data.all_decimal_colour_steps_to_loop_back
        for loop_back_steps in list(all_decimal_colour_steps_to_loop_back.values()):
            key = (graph_data.vertices_count, loop_back_steps)
            if key in graph_loop_back_count.keys():
                graph_loop_back_count[key] += 1
                continue
            graph_loop_back_count[key] = 1
    return graph_loop_back_count


def get_colour_iterations_data(graph_data_list: list):
    """
    Calculates and returns a dictionary of the number of colour iterations,
    and the number of uniquely coloured graphs that has that many colour iterations
    :param graph_data_list: A list of GraphData
    :returns graphs_iterations_counts: A dictionary -{(vertices, number of iterations/colour states seen): number of unique coloured graphs of n vertices that go through this number of iterations}
    """
    graphs_iterations_counts = {}

    for graph_data in graph_data_list:
        all_decimal_colour_states: dict = graph_data.all_decimal_colour_states
        for colour_states in list(all_decimal_colour_states.values()):
            iterations = len(colour_states)
            key = (graph_data.vertices_count, iterations)
            if key in graphs_iterations_counts.keys():
                graphs_iterations_counts[key] += 1
                continue
            graphs_iterations_counts[key] = 1
    return graphs_iterations_counts


# Collective bargraph dataframe functions
def create_colour_subject_bargraph_dataframe(graph_data_list: list, iterations: bool = False,
                                             loop_backs: bool = False):
    """
    FOR ALL GRAPHS IN LIST. Creates and returns the dataframe containing the frequency(# of graphs) and \
    number of colour iterations/steps taken to go back to a previously witnessed colour
    """
    y_label = "Frequency"
    y_label_normalised = "Normalised Frequency"
    z_label = "V"
    data_retrieval_function = None
    x_label = None
    if iterations:
        x_label = "Colour Steps"
        data_retrieval_function = get_colour_iterations_data
    elif loop_backs:
        x_label = "Loop-Back Steps"
        data_retrieval_function = get_colour_loop_back_data
    elif iterations and loop_backs:
        return [], None, None, None, None
    subject_data = data_retrieval_function(graph_data_list)
    frequency_data = list(subject_data.values())
    subject_count_data = [key[1] for key in list(subject_data.keys())]
    vertices_count_data = [key[0] for key in list(subject_data.keys())]
    # Calculating the max frequency per v so as to normalise each
    max_frequency_per_v = {}
    for key, value in subject_data.items():
        v = key[0]
        if v in max_frequency_per_v:
            max_frequency_per_v[v] = max_frequency_per_v[v] + value
            continue
        max_frequency_per_v[v] = value
    normalised_frequency = [round(frequency / max_frequency_per_v[key[0]], 3) for key, frequency in
                            subject_data.items()]
    # Calculating y_label_normalised(normalised frequency) as a fraction of the max frequency for that vertex
    subject_data = {y_label: frequency_data,
                    y_label_normalised: normalised_frequency,
                    x_label: subject_count_data,
                    z_label: vertices_count_data}
    df = pd.DataFrame(subject_data)
    df['threshold'] = df.groupby(z_label)[y_label].transform('max') / 10
    df = df[df[y_label] >= df['threshold']]
    df = df.drop(['threshold'], axis=1)
    return df, x_label, y_label, y_label_normalised, z_label


# Collective statistical measure dataframe functions
def calculate_stat_measure_per_n_linegraph_dataframe(graph_data_dict: dict, x_label: str, iterations: bool = False,
                                                     loop_backs: bool = False):
    """
    FOR ALL GRAPHS IN LIST. Creates and returns dataframe containing the number of vertices, and the mean
    colour iterations for that number of vertices
    """
    stat_measure_per_n_data = []
    stat_measure_per_n_batched_data = []
    columns = [x_label, 'mean', 'median', 'mode', 'variance', 'skewness']
    data_retrieval_function = None
    # picking the right data retrieval function based on the required subject
    if iterations:
        data_retrieval_function = get_colour_iterations_data
    elif loop_backs:
        data_retrieval_function = get_colour_loop_back_data
    elif iterations and loop_backs:
        return [], [], None
    # Getting the iteration count of each vertex count and calculating the mean of graphs with vertex = n
    for vertices_count, graphs_list in graph_data_dict.items():
        # get batched data
        batched_data, measure_data = _calculate_batch_stat_measure_dataframes(vertices_count=vertices_count,
                                                                              graphs_list=graphs_list,
                                                                              data_retrieval_func=data_retrieval_function)
        stat_measure_per_n_batched_data += batched_data
        if measure_data:
            stat_measure_per_n_data += measure_data
            continue
        retrieved_data = data_retrieval_function(graphs_list)
        frequency_data = list(retrieved_data.values())
        retrieved_data_counts = [key[1] for key in list(retrieved_data.keys())]
        if frequency_data:
            statistical_measures: dict = calculate_statistical_measures(x=retrieved_data_counts,
                                                                        frequency=frequency_data)
            stat_measure_per_n_data.append([vertices_count, statistical_measures['mean'],
                                            statistical_measures['median'], statistical_measures['mode'],
                                            statistical_measures['variance'],
                                            statistical_measures['skewness']])
    stat_measure_df = pd.DataFrame(data=stat_measure_per_n_data, columns=columns)
    batched_stat_measure_df = pd.DataFrame(data=stat_measure_per_n_batched_data, columns=columns)
    return stat_measure_df, batched_stat_measure_df


def _calculate_batch_stat_measure_dataframes(vertices_count: int, graphs_list: list, data_retrieval_func):
    """
    Calculates the batched statistical measures for plots.
    """
    iteration_stat_measure_per_n_batched_data = []
    iteration_stat_measure_per_n_data = []
    sample_batch_size = SAMPLE_BATCH_SIZE()
    if len(graphs_list) >= sample_batch_size:
        lower_batch_value = 0
        upper_batch_value = sample_batch_size
        for i in range(0, math.ceil(len(graphs_list) / sample_batch_size)):
            iteration_data = data_retrieval_func(graphs_list[lower_batch_value:upper_batch_value])
            frequency_data = list(iteration_data.values())
            iteration_count_data = [key[1] for key in list(iteration_data.keys())]
            lower_batch_value += sample_batch_size
            upper_batch_value += sample_batch_size
            if frequency_data:
                statistical_measures: dict = calculate_statistical_measures(x=iteration_count_data,
                                                                            frequency=frequency_data)
                iteration_stat_measure_per_n_batched_data.append([vertices_count, statistical_measures['mean'],
                                                                  statistical_measures['median'],
                                                                  statistical_measures['mode'],
                                                                  statistical_measures['variance'],
                                                                  statistical_measures['skewness']])
        batched_means, batched_medians, batched_modes, batched_variances, batched_skewness = [], [], [], [], []
        for data in iteration_stat_measure_per_n_batched_data:
            if data[0] == vertices_count:
                batched_means.append(data[1])
                batched_medians.append(data[2])
                batched_modes.append(data[3])
                batched_variances.append(data[4])
                batched_skewness.append(data[5])
        iteration_stat_measure_per_n_data.append([vertices_count, round(numpy.average(batched_means), 3),
                                                  round(numpy.average(batched_medians), 3),
                                                  round(numpy.average(batched_modes), 3),
                                                  round(numpy.average(batched_variances), 3),
                                                  round(numpy.average(batched_skewness), 3)])
    return iteration_stat_measure_per_n_batched_data, iteration_stat_measure_per_n_data


# Collective stats dataframe functions for the tables
def create_colour_loop_back_stats_dataframe(graph_data_list: list):
    """
    Creates a dataframe on the stats of the loop-back data
    :param graph_data_list: A list of GraphData
    :return loop_back_stats_df: A dataframe containing stats on the loop back data
    """
    loop_back_data = get_colour_loop_back_data(graph_data_list)
    frequency_data = list(loop_back_data.values())
    loop_back_count_data = [key[1] for key in list(loop_back_data.keys())]

    statistical_measures = calculate_statistical_measures(x=loop_back_count_data, frequency=frequency_data)

    loop_back_stats_properties = ["Mean", "Median", "Mode", "Variance", "Skewness"]
    loop_back_stats_values = [statistical_measures["mean"], statistical_measures["median"],
                              statistical_measures["mode"], statistical_measures["variance"],
                              statistical_measures["skewness"]]
    loop_back_stats_data = {"Statistical Measure": loop_back_stats_properties,
                            "Value (Loop-Back Iterations)": loop_back_stats_values}
    loop_back_stats_df = pd.DataFrame(data=loop_back_stats_data)
    return loop_back_stats_df


def create_colour_iterations_stats_dataframe(graph_data_list: list):
    """
    Creates a dataframe on the stats of the iterations data
    :param graph_data_list: A list of GraphData
    :return loop_back_stats_df: A dataframe containing stats on the iterations data
    """
    iterations_data = get_colour_iterations_data(graph_data_list)
    frequency_data = list(iterations_data.values())
    iteration_count_data = [key[1] for key in list(iterations_data.keys())]

    statistical_measures = calculate_statistical_measures(x=iteration_count_data, frequency=frequency_data)

    iterations_stats_properties = ["Mean", "Median", "Mode", "Variance", "Skewness"]
    iterations_stats_values = [statistical_measures["mean"], statistical_measures["median"],
                               statistical_measures["mode"], statistical_measures["variance"],
                               statistical_measures["skewness"]]
    iterations_stats_data = {"Statistical Measure": iterations_stats_properties,
                             "Value (Iterations)": iterations_stats_values}
    iterations_stats_df = pd.DataFrame(data=iterations_stats_data)
    return iterations_stats_df


# Individual graph data functions
def get_graph_node_coords(graph: igraph.Graph):
    """
    Creates graph coordinates for each node for graph plot.
    """
    vertices_count = graph.vcount()
    edge_list = graph.get_edgelist()
    # creates layout that has coords for vertices
    g_layout = graph.layout("kk")
    x_coords = [g_layout[vertex][0] for vertex in range(vertices_count)]
    y_coords = [g_layout[vertex][1] for vertex in range(vertices_count)]
    x_edges = []
    y_edges = []
    for e in edge_list:
        x_edges += [g_layout[e[0]][0], g_layout[e[1]][0], None]
        y_edges += [g_layout[e[0]][1], g_layout[e[1]][1], None]
    return x_coords, x_edges, y_coords, y_edges


# Graph Data Stats for Individual Graph table
def create_graph_data_dataframe(graph_data: graph_generator.GraphData, colour_step: int = 0,
                                current_graph_colour: list = None, colour_states: list = None,
                                decimal_colour_states: list = None):
    """
    Creates a dataframe of the GraphData in order for it to be displayed in Individual Graph Table. Not all properties
    of GraphData are included in dataframe.
    """
    graph_properties = ["Vertices Count", "Mean Graph Degree", "Max Graph Degree", "Min Graph Degree",
                        "Mean Graph Eccentricity"]
    property_values = [graph_data.vertices_count, graph_data.graph_degree,
                       graph_data.max_degree,
                       graph_data.min_degree, graph_data.graph_eccentricity]

    if colour_states:
        total_states_count = len(colour_states)
        if colour_states[len(colour_states) - 1] == colour_states[len(colour_states) - 2]:
            total_states_count -= 1
        if colour_step > total_states_count:
            colour_step = total_states_count
        colour_step_fraction = f"{colour_step}/{total_states_count}"
        colour_properties = ["Colour Step"]
        colour_values = [colour_step_fraction]
        if decimal_colour_states:
            loop_back_steps = graph_data.all_decimal_colour_steps_to_loop_back[decimal_colour_states[0]]
            colour_properties.append("Loop-Back Steps")
            colour_values.append(loop_back_steps)
            colour_ratio = graph_data.colour_state_change_ratios[decimal_colour_states[0]]
            colour_properties.append("Colour Change Ratio")
            colour_values.append(colour_ratio)
        insertion_index = 1
        graph_properties = graph_properties[:insertion_index] + colour_properties + graph_properties[insertion_index:]
        property_values = property_values[:insertion_index] + colour_values + property_values[insertion_index:]

    if current_graph_colour:
        colour1 = numpy.unique(current_graph_colour)[0]
        colour1_count = f"{current_graph_colour.count(colour1)}/{graph_data.vertices_count}"
        colour_properties = [f"{colour1.capitalize()} Vertices Count"]
        colour_values = [colour1_count]
        if numpy.unique(current_graph_colour).size > 1:
            colour2 = numpy.unique(current_graph_colour)[1]
            colour2_count = f"{current_graph_colour.count(colour2)}/{graph_data.vertices_count}"
            colour_properties.append(f"{colour2.capitalize()} Vertices Count")
            colour_values.append(colour2_count)
        insertion_index = 2
        graph_properties = graph_properties[:insertion_index] + colour_properties + graph_properties[insertion_index:]
        property_values = property_values[:insertion_index] + colour_values + property_values[insertion_index:]
    basic_graph_data = {"Graph Property": graph_properties, "Values": property_values}
    df = pd.DataFrame(data=basic_graph_data)
    return df
