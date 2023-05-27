import dash
import numpy
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.constants import SAMPLE_BATCH_SIZE
from utils.dataframe_generator import create_colour_subject_bargraph_dataframe, \
    calculate_stat_measure_per_n_linegraph_dataframe, get_graph_node_coords
from utils.graph_generator import GraphData
from utils.utils import decimal_to_colour_vector

global colour_step_stat_df
colour_step_stat_df = None
global colour_step_stat_batched_df
colour_step_stat_batched_df = None
global loop_back_stat_df
loop_back_stat_df = None
global loop_back_stat_batched_df
loop_back_stat_batched_df = None


def get_colour_iteration_bar_figure(graph_data_list: list):
    """
    Creates a figure to be used in the bar graph for colour iteration frequency
    :return figure: Bargraph Figure to be used in iterations bar graph
    """
    if graph_data_list:
        data_df, x, y, y_normalised, z = create_colour_subject_bargraph_dataframe(graph_data_list, iterations=True)
        figure = px.bar(data_df, x=x, y=y_normalised, facet_row=z, title="Colour Step Frequency", barmode="group",
                        hover_data={y_normalised: False, 'Freq': data_df[y]})
        figure.update_layout(
            xaxis=dict(tickvals=list(range(min(list(data_df[x])), max(list(data_df[x])))))
        )
        figure.update_yaxes(matches=None, range=[0, 1], tickfont=dict(size=10))
        figure.update_layout(
            xaxis=dict(tickmode="array",
                       tickvals=list(range(min(list(data_df[x])), max(list(data_df[x])) + 1))))
        figure = _mediate_yaxis_label(figure)
        return figure
    else:
        return {}


def get_colour_loop_back_bar_figure(graph_data_list: list):
    """
    Creates a figure to be used in the bar graph for colour loop back frequency
    :return figure: Bargraph Figure to be used in loop-back bar graph
    """
    if graph_data_list:
        data_df, x, y, y_normalised, z = create_colour_subject_bargraph_dataframe(graph_data_list, loop_backs=True)
        figure = px.bar(data_df, x=x, y=y_normalised, facet_row=z, title="Colour Loop-Back Step Frequency",
                        barmode="group",
                        hover_data={y_normalised: False, 'Freq': data_df[y]})
        figure.update_layout(
            xaxis=dict(tickmode="array", tickvals=list(range(min(list(data_df[x])), max(list(data_df[x])) + 1))))
        figure.update_yaxes(matches=None, range=[0, 1], tickfont=dict(size=10))
        figure.update_layout(
            xaxis=dict(tickmode="array",
                       tickvals=list(range(min(list(data_df[x])), max(list(data_df[x])) + 1))))
        figure = _mediate_yaxis_label(figure)
        return figure
    else:
        return {}


def get_colour_iteration_stat_measure_per_n_figure(graph_data_dict: dict, statistical_measure: str):
    """
    Creates a figure to be used in the line graph for colour iteration means across vertices of count n
    :return figure: Line Graph Figure to be used in iterations mean line graph
    """
    global colour_step_stat_df
    global colour_step_stat_batched_df
    x = "Vertices"
    if graph_data_dict:
        if colour_step_stat_df is None or colour_step_stat_batched_df is None:
            colour_step_stat_df, colour_step_stat_batched_df = calculate_stat_measure_per_n_linegraph_dataframe(graph_data_dict,
                                                                                                                   iterations=True,
                                                                                                                   x_label=x)
        else:
            if max(graph_data_dict.keys()) not in colour_step_stat_df[x]:
                colour_step_stat_df, colour_step_stat_batched_df = calculate_stat_measure_per_n_linegraph_dataframe(graph_data_dict,
                                                                                                                       iterations=True,
                                                                                                                       x_label=x)
        figure = px.line(colour_step_stat_df, x=x, y=statistical_measure,
                         title="Colour Step Statistical Measures Per Vertices(n)", markers=True)
        figure.add_trace(go.Scatter(x=colour_step_stat_batched_df[x], y=colour_step_stat_batched_df[statistical_measure],
                                    mode="markers", hoverinfo="none", marker=dict(size=4),
                                    name=f"n(Sampled Graphs) <= {SAMPLE_BATCH_SIZE()}"))
        vertex_list = [int(v) for v in colour_step_stat_df[x]]
        figure.update_layout(
            xaxis=dict(tickmode="array", tickvals=list(range(min(vertex_list), max(vertex_list) + 1))))
        return figure
    else:
        return {}


def get_colour_loop_back_stat_measure_per_n_figure(graph_data_dict: dict, statistical_measure: str):
    """
    Creates a figure to be used in the line graph for colour loop-back means across vertices of count n
    :return figure: Line Graph Figure to be used in iterations mean line graph
    """
    global loop_back_stat_df
    global loop_back_stat_batched_df
    x = "Vertices"
    if graph_data_dict:
        if loop_back_stat_df is None or loop_back_stat_batched_df is None:
            loop_back_stat_df, loop_back_stat_batched_df = calculate_stat_measure_per_n_linegraph_dataframe(graph_data_dict, loop_backs=True,
                                                                                        x_label=x)
        else:
            if max(graph_data_dict.keys()) not in loop_back_stat_df[x]:
                loop_back_stat_df, loop_back_stat_batched_df = calculate_stat_measure_per_n_linegraph_dataframe(
                    graph_data_dict, loop_backs=True,
                    x_label=x)
        figure = px.line(loop_back_stat_df, x=x, y=statistical_measure,
                         title="Colour Loop-Back Statistical Measures Per Vertices(n)", markers=True)
        figure.add_trace(go.Scatter(x=loop_back_stat_batched_df[x], y=loop_back_stat_batched_df[statistical_measure], mode="markers",
                                    hoverinfo="none", marker=dict(size=4),
                                    name=f"n(Sampled Graphs) <= {SAMPLE_BATCH_SIZE()}"))
        figure.update_layout(
            xaxis=dict(tickmode="array", tickvals=list(range(min(list(loop_back_stat_df[x])), max(list(loop_back_stat_df[x])) + 1))))
        return figure
    else:
        return {}


def get_dash_table_children(dataframe: pd.DataFrame, row_height: str = '50px'):
    """
    Given a dataframe, breaks down its column heads and data to return children accepted by Dash tables.
    This function can be best used when needing to update a dbc table, which does not have data as a subcomponent.
    :return dash.dbc.children: Children of dash table, containing the data from the dataframe.
    """
    style = {}
    if row_height:
        style['height'] = row_height
    return [dash.html.Tr([dash.html.Th(col) for col in dataframe.columns])] + [
        dash.html.Tr([dash.html.Td(dataframe.iloc[i][col], style=style) for col in dataframe.columns]) for i in
        range(len(dataframe))]


def _mediate_yaxis_label(figure):
    """
    Takes a figure and if a figure with multiple facets, will position the y label at the middle of all facets
    """
    yaxis_count_list = [1 for axis in figure.layout if type(figure.layout[axis]) == go.layout.YAxis]
    median_yaxis = round(numpy.median(list(range(1, len(yaxis_count_list) + 1))), 0)
    for yaxis_count in range(1, len(yaxis_count_list) + 1):
        if yaxis_count == median_yaxis or len(yaxis_count_list) == 1:
            continue
        yaxis = f"yaxis{yaxis_count}"
        if yaxis_count == 1:
            yaxis = "yaxis"
        figure.layout[yaxis].title.text = ''
    return figure


def convert_decimals_to_graph_colours(decimal_colour_states: list, vertices_count: int, colour1: str = "red",
                                      colour2: str = "blue"):
    """
    Takes the colour states of a graph over time and converts it to a list of colours
    :param decimal_colour_states: The colour states (list of decimal numbers) to be converted to vertex colour states
    :param vertices_count: The number of vertices the graph has
    :param colour1: Colour of Opinion1
    :param colour2: Colour of Opinion2
    :returns vertex_colours: A list of colours that the vertices will be represented as in the visualisation
    """
    vertex_colour_states = []
    # going through each state(decimal) and getting the colour_vector(made of 1/-1) that it represents
    for dec in decimal_colour_states:
        colour_vector = decimal_to_colour_vector(dec, vertices_count)
        # going through the colour_vector and creating the string version of itself with actual colours
        vertex_colours = [colour1 if dec == -1 else colour2 for dec in colour_vector]
        vertex_colour_states.append(vertex_colours)
    return vertex_colour_states


def create_graph_figure(graph_data: GraphData):
    """
    Creates a Figure object based on the provided graph data, along with all customisation for that figure
    :param graph_data: GraphData
    """
    graph = graph_data.igraph
    vertices_colour_states = graph_data.all_decimal_colour_states
    x_coords, x_edges, y_coords, y_edges = get_graph_node_coords(graph)
    # CREATING PLOT AND LAYOUT
    edge_scatter = go.Scatter(x=x_edges,
                              y=y_edges,
                              mode='lines',
                              line=dict(color='rgb(90,90,90)', width=3),
                              hoverinfo='none'
                              )
    vertex_scatter = go.Scatter(x=x_coords,
                                y=y_coords,
                                mode='markers',
                                name='ntw',
                                marker=dict(symbol='circle',
                                            size=20,
                                            color='rgb(160,160,160)',
                                            line=dict(color='rgb(50,50,50)', width=0.5)
                                            ),
                                hoverinfo='text'
                                )
    axis = dict(showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False
                )
    figure_width = 500
    figure_height = 500
    go_layout = go.Layout(title="",
                          font=dict(size=12),
                          showlegend=False,
                          autosize=False,
                          width=figure_width,
                          height=figure_height,
                          xaxis=go.layout.XAxis(axis),
                          yaxis=go.layout.YAxis(axis),
                          margin=go.layout.Margin(
                              l=40,
                              r=40,
                              b=30,
                              t=80,
                          ),
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='white',
                          hovermode='closest'
                          )
    data = [edge_scatter, vertex_scatter]
    figure = go.Figure(data=data, layout=go_layout)
    return figure
