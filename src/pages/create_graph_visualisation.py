import logging

import dash
import dash_bootstrap_components as dbc
import numpy
from dash import dcc, callback

from utils.constants import MINIMUM_VERTICES_SAMPLE_SIZE, SAMPLE_GRAPH_COUNT
from utils.dataframe_generator import create_colour_loop_back_stats_dataframe, \
    create_colour_iterations_stats_dataframe
from utils.graph_generator import create_matrices, get_sampled_matrices, GraphData, read_graph_data_from_csv
from utils.utils import load_written_adj_matrix_data, get_sequenced_graph_count, write_all_data_to_csv
from utils.visualisation_utils import get_colour_iteration_bar_figure, get_colour_loop_back_bar_figure, \
    get_dash_table_children, get_colour_iteration_stat_measure_per_n_figure, \
    get_colour_loop_back_stat_measure_per_n_figure

_logger = logging.Logger("OpinionForming", level="INFO")
dash.register_page(__name__, order=1)

global connected_nonisograph_vertex_mapping
connected_nonisograph_vertex_mapping = {}


def create_and_sample_vertices(vertices_max_value: int, vertices_min_value: int = 1):
    noniso_graphdata_mapping = {}
    stored_graph_data = read_graph_data_from_csv(vertex_max_value=vertices_max_value,
                                                 vertex_min_value=vertices_min_value)
    for vertex_count in range(vertices_min_value, vertices_max_value + 1):
        new_connected_non_isomorphic_matrices = []
        graph_data_list = []
        # Appending csv stored graph data to graph data list
        if vertex_count in stored_graph_data:
            graph_data_list += stored_graph_data[vertex_count]
        graph_data_matrix_list = [graph_data.adjacency_matrix for graph_data in graph_data_list]
        # Sample matrices to be created here
        if vertex_count >= MINIMUM_VERTICES_SAMPLE_SIZE():
            graph_count = SAMPLE_GRAPH_COUNT()
            if len(graph_data_matrix_list) < graph_count:
                sampled_matrix_data = load_written_adj_matrix_data(vertices=vertex_count)
                new_connected_non_isomorphic_matrices = get_sampled_matrices(vertices=vertex_count,
                                                                         graph_count=graph_count,
                                                                         already_sampled_data=sampled_matrix_data,
                                                                         already_sampled_matrices=graph_data_matrix_list)
            # Removing already graphed data from new matrix list.
            # Function acceptance of already sampled matrices should minimize occurance of duplicate error.
            for mat in graph_data_matrix_list:
                if tuple(mat.flatten()) in [tuple(new_mat.flatten()) for new_mat in new_connected_non_isomorphic_matrices]:
                    new_connected_non_isomorphic_matrices.remove(mat)
            graph_data_list = graph_data_list[:abs(len(new_connected_non_isomorphic_matrices)-graph_count)]
        # Only creating new matrices if they are below sample and not in csv
        else:
            if len(graph_data_list) < get_sequenced_graph_count(vertices=vertex_count):
                _logger.info(f"CREATING MATRICES FOR VERTEX SIZE: {vertex_count}")
                new_connected_matrices, new_connected_non_isomorphic_matrices = create_matrices(vertices=vertex_count)
        _logger.info(f"GENERATING GRAPH DATA FOR {len(new_connected_non_isomorphic_matrices)} MATRICES [V: {vertex_count}]")
        noniso_graphdata_mapping[vertex_count] = [GraphData(numpy.array(matrix)).fill_data() for matrix
                                                                  in
                                                                  new_connected_non_isomorphic_matrices] + graph_data_list
    return noniso_graphdata_mapping


@callback(
    [dash.Output('post-graph-creation-text', 'children'),
     dash.Output('spinner-button-div', 'style'),
     dash.Output('all-graphs-data-div', 'style'),
     dash.Output('post-graph-creation-text-div', 'style'),
     dash.Output('colour-iterations-frequency-graph', 'figure'),
     dash.Output('colour-loop-size-frequency-graph', 'figure'),
     dash.Output('iterations-stats-table', 'children'),
     dash.Output('loop-back-stats-table', 'children')],
    dash.Input('user-max-vertices-count', 'value'),
    dash.Input('create-graphs-button', 'n_clicks'),
    dash.Input('csv-button', 'n_clicks')
)
def create_graph_callback(vertices_max_value, create_graphs_button_n_cicks,
                          csv_button_n_clicks):
    clicked_button = list(dash.callback_context.triggered_prop_ids.values())
    if 'create-graphs-button' not in clicked_button or not vertices_max_value:
        if 'csv-button' not in clicked_button or not vertices_max_value:
            return (dash.no_update,) * 8

    global connected_nonisograph_vertex_mapping
    if 'create-graphs-button' in clicked_button:
        vertices_min_value = 1
        if list(connected_nonisograph_vertex_mapping.keys()):
            vertices_min_value = numpy.max(list(connected_nonisograph_vertex_mapping.keys())) + 1
        connected_nonisograph_vertex_mapping.update(create_and_sample_vertices(vertices_max_value=vertices_max_value, vertices_min_value=vertices_min_value))

        connected_nonisograph_list = [graph_data for graph_data_list in
                                      list(connected_nonisograph_vertex_mapping.values())
                                      for graph_data in graph_data_list]
        _logger.info(f"Displaying graph data for graphs with vertices: {list(connected_nonisograph_vertex_mapping.keys())}")

        connected_graph_count = numpy.sum([get_sequenced_graph_count(vertices=vertices, connected_graphs=True)
                                           for vertices in range(1, vertices_max_value + 1)])
        post_graph_creation_text = [f"Vertices Less than or Equal to: {vertices_max_value}", dash.html.Br(),
                                    f"Total Number of Connected Graphs: {connected_graph_count}", dash.html.Br(),
                                    f"Total Number of Connected Non-Isomorphic Graphs: {len(connected_nonisograph_list)}"]
        colour_iterations_bar_figure = get_colour_iteration_bar_figure(connected_nonisograph_list)
        colour_loop_size_bar_figure = get_colour_loop_back_bar_figure(connected_nonisograph_list)
        iterations_stats_table_children = get_dash_table_children(
            create_colour_iterations_stats_dataframe(connected_nonisograph_list))
        loop_back_stats_table_children = get_dash_table_children(
            create_colour_loop_back_stats_dataframe(connected_nonisograph_list))
        output = post_graph_creation_text, {'display': 'none'}, {'display': 'block'}, {'display': 'block'}, \
                 colour_iterations_bar_figure, colour_loop_size_bar_figure, iterations_stats_table_children, \
                 loop_back_stats_table_children
        return output
    elif 'csv-button' in clicked_button:
        write_all_data_to_csv(graph_data_mapping=connected_nonisograph_vertex_mapping)
        return (dash.no_update,) * 8


@callback(
    [dash.Output('colour-iterations-stat-measure-graph', 'figure'),
     dash.Output('colour-loop-size-stat-measure-graph', 'figure')],
    dash.Input('colour-iterations-stat-measure-dropdown', 'value'),
    dash.Input('colour-loop-size-stat-measure-dropdown', 'value'),
    dash.Input('colour-iterations-frequency-graph', 'figure'),
    dash.Input('colour-loop-size-frequency-graph', 'figure')
)
def update_statistical_measure_line_graphs(iterations_stat_dropdown_value, loop_size_stat_dropdown_value,
                                           iteration_bargraph_figure, loop_back_bargraph_figure):
    global connected_nonisograph_vertex_mapping
    if iteration_bargraph_figure and loop_back_bargraph_figure:
        colour_iterations_stat_measure_line_figure = get_colour_iteration_stat_measure_per_n_figure(
            connected_nonisograph_vertex_mapping, iterations_stat_dropdown_value)
        colour_loop_back_stat_measure_line_figure = get_colour_loop_back_stat_measure_per_n_figure(
            connected_nonisograph_vertex_mapping, loop_size_stat_dropdown_value)
        return colour_iterations_stat_measure_line_figure, colour_loop_back_stat_measure_line_figure
    return dash.no_update, dash.no_update


@callback(
    dash.Output('spinner-button-inner-div', 'style'),
    [dash.Input('create-graphs-button', 'n_clicks')]
)
def toggle_spinner(n_clicks):
    clicked_button = list(dash.callback_context.triggered_prop_ids.values())
    if 'create-graphs-button' in clicked_button:
        return {'diplay': 'block'}
    return dash.no_update


layout = dbc.Container([
    # Flex Container
    dash.html.Div([
        # Select Graph Specifications and Overall graph counts
        dash.html.Div([
            # Graph
            dbc.InputGroup([dbc.InputGroupText("Select Graph Vertices Count:"),
                            dbc.Input(id='user-max-vertices-count', type='number', min=2, max=16,
                                      style={'width': '60px'}),
                            dbc.Button('Create Graphs', id='create-graphs-button', n_clicks=0)]),
        ], style={'display': 'flex', 'justify-content': 'center'}),

        # Post Graph Creation Text
        dbc.Row(dbc.Col(
            dash.html.Div([
                dbc.Badge(id='post-graph-creation-text', color='primary')
            ], id='post-graph-creation-text-div', style={'display': 'none'}),
            width={"size": 6, "offset": 4}
        )),

        dash.html.Hr(style={'border': '0', 'height': '1px',
                            'background-image': 'linear-gradient(to right, rgba(0, 0, 0, 0), var(--primary), rgba(0, 0, 0, 0))'}),

        # Creating Graphs
        dbc.Row(dbc.Col(
            dash.html.Div([
                dash.html.Div(
                    # Graph Creation Wait Spinner
                    dbc.Button([dbc.Spinner(), " Creating Graphs. Please Wait..."], disabled=True),
                    id='spinner-button-inner-div', style={'display': 'none'})
            ], id='spinner-button-div'),
            width={"size": 6, "offset": 4}
        )),

        # Display graph data graphs for all vertices of vertices <= n
        dash.html.Div([
            # iterations graph and stats table
            dbc.Row([
                dbc.Col(dcc.Graph(id='colour-iterations-frequency-graph'), width='auto'),
                dbc.Col(dbc.Table(id="iterations-stats-table", color='primary', striped=True, bordered=True,
                                  hover=True, style={'font-size': 'large'}), width=3)
            ]),

            # space
            dash.html.Hr(style={'border': '0', 'height': '1px',
                                'background-image': 'linear-gradient(to right, rgba(0, 0, 0, 0), var(--primary), rgba(0, 0, 0, 0))'}),

            # iterations statistic measures dropdown and graph
            dbc.Row([
                dbc.Col(
                    dash.html.Div([
                        dcc.Dropdown(id='colour-iterations-stat-measure-dropdown',
                                     options=[{'label': 'Mean', 'value': 'mean'},
                                              {'label': 'Median', 'value': 'median'},
                                              {'label': 'Mode', 'value': 'mode'},
                                              {'label': 'Variance', 'value': 'variance'},
                                              {'label': 'Skewness', 'value': 'skewness'}],
                                     value='mean', clearable=False,
                                     style={'text-align': 'center', 'color': 'black'}),
                        dcc.Graph(id='colour-iterations-stat-measure-graph'),
                    ]), width='auto'
                ), dbc.Col(width=3)
            ]),

            # space
            dash.html.Hr(style={'border': '0', 'height': '1px',
                                'background-image': 'linear-gradient(to right, rgba(0, 0, 0, 0), var(--primary), rgba(0, 0, 0, 0))'}),

            # loop-back graph and stats table
            dbc.Row([
                dbc.Col(dcc.Graph(id='colour-loop-size-frequency-graph'), width='auto'),
                dbc.Col(
                    dbc.Table(id="loop-back-stats-table", color='primary', striped=True, bordered=True,
                              hover=True, style={'font-size': 'large'}),
                    width=3
                )
            ]),

            # space
            dash.html.Hr(style={'border': '0', 'height': '1px',
                                'background-image': 'linear-gradient(to right, rgba(0, 0, 0, 0), var(--primary), rgba(0, 0, 0, 0))'}),

            # loop-back statistic measures dropdown and graph
            dbc.Row([
                dbc.Col(
                    dash.html.Div([
                        dcc.Dropdown(id='colour-loop-size-stat-measure-dropdown',
                                     options=[{'label': 'Mean', 'value': 'mean'},
                                              {'label': 'Median', 'value': 'median'},
                                              {'label': 'Mode', 'value': 'mode'},
                                              {'label': 'Variance', 'value': 'variance'},
                                              {'label': 'Skewness', 'value': 'skewness'}],
                                     value='mean', clearable=False,
                                     style={'text-align': 'center', 'color': 'black'}),
                        dcc.Graph(id='colour-loop-size-stat-measure-graph'),
                    ]), width='auto'
                ), dbc.Col(width=3)
            ]),

            # space
            dash.html.Hr(style={'border': '0', 'height': '1px',
                                'background-image': 'linear-gradient(to right, rgba(0, 0, 0, 0), var(--primary), rgba(0, 0, 0, 0))'}),

            # Write to excel button
            dbc.Row(dbc.Col(
                dash.html.Div(
                    # Graph Creation Wait Spinner
                    dbc.Button('Export Data to CSV', id='csv-button')
                ),
                width={"size": 6, "offset": 4}
            ))
        ], id='all-graphs-data-div', style={'display': 'none'}),
    ], style={'display': 'flex', 'justified-content': 'center', 'flex-direction': 'column'}),
])
