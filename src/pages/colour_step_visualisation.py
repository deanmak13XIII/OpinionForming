import dash
import dash_bootstrap_components as dbc
from dash import dcc, callback

from utils.dataframe_generator import create_graph_data_dataframe
from utils.graph_generator import GraphData, read_graph_data_from_csv
from utils.visualisation_utils import convert_decimals_to_graph_colours, create_graph_figure, get_dash_table_children

dash.register_page(__name__, order=2)

global stored_graph_data
stored_graph_data = {}


@callback(
    dash.Output("graph-table-div", "style"),
    dash.Output("graph-pagination", "max_value"),
    dash.Output("colour-pagination", "max_value"),
    dash.Output("colour-step-graph", "figure"),
    dash.Output("graph-table", "children"),
    dash.Output("automation-interval", "disabled"),
    dash.Output("automation-interval", "interval"),
    dash.Output("csv-alert-div", "style"),
    dash.Output("loading-vis-div", "style"),
    dash.Input("vertices-input", "value"),
    dash.Input("view-graph-button", "n_clicks"),
    dash.Input("next-state-button", "n_clicks"),
    dash.Input("opinion-reset-button", "n_clicks"),
    dash.Input("automation-time-input", "value"),
    dash.Input("start-automation-button", "n_clicks"),
    dash.Input("automation-interval", "n_intervals"),
    dash.Input("automation-interval", "disabled"),
    dash.Input("graph-pagination", "active_page"),
    dash.Input("colour-pagination", "active_page"),
    dash.State("colour-step-graph", "figure"),
)
def update_main_graph(vertices_value, view_graph_button, next_state_n_clicks, opinion_reset_n_clicks,
                      automation_input_value,
                      automation_n_clicks, automation_n_interval, disabled_interval, graph_pagination,
                      colour_pagination, figure):
    clicked_button = list(dash.callback_context.triggered_prop_ids.values())
    show_csv_alert = {'display': 'flex', 'justify-content': 'center'}
    hide_div = {'display': 'none'}

    # if neither the graph data nor vertices colour, do nothing
    if not vertices_value:
        return (dash.no_update, ) * 9

    global stored_graph_data
    if not stored_graph_data or vertices_value not in stored_graph_data:
        if "view-graph-button" in clicked_button and vertices_value:
            stored_graph_data = read_graph_data_from_csv(vertex_max_value=vertices_value, vertex_min_value=vertices_value)
            # Showing only csv reminder alert
            if not stored_graph_data:
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, show_csv_alert, hide_div
        else:
            return (dash.no_update, ) * 9
    # Only filling the data in for the focus graph data
    focus_graph_data: GraphData = stored_graph_data[vertices_value][graph_pagination - 1]
    focus_graph_colour_states = list(focus_graph_data.all_decimal_colour_states.values())[colour_pagination - 1]
    # Retrieving graph count and colour count for pagination
    graph_pagination_max_value = len(stored_graph_data[vertices_value])
    colour_pagination_max_value = len(list(focus_graph_data.all_decimal_colour_states.values()))
    # Getting up to date graph colours and table data
    vertices_colour_states = convert_decimals_to_graph_colours(decimal_colour_states=focus_graph_colour_states,
                                                               vertices_count=focus_graph_data.vertices_count)
    graph_data_table = get_dash_table_children(
        create_graph_data_dataframe(focus_graph_data, decimal_colour_states=focus_graph_colour_states))

    # RETURN INITIAL UNCOLOURED GRAPH
    if "view-graph-button" in clicked_button or "graph-pagination" in clicked_button:
        figure = create_graph_figure(focus_graph_data)
        return {'display': 'block'}, graph_pagination_max_value, colour_pagination_max_value, figure, graph_data_table, dash.no_update, dash.no_update, dash.no_update, hide_div

    # UPDATE GRAPH FIGURE AND TABLE IF NEXT STATE BUTTON CLICKED, OR IF START AUTOMATION CLICKED, OR IF AUTOMATION INTERVAL IS ENABLED
    if "next-state-button" in clicked_button or "start-automation-button" in clicked_button or not disabled_interval:
        continue_disabled_interval = dash.no_update
        automation_interval = dash.no_update
        next_state_index = 0
        if figure['data'][1]['marker']['color'] in vertices_colour_states:
            next_state_index = vertices_colour_states.index(figure['data'][1]['marker']['color']) + 1
        if "start-automation-button" in clicked_button and automation_input_value and next_state_index > 0:
            continue_disabled_interval = False
            automation_interval = automation_input_value * 1000

        if next_state_index < len(vertices_colour_states):
            next_colour = vertices_colour_states[next_state_index]
            graph_data_table = get_dash_table_children(create_graph_data_dataframe(focus_graph_data,
                                                                                   colour_step=next_state_index + 1,
                                                                                   current_graph_colour=next_colour,
                                                                                   colour_states=vertices_colour_states,
                                                                                   decimal_colour_states=focus_graph_colour_states))
            figure['data'][1]['marker']['color'] = next_colour
            if continue_disabled_interval and next_state_index == len(vertices_colour_states) - 1:
                continue_disabled_interval = True
            return {'display': 'block'}, dash.no_update, dash.no_update, figure, graph_data_table, continue_disabled_interval, automation_interval, dash.no_update, dash.no_update

    # RESET GRAPH FIGURE COLOURS TO CURRENT COLOUR PAGINATION
    if "opinion-reset-button" in clicked_button or "colour-pagination" in clicked_button:
        reset_colour = vertices_colour_states[0]
        graph_data_table = get_dash_table_children(create_graph_data_dataframe(focus_graph_data,
                                                                               colour_step=1,
                                                                               current_graph_colour=reset_colour,
                                                                               colour_states=vertices_colour_states,
                                                                               decimal_colour_states=focus_graph_colour_states))
        figure['data'][1]['marker']['color'] = reset_colour
        return {'display': 'block'}, dash.no_update, dash.no_update, figure, graph_data_table, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    return (dash.no_update, ) * 9


@callback(
    dash.Output('loading-vis-inner-div', 'style'),
    [dash.Input('view-graph-button', 'n_clicks')]
)
def toggle_spinner(n_clicks):
    clicked_button = list(dash.callback_context.triggered_prop_ids.values())
    if 'view-graph-button' in clicked_button:
        return {'diplay': 'block'}
    return dash.no_update


layout = dbc.Container(
    # Flex container
    dash.html.Div([
        # Select Graph Vertices Count
        dash.html.Div([
            dbc.InputGroup([dbc.InputGroupText("Select Graph Vertices Count:"),
                            dbc.Input(id='vertices-input', type='number', min=2, max=16,
                                      style={'width': '60px'}),
                            dbc.Button("View Graph", id='view-graph-button', n_clicks=0)]),
        ], style={'display': 'flex', 'justify-content': 'center'}),

        dash.html.Div([
            dbc.Alert("Please ensure you have exported created graphs to CSV file...",
                      color="primary")
        ], style={'display': 'none', 'justify-content': 'center'}, id="csv-alert-div"),

        dash.html.Hr(style={'border': '0', 'height': '1px',
                            'background-image': 'linear-gradient(to right, rgba(0, 0, 0, 0), var(--primary), rgba(0, 0, 0, 0))'}),

        # Loading Graphs Spinner
        dbc.Row(dbc.Col(
            dash.html.Div([
                dash.html.Div(
                    # Graph Creation Wait Spinner
                    dbc.Button([dbc.Spinner(), "Loading Graphs. Please Wait..."], disabled=True),
                    id='loading-vis-inner-div', style={'display': 'none'})
            ], id='loading-vis-div'),
            width={"size": 6, "offset": 4}
        )),

        # Graph and Table Container
        dash.html.Div([
            # Graph Pagination
            dash.html.Div([
                dash.html.Div("Select Graph Structure: "),
                dbc.Pagination(id='graph-pagination', active_page=1, size='sm', max_value=1, fully_expanded=False,
                               previous_next=True),
            ], style={'display': 'flex', 'justify-content': 'center'}),

            dash.html.Div([
                dash.html.Div("Select Initial Colour: "),
                dbc.Pagination(id='colour-pagination', active_page=1, size='sm', max_value=1, fully_expanded=False,
                               previous_next=True),
            ], style={'display': 'flex', 'justify-content': 'center'}),

            dash.html.Hr(style={'border': '0', 'height': '1px',
                                'background-image': 'linear-gradient(to right, rgba(0, 0, 0, 0), var(--primary), rgba(0, 0, 0, 0))'}),

            dbc.Row([
                # Graph Container
                dbc.Col([
                    dcc.Graph(id="colour-step-graph"),
                    # Button Container
                    dash.html.Div([
                        dbc.ButtonGroup([dbc.Button("Next Colour State", id="next-state-button", n_clicks=0),
                                         dbc.Button("Reset Opinions", id="opinion-reset-button", n_clicks=0)]
                                        ),
                        dbc.InputGroup(
                            [dbc.InputGroupText("Time Interval (s): "),
                             dbc.Input(id="automation-time-input", type='number', min=1, max=60,
                                       style={'width': '60px'}, value=1),
                             dbc.Button("Start Automation", id="start-automation-button", n_clicks=0)],
                            id="automation-group",
                            style={'display': 'grid', 'grid-template-columns': 'auto auto',
                                   'align-items': 'center', 'grid-column': '3/3', 'grid-column-gap': '0px'}
                        )
                    ], style={'display': 'grid', 'grid-template-columns': 'auto auto auto',
                              'grid-column-gap': '5px', 'align-items': 'center', 'grid-row-gap': '0px'}
                    )
                ], width=6),
                # Table Container
                dbc.Col([
                    dash.html.Div([
                        dbc.Table(id="graph-table", color='primary', striped=True, bordered=True, hover=True,
                                  style={'font-size': 'large'})
                    ])
                ], width=3)
            ]),
        ], id="graph-table-div", style={'display': 'none'}),
        dcc.Interval(id='automation-interval', disabled=True)
    ], style={'display': 'flex', 'justified-content': 'center', 'flex-direction': 'column'})
)
