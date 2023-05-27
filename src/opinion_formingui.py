import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, callback
from dash_bootstrap_templates import load_figure_template

app = dash.Dash(external_stylesheets=[dbc.themes.SUPERHERO], use_pages=True)
server = app.server
load_figure_template('SUPERHERO')


@callback(
    dash.Output('url', 'pathname'),
    dash.Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == "/":
        return "/create-graph-visualisation"
    else:
        return dash.no_update


app.layout = dbc.Container([
    dcc.Location(id='url', refresh="callback-nav"),
    html.Div([
        html.H1('Opinion Forming with Network Theory', style={'display': 'flex', 'justify-content': 'center'}),

        html.Div(
            [
                dbc.Row([
                    dbc.Col(
                        html.Div(
                            dcc.Link(
                                dbc.Button(f"{page['name']}", id=f"{page['path']}", size='large'), href=page["relative_path"]
                            )
                        ), width="auto") for page in dash.page_registry.values()
                ])
            ], style={'display': 'flex', 'justify-content': 'center'}
        ),

        dash.page_container,
        dcc.Store(id='connected_nonisomatrix_vertex_mapping', storage_type='session')
    ], style={'display': 'flex', 'justified-content': 'center', 'flex-direction': 'column'})
])


if __name__ == '__main__':
    app.run_server(debug=True)
