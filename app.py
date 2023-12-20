import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html
from dash import dcc, html, dash_table, Output, Input, State
from dash.exceptions import PreventUpdate
import pandas as pd
import io
import base64
from PIL import Image

external_stylesheets = ['https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap']

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP,external_stylesheets], use_pages=True,)

img = Image.open('./5araf2.png')

sidebar = html.Div(
    [
        html.Img(src=img,width=200,height=200),
        #html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Date Type", href="/", active="exact", className="nav_link"),
                dbc.NavLink("Date Grade", href="/dates_grade", active="exact", className="nav_link"),
                dbc.NavLink("Date Maturity", href="/fruit_type", active="exact", className="nav_link"),
                dbc.NavLink("Palm Disease", href="/tree_disease", active="exact", className="nav_link"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    className="sidebar_div",
)

content = html.Div(id="page-content", children=[html.Div(dash.page_container)], className="content_div")

header = html.Div(
    [
        html.Img(src=app.get_asset_url("header.jpg"), className="header_image"),
    ],
    className="header",
)

app.layout = html.Div([dcc.Location(id="url"), sidebar, header, content])


if __name__ == "__main__":
    app.run_server(debug=True)