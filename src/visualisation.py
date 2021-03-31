import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output
import sqlite3
import os
import plotly.graph_objects as go
import numpy as np
import pickle
import dash_daq as daq

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets = external_stylesheets)

conn = sqlite3.connect('../SQL_Data/live_data/liveData.sqlite3')
cur =  conn.cursor()
cur.execute('SELECT * FROM Live')
df = pd.DataFrame(cur.fetchall())
names = list(map(lambda x: x[0], cur.description))
df.columns = names
conn.close()
available_indicators = df.columns


theme = {
    'dark': True,
    'detail': '#007439',
    'primary': '#00EA64',
    'secondary': '#6E6E6E'
}

interval = 1000

app.layout = html.Div([

    html.Div([
        dcc.Graph(
            id='positional-graph',
            figure={}
            ),
        dcc.Interval(
            id='positional-interval',
            interval=interval,
            n_intervals=0,
        )
    ]),

    html.Div([
        dcc.Graph(
            id='angle-figure',
            figure = {}
            ),
        dcc.Interval(
            id='angle-interval',
            interval=interval,
            n_intervals=0,
            )
    ]),

    html.Div([
        dcc.Graph(
            id='prediction-graph',
            figure = {}
            ),
        dcc.Interval(
            id='prediction-interval',
            interval=interval,
            n_intervals=0,
            )
    ]),

    html.Div([
        daq.Gauge(
            id='rev-count',
            label = 'RPM',
            value=0
            ),
        # daq.LEDDisplay(
        #     id='prediction-led',
        #     label = "Lap Time Prediction",
        #     labelPosition = "top",
        #     size = "Large",
        #     value = 0,
        # ),
        dcc.Interval(
            id='rev-interval',
            interval=interval,
            n_intervals=interval,
        ),

    ],),

],)


def update_positional_figure(x_axis, y_axis, df):
    try:
        fig = px.line(df, x=x_axis, y=y_axis, template='plotly_dark')
        return fig
    except Exception as e:
        print(f'positional {e}')
        return dash.no_update

def update_angle_figure(x_axis, y_axis, df):
    try:
        fig = px.line(df, x=x_axis, y=y_axis, template='plotly_dark')
        return fig
    except Exception as e:
        print(f'angle {e}')
        return dash.no_update

def update_rev_count(df):
    try:
        rpm = df['engineRPM'].to_list()
        return rpm[-1]
    except Exception as e:
        print(f'Rev count {e}')
        return dash.no_update

def update_prediction(df):
    try:
        predictions = df['Prediction'].to_list()
        fig = px.line(df, x = 'currentLapTime', y='Prediction', template = 'plotly_dark')
        return fig
    except Exception as e:
        print(f'prediction {e}')
        return dash.no_update

@app.callback(
[   Output('rev-count', 'value'),
    Output('positional-graph', 'figure'),
    Output('angle-figure', 'figure'),
    Output('prediction-graph', 'figure'),

],
[   Input('rev-interval', 'n_intervals'),
    Input('positional-interval', 'n_intervals'),
    Input('angle-interval','n_intervals'),
    Input('prediction-interval', 'n_intervals'),
]
)
def update_graph(n, i, j, k):

    df = pd.read_json('../SQL_Data/live_data/live_json.json')
    rev_count = update_rev_count(df)

    positional_x_col, positional_y_col = 'worldPositionX', ['worldPositionZ']
    positional_fig = update_positional_figure(positional_x_col, positional_y_col, df)

    angle_x_col, angle_y_col = 'currentLapTime',  ['yaw', 'pitch', 'roll']
    angle_fig = update_angle_figure(angle_x_col, angle_y_col, df)

    prediction = update_prediction(df)

    return rev_count, positional_fig, angle_fig, prediction

def gui():
    app.run_server(debug = True)

if __name__ == '__main__':
    gui()




# @app.callback(
#     Output('rev-count', 'value'),
#     [Input('rev-interval', 'n_intervals')]
# )
# def rev_update(n):
#     return update_rev_count()
#
# @app.callback(
#     Output('positional-graph', 'figure'),
#     [Input('positional-interval', 'n_intervals')]
# )
# def positional_update(n):
#     positional_x_col, positional_y_col = 'worldPositionX', ['worldPositionZ']
#     positional_fig = update_positional_figure(positional_x_col, positional_y_col )
#     return positional_fig
#
# @app.callback(
# Output('angle-figure', 'figure'),
# [Input('angle-interval','n_intervals')]
# )
# def angle_update(n):
#     angle_x_col, angle_y_col = 'currentLapTime',  ['yaw', 'pitch', 'roll']
#     angle_fig = update_angle_figure(angle_x_col, angle_y_col)
#     return angle_fig
#

    # html.Div([
    #     html.Div([dcc.Dropdown(
    #         id = 'positional-xaxis-column',
    #         options = [{'label':i, 'value': i} for i in available_indicators],
    #         value = 'worldPositionX'
    #         )],
    #     style = {'width':'49%', 'display':'inline-block'}),
    #     html.Div([dcc.Dropdown(
    #         id = 'positional-yaxis-column',multi=True,
    #         options = [{'label':i, 'value': i} for i in available_indicators],
    #         value = ['worldPositionZ']
    #         )],
    #     style = {'width':'49%', 'float':'right', 'display':'inline-block'}),
    # ],
    # style={
    #     'borderBottom': 'thin lightgrey solid',
    #     'backgroundColor': 'rgb(250, 250, 250)',
    #     'padding': '10px 5px'
    # }),
