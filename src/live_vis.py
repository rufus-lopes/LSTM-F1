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

app.layout = html.Div([
    html.Div([
        html.Div([dcc.Dropdown(
            id = 'crossfilter-xaxis-column',
            options = [{'label':i, 'value': i} for i in available_indicators],
            value = 'currentLapTime'
            )],
        style = {'width':'49%', 'display':'inline-block'}),
        html.Div([dcc.Dropdown(
            id = 'crossfilter-yaxis-column',multi=True,
            options = [{'label':i, 'value': i} for i in available_indicators],
            value = ['speed', 'worldPositionX']
            )],
        style = {'width':'49%', 'float':'right', 'display':'inline-block'}),
    ],
    style={
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'padding': '10px 5px'
    }),

    html.Div([
        dcc.Graph(
            id='live-update-graph',
            figure={}
            ),
        dcc.Interval(
            id='interval-component',
            interval=500,
            n_intervals=0,
        )
        ])
])

def update_figure(x_axis, y_axis):
    try:
        #df= pickle.load(open('../SQL_Data/live_data/pickle.pkl', 'rb'))
        df = pd.read_json('../SQL_Data/live_data/live_json.json')
        fig = px.line(df, x=x_axis, y=y_axis)
        return fig
    except Exception as e:
        print(e)
        return dash.no_update

@app.callback(
Output('live-update-graph', 'figure'),
[Input('interval-component', 'n_intervals'),
Input('crossfilter-xaxis-column', 'value'),
Input('crossfilter-yaxis-column', 'value'),]
)
def live_update(n, x_col, y_col):
    fig = update_figure(x_col, y_col)
    return fig

def gui():
    app.run_server(debug = True)

if __name__ == '__main__':
    gui()
