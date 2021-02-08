import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output
import queue
import sqlite3
from tkinter import *
from tkinter import filedialog
import os

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets = external_stylesheets)
file = filedialog.askopenfilename(initialdir='/home/rufus/Documents/F1-Project/SQL_Data/constant_setup')
path = os.path.relpath(file)
conn = sqlite3.connect(path)
cur =  conn.cursor()
cur.execute('SELECT * FROM TrainingData')
df = pd.DataFrame(cur.fetchall())
print(df.head())
names = list(map(lambda x: x[0], cur.description))
df.columns = names

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
            id = 'crossfilter-yaxis-column',
            options = [{'label':i, 'value': i} for i in available_indicators],
            value = 'speed'
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
            id='crossfilter-indicator-line'
            )
        ], style={'width': '90%', 'padding': '0 20'}
    )
])

def get_figure(df, x_col, y_col):
    fig = px.line(df, x=df[x_col], y = df[y_col])
    return fig

@app.callback(
Output('crossfilter-indicator-line', 'figure'),
Input('crossfilter-xaxis-column', 'value'),
Input('crossfilter-yaxis-column', 'value')
)
def callback(xaxis, yaxis):
    return get_figure(df, xaxis, yaxis)

def gui():
    app.run_server(debug = True)

if __name__ == '__main__':
    gui()
