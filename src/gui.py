import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output
import queue




class GUI(object):
    def __init__(self, parent):
        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
        self.parent = parent
        self.df = self.get_df()
        self.indicators = self.df.columns
        self.app = dash.Dash(name=__name__, external_stylesheets=external_stylesheets)
        self.app.callback(
        Output('crossfilter-indicator-line', 'figure'),
        [Input('crossfilter-xaxis-column', 'value'),
        Input('crossfilter-yaxis-column', 'value')]
        )(self.callback)
        self.app.layout = self.build_layout()
    def get_df(self):
        '''gets the live dataframe'''
        pass
    def build_layout(self):
        available_indicators = self.indicators
        layout = html.Div([
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
                ], style={'width': '90%', 'display': 'inline-block', 'padding': '0 20'}
            )
        ])
        return layout
    def callback(self, xaxis, yaxis):
        return self.get_figure(xaxis, yaxis)
    def get_figure(self, x_col, y_col):
        df = self.df
        fig = px.line(df, x=df[x_col], y = df[y_col])
        return fig
    def run(self):
        print('running')
        self.app.run_server(debug = True)
