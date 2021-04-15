import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import dash_table
from PIL import Image


scale_factor = 2.5
# url = 'https://raw.githubusercontent.com/rufus-lopes/csv_F1/master/data_sampled.csv'
img = Image.open('T6150292-Silverstone_race_track,_aerial_image.jpg')
def get_data():
    ''' accesses data and cleans up first axis '''
    df = pd.read_csv('data_sampled.csv')
    df.drop([df.columns[0]], axis=1, inplace=True)
    df.reset_index(inplace=True, drop=True)
    df['worldPositionZ'] = -1*df["worldPositionZ"]
    return df

def plot_position(data):
    ''' takes in dataframe and plots the X and Z positions '''
    x = data['worldPositionX']
    y = data['worldPositionZ']

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
    x=x,
    y=y,
    mode='markers',
    marker = dict(
            color = data['finalLapTime'])))
    fig.update_traces(marker_showscale=True,
        marker_size = 6,
        opacity = 0.6,
        selector=dict(type='scattergl'))
    fig.update_layout(height=700*1.15,width=600*1.2,
        title=dict(text = 'Positional Data'),
        yaxis = dict(title='Y-Position (m)'),
        xaxis= dict(title = 'X-Position (m)'),)

#     fig.add_layout_image(
#         dict(
#             source=img,
#             xref="x",
#             yref="y",
#             x=-620,
#             y=750,
#             sizex=606*scale_factor,
#             sizey=800*scale_factor,
#             sizing="stretch",
#             opacity=0.8,
#             layer="below")
# )

    return fig

def plot_speed(data):
    '''takes dataframe and plots the speed against distance through lap '''
    x = data['lapDistance']
    y = data['speed']

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
    x=x,
    y=y,
    mode='markers',
    marker = dict(
            color = data['finalLapTime'])))
    fig.update_traces(marker_showscale=True,
        marker_size = 6,
        opacity = 0.4,
        marker_symbol = 'circle-dot',
        selector=dict(type='scattergl'))
    fig.update_layout(height=700*1.15,width=800*1.2,
        title=dict(text ='Speed Data'),
        yaxis = dict(title='Speed (KM/h)'),
        xaxis= dict(title = 'Lap Distance (m)'),)

    return fig

def sector_splits(data):

    '''splits the laps by sector and returns each sectors data as dataframe'''
    sector_1 = []
    sector_2 = []
    sector_3 = []
    sessions = data.groupby('sessionUID')
    for s in list(sessions.groups):
        session = sessions.get_group(s)
        laps = session.groupby('currentLapNum')
        for l in list(laps.groups):
            lap = laps.get_group(l)
            sectors = lap.groupby('sector')
            for i in list(sectors.groups):
                sector = sectors.get_group(i)
                if i == 0:
                    sector_1.append(sector)
                elif i == 1:
                    sector_2.append(sector)
                elif i == 2:
                    sector_3.append(sector)
                else:
                    print('Bad sector')
    sector_1 = pd.concat(sector_1)
    sector_2 = pd.concat(sector_2)
    sector_3 = pd.concat(sector_3)
    return sector_1, sector_2, sector_3

def bucket_names(buckets):
    ''' creates the names of buckets for use in speed group function '''
    names = []
    for i in range(len(buckets)-1):
        name = f'{buckets[i]} < time < {buckets[i+1]}'
        names.append(name)
    return names

def speed_table(selected_data):
    ''' returns dataframe table of speed, brake, throttle, gear and laps
    grouped by the time of the lap
    '''
    # fastest = min(data['finalLapTime'])
    # slowest = max(data['finalLapTime'])
    # lap_times = data['finalLapTime'].unique()
    # x = [i for i in range(len(lap_times))]
    # lap_times = sorted(lap_times)
    # lap_time_dict = group_by_lap_time(data)

    buckets = [90, 92, 94, 96, 100, 110, 120] #range(80, 140, 5) # 
    names = bucket_names(buckets)
    selected_data['range'] = pd.cut(selected_data['finalLapTime'], buckets, labels=names)
    ranges = selected_data.groupby('range')
    unique_ranges = list(selected_data['range'].unique())
    laps = data.groupby('finalLapTime')
    # print(f'total num laps: {len(laps)}')

    unique_ranges = sorted(unique_ranges)

    tens = [x for x in unique_ranges if x[0] == '1']
    nines = [x for x in unique_ranges if x[0] == '9']

    nines = sorted(nines)
    tens = sorted(tens)

    unique_ranges = nines+tens

    speed = {}
    brake = {}
    throttle = {}
    gear = {}
    steering = {}
    laps = {}
    for r in unique_ranges:
        time_range = ranges.get_group(r)
        speed[r] = time_range['speed'].mean()
        brake[r] = time_range['brake'].mean()*100
        throttle[r] = time_range['throttle'].mean()*100
        gear[r] = time_range['gear'].mean()
        steering[r] = time_range['steer'].mean()
        laps[r] = time_range['finalLapTime'].nunique()

    names = ['speed (KM/h)', 'brake %', 'throttle %', 'gear', 'steering ratio', 'laps']
    new_data = [speed, brake, throttle, gear, steering, laps]
    df2 = pd.DataFrame(new_data, index=names)
    df2.reset_index(inplace=True)
    df2 = df2.round(3)
    df2.to_latex('table.tex')
    return df2

#currently not used
def group_by_lap_time(data):

    ''' creates a dataframe of each lap grouped by the lap time '''
    lap_time_dict = {}
    times = data.groupby('finalLapTime')
    for t in list(times.groups):
        time = times.get_group(t)
        lap_time_dict[t] = time
    return lap_time_dict

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets = external_stylesheets)
data = get_data()
positional_fig = plot_position(data)
speed_fig = plot_speed(data)
speedtable = speed_table(data)

theme = {
    'dark': True,
    'detail': '#007439',
    'primary': '#00EA64',
    'secondary': '#6E6E6E'
}

app.layout = html.Div([


    html.H1(children='Analysing Lap Performance'),

    html.Div(children=[
        dcc.Graph(
            id='positional-graph',
            figure = positional_fig,
            style={'display': 'inline-block'}
            ),
        dcc.Graph(
            id = 'speed-graph',
            figure = speed_fig,
            style={'display': 'inline-block'}
            ),
        html.Div([
                # dcc.Markdown("""
                #     **Selection Data**
                #
                #     Choose the lasso or rectangle tool in the graph's menu
                #     bar and then select points in the graph.
                #
                #     The table below shows average values for lap time ranges based on the data selected.
                #
                # """),
                dash_table.DataTable(
                    id='selected-data',
                    columns = [{'name':i, 'id': i} for i in speedtable.columns],
                    data = speedtable.to_dict('records')
                    ),
            ]),
    ]),

])



#Sorting datatable info

def get_dash_table(selection, position, speed):
    if position:
        position_df = filter_position_df(selection)
        speed_tab = speed_table(position_df)
        speed_tab.reset_index(inplace=True)
        dash_table = speed_tab.to_dict('records')
    else:
        speed_df = filter_speed_distance(selection)
        speed_tab = speed_table(speed_df)
        speed_tab.reset_index(inplace=True)
        dash_table = speed_tab.to_dict('records')
    return dash_table

def filter_speed_distance(selection):
    xy_data = get_speed_distance_df(selection)
    keys = list(xy_data.columns.values)
    i1 = data.set_index(keys).index
    i2 = xy_data.set_index(keys).index
    df = data[i1.isin(i2)]
    return df

def get_speed_distance_df(selection):
    x,y = get_speed_distance(selection)
    xy_dataframe = pd.DataFrame(columns = ['lapDistance', 'speed'])
    xy_dataframe['lapDistance'] = x
    xy_dataframe['speed'] = y
    return xy_dataframe

def get_speed_distance(selection):
    x = []
    y = []
    for point in selection['points']:
        x.append(point['x'])
        y.append(point['y'])
    return x,y

def filter_position_df(selection):
    xy_data = xy_position_dataframe(selection)
    keys = list(xy_data.columns.values)
    i1 = data.set_index(keys).index
    i2 = xy_data.set_index(keys).index
    df = data[i1.isin(i2)]
    return df

def xy_position_dataframe(selection):
    x,y = get_position_xy(selection)
    xy_dataframe = pd.DataFrame(columns = ['worldPositionX', 'worldPositionZ'])
    xy_dataframe['worldPositionX'] = x
    xy_dataframe['worldPositionZ'] = y
    # print(xy_dataframe.info())
    return xy_dataframe

def get_position_xy(selection):
    x = []
    y = []
    for point in selection['points']:
        x.append(point['x'])
        y.append(point['y'])
    return x,y



@app.callback(
    Output('selected-data', 'data'),
    [Input('positional-graph', 'selectedData'),
    Input('speed-graph', 'selectedData')]
    )
def display_selected_position_data(position_data, speed_data):

    context = dash.callback_context
    trigger = context.triggered[0]['prop_id'].split('.')[0]
    if trigger == 'positional-graph':
        if position_data:
            dash_table = get_dash_table(position_data, True, False)
            return dash_table
        else:
            return dash.no_update
    elif trigger == 'speed-graph':
        if speed_data:
            dash_table = get_dash_table(speed_data, False, True)
            return dash_table
        else:
            return dash.no_update
    else:
        return dash.no_update



if __name__ == '__main__':
    app.run_server(debug=True)
    # sector_1, sector_2, sector_3 = sector_splits(data)

    # plot_position(sector_1)

    # speed_group_table = speed_groups(data)

    # plot_speed(sector_1)
