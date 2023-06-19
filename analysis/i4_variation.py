import pandas as pd
import plotly.express as px
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from plotly.validators.box.marker import SymbolValidator

path2 = '/Users/zhaoyu/PycharmProjects/T4Fire/svg/afdes2.csv'
path1 = '/Users/zhaoyu/PycharmProjects/T4Fire/svg/afdes1.csv'
if __name__=='__main__':
    df1 = pd.read_csv(path1)
    df1['Band I4 Post Fire']=df1['b4'].rolling(1).sum()
    df1['Band I5 Post Fire']=df1['b5'].rolling(1).sum()
    df1['Date']=pd.to_datetime(df1['system:time_start']).dt.strftime('%y-%m-%d')

    df2 = pd.read_csv(path2)
    df2['Band I4 Pre Fire']=df2['b4'].rolling(1).sum()
    df2['Band I5 Pre Fire']=df2['b5'].rolling(1).sum()
    df2['Date']=pd.to_datetime(df2['system:time_start']).dt.strftime('%y-%m-%d')
    df = df1[['Date', 'Band I4 Post Fire', 'Band I5 Post Fire']].merge(df2[['Date', 'Band I4 Pre Fire', 'Band I5 Pre Fire']])
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df["Date"], y=df['Band I4 Pre Fire'], name='Non Fire', marker_size=15, marker_symbol='circle'))
    fig.add_trace(go.Scatter(x=df["Date"], y=df['Band I4 Post Fire'], name='Active Fire', marker_size=15, marker_symbol='square'))
    fig.update_yaxes(title_text="Band I4 Value(Kalvin)")
    # line1, = ax.plot(df['Band I4 Post Fire'], marker='*', color='red')
    # line2, = ax.plot(df['Band I4 Pre Fire'], marker='o', color='blue')
    # ax.legend(handles=[line1, line2])
    # ax.legend(handles=['Band I4 Post Fire', 'Band I4 Pre Fire'])
    fig.update_layout(width=700,
    height=360,margin=dict(l=20, r=20, t=20, b=20),legend=dict(
        bgcolor="rgba(0,0,0,0)",
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))
    fig.write_image('/Users/zhaoyu/PycharmProjects/T4Fire/svg/afdes.svg')
    fig.show()