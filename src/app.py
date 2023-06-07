from dash import Dash, dcc, html, Input, Output, State, exceptions
import dash_bootstrap_components as dbc

import numpy as np
import pandas as pd

import yfinance as yf

from prophet import Prophet
from plotly import graph_objs as go

import datetime as dt

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', '/assets/style.css', dbc.icons.FONT_AWESOME, dbc.icons.BOOTSTRAP]

app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div(
    style={
        'margin': '0',
        'padding': '0',
        'textAlign': 'center',
        'background-image': "url('/assets/pexels-pixabay-210607.jpg')",
        'background-repeat': 'no-repeat',
        'background-size': 'cover',
        'background-color': 'rgba(18, 18, 18)',
        'height': '100vh',
        'outline': 'rgba(18, 18, 18)',
        'border': 'none'

    },
    children=[
        html.Div(
            className="container",
            children=[
                html.H1("Stock Market Predictions", className="title"),
                html.P("Analyze and predict company's stock prices through machine learning", className="subtitle"),
                html.Div(
                    className="input-container",
                    children=[
                        html.Div(
                            className="date-input",
                            children=[
                                html.H4('Start Date', className="date"),
                                dcc.Input(
                                    id="year1_slider",
                                    value='',
                                    type='text',
                                    placeholder='YYYY-MM-DD',
                                    className="input-field"
                                ),
                            ],
                        ),
                        html.Div(
                            className="date-input",
                            children=[
                                html.H4('End Date', className="date"),
                                dcc.Input(
                                    id="year2_slider",
                                    value='',
                                    type='text',
                                    placeholder='YYYY-MM-DD',
                                    className="input-field"
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    className="ticker-input",
                    children=[
                        html.H4('Please enter the ticker symbol of a company', className="date"),
                        dcc.Input(
                            id="stock_input",
                            value='',
                            type='text',
                            placeholder='ex: AMZN, GOOG, etc...',
                            className="input-field"
                        ),
                    ],
                ),
                dbc.Button(
                    [html.I(className="bi bi-search"), 'Search'],
                    id='button_input',
                    className="search-button",
                ),
                html.Label("Need help? Click here and look up the company's name", style={'color': 'rgba(223, 223, 223)'}),
                html.A(
                    html.Label("https://finance.yahoo.com/lookup", className="link-label"),
                    href='https://finance.yahoo.com/lookup',
                    target='_blank',
                ),
                html.Label(id="error_message", className="error-message"),
            ],
        ),
        dcc.Loading(
                    id="loading-container",
                    type="circle",
                    style={'marginTop': '40px'},
                    children=[
                        html.Div(id='year-container', className="graph-container", style={'marginTop': '25px'}),
                        html.Div(id='prediction-container', className="graph-container"),
                    ]
            ),
    ],
)

@app.callback(
    Output('year-container', 'children'),
    Output('prediction-container', 'children'),
    Input('button_input', 'n_clicks'),
    State('stock_input', 'value'),
    State('year1_slider', 'value'),
    State('year2_slider', 'value'),
    prevent_initial_call=True,
)

def update_and_prediction_stock(n_clicks, stock_input, year1_slider, year2_slider):
    if n_clicks is None:
        raise exceptions.PreventUpdate
    
    start_date = year1_slider
    end_date = year2_slider
    
    data = yf.download(stock_input, start_date, end_date)
    data.reset_index(inplace=True)
    
    if start_date > end_date:
        return [html.Label("Invalid date range... Start date should be earlier than end date!", id="error_message", style={
            'width': '30%',
            'color': 'rgba(223, 223, 223)',
            'background-color': 'rgba(26, 24, 25)',
            'border': '2px solid rgba(87, 86, 87)',
            'border-radius': '5px',
            'margin': 'auto'
            })], None
    
    if data.empty:
        return [html.Label("Invalid ticker... Please try again!", id="error_message", style={
            'width': '20%',
            'color': 'rgba(223, 223, 223)',
            'background-color': 'rgba(26, 24, 25)',
            'border': '2px solid rgba(87, 86, 87)',
            'border-radius': '5px',
            'margin': 'auto'
        })], None
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.Date, y=data.Close, name='Close', mode='lines'))
    fig.add_trace(go.Scatter(x=data.Date, y=data.Open, name='Open', mode='lines'))
    fig.update_layout(
        title_text='{} Selected Stock Price'.format(stock_input), 
        xaxis=dict(title="Date"), yaxis=dict(title="Stock Price"), 
        plot_bgcolor='rgb(223, 223, 223)',
        paper_bgcolor='rgb(18, 18, 18)',
        font=dict(color='rgb(223, 223, 223)')
    )

    company_data = data[["Date", "Close"]]
    company_data.columns = ["ds", "y"]

    company_data_prophet = Prophet(daily_seasonality=True)
    company_data_prophet.fit(company_data)

    future_stock_company_data = company_data_prophet.make_future_dataframe(periods=365)
    company_predictions = company_data_prophet.predict(future_stock_company_data)

    prediction_fig = go.Figure()
    prediction_fig.add_trace(go.Scatter(x=company_predictions['ds'], y=company_predictions['yhat'], name='Prediction', mode='lines'))
    prediction_fig.add_trace(go.Scatter(x=company_data['ds'], y=company_data['y'], name='Actual', mode='lines'))
    prediction_fig.update_layout(
    title_text='{} Stock Price Prediction'.format(stock_input),
    xaxis=dict(title="Date"),
    yaxis=dict(title="Stock Price"),
    plot_bgcolor='rgb(223, 223, 223)',
    paper_bgcolor='rgb(18, 18, 18)',
    font=dict(color='rgb(223, 223, 223)')
    )

    return dcc.Graph(figure=fig), dcc.Graph(figure=prediction_fig)

if __name__ == "__main__":
    app.run_server(debug=True)