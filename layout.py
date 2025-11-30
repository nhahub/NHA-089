# src/layout.py
from dash import html, dcc
import plotly.express as px
import pandas as pd
from .constants import *
from .components import generate_input
from .data_manager import data_manager

def create_layout():
    # Build Sections
    sections = []
    for group_name, features in GROUPS.items():
        inputs = [generate_input(f, data_manager.defaults) for f in features]
        grid = html.Div(inputs, style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fill, minmax(200px, 1fr))', 'gap': '15px'})
        sections.append(html.Div([html.H4(group_name, style=S_GROUP_TITLE), grid], style=S_CARD))

    return html.Div([
        # Header
        html.Div([
            html.H1("CardioGuard AI", style={'margin': 0, 'fontSize': '2.2rem'}),
            html.P("Clinical Risk Assessment System", style={'opacity': 0.7, 'margin': '5px 0 0 0'})
        ], style=S_HEADER),

        html.Div([
            # Left Column: Inputs
            html.Div([
                *sections, 
                html.Div([html.Button("ANALYZE PATIENT", id='btn-predict', n_clicks=0, style=S_BTN)], style=S_CARD)
            ], style={'flex': '3', 'minWidth': '600px'}),
            
            # Right Column: Results
            html.Div([
                html.Div(id='output-result', style={**S_CARD, 'textAlign': 'center', 'minHeight': '200px', 'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center'}),
                html.Div([
                    html.H4("Model Weights (Feature Importance)", style=S_GROUP_TITLE),
                    dcc.Graph(
                        figure=px.bar(
                            data_manager.feature_importance_df.tail(10) if data_manager.feature_importance_df is not None else pd.DataFrame({'feature':['No Data'], 'importance':[0]}), 
                            x='importance', y='feature', orientation='h', 
                            template='plotly_white', color_discrete_sequence=['#3b82f6']
                        ).update_layout(margin=dict(l=0,r=0,t=0,b=0), height=350, xaxis={'visible': False}),
                        config={'displayModeBar': False}
                    )
                ], style=S_CARD)
            ], style={'flex': '1', 'minWidth': '350px', 'position': 'sticky', 'top': '20px', 'alignSelf': 'start'})
        ], style=S_CONTAINER)
    ], style=S_MAIN)