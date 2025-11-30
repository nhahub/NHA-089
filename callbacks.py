# src/callbacks.py
from dash import Input, Output, State, html
import pandas as pd
import numpy as np
from .constants import ALL_MODEL_FEATURES, STRING_CATEGORICALS, BINARY_NUMERIC_FEATURES, NUMERIC_FEATURES
from .data_manager import data_manager

def register_callbacks(app):
    @app.callback(
        Output('output-result', 'children'),
        [Input('btn-predict', 'n_clicks')],
        [State(f'input-{f}', 'value') for f in ALL_MODEL_FEATURES]
    )
    def predict(n, *vals):
        if not n: 
            return html.Div([html.H3("Ready", style={'color': '#94a3b8'}), html.P("Enter data to begin.")])
        
        if data_manager.model is None: 
            return html.Div("Model Error")
        
        try:
            data = { feat: vals[i] for i, feat in enumerate(ALL_MODEL_FEATURES) }
            df_pred = pd.DataFrame([data])
            
            # Hybrid Casting to match Model
            for col in STRING_CATEGORICALS: 
                df_pred[col] = df_pred[col].apply(lambda x: np.nan if x == '' else (str(x) if pd.notnull(x) else np.nan))
            for col in BINARY_NUMERIC_FEATURES: df_pred[col] = pd.to_numeric(df_pred[col]).astype(int)
            for col in NUMERIC_FEATURES: df_pred[col] = pd.to_numeric(df_pred[col]).astype(float)

            prob = data_manager.model.predict_proba(df_pred)[0][1] * 100
            pred = data_manager.model.predict(df_pred)[0]
            
            color = '#ef4444' if pred == 1 else '#10b981'
            text = "HIGH RISK" if pred == 1 else "LOW RISK"
            bg_color = '#fef2f2' if pred == 1 else '#ecfdf5'
            
            return html.Div([
                html.Div(text, style={'color': color, 'fontSize': '1.8rem', 'fontWeight': '800', 'letterSpacing': '1px'}),
                html.Div(f"{prob:.1f}%", style={'color': '#1e293b', 'fontSize': '4rem', 'fontWeight': '800', 'lineHeight': '1'}),
                html.Div([html.Div(style={'width': f'{prob}%', 'backgroundColor': color, 'height': '100%'})], style={'height': '12px', 'backgroundColor': '#e2e8f0', 'borderRadius': '6px', 'overflow': 'hidden', 'margin': '20px 0'}),
                html.P("Immediate medical attention recommended." if pred == 1 else "Keep healthy habits.", style={'color': '#64748b'})
            ], style={'backgroundColor': bg_color, 'padding': '20px', 'borderRadius': '8px'})
            
        except Exception as e: 
            return html.Div(f"Error: {str(e)}", style={'color': 'red'})