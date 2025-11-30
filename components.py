from dash import dcc, html
from .constants import STRING_CATEGORICALS, BINARY_NUMERIC_FEATURES, available_features_ui, S_LABEL

def generate_input(feature, defaults):
    lbl = available_features_ui.get(feature, feature)
    
    is_categorical = (feature in STRING_CATEGORICALS) or (feature in BINARY_NUMERIC_FEATURES)
    
    if is_categorical:
        if 'gender' == feature.lower(): opts = [{'label': 'Female', 'value': 'Female'}, {'label': 'Male', 'value': 'Male'}]
        elif 'region' == feature.lower(): opts = [{'label': 'Urban', 'value': 'Urban'}, {'label': 'Rural', 'value': 'Rural'}]
        elif feature.lower() in ['physical_activity', 'stress_level', 'air_pollution_exposure']: opts = [{'label': l, 'value': l} for l in ['Low', 'Moderate', 'High']]
        elif 'income' in feature.lower(): opts = [{'label': l, 'value': l} for l in ['Low', 'Middle', 'High']]
        elif 'ekg' in feature.lower(): opts = [{'label': 'Normal', 'value': 'Normal'}, {'label': 'Abnormal', 'value': 'Abnormal'}]
        elif 'diet' in feature.lower(): opts = [{'label': 'Unhealthy', 'value': 'Unhealthy'}, {'label': 'Healthy', 'value': 'Healthy'}]
        elif 'smoking' in feature.lower(): opts = [{'label': l, 'value': l} for l in ['Never', 'Past', 'Current']]
        elif 'alcohol' in feature.lower(): opts = [{'label': 'Never', 'value': ''}, {'label': 'Occasionally', 'value': 'Moderate'}, {'label': 'Regular', 'value': 'High'}]
        elif feature in BINARY_NUMERIC_FEATURES: opts = [{'label': 'No', 'value': 0}, {'label': 'Yes', 'value': 1}]
        else: opts = [{'label': 'Low', 'value': 'Low'}, {'label': 'High', 'value': 'High'}]

        fallback = opts[0]['value']
        def_val = defaults.get(feature, fallback)
        if def_val not in [o['value'] for o in opts]: def_val = fallback
        
        comp = dcc.Dropdown(id=f'input-{feature}', options=opts, value=def_val, clearable=False, style={'fontSize': '14px'})
    else:
        val = defaults.get(feature, 0)
        comp = dcc.Input(id=f'input-{feature}', type='number', value=round(val, 2), style={'width': '100%', 'padding': '8px', 'borderRadius': '4px', 'border': '1px solid #cbd5e1', 'boxSizing': 'border-box'})

    return html.Div([html.Label(lbl, style=S_LABEL), comp], style={'marginBottom': '15px'})
