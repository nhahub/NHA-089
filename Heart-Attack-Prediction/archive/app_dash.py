import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from scipy import stats
import webbrowser 
import threading
import time
import joblib
import os

# 1. DATA LOADING AND PREPROCESSING

file_name = 'heart_attack_prediction_indonesia.csv'

# Load data
df = pd.read_csv(file_name)

# Data Cleaning and Feature Preparation
df.columns = df.columns.str.strip().str.replace(' ', '_', regex=False).str.replace('[^A-Za-z0-9_]+', '', regex=True)

def encode_categorical_columns(df):
    df_encoded = df.copy()
    
    if 'gender' in df_encoded.columns:
        df_encoded['gender'] = df_encoded['gender'].astype(str).str.lower().str.strip()
        df_encoded['gender'] = df_encoded['gender'].apply(lambda x: 1 if x == 'male' else (0 if x == 'female' else np.nan))
        df_encoded['gender'] = df_encoded['gender'].fillna(df_encoded['gender'].median() if not df_encoded['gender'].empty else 0).astype(int)
    
    categorical_columns = ['region', 'income_level', 'smoking_status', 'alcohol_consumption', 
                          'physical_activity', 'dietary_habits', 'air_pollution_exposure', 
                          'stress_level', 'EKG_results', 'previous_heart_disease', 
                          'medication_usage', 'participated_in_free_screening']
    
    for col in categorical_columns:
        if col in df_encoded.columns:
            df_encoded[col] = pd.Categorical(df_encoded[col]).codes

    return df_encoded

df = encode_categorical_columns(df)

COLUMN_RENAME_MAP = {
    'age': 'Age',
    'gender': 'Gender',
    'blood_pressure_systolic': 'Systolic_blood_pressure',
    'blood_pressure_diastolic': 'Diastolic_blood_pressure',
    'fasting_blood_sugar': 'Blood_sugar',
}
df.rename(columns=COLUMN_RENAME_MAP, inplace=True)

result_column = None
for col in df.columns:
    if 'result' in col.lower() or 'outcome' in col.lower() or 'target' in col.lower() or 'diagnosis' in col.lower() or 'heart_attack' in col.lower():
        result_column = col
        break

df['Result'] = df[result_column].astype(str).str.lower().str.strip()    
positive_indicators = ['positive', 'pos', '1', 'true', 'yes', 'high', 'abnormal']
negative_indicators = ['negative', 'neg', '0', 'false', 'no', 'low', 'normal']    
df['Result'] = df['Result'].apply(lambda x: 1 if x in positive_indicators else (0 if x in negative_indicators else np.nan))
    
df = df.dropna(subset=['Result'])
    
if not df.empty:
    df['Result'] = df['Result'].astype(int)

POTENTIAL_FEATURES = [
    'Age', 
    'Gender', 
    'Systolic_blood_pressure', 
    'Diastolic_blood_pressure', 
    'Blood_sugar',
    'hypertension',
    'diabetes', 
    'cholesterol_level',
    'obesity',
    'family_history'
]

available_features = [f for f in POTENTIAL_FEATURES if f in df.columns]
print(f"Available features: {available_features}")

HUMAN_READABLE_MAP = {
    'Age': 'Age (Years)', 
    'Gender': 'Gender (0=F, 1=M)', 
    'Systolic_blood_pressure': 'Systolic BP (mmHg)', 
    'Diastolic_blood_pressure': 'Diastolic BP (mmHg)',
    'Blood_sugar': 'Blood Sugar (mg/dL)',
    'hypertension': 'Hypertension',
    'diabetes': 'Diabetes',
    'cholesterol_level': 'Cholesterol Level',
    'obesity': 'Obesity',
    'family_history': 'Family History'
}
available_features_ui = {col: HUMAN_READABLE_MAP.get(col, col.replace('_', ' ').title()) for col in available_features}

# 2. MODEL TRAINING (HistGradientBoosting)

hgb_model = None
feature_importance_df = None

if len(available_features) > 0 and 'Result' in df.columns and not df.empty:
    X_train = df[available_features].copy()
    y_train = df['Result']

    for col in X_train.columns:
        if pd.api.types.is_numeric_dtype(X_train[col]):
            X_train[col] = X_train[col].fillna(X_train[col].median() if not X_train[col].empty else 0)
        else:
            print(f"Warning: Non-numeric feature {col} found. Attempting to convert...")
            X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
            X_train[col] = X_train[col].fillna(X_train[col].median() if not X_train[col].empty else 0)

    available_features = X_train.columns.tolist()

    if len(X_train) > 0 and len(X_train.columns) > 0 and len(y_train.unique()) > 1:
        # Best Hyperparameters from RandomizedSearchCV
        best_params = {
            'learning_rate': 0.0723,
            'max_depth': 3,
            'max_iter': 103,
            'min_samples_leaf': 23,
            'random_state': 42
        }
        
        hgb_model = HistGradientBoostingClassifier(**best_params)
        hgb_model.fit(X_train, y_train)
        
        available_features_ui_corrected = {col: HUMAN_READABLE_MAP.get(col, col.replace('_', ' ').title()) for col in available_features}
        
        # Calculate permutation importance for HistGradientBoosting
        perm_importance = permutation_importance(hgb_model, X_train, y_train, n_repeats=10, random_state=42)
        
        feature_importance_df = pd.DataFrame({
            'feature': [available_features_ui_corrected[f] for f in available_features],
            'importance': perm_importance.importances_mean
        }).sort_values('importance', ascending=True)
        
        print("Model trained successfully!")
        print(f"Features used: {available_features}")

# 3. THEME & STYLING 

LIGHT_BG = '#f0f2f5'         
CONTAINER_BG = '#ffffff'     
TEXT_COLOR = '#333333'       
ACCENT_BLUE = '#1f77b4'      
ACCENT_RED = '#d62728'       
ACCENT_HEADER = '#0056b3'    
BEST_THRESHOLD = 0.3  # Optimal threshold from notebook

def get_empty_fig(title="No Data Available", color=TEXT_COLOR):
    fig = px.scatter(title=title, template='plotly_white')
    fig.update_layout(
        plot_bgcolor=LIGHT_BG,
        paper_bgcolor=CONTAINER_BG,
        font=dict(color=color, family="Arial, Tahoma"),
        title_font=dict(color=ACCENT_HEADER, size=18)
    )
    return fig

# Custom styles
MAIN_LAYOUT_STYLE = {
    'padding': '20px', 
    'maxWidth': '1300px', 
    'margin': 'auto', 
    'backgroundColor': LIGHT_BG, 
    'direction': 'ltr' 
}

CONTAINER_STYLE = {
    'padding': '25px', 
    'backgroundColor': CONTAINER_BG, 
    'borderRadius': '10px', 
    'boxShadow': '0 4px 10px rgba(0,0,0,0.1)', 
    'marginBottom': '30px', 
    'color': TEXT_COLOR, 
    'border': '1px solid #e0e0e0', 
    'direction': 'ltr'
}

HEADER_CONTAINER_STYLE = {
    'backgroundColor': CONTAINER_BG, 
    'backgroundImage': 'linear-gradient(135deg, #ffffff, #e0e0e0)', 
    'marginBottom': '30px', 
    'borderRadius': '10px', 
    'boxShadow': '0 8px 15px rgba(0,0,0,0.15)', 
    'direction': 'ltr'
}

PREDICT_BUTTON_STYLE = {
    'backgroundColor': ACCENT_RED, 
    'color': 'white', 
    'padding': '12px 25px', 
    'border': 'none', 
    'borderRadius': '6px', 
    'cursor': 'pointer', 
    'fontWeight': 'bold', 
    'textAlign': 'center', 
    'width': '100%', 
    'fontSize': '1.1em', 
    'boxShadow': '0 4px #a02020', 
    'transition': 'all 0.2s'
}

# 4. INITIAL PLOTS

fig_importance = get_empty_fig(title="Feature Importance - No Data")
fig_stats = get_empty_fig(title="Statistical Significance - No Data")

if hgb_model is not None and available_features and feature_importance_df is not None:
    # A. Feature Importance Plot
    fig_importance = px.bar(feature_importance_df, x='importance', y='feature',
                             title='Feature Importance in Prediction (Permutation-Based)', 
                             template='plotly_white', 
                             color='importance', 
                             color_continuous_scale='Viridis',
                             orientation='h')
    fig_importance.update_layout(
        xaxis_title="Importance Score", yaxis_title="", plot_bgcolor=LIGHT_BG, 
        paper_bgcolor=CONTAINER_BG, font=dict(color=TEXT_COLOR),
        title_font=dict(color=ACCENT_HEADER, size=18)
    )
    fig_importance.update_yaxes(automargin=True)

    # B. T-Test for Statistical Significance
    t_test_results = []
    numeric_features = [f for f in available_features if pd.api.types.is_numeric_dtype(df[f])]
    for feature in numeric_features:
        group_0 = df[df['Result'] == 0][feature].dropna()
        group_1 = df[df['Result'] == 1][feature].dropna()
        if len(group_0) > 1 and len(group_1) > 1:
            t_stat, p_value = stats.ttest_ind(group_0, group_1, equal_var=False, nan_policy='omit')
        else:
            p_value = 1.0 
        t_test_results.append({'Feature': available_features_ui.get(feature, feature), 'P-Value': p_value, 'Significant': p_value < 0.05})

    t_test_df = pd.DataFrame(t_test_results)
    if not t_test_df.empty:
        fig_stats = px.scatter(t_test_df, x='P-Value', y='Feature', color='Significant',
                                 template='plotly_white', title='Statistical Significance (T-Test)', 
                                 color_discrete_map={True: ACCENT_RED, False: ACCENT_BLUE}, 
                                 hover_data=['P-Value'])
        fig_stats.add_vline(x=0.05, line_dash="dash", line_color="#808080", annotation_text="Significance Level (0.05)") 
        fig_stats.update_layout(
            xaxis_title="P-Value", yaxis_title="", plot_bgcolor=LIGHT_BG, 
            paper_bgcolor=CONTAINER_BG, font=dict(color=TEXT_COLOR),
            title_font=dict(color=ACCENT_HEADER, size=18)
        )
        fig_stats.update_yaxes(automargin=True)

# 5. DASH LAYOUT

app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
app.title = 'Cardiovascular Disease Analysis Dashboard'

feature_means = df[available_features].mean(numeric_only=True).to_dict() if available_features and not df.empty else {}

# Create prediction inputs
prediction_inputs = []

for feature in available_features:
    readable_name = available_features_ui.get(feature, feature)
    default_value = round(feature_means.get(feature, 0.0), 2)
    
    input_style_ltr = {
        'width': '100%', 
        'padding': '10px', 
        'borderRadius': '6px', 
        'border': '1px solid #cccccc', 
        'backgroundColor': '#f9f9f9', 
        'color': TEXT_COLOR, 
        'boxShadow': 'inset 0 1px 3px rgba(0,0,0,0.1)',
        'direction': 'ltr',
        'textAlign': 'left'
    }

    step = 1.0
    min_val = 0
    max_val = None

    if feature == 'Gender':
        default_value = 1 if default_value > 0.5 else 0 
        step = 1
        min_val = 0
        max_val = 1
    elif feature in ['hypertension', 'diabetes', 'obesity', 'family_history']:
        default_value = 0
        step = 1
        min_val = 0
        max_val = 1
        
    prediction_inputs.append(
        html.Div([
            html.Label(f"{readable_name}:", 
                        style={'fontWeight': 'bold', 'marginTop': '10px', 'color': TEXT_COLOR, 'fontSize': '0.9em', 'textAlign': 'left', 'display': 'block'}),
            dcc.Input(
                id=f'input-{feature}',
                type='number',
                placeholder=f'Enter value for {readable_name}',
                value=default_value, 
                min=min_val, 
                max=max_val,
                step=step,
                style={**input_style_ltr, 'color': TEXT_COLOR}
            )
        ], className='three columns', style={'padding': '0 10px', 'marginBottom': '15px', 'direction': 'ltr'})
    )

row1_inputs = prediction_inputs[:4]
row2_inputs = prediction_inputs[4:8] if len(prediction_inputs) > 4 else []
row3_inputs = prediction_inputs[8:] if len(prediction_inputs) > 8 else []

# Get Age range for slider
age_min, age_max, age_marks = 20, 80, {}
if 'Age' in df.columns and not df['Age'].empty:
    age_min = int(df['Age'].min().item())
    age_max = int(df['Age'].max().item())
    age_range_diff = age_max - age_min
    age_step = max(10, age_range_diff // 5) if age_range_diff > 10 else 1
    age_marks = {str(age): {'label': str(age), 'style': {'color': ACCENT_HEADER, 'fontSize': '0.8em'}} 
                  for age in range(age_min, age_max + 1, age_step)}

# Main Layout
app.layout = html.Div([
    
    html.Div(className='header-container', children=[
        html.H1("Cardiovascular Disease Analysis Dashboard", 
                style={'textAlign': 'center', 'color': ACCENT_HEADER, 'paddingTop': '15px', 'marginBottom': '5px'}), 
    ], style=HEADER_CONTAINER_STYLE),
    
    # Data Summary
    html.Div(className='row', children=[
        html.Div(className='twelve columns', 
                  style={**CONTAINER_STYLE, 'color': TEXT_COLOR, 'backgroundColor': CONTAINER_BG}, 
                  children=[
            html.H4("Data Summary", style={'color': ACCENT_HEADER, 'marginBottom': '10px', 'textAlign': 'left'}),
            html.P(f"Total Records: {len(df)} | Positive Cases (High Risk): {df['Result'].sum() if 'Result' in df.columns else 'N/A'} | "
                   f"Available Features: {len(available_features)}", 
                   style={'margin': '5px 0', 'fontSize': '0.9em', 'textAlign': 'left'})
        ])
    ], style={'direction': 'ltr'}),
    
    # Prediction Section
    html.Div(className='row', children=[
        html.Div(className='twelve columns', 
                  style=CONTAINER_STYLE, 
                  children=[
            html.H3("Instant Heart Attack Risk Prediction System", 
                     style={'color': ACCENT_HEADER, 'borderBottom': '1px solid #cccccc', 'paddingBottom': '10px', 'textAlign': 'left'}),
            
            # Input Rows
            html.Div(className='row', children=row1_inputs, style={'marginBottom': '5px'}),
            html.Div(className='row', children=row2_inputs, style={'marginBottom': '5px'}),
            html.Div(className='row', children=row3_inputs, style={'marginBottom': '20px'}),
            
            html.Div(className='row', children=[
                html.Div(className='three columns', children=[
                    html.Button('Predict Outcome', id='predict-button', n_clicks=0, style=PREDICT_BUTTON_STYLE),
                ], style={'padding': '0 10px'}),
                
                html.Div(className='nine columns', children=[
                    html.Div(id='prediction-output', 
                              style={'padding': '15px', 'border': '2px solid #cccccc', 'backgroundColor': LIGHT_BG, 'borderRadius': '6px', 'minHeight': '50px', 'textAlign': 'left', 'fontSize': '1.1em', 'fontWeight': 'normal', 'color': TEXT_COLOR, 'boxShadow': '0 2px 5px rgba(0,0,0,0.1)'})
                ], style={'padding': '0 10px'})
            ])
        ])
    ], style={'direction': 'ltr'}),
    
    # Interactive Analysis Section
    html.Div(className='row', children=[
        html.Div(className='six columns', children=[
            html.Div(
                style={**CONTAINER_STYLE, 'height': '500px', 'marginBottom': '0px'},
                children=[
                    html.H3("Interactive Analysis Tools", 
                             style={'color': ACCENT_HEADER, 'borderBottom': '1px solid #cccccc', 'paddingBottom': '10px', 'textAlign': 'left'}),
                    
                    html.Div(children=[
                        html.Label("Select X-Axis Feature:", style={'fontWeight': 'bold', 'marginTop': '15px', 'textAlign': 'left', 'display': 'block', 'color': TEXT_COLOR}),
                        dcc.Dropdown(
                            id='xaxis-feature',
                            options=[{'label': available_features_ui.get(val, val), 'value': val} for val in available_features],
                            value=available_features[0] if available_features else None,
                            style={'marginBottom': '20px', 'textAlign': 'left', 'color': TEXT_COLOR},
                            clearable=False,
                            className='light-theme-dropdown' 
                        ),
                        html.Label("Select Y-Axis Feature:", style={'fontWeight': 'bold', 'textAlign': 'left', 'display': 'block', 'color': TEXT_COLOR}),
                        dcc.Dropdown(
                            id='yaxis-feature',
                            options=[{'label': available_features_ui.get(val, val), 'value': val} for val in available_features],
                            value=available_features[1] if len(available_features) > 1 else (available_features[0] if available_features else None),
                            style={'marginBottom': '20px', 'textAlign': 'left', 'color': TEXT_COLOR},
                            clearable=False,
                            className='light-theme-dropdown'
                        ),
                        html.Label(f"Filter by Age: ({age_min} - {age_max})", style={'fontWeight': 'bold', 'textAlign': 'left', 'display': 'block', 'color': TEXT_COLOR}),
                        dcc.RangeSlider(
                            id='age-slider',
                            min=age_min,
                            max=age_max,
                            value=[age_min, age_max],
                            marks=age_marks,
                            step=1,
                            className='light-theme-slider'
                        )
                    ])
                ])
        ], style={'padding': '0 10px'}),
        
        html.Div(className='six columns', children=[
            dcc.Graph(id='scatter-plot', 
                      config={'responsive': True}, 
                      figure=get_empty_fig(title="Interactive Scatter Plot"), 
                      style={'height': '500px', 'boxShadow': '0 4px 10px rgba(0,0,0,0.1)', 'borderRadius': '10px', 'overflow': 'hidden'})
        ], style={'padding': '0 10px'})
    ], style={'direction': 'ltr', 'marginBottom': '30px'}),

    # Charts Section
    html.Div(className='row', children=[
        html.Div(className='six columns', children=[
            dcc.Graph(id='feature-importance-chart', figure=fig_importance, 
                      config={'responsive': True},
                      style={'height': '450px', 'boxShadow': '0 4px 10px rgba(0,0,0,0.1)', 'borderRadius': '10px', 'overflow': 'hidden'})
        ], style={'padding': '0 10px'}),
        
        html.Div(className='six columns', children=[
            dcc.Graph(id='statistical-significance-chart', figure=fig_stats, 
                      config={'responsive': True},
                      style={'height': '450px', 'boxShadow': '0 4px 10px rgba(0,0,0,0.1)', 'borderRadius': '10px', 'overflow': 'hidden'})
        ], style={'padding': '0 10px'})
    ], style={'direction': 'ltr', 'marginBottom': '30px'})
], style=MAIN_LAYOUT_STYLE)

# 6. DASH CALLBACKS

@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('xaxis-feature', 'value'),
     Input('yaxis-feature', 'value'),
     Input('age-slider', 'value')]
)
def update_scatter_plot(x_feature, y_feature, age_range):
    try:
        if not x_feature or not y_feature or not available_features or df.empty:
            return get_empty_fig(title="Select Features to Display")

        x_label = available_features_ui.get(x_feature, x_feature)
        y_label = available_features_ui.get(y_feature, y_feature)
        
        filtered_df = df.copy()
        if 'Age' in df.columns and age_range:
            filtered_df = filtered_df[(filtered_df['Age'] >= age_range[0]) & (filtered_df['Age'] <= age_range[1])]
        
        if filtered_df.empty:
            return get_empty_fig(title="No data in selected age range")
        
        # Create scatter plot
        fig = px.scatter(
            filtered_df, x=x_feature, y=y_feature, 
            color='Result' if 'Result' in df.columns else None,
            title=f'{y_label} vs. {x_label} (Ages {age_range[0]}-{age_range[1]})', 
            color_discrete_map={0: ACCENT_BLUE, 1: ACCENT_RED} if 'Result' in df.columns else None, 
            template='plotly_white',
            size_max=15
        )
        
        if 'Result' in df.columns:
            fig.update_layout(legend_title_text='Heart Attack Outcome')
            fig.update_traces(
                marker=dict(size=8, opacity=0.7),
                selector=dict(mode='markers')
            )
        
        fig.update_layout(
            plot_bgcolor=LIGHT_BG, 
            paper_bgcolor=CONTAINER_BG, 
            xaxis_title=x_label,
            yaxis_title=y_label,
            font=dict(color=TEXT_COLOR, family="Arial, Tahoma"),
            title_font=dict(color=ACCENT_HEADER, size=18),
            hovermode='closest'
        )
        return fig
    except Exception as e:
        print(f"Error in scatter plot: {e}")
        return get_empty_fig(title=f"Error: {str(e)}")

@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State(f'input-{feature}', 'value') for feature in available_features]
)
def predict_heart_attack(n_clicks, *input_values):
    try:
        if n_clicks is None or n_clicks < 1:
            return dash.no_update
            
        if hgb_model is None:
            return html.Div("Prediction model is unavailable. Please check your data and train the model.", 
                              style={'color': ACCENT_RED, 'fontWeight': 'bold'})
        
        if any(val is None for val in input_values):
            return html.Div("Please ensure all fields have valid values.", 
                              style={'color': ACCENT_RED, 'fontWeight': 'bold'})

        input_data = {}
        for i, feature in enumerate(available_features):
            input_data[feature] = float(input_values[i])

        X_predict = pd.DataFrame([input_data])
        X_predict = X_predict.reindex(columns=available_features)
        
        for col in X_predict.columns:
            if X_predict[col].isnull().any():
                X_predict[col] = X_predict[col].fillna(df[col].median())
        
        proba = hgb_model.predict_proba(X_predict)[0]
        prob_negative = proba[0] * 100
        prob_positive = proba[1] * 100
        
        # Apply optimized threshold
        prediction = (prob_positive / 100 >= BEST_THRESHOLD).astype(int)
        
        if prediction == 1:
            result_text = "Positive (High Risk)" 
            risk_percent = prob_positive
            safe_percent = prob_negative
            color_style = {'color': ACCENT_RED, 'fontWeight': 'bold', 'fontSize': '1.3em', 'padding': '0 5px'}
        else:
            result_text = "Negative (Low Risk)" 
            risk_percent = prob_positive
            safe_percent = prob_negative
            color_style = {'color': ACCENT_BLUE, 'fontWeight': 'bold', 'fontSize': '1.3em', 'padding': '0 5px'}

        return html.Div([
            html.Span("Predicted Outcome:", style={'color': ACCENT_HEADER, 'fontWeight': 'normal', 'marginRight': '10px'}),
            html.Span(result_text, style=color_style),
            html.Br(),
            
            # Probability Bar
            html.Div(style={'display': 'flex', 'marginTop': '10px', 'height': '15px', 'borderRadius': '4px', 'overflow': 'hidden', 'boxShadow': '0 1px 3px rgba(0,0,0,0.2)', 'direction': 'ltr'}, children=[
                html.Div(style={'width': f'{safe_percent:.0f}%', 'backgroundColor': ACCENT_BLUE, 'height': '100%'}), 
                html.Div(style={'width': f'{risk_percent:.0f}%', 'backgroundColor': ACCENT_RED, 'height': '100%'}) 
            ]),
            
            # Probability Text
            html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'marginTop': '5px', 'fontSize': '0.9em', 'direction': 'ltr'}, children=[
                html.Span(f"Low Risk: {safe_percent:.2f}%", style={'color': ACCENT_BLUE, 'textAlign': 'left'}), 
                html.Span(f"High Risk: {risk_percent:.2f}%", style={'color': ACCENT_RED, 'textAlign': 'right'}) 
            ])
        ], style={'textAlign': 'left'})
    except Exception as e:
        print(f"Error in prediction: {e}")
        return html.Div(f"Prediction error: {str(e)}", 
                          style={'color': ACCENT_RED, 'fontWeight': 'bold'})

def open_browser():
    time.sleep(3)
    webbrowser.open_new_tab("http://127.0.0.1:8050")

if __name__ == '__main__':
    HOST = '127.0.0.1'
    PORT = 8050
    threading.Thread(target=open_browser).start()
    app.run(debug=True, host=HOST, port=PORT, use_reloader=False)