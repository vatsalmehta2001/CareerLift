import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to the path if needed
if '.' not in sys.path:
    sys.path.append('.')

from utils.data_processor import DataProcessor
from models.income_predictor import IncomePredictor
from optimizer.career_optimizer import CareerPathOptimizer
from optimizer.roi_calculator import ROICalculator


def create_app(model_path='../models/saved/random_forest', data_path='../data/adult.csv'):
    """
    Create the Dash application for the Career Path Optimizer.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model.
    data_path : str
        Path to the data file.
        
    Returns:
    --------
    dash.Dash
        Dash application.
    """
    try:
        print(f"Initializing optimizer with model path: {model_path} and data path: {data_path}")
        
        # Check if paths are valid
        if not os.path.exists(data_path):
            print(f"WARNING: Data file not found at {data_path}")
            # Try to find the data file in other locations
            possible_data_paths = [
                'data/adult.csv',
                '../data/adult.csv',
                './data/adult.csv'
            ]
            for path in possible_data_paths:
                if os.path.exists(path):
                    data_path = path
                    print(f"Found data file at {data_path}")
                    break
        
        # Initialize optimizer and ROI calculator
        optimizer = CareerPathOptimizer(model_path, data_path)
        roi_calculator = ROICalculator(optimizer)
        
        # Get the dataset for exploration
        df = optimizer.df.copy()
        
        print("Optimizer and ROI calculator initialized successfully")
    except Exception as e:
        print(f"ERROR initializing optimizer: {str(e)}")
        # Create a minimal DataFrame for the UI
        df = pd.DataFrame({
            'education': ['HS-grad', 'Bachelors', 'Masters', 'Doctorate'],
            'occupation': ['Sales', 'Tech-support', 'Exec-managerial', 'Prof-specialty'],
            'workclass': ['Private', 'Self-emp-not-inc', 'Local-gov', 'Federal-gov'],
            'marital_status': ['Never-married', 'Married-civ-spouse', 'Divorced', 'Separated']
        })
        
        # Create dummy optimizer and calculator that return safe values
        class DummyOptimizer:
            def get_income_probability(self, _):
                return 0.5
                
            def optimize_career_path(self, _):
                return {
                    'education_suggestions': [],
                    'occupation_suggestions': [],
                    'hours_suggestions': [],
                    'combined_suggestions': []
                }
                
            def calculate_optimal_path(self, _, max_steps=3):
                return [{'step': 0, 'probability': 0.5}]
                
        optimizer = DummyOptimizer()
        
        class DummyROICalculator:
            def __init__(self):
                pass
                
            def calculate_comprehensive_roi(self, _):
                return {
                    'education_roi': [],
                    'occupation_roi': [],
                    'combined_roi': []
                }
                
        roi_calculator = DummyROICalculator()
    
    # Create Dash app with external stylesheets
    app = dash.Dash(
        __name__, 
        title='Career Path Optimizer',
        assets_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets'),
        meta_tags=[
            {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
        ]
    )
    
    # Define app layout with improved UI
    app.layout = html.Div([
        # Header
        html.Div([
            html.H1('Career Path Optimizer', className='app-header'),
            html.P('Optimize your career path to maximize income potential using machine learning', className='app-description')
        ], className='header-container'),
        
        # Main container
        html.Div([
            # Left panel (inputs)
            html.Div([
                html.H3('Your Profile'),
                
                # Age input
                html.Label('Age'),
                dcc.Slider(
                    id='age-slider',
                    min=18,
                    max=65,
                    value=35,
                    marks={i: str(i) for i in range(20, 70, 10)},
                    step=1,
                    className='slider'
                ),
                html.Div(id='age-output'),
                
                # Education input
                html.Label('Education'),
                dcc.Dropdown(
                    id='education-dropdown',
                    options=[
                        {'label': education, 'value': education}
                        for education in sorted(df['education'].unique())
                    ],
                    value='Bachelors',
                    clearable=False
                ),
                
                # Occupation input
                html.Label('Occupation'),
                dcc.Dropdown(
                    id='occupation-dropdown',
                    options=[
                        {'label': occupation, 'value': occupation}
                        for occupation in sorted(df['occupation'].unique()) if pd.notna(occupation)
                    ],
                    value='Prof-specialty',
                    clearable=False
                ),
                
                # Work class input
                html.Label('Work Class'),
                dcc.Dropdown(
                    id='workclass-dropdown',
                    options=[
                        {'label': workclass, 'value': workclass}
                        for workclass in sorted(df['workclass'].unique()) if pd.notna(workclass)
                    ],
                    value='Private',
                    clearable=False
                ),
                
                # Hours per week input
                html.Label('Hours per Week'),
                dcc.Slider(
                    id='hours-slider',
                    min=20,
                    max=80,
                    value=40,
                    marks={i: str(i) for i in range(20, 90, 10)},
                    step=5,
                    className='slider'
                ),
                html.Div(id='hours-output'),
                
                # Other demographic inputs
                html.Label('Gender'),
                dcc.RadioItems(
                    id='gender-radio',
                    options=[
                        {'label': 'Male', 'value': 'Male'},
                        {'label': 'Female', 'value': 'Female'}
                    ],
                    value='Male',
                    className='radio-group'
                ),
                
                html.Label('Marital Status'),
                dcc.Dropdown(
                    id='marital-dropdown',
                    options=[
                        {'label': status, 'value': status}
                        for status in sorted(df['marital_status'].unique())
                    ],
                    value='Married-civ-spouse',
                    clearable=False
                ),
                
                # Calculate button
                html.Button('Calculate Optimal Career Path', id='calculate-button', className='calculate-button'),
                
            ], className='input-panel'),
            
            # Right panel (results)
            html.Div([
                html.Div([
                    html.H3('Income Probability'),
                    dcc.Graph(id='income-gauge', config={'displayModeBar': False})
                ], className='result-section'),
                
                html.Div([
                    html.H3('Career Optimization Suggestions'),
                    dcc.Graph(id='optimization-chart', config={'displayModeBar': False})
                ], className='result-section'),
                
                html.Div([
                    html.H3('Return on Investment (ROI) Analysis'),
                    dcc.Graph(id='roi-chart', config={'displayModeBar': False})
                ], className='result-section'),
                
                html.Div([
                    html.H3('Optimal Career Path'),
                    html.Div(id='optimal-path-output', className='path-output')
                ], className='result-section')
            ], className='results-panel'),
        ], className='main-container'),
        
        # Disclaimer
        html.Div([
            html.H4("Important Disclaimer", className="disclaimer-title"),
            html.P([
                "This application uses the 1994 Census Income dataset which is ", 
                html.Strong("over 30 years old"), 
                ". The economic landscape and job market have changed significantly since then.",
                " Modern occupations, current income levels, and new educational pathways are not reflected in this historical data.",
                " This tool is a portfolio demonstration and should not be used for actual career planning without considering current economic realities."
            ], className="disclaimer-text"),
        ], className="disclaimer-container"),
        
        # Footer
        html.Div([
            html.P([
                'Career Path Optimizer | Built with Python, Dash, and Machine Learning | ',
                html.A('Portfolio Project', href='#', style={'color': '#4CAF50', 'text-decoration': 'none'}),
                ' by Vatsal Mehta'
            ])
        ], className='footer')
    ])
    
    # Define callback to update age output
    @app.callback(
        Output('age-output', 'children'),
        Input('age-slider', 'value')
    )
    def update_age_output(value):
        return f'Age: {value}'
    
    # Define callback to update hours output
    @app.callback(
        Output('hours-output', 'children'),
        Input('hours-slider', 'value')
    )
    def update_hours_output(value):
        return f'Hours per Week: {value}'
    
    # Define callback to update income gauge
    @app.callback(
        Output('income-gauge', 'figure'),
        Input('calculate-button', 'n_clicks'),
        State('age-slider', 'value'),
        State('education-dropdown', 'value'),
        State('occupation-dropdown', 'value'),
        State('workclass-dropdown', 'value'),
        State('hours-slider', 'value'),
        State('gender-radio', 'value'),
        State('marital-dropdown', 'value')
    )
    def update_income_gauge(n_clicks, age, education, occupation, workclass, hours, gender, marital):
        # Default figure if button not clicked
        if n_clicks is None:
            return create_gauge(0.5, 'Income Probability')
        
        try:
            # Create person data
            person_data = create_person_data(age, education, occupation, workclass, hours, gender, marital)
            
            # Get income probability
            prob = optimizer.get_income_probability(person_data)
            
            return create_gauge(prob, 'Probability of Income >$50K')
        except Exception as e:
            print(f"Error updating income gauge: {str(e)}")
            # Return default gauge on error
            return create_gauge(0.5, 'Error: Could not calculate probability')
    
    # Define callback to update optimization chart
    @app.callback(
        Output('optimization-chart', 'figure'),
        Input('calculate-button', 'n_clicks'),
        State('age-slider', 'value'),
        State('education-dropdown', 'value'),
        State('occupation-dropdown', 'value'),
        State('workclass-dropdown', 'value'),
        State('hours-slider', 'value'),
        State('gender-radio', 'value'),
        State('marital-dropdown', 'value')
    )
    def update_optimization_chart(n_clicks, age, education, occupation, workclass, hours, gender, marital):
        # Default figure if button not clicked
        if n_clicks is None:
            return px.bar(
                title='Click Calculate to see optimization suggestions',
                template="plotly_dark"
            )
        
        try:
            # Create person data
            person_data = create_person_data(age, education, occupation, workclass, hours, gender, marital)
            
            # Get optimization results
            results = optimizer.optimize_career_path(person_data)
            
            # Create data for chart
            chart_data = create_optimization_chart_data(results)
            
            if chart_data.empty:
                return px.bar(
                    title='No optimization suggestions available',
                    template="plotly_dark"
                )
            
            # Create figure
            fig = px.bar(
                chart_data, 
                x='Improvement', 
                y='Description', 
                color='Type',
                title='Career Optimization Suggestions (% Improvement in High Income Probability)',
                labels={'Improvement': '% Improvement', 'Description': ''},
                orientation='h',
                template="plotly_dark",
                color_discrete_map={
                    'Education': '#4CAF50',
                    'Occupation': '#2196F3',
                    'Hours': '#FFC107',
                    'Combined': '#9C27B0'
                }
            )
            
            # Add current probability line
            fig.add_vline(x=0, line_dash='dash', line_color='white')
            
            # Customize layout for dark theme
            fig.update_layout(
                paper_bgcolor='#1e1e1e',
                plot_bgcolor='#1e1e1e',
                font_color='white',
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Format the y-axis
            fig.update_yaxes(tickfont=dict(size=12))
            
            # Add percentage formatting
            fig.update_traces(
                texttemplate='%{x:.1f}%',
                textposition='outside'
            )
            
            return fig
        except Exception as e:
            print(f"Error updating optimization chart: {str(e)}")
            # Return empty chart on error
            return px.bar(
                title=f'Error: Could not generate optimization suggestions',
                template="plotly_dark"
            )
    
    # Define callback to update ROI chart
    @app.callback(
        Output('roi-chart', 'figure'),
        Input('calculate-button', 'n_clicks'),
        State('age-slider', 'value'),
        State('education-dropdown', 'value'),
        State('occupation-dropdown', 'value'),
        State('workclass-dropdown', 'value'),
        State('hours-slider', 'value'),
        State('gender-radio', 'value'),
        State('marital-dropdown', 'value')
    )
    def update_roi_chart(n_clicks, age, education, occupation, workclass, hours, gender, marital):
        # Default figure if button not clicked
        if n_clicks is None:
            return px.bar(
                title='Click Calculate to see ROI analysis',
                template="plotly_dark"
            )
        
        try:
            # Create person data
            person_data = create_person_data(age, education, occupation, workclass, hours, gender, marital)
            
            # Calculate ROI
            roi_results = roi_calculator.calculate_comprehensive_roi(person_data)
            
            # Create data for chart
            chart_data = create_roi_chart_data(roi_results)
            
            if chart_data.empty:
                return px.bar(
                    title='No ROI data available',
                    template="plotly_dark"
                )
            
            # Create figure
            fig = px.bar(
                chart_data, 
                x='5-Year ROI', 
                y='Description', 
                color='Category',
                title='5-Year Return on Investment (ROI)',
                labels={'5-Year ROI': 'ROI (%)', 'Description': ''},
                orientation='h',
                template="plotly_dark",
                color_discrete_map={
                    'Education': '#4CAF50',
                    'Occupation': '#2196F3',
                    'Combined': '#9C27B0'
                }
            )
            
            # Customize layout for dark theme
            fig.update_layout(
                paper_bgcolor='#1e1e1e',
                plot_bgcolor='#1e1e1e',
                font_color='white',
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Format the y-axis
            fig.update_yaxes(tickfont=dict(size=12))
            
            # Add percentage formatting
            fig.update_traces(
                texttemplate='%{x:.1f}%',
                textposition='outside'
            )
            
            return fig
        except Exception as e:
            print(f"Error updating ROI chart: {str(e)}")
            # Return empty chart on error
            return px.bar(
                title=f'Error: Could not generate ROI analysis',
                template="plotly_dark"
            )
    
    # Define callback to update optimal path output
    @app.callback(
        Output('optimal-path-output', 'children'),
        Input('calculate-button', 'n_clicks'),
        State('age-slider', 'value'),
        State('education-dropdown', 'value'),
        State('occupation-dropdown', 'value'),
        State('workclass-dropdown', 'value'),
        State('hours-slider', 'value'),
        State('gender-radio', 'value'),
        State('marital-dropdown', 'value')
    )
    def update_optimal_path(n_clicks, age, education, occupation, workclass, hours, gender, marital):
        # Default output if button not clicked
        if n_clicks is None:
            return html.P('Click Calculate to see the optimal career path')
        
        try:
            # Create person data
            person_data = create_person_data(age, education, occupation, workclass, hours, gender, marital)
            
            # Calculate optimal path
            path = optimizer.calculate_optimal_path(person_data, max_steps=3)
            
            # Create output
            return create_path_output(path)
        except Exception as e:
            print(f"Error updating optimal path: {str(e)}")
            # Return error message
            return html.P(f'Error: Could not calculate optimal career path')
    
    return app


def create_person_data(age, education, occupation, workclass, hours, gender, marital):
    """
    Create a person data dictionary from input values.
    
    Parameters:
    -----------
    Various input parameters from the form.
    
    Returns:
    --------
    dict
        Dictionary containing the person's feature values.
    """
    # Education num mapping
    education_num_mapping = {
        'Preschool': 1,
        '1st-4th': 2,
        '5th-6th': 3,
        '7th-8th': 4,
        '9th': 5,
        '10th': 6,
        '11th': 7,
        '12th': 8,
        'HS-grad': 9,
        'Some-college': 10,
        'Assoc-voc': 11,
        'Assoc-acdm': 11,
        'Bachelors': 13,
        'Masters': 14,
        'Prof-school': 15,
        'Doctorate': 16
    }
    
    return {
        'age': age,
        'workclass': workclass,
        'education': education,
        'education_num': education_num_mapping.get(education, 9),
        'marital_status': marital,
        'occupation': occupation,
        'relationship': 'Husband' if gender == 'Male' else 'Wife',
        'race': 'White',  # Default
        'sex': gender,
        'capital_gain': 0,
        'capital_loss': 0,
        'hours_per_week': hours,
        'native_country': 'United-States'  # Default
    }


def create_gauge(value, title):
    """
    Create a gauge chart.
    
    Parameters:
    -----------
    value : float
        Value to display (between 0 and 1).
    title : str
        Chart title.
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Gauge chart figure.
    """
    # Set theme colors
    bg_color = "#1e1e1e"
    text_color = "#ffffff"
    gauge_bg = "#2a2a2a"
    gauge_border = "#444444"
    
    # Color gradient for the gauge
    colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#27ae60"]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': title,
            'font': {'color': text_color, 'size': 18}
        },
        number={
            'font': {'color': text_color, 'size': 40},
            'suffix': '%'
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 1,
                'tickcolor': gauge_border,
                'tickfont': {'color': text_color}
            },
            'bar': {'color': "#4CAF50"},
            'bgcolor': gauge_bg,
            'borderwidth': 2,
            'bordercolor': gauge_border,
            'steps': [
                {'range': [0, 20], 'color': colors[0]},
                {'range': [20, 40], 'color': colors[1]},
                {'range': [40, 60], 'color': colors[2]},
                {'range': [60, 80], 'color': colors[3]},
                {'range': [80, 100], 'color': colors[4]}
            ],
            'threshold': {
                'line': {'color': "#ffffff", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    # Update layout with dark theme
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font={'color': text_color}
    )
    
    return fig


def create_optimization_chart_data(optimization_results):
    """
    Create data for the optimization chart.
    
    Parameters:
    -----------
    optimization_results : dict
        Results from the optimize_career_path method.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the chart data.
    """
    # Collect all suggestions
    all_suggestions = []
    
    # Safely get collections with defaults
    education_suggestions = optimization_results.get('education_suggestions', [])
    occupation_suggestions = optimization_results.get('occupation_suggestions', [])
    hours_suggestions = optimization_results.get('hours_suggestions', [])
    combined_suggestions = optimization_results.get('combined_suggestions', [])
    
    # Education suggestions
    for sugg in education_suggestions:
        try:
            all_suggestions.append({
                'Type': 'Education',
                'Description': f"Education: {sugg.get('education', 'Unknown')}",
                'Improvement': sugg.get('improvement', 0)
            })
        except Exception as e:
            print(f"Error processing education suggestion: {str(e)}")
    
    # Occupation suggestions
    for sugg in occupation_suggestions:
        try:
            all_suggestions.append({
                'Type': 'Occupation',
                'Description': f"Occupation: {sugg.get('occupation', 'Unknown')}",
                'Improvement': sugg.get('improvement', 0)
            })
        except Exception as e:
            print(f"Error processing occupation suggestion: {str(e)}")
    
    # Hours suggestions
    for sugg in hours_suggestions:
        try:
            all_suggestions.append({
                'Type': 'Hours',
                'Description': f"Hours: {sugg.get('hours_per_week', 'Unknown')}",
                'Improvement': sugg.get('improvement', 0)
            })
        except Exception as e:
            print(f"Error processing hours suggestion: {str(e)}")
    
    # Combined suggestions
    for sugg in combined_suggestions:
        try:
            all_suggestions.append({
                'Type': 'Combined',
                'Description': f"Edu: {sugg.get('education', 'Unknown')} + Occ: {sugg.get('occupation', 'Unknown')}",
                'Improvement': sugg.get('improvement', 0)
            })
        except Exception as e:
            print(f"Error processing combined suggestion: {str(e)}")
    
    # Create DataFrame
    sugg_df = pd.DataFrame(all_suggestions)
    
    if not sugg_df.empty:
        # Sort by improvement
        sugg_df = sugg_df.sort_values('Improvement', ascending=False)
    
    return sugg_df


def create_roi_chart_data(roi_results):
    """
    Create data for the ROI chart.
    
    Parameters:
    -----------
    roi_results : dict
        Results from the calculate_comprehensive_roi method.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the chart data.
    """
    # Extract ROI data for comparison
    comparison_data = []
    
    # Safely get collections with defaults
    education_roi = roi_results.get('education_roi', [])
    occupation_roi = roi_results.get('occupation_roi', [])
    combined_roi = roi_results.get('combined_roi', [])
    
    # Education ROI
    for roi in education_roi:
        try:
            comparison_data.append({
                'Category': 'Education',
                'Description': f"Education: {roi.get('suggested_education', 'Unknown')}",
                '5-Year ROI': roi.get('roi_5yr', 0)
            })
        except Exception as e:
            print(f"Error processing education ROI: {str(e)}")
    
    # Occupation ROI
    for roi in occupation_roi:
        try:
            comparison_data.append({
                'Category': 'Occupation',
                'Description': f"Occupation: {roi.get('suggested_occupation', 'Unknown')}",
                '5-Year ROI': roi.get('roi_5yr', 0)
            })
        except Exception as e:
            print(f"Error processing occupation ROI: {str(e)}")
    
    # Combined ROI
    for roi in combined_roi:
        try:
            comparison_data.append({
                'Category': 'Combined',
                'Description': f"Edu: {roi.get('education', 'Unknown')} + Occ: {roi.get('occupation', 'Unknown')}",
                '5-Year ROI': roi.get('roi_10yr', 0) / 2  # Approximate 5-year from 10-year
            })
        except Exception as e:
            print(f"Error processing combined ROI: {str(e)}")
    
    # Create DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    if not comparison_df.empty:
        # Sort by ROI
        comparison_df = comparison_df.sort_values('5-Year ROI', ascending=False)
    
    return comparison_df


def create_path_output(path):
    """
    Create output elements for the optimal career path.
    
    Parameters:
    -----------
    path : list
        List of steps in the optimal career path.
        
    Returns:
    --------
    list
        List of HTML elements.
    """
    if not path:
        return [html.P("No path data available", className="no-data-message")]
    
    output_elements = []
    
    # Add path header
    output_elements.append(html.Div([
        html.H4("Your Personalized Career Path", style={'margin-top': '0', 'margin-bottom': '15px', 'color': '#4CAF50'})
    ]))
    
    # Create the path visualization
    for i, step in enumerate(path):
        step_class = "path-step"
        if i == 0:
            # Starting point
            output_elements.append(html.Div([
                html.Div([
                    html.Span("Step 0", className="step-number"),
                    html.H4("Starting Point", className="step-title"),
                ], className="step-header"),
                html.Div([
                    html.P([
                        html.Strong("Current Profile: "), 
                        f"Income >$50K probability: ",
                        html.Span(f"{step['probability']:.2%}", style={'color': '#4CAF50'})
                    ]),
                ], className="step-content")
            ], className=step_class))
            
            # Add arrow if not the last step
            if i < len(path) - 1:
                output_elements.append(html.Div("↓", className="path-arrow"))
        else:
            # Action steps
            improvement_color = '#4CAF50' if step.get('improvement', 0) > 0 else '#e74c3c'
            
            output_elements.append(html.Div([
                html.Div([
                    html.Span(f"Step {step['step']}", className="step-number"),
                    html.H4(step.get('action', 'Action'), className="step-title"),
                ], className="step-header"),
                html.Div([
                    html.P([
                        html.Strong("Action: "), 
                        step.get('action', 'Improve career path')
                    ]),
                    html.P([
                        html.Strong("New Probability: "), 
                        html.Span(f"{step['probability']:.2%}", style={'color': '#4CAF50'})
                    ]),
                    html.P([
                        html.Strong("Improvement: "), 
                        html.Span(f"{step.get('improvement', 0):.2f}%", style={'color': improvement_color})
                    ])
                ], className="step-content")
            ], className=step_class))
            
            # Add arrow if not the last step
            if i < len(path) - 1:
                output_elements.append(html.Div("↓", className="path-arrow"))
    
    # Add a call to action at the end
    output_elements.append(html.Div([
        html.P([
            "Following this path could maximize your income potential. ",
            html.Strong("Need more details? "), 
            "Contact us for personalized recommendations."
        ], style={'margin-top': '20px', 'font-style': 'italic'})
    ], className="path-footer"))
    
    return output_elements


def run_dashboard(model_path='../models/saved/random_forest', data_path='../data/adult.csv'):
    """
    Run the Dash application.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model.
    data_path : str
        Path to the data file.
    """
    try:
        # Convert relative paths to absolute if needed
        if model_path.startswith('../') or model_path.startswith('./'):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.normpath(os.path.join(base_dir, model_path))
        
        if data_path.startswith('../') or data_path.startswith('./'):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.normpath(os.path.join(base_dir, data_path))
        
        print("\n=== Launching Career Path Optimizer Dashboard ===")
        print(f"Model path: {model_path}")
        print(f"Data path: {data_path}")
        
        # Verify model and data paths
        if not os.path.exists(model_path):
            print(f"WARNING: Model not found at {model_path}")
        
        if not os.path.exists(data_path):
            print(f"WARNING: Data file not found at {data_path}")
            alternative_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'adult.csv')
            if os.path.exists(alternative_data_path):
                data_path = alternative_data_path
                print(f"Using alternative data path: {data_path}")
        
        # Create the app
        app = create_app(model_path, data_path)
        
        # Run the app
        print("\nStarting dashboard - press Ctrl+C to exit")
        print("Dashboard will be available at http://127.0.0.1:8050/ once loaded...")
        app.run(debug=True, host='127.0.0.1', port=8050)
        
        return app
    except Exception as e:
        print(f"Error starting dashboard: {str(e)}")
        print("If the error is about the port being in use, try closing other applications or using a different port.")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    run_dashboard() 