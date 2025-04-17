import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys
sys.path.append('..')


def create_income_distribution_plot(df):
    """
    Create an interactive income distribution plot.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset DataFrame.
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive plot.
    """
    # Convert income to binary for calculations
    binary_income = df['income'].map({'>50K': 1, '<=50K': 0})
    
    # Calculate income distribution by feature
    fig = go.Figure()
    
    # Add income distribution by age
    age_groups = pd.cut(df['age'], bins=[15, 25, 35, 45, 55, 65, 100],
                        labels=['16-25', '26-35', '36-45', '46-55', '56-65', '65+'])
    income_by_age = df.groupby(age_groups)['income'].apply(
        lambda x: (x == '>50K').mean() * 100
    ).reset_index()
    income_by_age.columns = ['Age Group', 'High Income %']
    
    fig.add_trace(go.Bar(
        x=income_by_age['Age Group'],
        y=income_by_age['High Income %'],
        name='By Age',
        marker_color='blue',
        visible=True
    ))
    
    # Add income distribution by education
    if 'education' in df.columns and 'education_num' in df.columns:
        education_order = df.groupby('education')['education_num'].mean().sort_values().index
        income_by_education = df.groupby('education')['income'].apply(
            lambda x: (x == '>50K').mean() * 100
        ).reset_index()
        income_by_education.columns = ['Education', 'High Income %']
        income_by_education['Education'] = pd.Categorical(
            income_by_education['Education'], 
            categories=education_order, 
            ordered=True
        )
        income_by_education = income_by_education.sort_values('Education')
        
        fig.add_trace(go.Bar(
            x=income_by_education['Education'],
            y=income_by_education['High Income %'],
            name='By Education',
            marker_color='green',
            visible=False
        ))
    
    # Add income distribution by occupation
    if 'occupation' in df.columns:
        income_by_occupation = df.groupby('occupation')['income'].apply(
            lambda x: (x == '>50K').mean() * 100
        ).reset_index()
        income_by_occupation.columns = ['Occupation', 'High Income %']
        income_by_occupation = income_by_occupation.sort_values('High Income %', ascending=False)
        
        fig.add_trace(go.Bar(
            x=income_by_occupation['Occupation'],
            y=income_by_occupation['High Income %'],
            name='By Occupation',
            marker_color='red',
            visible=False
        ))
    
    # Add income distribution by hours per week
    if 'hours_per_week' in df.columns:
        hours_groups = pd.cut(df['hours_per_week'], 
                             bins=[0, 20, 30, 40, 50, 60, 100],
                             labels=['0-20', '21-30', '31-40', '41-50', '51-60', '60+'])
        income_by_hours = df.groupby(hours_groups)['income'].apply(
            lambda x: (x == '>50K').mean() * 100
        ).reset_index()
        income_by_hours.columns = ['Hours per Week', 'High Income %']
        
        fig.add_trace(go.Bar(
            x=income_by_hours['Hours per Week'],
            y=income_by_hours['High Income %'],
            name='By Hours',
            marker_color='purple',
            visible=False
        ))
    
    # Add income distribution by gender
    if 'sex' in df.columns:
        income_by_sex = df.groupby('sex')['income'].apply(
            lambda x: (x == '>50K').mean() * 100
        ).reset_index()
        income_by_sex.columns = ['Gender', 'High Income %']
        
        fig.add_trace(go.Bar(
            x=income_by_sex['Gender'],
            y=income_by_sex['High Income %'],
            name='By Gender',
            marker_color='orange',
            visible=False
        ))
    
    # Create dropdown menu for switching between plots
    fig.update_layout(
        title='Income Distribution by Different Factors',
        xaxis_title='',
        yaxis_title='% with Income >$50K',
        updatemenus=[
            dict(
                active=0,
                buttons=list([
                    dict(label="By Age",
                         method="update",
                         args=[{"visible": [True, False, False, False, False]},
                               {"title": "Income Distribution by Age Group",
                                "xaxis": {"title": "Age Group"}}]),
                    dict(label="By Education",
                         method="update",
                         args=[{"visible": [False, True, False, False, False]},
                               {"title": "Income Distribution by Education",
                                "xaxis": {"title": "Education Level"}}]),
                    dict(label="By Occupation",
                         method="update",
                         args=[{"visible": [False, False, True, False, False]},
                               {"title": "Income Distribution by Occupation",
                                "xaxis": {"title": "Occupation"}}]),
                    dict(label="By Hours",
                         method="update",
                         args=[{"visible": [False, False, False, True, False]},
                               {"title": "Income Distribution by Hours Worked",
                                "xaxis": {"title": "Hours per Week"}}]),
                    dict(label="By Gender",
                         method="update",
                         args=[{"visible": [False, False, False, False, True]},
                               {"title": "Income Distribution by Gender",
                                "xaxis": {"title": "Gender"}}]),
                ]),
                x=0.1,
                xanchor="left",
                y=1.15,
                yanchor="top"
            ),
        ]
    )
    
    return fig


def create_feature_importance_plot(importance_df, top_n=15):
    """
    Create an interactive feature importance plot.
    
    Parameters:
    -----------
    importance_df : pandas.DataFrame
        DataFrame containing feature importance values.
    top_n : int
        Number of top features to show.
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive plot.
    """
    # Get top N features
    top_features = importance_df.head(top_n)
    
    # Create horizontal bar plot
    fig = px.bar(
        top_features, 
        x='Importance', 
        y='Feature',
        orientation='h',
        title=f'Top {top_n} Feature Importance',
        labels={'Importance': 'Importance Score', 'Feature': 'Feature'},
        color='Importance',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=500,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def create_income_prediction_sunburst(df):
    """
    Create a sunburst plot showing income prediction by education and occupation.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset DataFrame.
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive plot.
    """
    # Prepare data
    sunburst_data = df.groupby(['education', 'occupation'])['income'].apply(
        lambda x: (x == '>50K').mean() * 100
    ).reset_index()
    sunburst_data.columns = ['education', 'occupation', 'high_income_rate']
    
    # Create sunburst chart
    fig = px.sunburst(
        sunburst_data,
        path=['education', 'occupation'],
        values='high_income_rate',
        color='high_income_rate',
        color_continuous_scale='RdYlGn',
        title='Income >$50K Rate by Education and Occupation',
        labels={'high_income_rate': 'High Income Rate (%)'}
    )
    
    return fig


def create_scatter_matrix(df, numeric_features):
    """
    Create a scatter matrix of numeric features colored by income.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset DataFrame.
    numeric_features : list
        List of numeric features to include.
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive plot.
    """
    # Create scatter matrix
    fig = px.scatter_matrix(
        df,
        dimensions=numeric_features,
        color='income',
        title='Relationships Between Numeric Features',
        labels={col: col.replace('_', ' ').title() for col in numeric_features}
    )
    
    fig.update_layout(
        height=800,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def create_what_if_scenario_plot(optimizer, person_data, feature_name, value_range, step=1):
    """
    Create a what-if scenario plot for a specific feature.
    
    Parameters:
    -----------
    optimizer : CareerPathOptimizer
        Optimizer instance.
    person_data : dict
        Base person data.
    feature_name : str
        Feature to vary.
    value_range : list or range
        Range of values to simulate.
    step : int or float
        Step size for numeric features.
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive plot.
    """
    # Generate scenarios
    scenarios = []
    
    for value in value_range:
        # Create modified person data
        modified_data = person_data.copy()
        modified_data[feature_name] = value
        
        # Get probability
        prob = optimizer.get_income_probability(modified_data)
        
        scenarios.append({
            'Value': value,
            'Probability': prob
        })
    
    # Create DataFrame
    scenario_df = pd.DataFrame(scenarios)
    
    # Create line plot
    fig = px.line(
        scenario_df,
        x='Value',
        y='Probability',
        title=f'What-If Scenario: Impact of {feature_name} on Income Probability',
        labels={
            'Value': feature_name.replace('_', ' ').title(),
            'Probability': 'Probability of Income >$50K'
        },
        markers=True
    )
    
    fig.update_layout(
        yaxis=dict(
            tickformat='.0%',
            range=[0, 1]
        )
    )
    
    return fig


def create_policy_impact_simulation(df, policy_param, change_range):
    """
    Create a policy impact simulation plot.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset DataFrame.
    policy_param : str
        Policy parameter to simulate (e.g., 'education_num', 'hours_per_week').
    change_range : list
        Range of policy changes to simulate.
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive plot.
    """
    # Current baseline
    current_high_income_rate = (df['income'] == '>50K').mean() * 100
    
    # Simulate policy changes
    simulations = []
    
    for change in change_range:
        # Create modified DataFrame based on policy
        df_modified = df.copy()
        
        if policy_param == 'education_num':
            # Simulate increase in education level
            df_modified[policy_param] = df[policy_param].apply(
                lambda x: min(x + change, 16)  # Cap at maximum education level
            )
            
            # Estimate new income distribution
            # (Simplified: assume linear relationship between education and income)
            additional_high_income = 0.05 * change  # 5% increase per education level
            new_high_income_rate = min(current_high_income_rate * (1 + additional_high_income), 100)
            
        elif policy_param == 'hours_per_week':
            # Simulate change in working hours (e.g., minimum wage increase allowing fewer hours)
            df_modified[policy_param] = df[policy_param].apply(
                lambda x: max(x - change, 20)  # Minimum 20 hours
            )
            
            # Estimate new income distribution based on hours reduction
            # (Simplified: assume reduction in hours leads to some income improvement due to better wages)
            additional_high_income = 0.02 * change  # 2% increase per hour reduction
            new_high_income_rate = min(current_high_income_rate * (1 + additional_high_income), 100)
        
        else:
            # Default: no change
            new_high_income_rate = current_high_income_rate
        
        simulations.append({
            'Policy Change': change,
            'High Income Rate': new_high_income_rate
        })
    
    # Create DataFrame
    simulation_df = pd.DataFrame(simulations)
    
    # Create line plot
    fig = px.line(
        simulation_df,
        x='Policy Change',
        y='High Income Rate',
        title=f'Policy Impact Simulation: Effect of {policy_param.replace("_", " ").title()} Changes',
        labels={
            'Policy Change': f'Change in {policy_param.replace("_", " ").title()}',
            'High Income Rate': '% Population with Income >$50K'
        },
        markers=True
    )
    
    # Add baseline
    fig.add_hline(
        y=current_high_income_rate,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Current: {current_high_income_rate:.1f}%",
        annotation_position="bottom right"
    )
    
    return fig 