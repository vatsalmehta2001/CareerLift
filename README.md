# Career Path Optimizer

A data-driven tool to optimize career paths using the Census Income (adult.csv) dataset.

## Project Overview

This application analyzes how factors like education, occupation, hours worked, and demographic features relate to income improvement. It provides:

1. **Career Path Optimizer**: A recommendation engine that suggests the most efficient career, education, or work-hour changes to maximize the probability of moving to a higher income bracket (>$50K).

2. **Visualization Tool**: Interactive visualizations showing how different decisions impact income trajectories, with "what-if" scenario modeling capabilities.

3. **Policy Impact Simulator**: Simulations of the effects of changes in education funding or minimum wage policy on income distribution.

## Features

- **Income Prediction**: Machine learning models that predict income bracket based on various factors
- **Career Path Optimization**: Personalized recommendations for improving income potential  
- **ROI Calculations**: Time-to-ROI for education or job transitions
- **Interactive Visualizations**: "What-if" scenario modeling for users
- **Policy Simulation**: Impact analysis of policy changes on income distribution
- **Feature Importance Analysis**: Understand which factors most influence income

## Project Structure

```
CareerLift/
├── data/                      # Data files
│   └── adult.csv              # Census Income dataset
├── models/                    # Machine learning models
│   ├── income_predictor.py    # Income prediction model
│   └── feature_importance.py  # Feature importance analysis
├── optimizer/                 # Career path optimization components
│   ├── career_optimizer.py    # Core optimization logic
│   └── roi_calculator.py      # Return on investment calculations
├── visualization/             # Visualization components
│   ├── dashboard.py           # Dash/Plotly dashboard
│   └── interactive_plots.py   # Interactive visualization components
├── simulator/                 # Policy impact simulation
│   └── policy_simulator.py    # Policy impact simulation logic
├── utils/                     # Utility functions
│   ├── data_processor.py      # Data processing utilities
│   └── evaluation.py          # Model evaluation utilities
├── plots/                     # Saved visualizations
├── app.py                     # Main application entry point
├── data_exploration.py        # Initial data exploration
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages listed in `requirements.txt`

### Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   python app.py
   ```

### Usage Options

The application provides several modes that can be specified as command-line arguments:

```
python app.py --mode [mode] --data [data_path] --model [model_path]
```

Available modes:
- `train`: Train and evaluate machine learning models
- `analyze`: Analyze feature importance
- `optimize`: Run example career path optimization
- `dashboard`: Launch the interactive dashboard
- `all`: Run all components (default)

Example:
```
python app.py --mode dashboard --data data/adult.csv --model models/saved/random_forest
```

## Dataset

The adult.csv dataset (Census Income) contains individual records with features such as:
- Age, workclass, education, marital status, occupation
- Relationship, race, sex, capital gains/losses
- Hours worked per week, native country
- Income bracket (<=50K or >50K)

## Interactive Dashboard

The interactive dashboard allows users to:
1. Input their personal profile (age, education, occupation, etc.)
2. View their current probability of having income >$50K
3. See optimization suggestions for improving income potential
4. Analyze the ROI of different career path options
5. Explore the optimal career path with multiple steps

To launch the dashboard:
```
python app.py --mode dashboard
```

## Policy Impact Simulator

The policy simulator allows analysis of how changes in policy could affect income distribution:
1. Education policy: Increasing access to higher education
2. Minimum wage policy: Boosting income for certain workclasses
3. Targeted subsidies: Supporting specific demographic groups

To run an example policy simulation:
```
python -m simulator.policy_simulator
```

## Development

### Model Training and Evaluation

The application uses multiple machine learning models to predict income:
- Random Forest
- Gradient Boosting
- Logistic Regression
- Support Vector Machine

To train and evaluate models:
```
python app.py --mode train
```

### Feature Importance Analysis

To analyze which features have the greatest impact on income:
```
python app.py --mode analyze
```

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Census Income dataset provided by the UCI Machine Learning Repository 

## Limitations and Considerations

### Dataset Limitations

- **Historical Data**: The Adult Census Income dataset was extracted from the 1994 Census bureau database and is now significantly outdated (30+ years old). The economic landscape, job market, and income distributions have changed dramatically since then.

- **Income Threshold**: The income classification threshold (>$50K) has not been adjusted for inflation. $50,000 in 1994 would be equivalent to approximately $100,000 in today's dollars, meaning the income classification boundary is not consistent with current economic realities.

- **Occupational Evolution**: Many modern occupations (data scientist, mobile app developer, social media manager, etc.) did not exist when this data was collected. The occupation categories do not reflect the current job market.

- **Educational Landscape**: The education system and its relationship to career outcomes have evolved significantly. Online learning, coding bootcamps, and alternative credentials were not represented in the original dataset.

### Model Limitations

- **Correlation vs. Causation**: The models identify correlations between features and income but cannot establish causal relationships. For example, higher education correlating with higher income doesn't necessarily mean that education directly causes higher income.

- **Demographic Bias**: The historical dataset may contain biases related to gender, race, and other protected characteristics that were more pronounced in the past. Models trained on this data may perpetuate these biases.

- **Limited Features**: Important factors that influence income, such as regional cost of living, industry-specific demands, economic cycles, and personal networks, are not captured in the dataset.

- **Static Analysis**: The model provides a static snapshot rather than accounting for dynamic changes in the job market, economic conditions, and individual career trajectories over time.

### Application Context

- **Illustrative Purpose**: This application is intended as a portfolio demonstration and educational tool rather than a production career advisory system. Any career decisions should consider multiple sources of current information.

- **Simplified ROI Calculations**: The return on investment calculations use simplified assumptions about education costs, career transition timelines, and income trajectories that may not reflect individual circumstances or current economic conditions.

- **Policy Simulation Limitations**: The policy impact simulations are highly simplified models that do not account for the complex economic and social dynamics that would occur with actual policy changes.

### Future Improvements

For a production version of this system, several improvements would be necessary:

1. **Updated Dataset**: Replace with current census or labor statistics data that reflects the modern economy and job market.

2. **Enhanced Features**: Include additional factors such as geographic location, industry trends, skill-specific demand, and economic indicators.

3. **Personalized Learning**: Implement a system that learns from user feedback and outcomes to improve recommendations over time.

4. **Ethical AI Framework**: Develop robust fairness metrics and bias detection to ensure the system provides equitable recommendations across demographic groups.

5. **Integration with Current Resources**: Connect with job posting APIs, educational institution databases, and salary information services to provide real-time insights.

This portfolio project demonstrates technical capabilities in machine learning, data visualization, and application development while acknowledging the significant limitations of applying historical data to current career planning. 