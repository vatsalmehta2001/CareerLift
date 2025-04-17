import os
import argparse
import warnings
warnings.filterwarnings('ignore')

from utils.data_processor import DataProcessor
from models.income_predictor import IncomePredictor, train_and_evaluate_models
from models.feature_importance import analyze_feature_importance
from optimizer.career_optimizer import CareerPathOptimizer
from optimizer.roi_calculator import ROICalculator
from visualization.dashboard import run_dashboard


def create_directories():
    """
    Create necessary directories for the project.
    """
    directories = [
        'data',
        'models/saved',
        'plots',
        'plots/model_evaluation',
        'plots/model_comparison',
        'plots/feature_importance',
        'plots/feature_importance/categorical',
        'plots/feature_importance/numeric'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


def train_models(data_path='data/adult.csv'):
    """
    Train and evaluate income prediction models.
    
    Parameters:
    -----------
    data_path : str
        Path to the data file.
    """
    print("\n=== Training and Evaluating Models ===")
    train_and_evaluate_models(file_path=data_path, output_dir='plots/model_comparison')


def analyze_features(model_path='models/saved/random_forest', data_path='data/adult.csv'):
    """
    Analyze feature importance for understanding income factors.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model.
    data_path : str
        Path to the data file.
    """
    print("\n=== Analyzing Feature Importance ===")
    analyze_feature_importance(model_path=model_path, file_path=data_path, output_dir='plots/feature_importance')


def optimize_career_example(model_path='models/saved/random_forest', data_path='data/adult.csv'):
    """
    Run an example career path optimization.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model.
    data_path : str
        Path to the data file.
    """
    print("\n=== Career Path Optimization Example ===")
    
    # Create optimizer
    optimizer = CareerPathOptimizer(model_path, data_path)
    
    # Example person data
    person_data = {
        'age': 35,
        'workclass': 'Private',
        'education': 'HS-grad',
        'education_num': 9,
        'marital_status': 'Married-civ-spouse',
        'occupation': 'Sales',
        'relationship': 'Husband',
        'race': 'White',
        'sex': 'Male',
        'capital_gain': 0,
        'capital_loss': 0,
        'hours_per_week': 40,
        'native_country': 'United-States'
    }
    
    # Get optimization suggestions
    results = optimizer.optimize_career_path(person_data)
    
    # Plot optimization impact
    optimizer.plot_optimization_impact(results, save_path='plots/optimization_impact.png')
    
    # Calculate optimal path
    path = optimizer.calculate_optimal_path(person_data, max_steps=3)
    
    print("Example completed. Optimization impact plot saved to plots/optimization_impact.png")
    
    # Calculate ROI
    print("\n=== ROI Calculation Example ===")
    roi_calculator = ROICalculator(optimizer)
    comprehensive_roi = roi_calculator.calculate_comprehensive_roi(person_data)
    roi_calculator.plot_roi_comparison(comprehensive_roi, save_path='plots/roi_comparison.png')
    
    print("ROI comparison plot saved to plots/roi_comparison.png")


def launch_dashboard(model_path='models/saved/random_forest', data_path='data/adult.csv'):
    """
    Launch the interactive dashboard.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model.
    data_path : str
        Path to the data file.
    """
    print("\n=== Launching Interactive Dashboard ===")
    print("Dashboard will be available at http://127.0.0.1:8050/ once loaded...")
    
    run_dashboard(model_path, data_path)


def main():
    """
    Main function to run the application.
    """
    parser = argparse.ArgumentParser(description='Career Path Optimizer')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['train', 'analyze', 'optimize', 'dashboard', 'all'],
                        help='Mode to run the application')
    parser.add_argument('--data', type=str, default='data/adult.csv',
                        help='Path to the data file')
    parser.add_argument('--model', type=str, default='models/saved/random_forest',
                        help='Path to the saved model')
    
    args = parser.parse_args()
    
    # Create directories
    create_directories()
    
    # Check if data file exists
    if not os.path.exists(args.data):
        print(f"Error: Data file not found at {args.data}")
        return
    
    # Run selected mode
    if args.mode == 'train' or args.mode == 'all':
        train_models(args.data)
    
    # Check if model exists for other modes
    if args.mode != 'train' and not os.path.exists(os.path.join(args.model, 'model_info.pickle')):
        if args.mode == 'all':
            print("Model not found. Training models first...")
            train_models(args.data)
        else:
            print(f"Error: Model not found at {args.model}")
            return
    
    if args.mode == 'analyze' or args.mode == 'all':
        analyze_features(args.model, args.data)
    
    if args.mode == 'optimize' or args.mode == 'all':
        optimize_career_example(args.model, args.data)
    
    if args.mode == 'dashboard' or args.mode == 'all':
        launch_dashboard(args.model, args.data)


if __name__ == "__main__":
    main() 