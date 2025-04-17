import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import os
import sys

# Add the parent directory to the path if needed
if '.' not in sys.path:
    sys.path.append('.')

from models.income_predictor import IncomePredictor
from utils.data_processor import DataProcessor


class CareerPathOptimizer:
    """
    Class for optimizing career paths based on income prediction models.
    """
    
    def __init__(self, model_path='../models/saved/random_forest', data_path='../data/adult.csv'):
        """
        Initialize the CareerPathOptimizer.
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model.
        data_path : str
            Path to the data file.
        """
        self.model_path = model_path
        self.data_path = data_path
        self.predictor = None
        self.data_processor = None
        self.df = None
        self.feature_ranges = {}
        self.categorical_mapping = {}
        self.education_order = []
        self.occupation_stats = {}
        self._load_model_and_data()
        self._analyze_feature_ranges()
    
    def _load_model_and_data(self):
        """
        Load the income prediction model and dataset.
        """
        # Load predictor
        self.predictor = IncomePredictor.load_model(self.model_path)
        
        # Load and process data
        self.data_processor = DataProcessor(self.data_path)
        self.df = self.data_processor.load_data()
        self.data_processor.clean_data()
        self.data_processor.prepare_features()
        
        # Make sure the data processor's preprocessor is fitted
        self.data_processor.split_data()
        self.data_processor.preprocess_data(fit=True)
    
    def _analyze_feature_ranges(self):
        """
        Analyze the ranges and distributions of features in the dataset.
        """
        # Get numeric feature ranges
        for feature in self.data_processor.numeric_features:
            self.feature_ranges[feature] = {
                'min': self.df[feature].min(),
                'max': self.df[feature].max(),
                'mean': self.df[feature].mean(),
                'median': self.df[feature].median(),
                'quantiles': self.df[feature].quantile([0.25, 0.5, 0.75]).to_dict()
            }
        
        # Get categorical feature values
        for feature in self.data_processor.categorical_features:
            value_counts = self.df[feature].value_counts(normalize=True)
            self.categorical_mapping[feature] = value_counts.to_dict()
        
        # Create ordered list of education levels
        if 'education' in self.df.columns and 'education_num' in self.df.columns:
            education_mapping = self.df.groupby('education')['education_num'].mean().sort_values()
            self.education_order = education_mapping.index.tolist()
        
        # Calculate occupation statistics
        if 'occupation' in self.df.columns:
            self.occupation_stats = {}
            for occupation in self.df['occupation'].unique():
                occupation_data = self.df[self.df['occupation'] == occupation]
                high_income_rate = (occupation_data['income'] == 1).mean() * 100
                
                self.occupation_stats[occupation] = {
                    'count': len(occupation_data),
                    'high_income_rate': high_income_rate,
                    'avg_education': occupation_data['education_num'].mean() if 'education_num' in self.df.columns else None,
                    'avg_hours': occupation_data['hours_per_week'].mean() if 'hours_per_week' in self.df.columns else None,
                    'avg_age': occupation_data['age'].mean() if 'age' in self.df.columns else None
                }
    
    def get_income_probability(self, person_data):
        """
        Calculate the probability of high income for a given person.
        
        Parameters:
        -----------
        person_data : dict
            Dictionary containing the person's feature values.
            
        Returns:
        --------
        float
            Probability of high income (>50K).
        """
        # Preprocess person data
        X = self.data_processor.preprocess_single_sample(person_data)
        
        # Get probability
        proba = self.predictor.predict_proba(X)[0, 1]
        
        return proba
    
    def suggest_education_improvement(self, person_data, max_levels=3):
        """
        Suggest education improvements to increase income probability.
        
        Parameters:
        -----------
        person_data : dict
            Dictionary containing the person's feature values.
        max_levels : int
            Maximum number of education levels to suggest.
            
        Returns:
        --------
        list
            List of suggested education improvements with associated probabilities.
        """
        if 'education' not in person_data or not self.education_order:
            return []
        
        current_education = person_data['education']
        current_proba = self.get_income_probability(person_data)
        
        # Find current education level index
        if current_education in self.education_order:
            current_index = self.education_order.index(current_education)
        else:
            return []
        
        # Generate suggestions for higher education levels
        suggestions = []
        
        for i in range(current_index + 1, min(current_index + max_levels + 1, len(self.education_order))):
            new_education = self.education_order[i]
            
            # Create modified person data
            modified_data = copy.deepcopy(person_data)
            modified_data['education'] = new_education
            
            # Find corresponding education_num if available
            if 'education_num' in person_data:
                # Get average education_num for this education level
                edu_num = self.df[self.df['education'] == new_education]['education_num'].mean()
                modified_data['education_num'] = edu_num
            
            # Calculate new probability
            new_proba = self.get_income_probability(modified_data)
            
            # Calculate improvement
            improvement = (new_proba - current_proba) * 100
            
            suggestions.append({
                'education': new_education,
                'probability': new_proba,
                'improvement': improvement
            })
        
        # Sort by probability
        suggestions.sort(key=lambda x: x['probability'], reverse=True)
        
        return suggestions
    
    def suggest_occupation_change(self, person_data, top_n=5):
        """
        Suggest occupation changes to increase income probability.
        
        Parameters:
        -----------
        person_data : dict
            Dictionary containing the person's feature values.
        top_n : int
            Number of top suggestions to return.
            
        Returns:
        --------
        list
            List of suggested occupation changes with associated probabilities.
        """
        if 'occupation' not in person_data:
            return []
        
        current_occupation = person_data['occupation']
        current_proba = self.get_income_probability(person_data)
        
        # Generate suggestions for different occupations
        suggestions = []
        
        for occupation in self.df['occupation'].unique():
            if occupation == current_occupation or pd.isna(occupation):
                continue
            
            # Create modified person data
            modified_data = copy.deepcopy(person_data)
            modified_data['occupation'] = occupation
            
            # Calculate new probability
            new_proba = self.get_income_probability(modified_data)
            
            # Calculate improvement
            improvement = (new_proba - current_proba) * 100
            
            # Add occupation statistics
            stats = self.occupation_stats.get(occupation, {})
            
            suggestions.append({
                'occupation': occupation,
                'probability': new_proba,
                'improvement': improvement,
                'high_income_rate': stats.get('high_income_rate', None),
                'avg_education': stats.get('avg_education', None),
                'avg_hours': stats.get('avg_hours', None)
            })
        
        # Sort by probability and get top N
        suggestions.sort(key=lambda x: x['probability'], reverse=True)
        
        return suggestions[:top_n]
    
    def suggest_hours_adjustment(self, person_data, step=5, max_hours=60):
        """
        Suggest adjustments to hours worked per week to increase income probability.
        
        Parameters:
        -----------
        person_data : dict
            Dictionary containing the person's feature values.
        step : int
            Step size for hours adjustment.
        max_hours : int
            Maximum hours to consider.
            
        Returns:
        --------
        list
            List of suggested hours adjustments with associated probabilities.
        """
        if 'hours_per_week' not in person_data:
            return []
        
        current_hours = person_data['hours_per_week']
        current_proba = self.get_income_probability(person_data)
        
        # Generate suggestions for different hours
        suggestions = []
        
        for hours in range(current_hours + step, max_hours + 1, step):
            # Create modified person data
            modified_data = copy.deepcopy(person_data)
            modified_data['hours_per_week'] = hours
            
            # Calculate new probability
            new_proba = self.get_income_probability(modified_data)
            
            # Calculate improvement
            improvement = (new_proba - current_proba) * 100
            
            suggestions.append({
                'hours_per_week': hours,
                'probability': new_proba,
                'improvement': improvement
            })
        
        # Sort by probability
        suggestions.sort(key=lambda x: x['probability'], reverse=True)
        
        return suggestions
    
    def optimize_career_path(self, person_data, max_suggestions=3):
        """
        Generate comprehensive career path optimization suggestions.
        
        Parameters:
        -----------
        person_data : dict
            Dictionary containing the person's feature values.
        max_suggestions : int
            Maximum number of suggestions per category.
            
        Returns:
        --------
        dict
            Dictionary containing optimization suggestions.
        """
        current_proba = self.get_income_probability(person_data)
        
        results = {
            'current_profile': person_data,
            'current_probability': current_proba,
            'education_suggestions': self.suggest_education_improvement(person_data)[:max_suggestions],
            'occupation_suggestions': self.suggest_occupation_change(person_data, top_n=max_suggestions),
            'hours_suggestions': self.suggest_hours_adjustment(person_data)[:max_suggestions]
        }
        
        # Find combined optimizations (education + occupation)
        combined_suggestions = []
        
        if results['education_suggestions'] and results['occupation_suggestions']:
            for edu_sugg in results['education_suggestions'][:2]:
                for occ_sugg in results['occupation_suggestions'][:2]:
                    # Create combined modified data
                    modified_data = copy.deepcopy(person_data)
                    modified_data['education'] = edu_sugg['education']
                    if 'education_num' in person_data:
                        modified_data['education_num'] = self.df[self.df['education'] == edu_sugg['education']]['education_num'].mean()
                    modified_data['occupation'] = occ_sugg['occupation']
                    
                    # Calculate combined probability
                    combined_proba = self.get_income_probability(modified_data)
                    
                    # Calculate improvement
                    improvement = (combined_proba - current_proba) * 100
                    
                    combined_suggestions.append({
                        'education': edu_sugg['education'],
                        'occupation': occ_sugg['occupation'],
                        'probability': combined_proba,
                        'improvement': improvement
                    })
        
        # Sort combined suggestions by probability
        combined_suggestions.sort(key=lambda x: x['probability'], reverse=True)
        results['combined_suggestions'] = combined_suggestions[:max_suggestions]
        
        return results
    
    def plot_optimization_impact(self, optimization_results, save_path=None):
        """
        Visualize the impact of optimization suggestions.
        
        Parameters:
        -----------
        optimization_results : dict
            Results from optimize_career_path method.
        save_path : str
            Path to save the plot. If None, the plot is displayed.
        """
        current_prob = optimization_results['current_probability']
        
        # Collect all suggestions
        all_suggestions = []
        
        # Education suggestions
        for sugg in optimization_results['education_suggestions']:
            all_suggestions.append({
                'Type': 'Education',
                'Description': f"Education: {sugg['education']}",
                'Probability': sugg['probability'],
                'Improvement': sugg['improvement']
            })
        
        # Occupation suggestions
        for sugg in optimization_results['occupation_suggestions']:
            all_suggestions.append({
                'Type': 'Occupation',
                'Description': f"Occupation: {sugg['occupation']}",
                'Probability': sugg['probability'],
                'Improvement': sugg['improvement']
            })
        
        # Hours suggestions
        for sugg in optimization_results['hours_suggestions']:
            all_suggestions.append({
                'Type': 'Hours',
                'Description': f"Hours: {sugg['hours_per_week']}",
                'Probability': sugg['probability'],
                'Improvement': sugg['improvement']
            })
        
        # Combined suggestions
        for sugg in optimization_results['combined_suggestions']:
            all_suggestions.append({
                'Type': 'Combined',
                'Description': f"Edu: {sugg['education']} + Occ: {sugg['occupation']}",
                'Probability': sugg['probability'],
                'Improvement': sugg['improvement']
            })
        
        # Create DataFrame
        sugg_df = pd.DataFrame(all_suggestions)
        
        if not sugg_df.empty:
            # Sort by improvement
            sugg_df = sugg_df.sort_values('Improvement', ascending=False)
            
            # Create plot
            plt.figure(figsize=(12, 8))
            
            ax = sns.barplot(x='Improvement', y='Description', hue='Type', data=sugg_df)
            plt.title('Career Optimization Suggestions (% Improvement in High Income Probability)')
            plt.xlabel('% Improvement')
            plt.ylabel('')
            
            # Add current probability line
            plt.axvline(0, color='gray', linestyle='--')
            plt.text(1, -0.5, f'Current probability: {current_prob:.2%}', fontsize=10)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()
    
    def calculate_optimal_path(self, person_data, max_steps=3):
        """
        Calculate the optimal career path with multiple steps.
        
        Parameters:
        -----------
        person_data : dict
            Dictionary containing the person's feature values.
        max_steps : int
            Maximum number of steps in the career path.
            
        Returns:
        --------
        list
            List of steps in the optimal career path.
        """
        current_data = copy.deepcopy(person_data)
        current_proba = self.get_income_probability(current_data)
        
        path = [{
            'step': 0,
            'profile': current_data,
            'probability': current_proba,
            'action': 'Starting Point',
            'improvement': 0.0
        }]
        
        for step in range(1, max_steps + 1):
            # Get optimization suggestions
            results = self.optimize_career_path(current_data)
            
            # Find best suggestion across all categories
            all_suggestions = (
                [{'category': 'education', 'suggestion': s} for s in results['education_suggestions']] +
                [{'category': 'occupation', 'suggestion': s} for s in results['occupation_suggestions']] +
                [{'category': 'hours', 'suggestion': s} for s in results['hours_suggestions']] +
                [{'category': 'combined', 'suggestion': s} for s in results['combined_suggestions']]
            )
            
            if not all_suggestions:
                break
            
            # Find suggestion with highest probability
            best = max(all_suggestions, key=lambda x: x['suggestion']['probability'])
            
            # Update current data based on the best suggestion
            if best['category'] == 'education':
                current_data['education'] = best['suggestion']['education']
                if 'education_num' in current_data:
                    current_data['education_num'] = self.df[
                        self.df['education'] == best['suggestion']['education']
                    ]['education_num'].mean()
                action = f"Improve education to {best['suggestion']['education']}"
            
            elif best['category'] == 'occupation':
                current_data['occupation'] = best['suggestion']['occupation']
                action = f"Change occupation to {best['suggestion']['occupation']}"
            
            elif best['category'] == 'hours':
                current_data['hours_per_week'] = best['suggestion']['hours_per_week']
                action = f"Adjust hours to {best['suggestion']['hours_per_week']} per week"
            
            elif best['category'] == 'combined':
                current_data['education'] = best['suggestion']['education']
                if 'education_num' in current_data:
                    current_data['education_num'] = self.df[
                        self.df['education'] == best['suggestion']['education']
                    ]['education_num'].mean()
                current_data['occupation'] = best['suggestion']['occupation']
                action = f"Improve education to {best['suggestion']['education']} and change occupation to {best['suggestion']['occupation']}"
            
            # Calculate new probability
            new_proba = self.get_income_probability(current_data)
            improvement = best['suggestion']['improvement']
            
            # Add step to path
            path.append({
                'step': step,
                'profile': copy.deepcopy(current_data),
                'probability': new_proba,
                'action': action,
                'improvement': improvement
            })
            
            # If high income probability is already very high, stop
            if new_proba > 0.9:
                break
        
        return path


def example_optimization(model_path='../models/saved/random_forest', data_path='../data/adult.csv'):
    """
    Run an example career path optimization.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model.
    data_path : str
        Path to the data file.
    """
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
    
    # Get current probability
    current_proba = optimizer.get_income_probability(person_data)
    print(f"Current probability of income >50K: {current_proba:.2%}")
    
    # Get education suggestions
    print("\n=== Education Improvement Suggestions ===")
    edu_suggestions = optimizer.suggest_education_improvement(person_data)
    for i, sugg in enumerate(edu_suggestions, 1):
        print(f"{i}. Improve to {sugg['education']}: {sugg['probability']:.2%} probability (improvement: {sugg['improvement']:.2f}%)")
    
    # Get occupation suggestions
    print("\n=== Occupation Change Suggestions ===")
    occ_suggestions = optimizer.suggest_occupation_change(person_data)
    for i, sugg in enumerate(occ_suggestions, 1):
        print(f"{i}. Change to {sugg['occupation']}: {sugg['probability']:.2%} probability (improvement: {sugg['improvement']:.2f}%)")
    
    # Get hours suggestions
    print("\n=== Hours Adjustment Suggestions ===")
    hours_suggestions = optimizer.suggest_hours_adjustment(person_data)
    for i, sugg in enumerate(hours_suggestions, 1):
        print(f"{i}. Adjust to {sugg['hours_per_week']} hours/week: {sugg['probability']:.2%} probability (improvement: {sugg['improvement']:.2f}%)")
    
    # Get comprehensive optimization suggestions
    print("\n=== Comprehensive Career Path Optimization ===")
    results = optimizer.optimize_career_path(person_data)
    
    # Plot results
    optimizer.plot_optimization_impact(results, save_path='../plots/optimization_impact.png')
    print("Optimization impact plot saved to plots/optimization_impact.png")
    
    # Calculate optimal career path
    print("\n=== Optimal Career Path (3 Steps) ===")
    path = optimizer.calculate_optimal_path(person_data, max_steps=3)
    
    for step in path:
        if step['step'] == 0:
            print(f"Step {step['step']}: Starting point - Income >50K probability: {step['probability']:.2%}")
        else:
            print(f"Step {step['step']}: {step['action']} - Income >50K probability: {step['probability']:.2%} (improvement: {step['improvement']:.2f}%)")


if __name__ == "__main__":
    # Run example optimization
    example_optimization() 