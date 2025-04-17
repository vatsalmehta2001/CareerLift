import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.inspection import permutation_importance

# Import from our utilities
import sys
# Add the parent directory to the path if needed
if '.' not in sys.path:
    sys.path.append('.')
from utils.data_processor import DataProcessor
from models.income_predictor import IncomePredictor


class FeatureImportanceAnalyzer:
    """
    Class for analyzing feature importance in income prediction models.
    """
    
    def __init__(self, model=None, data_processor=None):
        """
        Initialize the FeatureImportanceAnalyzer.
        
        Parameters:
        -----------
        model : object
            Trained machine learning model.
        data_processor : DataProcessor
            Data processor object with preprocessed data.
        """
        self.model = model
        self.data_processor = data_processor
        self.feature_names = None
        self.importance_values = None
    
    def set_model_and_processor(self, model, data_processor):
        """
        Set the model and data processor.
        
        Parameters:
        -----------
        model : object
            Trained machine learning model.
        data_processor : DataProcessor
            Data processor object with preprocessed data.
        """
        self.model = model
        self.data_processor = data_processor
    
    def get_feature_importance(self, method='built_in'):
        """
        Get feature importance values.
        
        Parameters:
        -----------
        method : str
            Method to use for feature importance calculation ('built_in', 'permutation').
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing feature importance values.
        """
        if self.model is None:
            raise ValueError("Model is not set. Use set_model_and_processor() first.")
        
        if self.data_processor is None:
            raise ValueError("Data processor is not set. Use set_model_and_processor() first.")
        
        # Get feature names
        if hasattr(self.data_processor, 'get_feature_names'):
            self.feature_names = self.data_processor.get_feature_names()
        else:
            raise ValueError("Data processor does not have a get_feature_names method.")
        
        # Calculate feature importance based on method
        if method == 'built_in':
            if hasattr(self.model, 'feature_importances_'):
                self.importance_values = self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                self.importance_values = np.abs(self.model.coef_[0])
            else:
                raise ValueError("Model does not have built-in feature importance attributes.")
        
        elif method == 'permutation':
            X_test = self.data_processor.X_test
            y_test = self.data_processor.y_test
            
            if self.data_processor.preprocessor is not None:
                X_test_processed = self.data_processor.preprocessor.transform(X_test)
            else:
                X_test_processed = X_test
            
            perm_importance = permutation_importance(
                self.model, X_test_processed, y_test, n_repeats=10, random_state=42
            )
            self.importance_values = perm_importance.importances_mean
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Create DataFrame with feature names and importance values
        feature_importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.importance_values
        })
        
        # Sort by importance
        feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
        
        return feature_importance_df
    
    def plot_feature_importance(self, top_n=20, figsize=(12, 10), save_path=None):
        """
        Plot feature importance.
        
        Parameters:
        -----------
        top_n : int
            Number of top features to plot.
        figsize : tuple
            Figure size.
        save_path : str
            Path to save the plot. If None, the plot is not saved.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing feature importance values.
        """
        if self.importance_values is None:
            feature_importance_df = self.get_feature_importance()
        else:
            feature_importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': self.importance_values
            }).sort_values('Importance', ascending=False).reset_index(drop=True)
        
        # Get top N features
        top_features = feature_importance_df.head(top_n)
        
        # Plot
        plt.figure(figsize=figsize)
        sns.barplot(x='Importance', y='Feature', data=top_features)
        plt.title(f'Top {top_n} Feature Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
        
        return feature_importance_df
    
    def analyze_categorical_importance(self, top_n=10, save_dir=None):
        """
        Analyze importance of categorical feature values.
        
        Parameters:
        -----------
        top_n : int
            Number of top features to analyze.
        save_dir : str
            Directory to save plots. If None, plots are not saved.
            
        Returns:
        --------
        dict
            Dictionary containing importance of categorical values.
        """
        if self.importance_values is None:
            feature_importance_df = self.get_feature_importance()
        else:
            feature_importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': self.importance_values
            }).sort_values('Importance', ascending=False).reset_index(drop=True)
        
        # Create save directory if specified
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Get categorical features from data processor
        categorical_features = self.data_processor.categorical_features
        
        # Dictionary to store category-level importance
        category_importance = {}
        
        # Analyze each categorical feature
        for feature in categorical_features:
            # Get one-hot encoded features for this category
            feature_cols = [col for col in feature_importance_df['Feature'] if col.startswith(f"{feature}_")]
            
            if not feature_cols:
                continue
            
            # Get importance values for these features
            category_values = {}
            for col in feature_cols:
                # Extract category value from column name (e.g., 'workclass_Private' -> 'Private')
                category = col.replace(f"{feature}_", '')
                importance = feature_importance_df.loc[feature_importance_df['Feature'] == col, 'Importance'].values[0]
                category_values[category] = importance
            
            # Sort by importance
            category_values = {k: v for k, v in sorted(category_values.items(), key=lambda item: item[1], reverse=True)}
            category_importance[feature] = category_values
            
            # Plot top categories for this feature
            plt.figure(figsize=(10, 6))
            categories = list(category_values.keys())[:top_n]
            importances = list(category_values.values())[:top_n]
            
            sns.barplot(x=importances, y=categories)
            plt.title(f'Top {top_n} Important Values for {feature}')
            plt.tight_layout()
            
            if save_dir:
                plt.savefig(os.path.join(save_dir, f'{feature}_importance.png'))
                plt.close()
            else:
                plt.show()
        
        return category_importance
    
    def analyze_numeric_feature_impact(self, save_dir=None):
        """
        Analyze the impact of numeric features on income.
        
        Parameters:
        -----------
        save_dir : str
            Directory to save plots. If None, plots are not saved.
            
        Returns:
        --------
        dict
            Dictionary containing impact analysis for numeric features.
        """
        if self.data_processor is None:
            raise ValueError("Data processor is not set. Use set_model_and_processor() first.")
        
        # Get numeric features
        numeric_features = self.data_processor.numeric_features
        
        # Get original data
        df = self.data_processor.df
        
        # Create save directory if specified
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Dictionary to store impact analysis
        numeric_impact = {}
        
        # Analyze each numeric feature
        for feature in numeric_features:
            plt.figure(figsize=(10, 6))
            
            # Group data by income and calculate mean
            means = df.groupby('income')[feature].mean()
            impact = (means[1] - means[0]) / means[0] * 100 if means[0] != 0 else float('inf')
            
            # Plot boxplot
            sns.boxplot(x='income', y=feature, data=df)
            plt.title(f'Impact of {feature} on Income\nMean difference: {impact:.2f}%')
            plt.tight_layout()
            
            if save_dir:
                plt.savefig(os.path.join(save_dir, f'{feature}_impact.png'))
                plt.close()
            else:
                plt.show()
            
            # Store impact values
            numeric_impact[feature] = {
                'mean_low_income': means[0],
                'mean_high_income': means[1],
                'percent_difference': impact
            }
        
        return numeric_impact


def analyze_feature_importance(model_path='models/saved/random_forest',
                             file_path='data/adult.csv',
                             output_dir='plots/feature_importance'):
    """
    Analyze feature importance for a trained model.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model.
    file_path : str
        Path to the data file.
    output_dir : str
        Directory to save output files and plots.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load model
    predictor = IncomePredictor.load_model(model_path)
    
    # Create data processor and load data
    data_processor = DataProcessor(file_path)
    data_processor.load_data()
    data_processor.clean_data()
    data_processor.prepare_features()
    X_train, X_test, y_train, y_test = data_processor.preprocess_data()
    
    # Create feature importance analyzer
    analyzer = FeatureImportanceAnalyzer(predictor.model, data_processor)
    
    # Get and plot overall feature importance
    print("Analyzing overall feature importance...")
    importance_df = analyzer.plot_feature_importance(
        save_path=os.path.join(output_dir, 'overall_feature_importance.png')
    )
    importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
    
    # Analyze categorical feature importance
    print("Analyzing categorical feature importance...")
    category_importance = analyzer.analyze_categorical_importance(
        save_dir=os.path.join(output_dir, 'categorical')
    )
    
    # Analyze numeric feature impact
    print("Analyzing numeric feature impact...")
    numeric_impact = analyzer.analyze_numeric_feature_impact(
        save_dir=os.path.join(output_dir, 'numeric')
    )
    
    # Save numeric impact to CSV
    numeric_impact_df = pd.DataFrame.from_dict(numeric_impact, orient='index')
    numeric_impact_df.to_csv(os.path.join(output_dir, 'numeric_feature_impact.csv'))
    
    # Print summary
    print("\n===== Feature Importance Summary =====")
    print("Top 10 most important features:")
    print(importance_df.head(10))
    
    print("\nNumeric feature impact:")
    for feature, impact in numeric_impact.items():
        print(f"{feature}: {impact['percent_difference']:.2f}% difference between income groups")
    
    return importance_df, category_importance, numeric_impact


if __name__ == "__main__":
    # Analyze feature importance
    analyze_feature_importance() 