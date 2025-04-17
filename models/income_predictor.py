import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib

# Import from our utilities
import sys
# Add the parent directory to the path if needed
if '.' not in sys.path:
    sys.path.append('.')
from utils.data_processor import DataProcessor
from utils.evaluation import ModelEvaluator


class IncomePredictor:
    """
    Class for predicting income bracket (<=50K or >50K) using various ML models.
    """
    
    def __init__(self, model_type='random_forest', model_params=None):
        """
        Initialize the IncomePredictor.
        
        Parameters:
        -----------
        model_type : str
            Type of model to use ('random_forest', 'gradient_boosting', 'logistic_regression', 'svc').
        model_params : dict
            Parameters for the model.
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.model = None
        self.data_processor = None
        self.evaluator = None
        self._initialize_model()
        self._initialize_evaluator()
    
    def _initialize_model(self):
        """
        Initialize the model based on the model_type.
        """
        if self.model_type == 'random_forest':
            default_params = {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            }
            params = {**default_params, **self.model_params}
            self.model = RandomForestClassifier(**params)
        
        elif self.model_type == 'gradient_boosting':
            default_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'random_state': 42
            }
            params = {**default_params, **self.model_params}
            self.model = GradientBoostingClassifier(**params)
        
        elif self.model_type == 'logistic_regression':
            default_params = {
                'C': 1.0,
                'penalty': 'l2',
                'solver': 'liblinear',
                'random_state': 42
            }
            params = {**default_params, **self.model_params}
            self.model = LogisticRegression(**params)
        
        elif self.model_type == 'svc':
            default_params = {
                'C': 1.0,
                'kernel': 'rbf',
                'probability': True,
                'random_state': 42
            }
            params = {**default_params, **self.model_params}
            self.model = SVC(**params)
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _initialize_evaluator(self):
        """
        Initialize the model evaluator.
        """
        self.evaluator = ModelEvaluator(self.model)
    
    def prepare_data(self, file_path='data/adult.csv'):
        """
        Prepare the data for model training and evaluation.
        
        Parameters:
        -----------
        file_path : str
            Path to the data file.
            
        Returns:
        --------
        tuple
            (X_train_processed, X_test_processed, y_train, y_test)
        """
        self.data_processor = DataProcessor(file_path)
        self.data_processor.load_data()
        self.data_processor.clean_data()
        self.data_processor.prepare_features()
        self.data_processor.split_data()
        X_train, X_test, y_train, y_test = self.data_processor.preprocess_data()
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train):
        """
        Train the model.
        
        Parameters:
        -----------
        X_train : array-like
            Training features.
        y_train : array-like
            Training target.
            
        Returns:
        --------
        object
            Trained model.
        """
        self.model.fit(X_train, y_train)
        return self.model
    
    def evaluate(self, X_test, y_test, output_dir='plots/model_evaluation'):
        """
        Evaluate the model.
        
        Parameters:
        -----------
        X_test : array-like
            Test features.
        y_test : array-like
            Test target.
        output_dir : str
            Directory to save evaluation plots.
            
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics.
        """
        self.evaluator.set_model(self.model)
        metrics = self.evaluator.evaluate_and_visualize(X_test, y_test, output_dir)
        return metrics
    
    def predict(self, X):
        """
        Make predictions.
        
        Parameters:
        -----------
        X : array-like
            Features to make predictions on.
            
        Returns:
        --------
        array-like
            Predicted classes.
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Make probability predictions.
        
        Parameters:
        -----------
        X : array-like
            Features to make predictions on.
            
        Returns:
        --------
        array-like
            Predicted probabilities for each class.
        """
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise ValueError(f"Model {self.model_type} does not support probability predictions.")
    
    def tune_hyperparameters(self, X_train, y_train, param_grid=None, n_iter=10, cv=5, 
                            scoring='roc_auc', random=True):
        """
        Tune hyperparameters using grid search or randomized search.
        
        Parameters:
        -----------
        X_train : array-like
            Training features.
        y_train : array-like
            Training target.
        param_grid : dict
            Dictionary of parameters to search.
        n_iter : int
            Number of iterations for randomized search.
        cv : int
            Number of cross-validation folds.
        scoring : str
            Scoring metric.
        random : bool
            Whether to use randomized search or grid search.
            
        Returns:
        --------
        object
            Best model found.
        """
        if param_grid is None:
            if self.model_type == 'random_forest':
                param_grid = {
                    'n_estimators': [50, 100, 150, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            elif self.model_type == 'gradient_boosting':
                param_grid = {
                    'n_estimators': [50, 100, 150, 200],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7, 9]
                }
            elif self.model_type == 'logistic_regression':
                param_grid = {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }
            elif self.model_type == 'svc':
                param_grid = {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': ['scale', 'auto', 0.1, 0.01]
                }
        
        if random:
            search = RandomizedSearchCV(
                self.model, param_grid, n_iter=n_iter, cv=cv, 
                scoring=scoring, n_jobs=-1, verbose=1, random_state=42
            )
        else:
            search = GridSearchCV(
                self.model, param_grid, cv=cv, 
                scoring=scoring, n_jobs=-1, verbose=1
            )
        
        search.fit(X_train, y_train)
        
        print(f"Best parameters: {search.best_params_}")
        print(f"Best {scoring} score: {search.best_score_:.4f}")
        
        self.model = search.best_estimator_
        self._initialize_evaluator()
        
        return self.model
    
    def save_model(self, model_path='models/saved'):
        """
        Save the trained model and preprocessor.
        
        Parameters:
        -----------
        model_path : str
            Directory to save the model.
        """
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        # Save the model
        model_file = os.path.join(model_path, f"{self.model_type}_model.joblib")
        joblib.dump(self.model, model_file)
        
        # Save the preprocessor if available
        if self.data_processor and self.data_processor.preprocessor:
            preprocessor_file = os.path.join(model_path, "preprocessor.joblib")
            joblib.dump(self.data_processor.preprocessor, preprocessor_file)
        
        # Save model info
        model_info = {
            'model_type': self.model_type,
            'model_params': self.model.get_params(),
            'feature_names': self.data_processor.get_feature_names() if self.data_processor else None
        }
        
        info_file = os.path.join(model_path, "model_info.pickle")
        with open(info_file, 'wb') as f:
            pickle.dump(model_info, f)
        
        print(f"Model saved to {model_file}")
        
    @classmethod
    def load_model(cls, model_path='models/saved'):
        """
        Load a saved model and preprocessor.
        
        Parameters:
        -----------
        model_path : str
            Directory where the model is saved.
            
        Returns:
        --------
        IncomePredictor
            IncomePredictor instance with loaded model.
        """
        # Load model info
        info_file = os.path.join(model_path, "model_info.pickle")
        with open(info_file, 'rb') as f:
            model_info = pickle.load(f)
        
        # Create instance
        instance = cls(model_type=model_info['model_type'])
        
        # Load model
        model_file = os.path.join(model_path, f"{model_info['model_type']}_model.joblib")
        instance.model = joblib.load(model_file)
        
        # Initialize evaluator
        instance._initialize_evaluator()
        
        # Load preprocessor if available
        preprocessor_file = os.path.join(model_path, "preprocessor.joblib")
        if os.path.exists(preprocessor_file):
            instance.data_processor = DataProcessor()
            instance.data_processor.preprocessor = joblib.load(preprocessor_file)
        
        return instance


def train_and_evaluate_models(file_path='data/adult.csv', output_dir='plots/model_comparison'):
    """
    Train and evaluate multiple models for comparison.
    
    Parameters:
    -----------
    file_path : str
        Path to the data file.
    output_dir : str
        Directory to save evaluation results.
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics for each model.
    """
    # Prepare data
    data_processor = DataProcessor(file_path)
    data_processor.load_data()
    data_processor.clean_data()
    data_processor.prepare_features()
    X_train, X_test, y_train, y_test = data_processor.preprocess_data()
    
    # Define models to train
    models = [
        ('random_forest', {}),
        ('gradient_boosting', {}),
        ('logistic_regression', {}),
        ('svc', {'C': 1.0, 'kernel': 'linear'})
    ]
    
    # Train and evaluate each model
    results = {}
    
    for model_type, model_params in models:
        print(f"\n===== Training {model_type} =====")
        
        # Create predictor
        predictor = IncomePredictor(model_type=model_type, model_params=model_params)
        predictor.data_processor = data_processor
        
        # Train model
        predictor.train(X_train, y_train)
        
        # Create model-specific output directory
        model_output_dir = os.path.join(output_dir, model_type)
        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)
        
        # Evaluate model
        metrics = predictor.evaluate(X_test, y_test, model_output_dir)
        results[model_type] = metrics
        
        # Save model
        predictor.save_model(os.path.join('models/saved', model_type))
    
    # Print comparison
    print("\n===== Model Comparison =====")
    comparison_df = pd.DataFrame(results).T
    print(comparison_df)
    
    # Save comparison results
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    comparison_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'))
    
    return results


if __name__ == "__main__":
    # Train and evaluate models
    train_and_evaluate_models() 