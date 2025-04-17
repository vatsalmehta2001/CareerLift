import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


class DataProcessor:
    """
    Utility class for loading, cleaning, and preprocessing the adult.csv dataset.
    """
    
    def __init__(self, file_path='data/adult.csv'):
        """
        Initialize the DataProcessor.
        
        Parameters:
        -----------
        file_path : str
            Path to the adult.csv dataset.
        """
        self.file_path = file_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.preprocessor = None
        self.categorical_features = None
        self.numeric_features = None
        
    def load_data(self):
        """
        Load the dataset from the specified file path.
        
        Returns:
        --------
        pandas.DataFrame
            The loaded dataset.
        """
        try:
            # Try different approaches due to potential issues with file format
            try:
                # First attempt with standard CSV read
                self.df = pd.read_csv(self.file_path)
            except:
                try:
                    # Try with different encoding
                    self.df = pd.read_csv(self.file_path, encoding='latin1')
                except:
                    # Try with different delimiter and handling potential leading spaces
                    self.df = pd.read_csv(self.file_path, sep=',', skipinitialspace=True, encoding='latin1')
            
            print(f"Loaded data with columns: {self.df.columns.tolist()}")
            return self.df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def clean_data(self):
        """
        Clean the loaded dataset, handling missing values, column names, etc.
        
        Returns:
        --------
        pandas.DataFrame
            The cleaned dataset.
        """
        if self.df is None:
            self.load_data()
        
        # Clean column names (remove spaces, lowercase)
        self.df.columns = [col.strip().lower().replace('.', '_') for col in self.df.columns]
        print(f"Cleaned column names: {self.df.columns.tolist()}")
        
        # Handle missing values (in this dataset, missing values are marked as '?')
        self.df.replace('?', np.nan, inplace=True)
        
        # Handle whitespace in string columns
        for col in self.df.select_dtypes(include='object').columns:
            self.df[col] = self.df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
        
        # Handle missing values
        # For categorical features, replace with the most frequent value
        for col in self.df.select_dtypes(include='object').columns:
            if self.df[col].isnull().sum() > 0:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        
        # Convert target variable to binary
        if 'income' in self.df.columns:
            self.df['income'] = self.df['income'].map({'>50K': 1, '<=50K': 0})
        
        return self.df
    
    def prepare_features(self):
        """
        Identify categorical and numeric features for preprocessing.
        
        Returns:
        --------
        tuple
            Lists of categorical and numeric feature names.
        """
        if self.df is None or 'income' not in self.df.columns:
            self.clean_data()
        
        # Remove fnlwgt as it's a sampling weight and not relevant for prediction
        if 'fnlwgt' in self.df.columns:
            self.df.drop('fnlwgt', axis=1, inplace=True)
        
        # Identify categorical and numeric features
        self.categorical_features = self.df.select_dtypes(include=['object']).columns.tolist()
        if 'income' in self.categorical_features:
            self.categorical_features.remove('income')
        
        self.numeric_features = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if 'income' in self.numeric_features:
            self.numeric_features.remove('income')
        
        print(f"Categorical features: {self.categorical_features}")
        print(f"Numeric features: {self.numeric_features}")
        
        return self.categorical_features, self.numeric_features
    
    def create_preprocessor(self):
        """
        Create a sklearn ColumnTransformer for preprocessing the features.
        
        Returns:
        --------
        sklearn.compose.ColumnTransformer
            The preprocessor for transforming the features.
        """
        if self.categorical_features is None or self.numeric_features is None:
            self.prepare_features()
        
        # Create preprocessing steps for numeric and categorical features
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            verbose_feature_names_out=False
        )
        
        return self.preprocessor
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Split the data into training and testing sets.
        
        Parameters:
        -----------
        test_size : float
            Proportion of the dataset to include in the test split.
        random_state : int
            Random seed for reproducibility.
            
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        if self.df is None or 'income' not in self.df.columns:
            self.clean_data()
        
        X = self.df.drop('income', axis=1)
        y = self.df['income']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def preprocess_data(self, fit=True):
        """
        Preprocess the data using the created preprocessor.
        
        Parameters:
        -----------
        fit : bool
            Whether to fit the preprocessor on the training data or just transform.
            
        Returns:
        --------
        tuple
            (X_train_processed, X_test_processed, y_train, y_test)
        """
        if self.X_train is None or self.X_test is None:
            self.split_data()
        
        if self.preprocessor is None:
            self.create_preprocessor()
        
        if fit:
            X_train_processed = self.preprocessor.fit_transform(self.X_train)
        else:
            X_train_processed = self.preprocessor.transform(self.X_train)
        
        X_test_processed = self.preprocessor.transform(self.X_test)
        
        return X_train_processed, X_test_processed, self.y_train, self.y_test
    
    def get_feature_names(self):
        """
        Get the names of the preprocessed features.
        
        Returns:
        --------
        list
            List of preprocessed feature names.
        """
        if self.preprocessor is None:
            self.create_preprocessor()
            self.preprocess_data()
        
        try:
            # Use get_feature_names_out for sklearn >= 1.0
            feature_names = self.preprocessor.get_feature_names_out()
            print(f"Generated feature names using get_feature_names_out: {len(feature_names)} features")
            return feature_names.tolist()
        except AttributeError:
            # Fallback for older sklearn versions
            print("Using fallback method to get feature names")
            numeric_features_out = self.numeric_features
            
            # Get one-hot encoded feature names for categorical features
            categorical_features_out = []
            for i, col in enumerate(self.categorical_features):
                encoder = self.preprocessor.transformers_[1][1].named_steps['onehot']
                feature_names = encoder.get_feature_names_out([col])
                categorical_features_out.extend(feature_names)
            
            combined_features = numeric_features_out + list(categorical_features_out)
            print(f"Generated feature names using fallback: {len(combined_features)} features")
            return combined_features
    
    def preprocess_single_sample(self, sample_dict):
        """
        Preprocess a single sample for prediction.
        
        Parameters:
        -----------
        sample_dict : dict
            Dictionary containing feature values for the sample.
            
        Returns:
        --------
        numpy.ndarray
            Preprocessed feature vector for the sample.
        """
        if self.preprocessor is None:
            self.create_preprocessor()
            self.preprocess_data()
        
        # Convert dict to DataFrame
        sample_df = pd.DataFrame([sample_dict])
        
        # Ensure all required columns are present
        for col in self.categorical_features + self.numeric_features:
            if col not in sample_df.columns:
                sample_df[col] = np.nan
        
        # Apply preprocessing
        return self.preprocessor.transform(sample_df) 