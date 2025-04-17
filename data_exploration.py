import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_and_clean_data(file_path):
    """
    Load and clean the adult.csv dataset
    """
    # Try different approaches due to potential issues with file format
    try:
        # First attempt with standard CSV read
        df = pd.read_csv(file_path)
    except:
        try:
            # Try with different encoding
            df = pd.read_csv(file_path, encoding='latin1')
        except:
            # Try with different delimiter and handling potential leading spaces
            df = pd.read_csv(file_path, sep=',', skipinitialspace=True, encoding='latin1')
    
    # Clean column names (remove spaces, lowercase)
    df.columns = [col.strip().lower().replace('.', '_') for col in df.columns]
    
    # Handle missing values (in this dataset, missing values are marked as '?')
    df.replace('?', np.nan, inplace=True)
    
    # Handle whitespace in string columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
    
    return df

def explore_dataset(df):
    """
    Provide basic exploratory analysis of the dataset
    """
    # Basic info and stats
    print("Dataset Shape:", df.shape)
    print("\nDataset Information:")
    print(df.info())
    
    print("\nBasic Statistics:")
    print(df.describe(include='all').to_string())
    
    print("\nMissing Values Count:")
    print(df.isnull().sum())
    
    print("\nIncome Distribution:")
    print(df['income'].value_counts(normalize=True) * 100)
    
    # Save summary stats to a file
    with open('dataset_summary.txt', 'w') as f:
        f.write(f"Dataset Shape: {df.shape}\n\n")
        f.write("Column Names:\n")
        for col in df.columns:
            f.write(f"- {col}\n")
        
        f.write("\nMissing Values Count:\n")
        for col, count in df.isnull().sum().items():
            f.write(f"- {col}: {count} ({count/len(df)*100:.2f}%)\n")
        
        f.write("\nCategory Distributions:\n")
        for col in df.select_dtypes(include=['object']).columns:
            f.write(f"\n{col.upper()}:\n")
            value_counts = df[col].value_counts()
            for val, count in value_counts.items():
                f.write(f"- {val}: {count} ({count/len(df)*100:.2f}%)\n")

def plot_distributions(df):
    """
    Create and save distribution plots for key variables
    """
    # Create a directory for plots if it doesn't exist
    import os
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # Numeric features distributions
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    for feature in numeric_features:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[feature], kde=True)
        plt.title(f'Distribution of {feature}')
        plt.tight_layout()
        plt.savefig(f'plots/{feature}_distribution.png')
        plt.close()
    
    # Income distribution by education
    plt.figure(figsize=(12, 8))
    education_order = df.groupby('education')['education_num'].mean().sort_values().index
    sns.countplot(y='education', hue='income', data=df, order=education_order)
    plt.title('Income Distribution by Education Level')
    plt.tight_layout()
    plt.savefig('plots/income_by_education.png')
    plt.close()
    
    # Income distribution by occupation
    plt.figure(figsize=(14, 10))
    occupation_counts = df.groupby('occupation').size().sort_values(ascending=False)
    top_occupations = occupation_counts.head(10).index
    occupation_data = df[df['occupation'].isin(top_occupations)]
    sns.countplot(y='occupation', hue='income', data=occupation_data, order=occupation_counts.index[:10])
    plt.title('Income Distribution by Top 10 Occupations')
    plt.tight_layout()
    plt.savefig('plots/income_by_occupation.png')
    plt.close()
    
    # Income distribution by age groups
    plt.figure(figsize=(12, 7))
    df['age_group'] = pd.cut(df['age'], bins=[15, 25, 35, 45, 55, 65, 100], 
                             labels=['16-25', '26-35', '36-45', '46-55', '56-65', '65+'])
    sns.countplot(x='age_group', hue='income', data=df)
    plt.title('Income Distribution by Age Group')
    plt.tight_layout()
    plt.savefig('plots/income_by_age.png')
    plt.close()
    
    # Income distribution by sex
    plt.figure(figsize=(10, 6))
    sns.countplot(x='sex', hue='income', data=df)
    plt.title('Income Distribution by Sex')
    plt.tight_layout()
    plt.savefig('plots/income_by_sex.png')
    plt.close()
    
    # Hours per week distribution by income
    plt.figure(figsize=(12, 7))
    sns.boxplot(x='income', y='hours_per_week', data=df)
    plt.title('Hours Worked per Week by Income')
    plt.tight_layout()
    plt.savefig('plots/hours_by_income.png')
    plt.close()

if __name__ == "__main__":
    # Load and clean the data
    df = load_and_clean_data('adult.csv')
    
    # Explore the dataset
    explore_dataset(df)
    
    # Create and save distribution plots
    plot_distributions(df)
    
    print("Exploration complete. Check dataset_summary.txt and plots/ directory for results.") 