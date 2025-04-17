import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import sys

# Add the parent directory to the path if needed
if '.' not in sys.path:
    sys.path.append('.')

from models.income_predictor import IncomePredictor
from utils.data_processor import DataProcessor


class PolicyImpactSimulator:
    """
    Class for simulating the impact of policy changes on income distribution.
    """
    
    def __init__(self, model_path='../models/saved/random_forest', data_path='../data/adult.csv'):
        """
        Initialize the PolicyImpactSimulator.
        
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
        self._load_model_and_data()
    
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
    
    def simulate_education_policy(self, min_education_level=10, affected_population=0.5, confidence_interval=0.95):
        """
        Simulate the impact of a policy that increases access to education.
        
        Parameters:
        -----------
        min_education_level : int
            Minimum education level to achieve (in education_num).
        affected_population : float
            Proportion of the population affected by the policy (0 to 1).
        confidence_interval : float
            Confidence interval for the simulation (0 to 1).
            
        Returns:
        --------
        dict
            Dictionary containing simulation results.
        """
        if 'education_num' not in self.df.columns:
            raise ValueError("education_num column not found in dataset")
        
        # Get current income distribution
        current_high_income_rate = (self.df['income'] == 1).mean()
        
        # Create a copy of the dataset for simulation
        df_simulated = self.df.copy()
        
        # Identify the affected population (those with education_num below min_education_level)
        affected_mask = df_simulated['education_num'] < min_education_level
        affected_count = affected_mask.sum()
        
        # Calculate how many people will be affected based on affected_population parameter
        target_affected_count = int(affected_count * affected_population)
        
        # Get random sample of affected people
        if target_affected_count > 0:
            affected_indices = df_simulated[affected_mask].sample(
                n=min(target_affected_count, affected_count)
            ).index
            
            # Update education level for the affected population
            df_simulated.loc[affected_indices, 'education_num'] = min_education_level
            
            # Update education category based on education_num
            education_num_to_category = self.df.groupby('education')['education_num'].mean().to_dict()
            
            # Invert the mapping (education_num -> education)
            category_by_num = {}
            for edu, num in education_num_to_category.items():
                category_by_num[round(num)] = edu
            
            # Apply the mapping to update education category
            for idx in affected_indices:
                edu_num = df_simulated.loc[idx, 'education_num']
                closest_edu_num = min(category_by_num.keys(), key=lambda x: abs(x - edu_num))
                df_simulated.loc[idx, 'education'] = category_by_num[closest_edu_num]
        
        # Preprocess the simulated data
        X_simulated = df_simulated.drop('income', axis=1)
        X_simulated_processed = self.data_processor.preprocessor.transform(X_simulated)
        
        # Predict income for the simulated population
        y_simulated_proba = self.predictor.model.predict_proba(X_simulated_processed)[:, 1]
        y_simulated = (y_simulated_proba > 0.5).astype(int)
        
        # Calculate new income distribution
        new_high_income_rate = y_simulated.mean()
        
        # Calculate absolute and relative change
        absolute_change = new_high_income_rate - current_high_income_rate
        relative_change = absolute_change / current_high_income_rate * 100
        
        # Calculate confidence intervals through bootstrap
        n_bootstrap = 1000
        bootstrap_results = []
        
        for _ in range(n_bootstrap):
            # Sample with replacement
            boot_indices = np.random.choice(len(y_simulated), size=len(y_simulated), replace=True)
            boot_high_income_rate = y_simulated[boot_indices].mean()
            bootstrap_results.append(boot_high_income_rate)
        
        bootstrap_results = np.array(bootstrap_results)
        ci_lower = np.percentile(bootstrap_results, (1 - confidence_interval) / 2 * 100)
        ci_upper = np.percentile(bootstrap_results, (1 + confidence_interval) / 2 * 100)
        
        return {
            'current_high_income_rate': current_high_income_rate,
            'new_high_income_rate': new_high_income_rate,
            'absolute_change': absolute_change,
            'relative_change': relative_change,
            'affected_population_count': len(affected_indices),
            'affected_population_percentage': len(affected_indices) / len(df_simulated) * 100,
            'confidence_interval_lower': ci_lower,
            'confidence_interval_upper': ci_upper,
            'confidence_level': confidence_interval
        }
    
    def simulate_minimum_wage_policy(self, min_income_boost=0.1, affected_workclasses=['Private'], confidence_interval=0.95):
        """
        Simulate the impact of a minimum wage increase policy.
        
        Parameters:
        -----------
        min_income_boost : float
            Minimum boost to the probability of high income (0 to 1).
        affected_workclasses : list
            List of workclasses affected by the policy.
        confidence_interval : float
            Confidence interval for the simulation (0 to 1).
            
        Returns:
        --------
        dict
            Dictionary containing simulation results.
        """
        if 'workclass' not in self.df.columns:
            raise ValueError("workclass column not found in dataset")
        
        # Get current income distribution
        current_high_income_rate = (self.df['income'] == 1).mean()
        
        # Create a copy of the dataset for simulation
        df_simulated = self.df.copy()
        
        # Identify the affected population (those in the specified workclasses)
        affected_mask = df_simulated['workclass'].isin(affected_workclasses)
        affected_count = affected_mask.sum()
        
        # Get affected indices
        affected_indices = df_simulated[affected_mask].index
        
        if len(affected_indices) > 0:
            # Preprocess the affected population data
            X_affected = df_simulated.loc[affected_indices].drop('income', axis=1)
            X_affected_processed = self.data_processor.preprocessor.transform(X_affected)
            
            # Get current income probabilities
            current_proba = self.predictor.model.predict_proba(X_affected_processed)[:, 1]
            
            # Apply income boost (simplified model: increase probability directly)
            boosted_proba = np.minimum(current_proba + min_income_boost, 1.0)
            
            # Prepare full dataset for prediction
            X_simulated = df_simulated.drop('income', axis=1)
            X_simulated_processed = self.data_processor.preprocessor.transform(X_simulated)
            
            # Get baseline predictions for unaffected population
            y_simulated_proba = self.predictor.model.predict_proba(X_simulated_processed)[:, 1]
            
            # Apply boosted probabilities for affected population
            y_simulated_proba[affected_indices] = boosted_proba
            
            # Convert to binary predictions
            y_simulated = (y_simulated_proba > 0.5).astype(int)
            
            # Calculate new income distribution
            new_high_income_rate = y_simulated.mean()
            
            # Calculate absolute and relative change
            absolute_change = new_high_income_rate - current_high_income_rate
            relative_change = absolute_change / current_high_income_rate * 100
            
            # Calculate confidence intervals through bootstrap
            n_bootstrap = 1000
            bootstrap_results = []
            
            for _ in range(n_bootstrap):
                # Sample with replacement
                boot_indices = np.random.choice(len(y_simulated), size=len(y_simulated), replace=True)
                boot_high_income_rate = y_simulated[boot_indices].mean()
                bootstrap_results.append(boot_high_income_rate)
            
            bootstrap_results = np.array(bootstrap_results)
            ci_lower = np.percentile(bootstrap_results, (1 - confidence_interval) / 2 * 100)
            ci_upper = np.percentile(bootstrap_results, (1 + confidence_interval) / 2 * 100)
            
            return {
                'current_high_income_rate': current_high_income_rate,
                'new_high_income_rate': new_high_income_rate,
                'absolute_change': absolute_change,
                'relative_change': relative_change,
                'affected_population_count': affected_count,
                'affected_population_percentage': affected_count / len(df_simulated) * 100,
                'confidence_interval_lower': ci_lower,
                'confidence_interval_upper': ci_upper,
                'confidence_level': confidence_interval
            }
        else:
            return {
                'current_high_income_rate': current_high_income_rate,
                'new_high_income_rate': current_high_income_rate,
                'absolute_change': 0,
                'relative_change': 0,
                'affected_population_count': 0,
                'affected_population_percentage': 0,
                'confidence_interval_lower': current_high_income_rate,
                'confidence_interval_upper': current_high_income_rate,
                'confidence_level': confidence_interval
            }
    
    def simulate_education_subsidy_by_demographic(self, target_demographic, education_increase=2, confidence_interval=0.95):
        """
        Simulate the impact of targeted education subsidies for specific demographics.
        
        Parameters:
        -----------
        target_demographic : dict
            Dictionary specifying the target demographic (e.g., {'sex': 'Female', 'race': 'Black'}).
        education_increase : int
            Increase in education level (in education_num units).
        confidence_interval : float
            Confidence interval for the simulation (0 to 1).
            
        Returns:
        --------
        dict
            Dictionary containing simulation results.
        """
        # Get current income distribution
        current_high_income_rate = (self.df['income'] == 1).mean()
        
        # Create a copy of the dataset for simulation
        df_simulated = self.df.copy()
        
        # Create mask for the target demographic
        demographic_mask = pd.Series(True, index=df_simulated.index)
        for feature, value in target_demographic.items():
            if feature in df_simulated.columns:
                demographic_mask &= (df_simulated[feature] == value)
        
        # Identify the affected population
        affected_indices = df_simulated[demographic_mask].index
        affected_count = len(affected_indices)
        
        if affected_count > 0:
            # Update education level for the affected population
            df_simulated.loc[affected_indices, 'education_num'] = df_simulated.loc[
                affected_indices, 'education_num'
            ].apply(lambda x: min(x + education_increase, 16))  # Cap at maximum education level
            
            # Update education category based on education_num
            education_num_to_category = self.df.groupby('education')['education_num'].mean().to_dict()
            
            # Invert the mapping (education_num -> education)
            category_by_num = {}
            for edu, num in education_num_to_category.items():
                category_by_num[round(num)] = edu
            
            # Apply the mapping to update education category
            for idx in affected_indices:
                edu_num = df_simulated.loc[idx, 'education_num']
                closest_edu_num = min(category_by_num.keys(), key=lambda x: abs(x - edu_num))
                df_simulated.loc[idx, 'education'] = category_by_num[closest_edu_num]
            
            # Preprocess the simulated data
            X_simulated = df_simulated.drop('income', axis=1)
            X_simulated_processed = self.data_processor.preprocessor.transform(X_simulated)
            
            # Predict income for the simulated population
            y_simulated_proba = self.predictor.model.predict_proba(X_simulated_processed)[:, 1]
            y_simulated = (y_simulated_proba > 0.5).astype(int)
            
            # Calculate new income distribution
            new_high_income_rate = y_simulated.mean()
            
            # Calculate absolute and relative change
            absolute_change = new_high_income_rate - current_high_income_rate
            relative_change = absolute_change / current_high_income_rate * 100
            
            # Calculate targeted demographic impact
            current_demographic_high_income_rate = (self.df.loc[affected_indices, 'income'] == 1).mean()
            new_demographic_high_income_rate = y_simulated[affected_indices].mean()
            demographic_absolute_change = new_demographic_high_income_rate - current_demographic_high_income_rate
            demographic_relative_change = (
                demographic_absolute_change / current_demographic_high_income_rate * 100
                if current_demographic_high_income_rate > 0 else 0
            )
            
            # Calculate confidence intervals through bootstrap
            n_bootstrap = 1000
            bootstrap_results = []
            demographic_bootstrap_results = []
            
            for _ in range(n_bootstrap):
                # Sample with replacement
                boot_indices = np.random.choice(len(y_simulated), size=len(y_simulated), replace=True)
                boot_high_income_rate = y_simulated[boot_indices].mean()
                bootstrap_results.append(boot_high_income_rate)
                
                # Calculate demographic-specific bootstrap results
                demo_boot_indices = [i for i in boot_indices if i in affected_indices]
                if demo_boot_indices:
                    demo_boot_rate = y_simulated[demo_boot_indices].mean()
                    demographic_bootstrap_results.append(demo_boot_rate)
                else:
                    demographic_bootstrap_results.append(new_demographic_high_income_rate)
            
            bootstrap_results = np.array(bootstrap_results)
            demographic_bootstrap_results = np.array(demographic_bootstrap_results)
            
            ci_lower = np.percentile(bootstrap_results, (1 - confidence_interval) / 2 * 100)
            ci_upper = np.percentile(bootstrap_results, (1 + confidence_interval) / 2 * 100)
            
            demo_ci_lower = np.percentile(demographic_bootstrap_results, (1 - confidence_interval) / 2 * 100)
            demo_ci_upper = np.percentile(demographic_bootstrap_results, (1 + confidence_interval) / 2 * 100)
            
            return {
                'current_high_income_rate': current_high_income_rate,
                'new_high_income_rate': new_high_income_rate,
                'absolute_change': absolute_change,
                'relative_change': relative_change,
                'affected_population_count': affected_count,
                'affected_population_percentage': affected_count / len(df_simulated) * 100,
                'confidence_interval_lower': ci_lower,
                'confidence_interval_upper': ci_upper,
                'demographic_current_high_income_rate': current_demographic_high_income_rate,
                'demographic_new_high_income_rate': new_demographic_high_income_rate,
                'demographic_absolute_change': demographic_absolute_change,
                'demographic_relative_change': demographic_relative_change,
                'demographic_confidence_interval_lower': demo_ci_lower,
                'demographic_confidence_interval_upper': demo_ci_upper,
                'confidence_level': confidence_interval
            }
        else:
            return {
                'current_high_income_rate': current_high_income_rate,
                'new_high_income_rate': current_high_income_rate,
                'absolute_change': 0,
                'relative_change': 0,
                'affected_population_count': 0,
                'affected_population_percentage': 0,
                'confidence_interval_lower': current_high_income_rate,
                'confidence_interval_upper': current_high_income_rate,
                'demographic_current_high_income_rate': 0,
                'demographic_new_high_income_rate': 0,
                'demographic_absolute_change': 0,
                'demographic_relative_change': 0,
                'demographic_confidence_interval_lower': 0,
                'demographic_confidence_interval_upper': 0,
                'confidence_level': confidence_interval
            }
    
    def plot_education_policy_impact(self, min_education_levels=range(9, 16), affected_proportions=[0.3, 0.5, 0.7, 1.0], save_path=None):
        """
        Plot the impact of different education policy scenarios.
        
        Parameters:
        -----------
        min_education_levels : list or range
            Range of minimum education levels to simulate.
        affected_proportions : list
            List of affected population proportions to simulate.
        save_path : str
            Path to save the plot. If None, the plot is displayed.
        """
        # Simulate different policy scenarios
        results = []
        
        for prop in affected_proportions:
            for level in min_education_levels:
                simulation = self.simulate_education_policy(
                    min_education_level=level,
                    affected_population=prop
                )
                
                results.append({
                    'Min Education Level': level,
                    'Affected Proportion': prop,
                    'High Income Rate': simulation['new_high_income_rate'] * 100,
                    'Relative Change': simulation['relative_change']
                })
        
        # Create DataFrame
        results_df = pd.DataFrame(results)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        for prop in affected_proportions:
            prop_data = results_df[results_df['Affected Proportion'] == prop]
            plt.plot(
                prop_data['Min Education Level'],
                prop_data['High Income Rate'],
                marker='o',
                label=f'{prop:.0%} Affected'
            )
        
        # Add baseline
        baseline = (self.df['income'] == 1).mean() * 100
        plt.axhline(y=baseline, color='red', linestyle='--', label=f'Baseline ({baseline:.1f}%)')
        
        plt.title('Impact of Education Policy on High Income Rate')
        plt.xlabel('Minimum Education Level')
        plt.ylabel('Population with Income >$50K (%)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_minimum_wage_policy_impact(self, income_boosts=np.arange(0.05, 0.31, 0.05), affected_workclasses=[['Private'], ['Private', 'Local-gov'], ['Private', 'Local-gov', 'State-gov']], save_path=None):
        """
        Plot the impact of different minimum wage policy scenarios.
        
        Parameters:
        -----------
        income_boosts : list or array
            Range of income boost values to simulate.
        affected_workclasses : list of lists
            List of affected workclass combinations to simulate.
        save_path : str
            Path to save the plot. If None, the plot is displayed.
        """
        # Simulate different policy scenarios
        results = []
        
        for classes in affected_workclasses:
            classes_str = ', '.join(classes)
            for boost in income_boosts:
                simulation = self.simulate_minimum_wage_policy(
                    min_income_boost=boost,
                    affected_workclasses=classes
                )
                
                results.append({
                    'Income Boost': boost,
                    'Affected Workclasses': classes_str,
                    'High Income Rate': simulation['new_high_income_rate'] * 100,
                    'Relative Change': simulation['relative_change']
                })
        
        # Create DataFrame
        results_df = pd.DataFrame(results)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        for classes_str in results_df['Affected Workclasses'].unique():
            class_data = results_df[results_df['Affected Workclasses'] == classes_str]
            plt.plot(
                class_data['Income Boost'],
                class_data['High Income Rate'],
                marker='o',
                label=f'Affected: {classes_str}'
            )
        
        # Add baseline
        baseline = (self.df['income'] == 1).mean() * 100
        plt.axhline(y=baseline, color='red', linestyle='--', label=f'Baseline ({baseline:.1f}%)')
        
        plt.title('Impact of Minimum Wage Policy on High Income Rate')
        plt.xlabel('Income Probability Boost')
        plt.ylabel('Population with Income >$50K (%)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


def run_policy_simulations(model_path='../models/saved/random_forest', data_path='../data/adult.csv'):
    """
    Run policy impact simulations and generate reports.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model.
    data_path : str
        Path to the data file.
    """
    # Create simulator
    simulator = PolicyImpactSimulator(model_path, data_path)
    
    # Create output directory
    output_dir = '../plots/policy_simulation'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Simulate education policy
    print("\n=== Simulating Education Policy Impact ===")
    education_results = simulator.simulate_education_policy(
        min_education_level=12,  # High school diploma
        affected_population=0.5
    )
    
    print(f"Current high income rate: {education_results['current_high_income_rate']:.2%}")
    print(f"New high income rate: {education_results['new_high_income_rate']:.2%}")
    print(f"Absolute change: {education_results['absolute_change']:.2%}")
    print(f"Relative change: {education_results['relative_change']:.2f}%")
    print(f"Affected population: {education_results['affected_population_count']} people ({education_results['affected_population_percentage']:.2f}%)")
    print(f"95% Confidence interval: [{education_results['confidence_interval_lower']:.2%}, {education_results['confidence_interval_upper']:.2%}]")
    
    # Plot education policy impact
    print("\nPlotting education policy impact...")
    simulator.plot_education_policy_impact(
        min_education_levels=range(9, 16),
        affected_proportions=[0.3, 0.5, 0.7, 1.0],
        save_path=os.path.join(output_dir, 'education_policy_impact.png')
    )
    
    # Simulate minimum wage policy
    print("\n=== Simulating Minimum Wage Policy Impact ===")
    wage_results = simulator.simulate_minimum_wage_policy(
        min_income_boost=0.1,
        affected_workclasses=['Private']
    )
    
    print(f"Current high income rate: {wage_results['current_high_income_rate']:.2%}")
    print(f"New high income rate: {wage_results['new_high_income_rate']:.2%}")
    print(f"Absolute change: {wage_results['absolute_change']:.2%}")
    print(f"Relative change: {wage_results['relative_change']:.2f}%")
    print(f"Affected population: {wage_results['affected_population_count']} people ({wage_results['affected_population_percentage']:.2f}%)")
    print(f"95% Confidence interval: [{wage_results['confidence_interval_lower']:.2%}, {wage_results['confidence_interval_upper']:.2%}]")
    
    # Plot minimum wage policy impact
    print("\nPlotting minimum wage policy impact...")
    simulator.plot_minimum_wage_policy_impact(
        income_boosts=np.arange(0.05, 0.31, 0.05),
        affected_workclasses=[['Private'], ['Private', 'Local-gov'], ['Private', 'Local-gov', 'State-gov']],
        save_path=os.path.join(output_dir, 'minimum_wage_policy_impact.png')
    )
    
    # Simulate targeted education subsidy
    print("\n=== Simulating Targeted Education Subsidy Impact ===")
    
    # For female population
    female_results = simulator.simulate_education_subsidy_by_demographic(
        target_demographic={'sex': 'Female'},
        education_increase=2
    )
    
    print("\nTargeted Education Subsidy for Females:")
    print(f"Current high income rate: {female_results['current_high_income_rate']:.2%}")
    print(f"New high income rate: {female_results['new_high_income_rate']:.2%}")
    print(f"Absolute change: {female_results['absolute_change']:.2%}")
    print(f"Demographic current high income rate: {female_results['demographic_current_high_income_rate']:.2%}")
    print(f"Demographic new high income rate: {female_results['demographic_new_high_income_rate']:.2%}")
    print(f"Demographic absolute change: {female_results['demographic_absolute_change']:.2%}")
    print(f"Demographic relative change: {female_results['demographic_relative_change']:.2f}%")
    
    # For minority population (example with 'Black' race)
    if 'race' in simulator.df.columns and 'Black' in simulator.df['race'].unique():
        minority_results = simulator.simulate_education_subsidy_by_demographic(
            target_demographic={'race': 'Black'},
            education_increase=2
        )
        
        print("\nTargeted Education Subsidy for Black Population:")
        print(f"Current high income rate: {minority_results['current_high_income_rate']:.2%}")
        print(f"New high income rate: {minority_results['new_high_income_rate']:.2%}")
        print(f"Absolute change: {minority_results['absolute_change']:.2%}")
        print(f"Demographic current high income rate: {minority_results['demographic_current_high_income_rate']:.2%}")
        print(f"Demographic new high income rate: {minority_results['demographic_new_high_income_rate']:.2%}")
        print(f"Demographic absolute change: {minority_results['demographic_absolute_change']:.2%}")
        print(f"Demographic relative change: {minority_results['demographic_relative_change']:.2f}%")
    
    print("\nSimulations complete. Plots saved to", output_dir)


if __name__ == "__main__":
    # Run policy simulations
    run_policy_simulations() 