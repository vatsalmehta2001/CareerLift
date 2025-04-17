import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import copy

# Add the parent directory to the path if needed
if '.' not in sys.path:
    sys.path.append('.')

from utils.data_processor import DataProcessor
from optimizer.career_optimizer import CareerPathOptimizer


class ROICalculator:
    """
    Class for calculating return on investment (ROI) for career path changes.
    """
    
    def __init__(self, optimizer=None, model_path='../models/saved/random_forest', data_path='../data/adult.csv'):
        """
        Initialize the ROICalculator.
        
        Parameters:
        -----------
        optimizer : CareerPathOptimizer
            Career path optimizer instance.
        model_path : str
            Path to the saved model.
        data_path : str
            Path to the data file.
        """
        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = CareerPathOptimizer(model_path, data_path)
        
        self.education_costs = self._initialize_education_costs()
        self.education_durations = self._initialize_education_durations()
        self.occupation_transitions = self._initialize_occupation_transitions()
        self.income_estimates = self._initialize_income_estimates()
    
    def _initialize_education_costs(self):
        """
        Initialize education costs based on typical US education costs.
        
        Returns:
        --------
        dict
            Dictionary mapping education level to estimated cost.
        """
        # Estimated total costs based on 2021-2022 data (approximations)
        return {
            'HS-grad': 0,  # Already completed
            'Some-college': 20000,  # 2 years of community college
            'Assoc-voc': 25000,  # Associate's degree (vocational)
            'Assoc-acdm': 25000,  # Associate's degree (academic)
            'Bachelors': 120000,  # 4-year Bachelor's degree
            'Masters': 60000,  # 2-year Master's degree (on top of Bachelor's)
            'Prof-school': 150000,  # Professional school (law, medical, etc.)
            'Doctorate': 120000  # PhD program
        }
    
    def _initialize_education_durations(self):
        """
        Initialize education durations in years.
        
        Returns:
        --------
        dict
            Dictionary mapping education level to estimated duration in years.
        """
        return {
            'HS-grad': 0,  # Already completed
            'Some-college': 2,  # 2 years of community college
            'Assoc-voc': 2,  # Associate's degree (vocational)
            'Assoc-acdm': 2,  # Associate's degree (academic)
            'Bachelors': 4,  # 4-year Bachelor's degree
            'Masters': 2,  # 2-year Master's degree (on top of Bachelor's)
            'Prof-school': 3,  # Professional school (law, medical, etc.)
            'Doctorate': 5  # PhD program
        }
    
    def _initialize_occupation_transitions(self):
        """
        Initialize occupation transition difficulty scores.
        Higher score means more difficult transition.
        
        Returns:
        --------
        dict
            Dictionary mapping occupation pairs to transition difficulty scores.
        """
        # Default difficulty score for all transitions
        default_score = 5  # Medium difficulty
        
        # Define some specific transition difficulties
        transitions = {}
        
        # Get all occupations from optimizer
        occupations = list(self.optimizer.occupation_stats.keys())
        
        # Initialize all transitions with default score
        for source in occupations:
            for target in occupations:
                if source != target:
                    transitions[(source, target)] = default_score
        
        # Define some easier transitions (similar skills/domains)
        easier_transitions = [
            ('Sales', 'Marketing'),
            ('Tech-support', 'Adm-clerical'),
            ('Craft-repair', 'Machine-op-inspct'),
            ('Exec-managerial', 'Prof-specialty'),
            ('Prof-specialty', 'Exec-managerial'),
            ('Adm-clerical', 'Sales'),
            ('Sales', 'Exec-managerial'),
            ('Protective-serv', 'Armed-Forces'),
            ('Armed-Forces', 'Protective-serv')
        ]
        
        # Define some harder transitions (very different skills/domains)
        harder_transitions = [
            ('Farming-fishing', 'Prof-specialty'),
            ('Priv-house-serv', 'Exec-managerial'),
            ('Handlers-cleaners', 'Prof-specialty'),
            ('Machine-op-inspct', 'Prof-specialty'),
            ('Armed-Forces', 'Farming-fishing')
        ]
        
        # Apply specific scores
        for source, target in easier_transitions:
            if (source, target) in transitions:
                transitions[(source, target)] = 3  # Easier transition
        
        for source, target in harder_transitions:
            if (source, target) in transitions:
                transitions[(source, target)] = 8  # Harder transition
        
        return transitions
    
    def _initialize_income_estimates(self):
        """
        Initialize income estimates based on the dataset.
        
        Returns:
        --------
        dict
            Dictionary with income estimates.
        """
        income_estimates = {}
        
        df = self.optimizer.df
        
        # Calculate average income by occupation and education
        for occupation in df['occupation'].unique():
            if pd.isna(occupation):
                continue
                
            # Get occupation data
            occ_data = df[df['occupation'] == occupation]
            
            # Calculate average income for low and high income groups
            low_income = 35000  # Default estimate for <=50K
            high_income = 75000  # Default estimate for >50K
            
            # Calculate proportion of high income in this occupation
            high_income_rate = (occ_data['income'] == 1).mean()
            
            # Store estimates
            income_estimates[occupation] = {
                'low_income': low_income,
                'high_income': high_income,
                'high_income_rate': high_income_rate,
                'expected_income': low_income * (1 - high_income_rate) + high_income * high_income_rate
            }
            
            # Calculate by education level if available
            if 'education' in df.columns:
                education_incomes = {}
                
                for education in df['education'].unique():
                    edu_occ_data = occ_data[occ_data['education'] == education]
                    
                    if len(edu_occ_data) > 0:
                        edu_high_income_rate = (edu_occ_data['income'] == 1).mean()
                        edu_expected_income = low_income * (1 - edu_high_income_rate) + high_income * edu_high_income_rate
                        
                        education_incomes[education] = {
                            'high_income_rate': edu_high_income_rate,
                            'expected_income': edu_expected_income
                        }
                
                income_estimates[occupation]['education_incomes'] = education_incomes
        
        return income_estimates
    
    def calculate_education_roi(self, person_data, suggested_education):
        """
        Calculate ROI for education improvement.
        
        Parameters:
        -----------
        person_data : dict
            Dictionary containing the person's feature values.
        suggested_education : str
            Suggested education level.
            
        Returns:
        --------
        dict
            Dictionary containing ROI calculations.
        """
        if 'education' not in person_data or suggested_education not in self.education_costs:
            return None
        
        current_education = person_data['education']
        current_occupation = person_data['occupation']
        
        # Calculate costs and time
        cost = self.education_costs.get(suggested_education, 0)
        time_years = self.education_durations.get(suggested_education, 0)
        
        # Calculate current and projected income
        current_income = self.income_estimates.get(current_occupation, {}).get('expected_income', 35000)
        
        # Create modified person data
        modified_data = copy.deepcopy(person_data)
        modified_data['education'] = suggested_education
        
        # Get corresponding education_num if available
        if 'education_num' in person_data:
            # Get average education_num for this education level
            edu_num = self.optimizer.df[self.optimizer.df['education'] == suggested_education]['education_num'].mean()
            modified_data['education_num'] = edu_num
        
        # Get new income probability
        new_proba = self.optimizer.get_income_probability(modified_data)
        
        # Estimate new income based on probability
        occupation_income = self.income_estimates.get(current_occupation, {})
        low_income = occupation_income.get('low_income', 35000)
        high_income = occupation_income.get('high_income', 75000)
        
        # Account for education-specific income if available
        edu_income = occupation_income.get('education_incomes', {}).get(suggested_education, {})
        if edu_income:
            new_expected_income = edu_income.get('expected_income')
        else:
            new_expected_income = low_income * (1 - new_proba) + high_income * new_proba
        
        # Calculate income improvement
        income_improvement = new_expected_income - current_income
        
        # Calculate opportunity cost during education
        opportunity_cost = current_income * time_years
        
        # Calculate total cost (direct cost + opportunity cost)
        total_cost = cost + opportunity_cost
        
        # Calculate simple payback period (years)
        if income_improvement > 0:
            payback_years = total_cost / income_improvement
        else:
            payback_years = float('inf')
        
        # Calculate ROI (5-year and 10-year)
        income_gain_5yr = income_improvement * (5 - time_years) if (5 - time_years) > 0 else 0
        income_gain_10yr = income_improvement * (10 - time_years) if (10 - time_years) > 0 else 0
        
        roi_5yr = (income_gain_5yr - total_cost) / total_cost * 100 if total_cost > 0 else 0
        roi_10yr = (income_gain_10yr - total_cost) / total_cost * 100 if total_cost > 0 else 0
        
        return {
            'current_education': current_education,
            'suggested_education': suggested_education,
            'direct_cost': cost,
            'time_years': time_years,
            'opportunity_cost': opportunity_cost,
            'total_cost': total_cost,
            'current_income': current_income,
            'new_expected_income': new_expected_income,
            'annual_income_improvement': income_improvement,
            'payback_years': payback_years,
            'roi_5yr': roi_5yr,
            'roi_10yr': roi_10yr
        }
    
    def calculate_occupation_roi(self, person_data, suggested_occupation):
        """
        Calculate ROI for occupation change.
        
        Parameters:
        -----------
        person_data : dict
            Dictionary containing the person's feature values.
        suggested_occupation : str
            Suggested occupation.
            
        Returns:
        --------
        dict
            Dictionary containing ROI calculations.
        """
        if 'occupation' not in person_data:
            return None
        
        current_occupation = person_data['occupation']
        current_education = person_data['education']
        
        # Get transition difficulty
        transition_difficulty = self.occupation_transitions.get((current_occupation, suggested_occupation), 5)
        
        # Estimated transition time based on difficulty (in months)
        transition_time_months = transition_difficulty * 3  # 3 months per difficulty point
        transition_time_years = transition_time_months / 12
        
        # Estimated transition cost (education, certifications, etc.)
        transition_cost = transition_difficulty * 2000  # $2,000 per difficulty point
        
        # Calculate current and projected income
        current_income = self.income_estimates.get(current_occupation, {}).get('expected_income', 35000)
        
        # Create modified person data
        modified_data = copy.deepcopy(person_data)
        modified_data['occupation'] = suggested_occupation
        
        # Get new income probability
        new_proba = self.optimizer.get_income_probability(modified_data)
        
        # Estimate new income based on probability
        occupation_income = self.income_estimates.get(suggested_occupation, {})
        low_income = occupation_income.get('low_income', 35000)
        high_income = occupation_income.get('high_income', 75000)
        
        # Account for education-specific income if available
        edu_income = occupation_income.get('education_incomes', {}).get(current_education, {})
        if edu_income:
            new_expected_income = edu_income.get('expected_income')
        else:
            new_expected_income = low_income * (1 - new_proba) + high_income * new_proba
        
        # Calculate income improvement
        income_improvement = new_expected_income - current_income
        
        # Calculate opportunity cost during transition (assuming partial income during transition)
        opportunity_cost = current_income * transition_time_years * 0.5  # Assuming 50% income during transition
        
        # Calculate total cost (direct cost + opportunity cost)
        total_cost = transition_cost + opportunity_cost
        
        # Calculate simple payback period (years)
        if income_improvement > 0:
            payback_years = total_cost / income_improvement
        else:
            payback_years = float('inf')
        
        # Calculate ROI (3-year and 5-year)
        income_gain_3yr = income_improvement * (3 - transition_time_years) if (3 - transition_time_years) > 0 else 0
        income_gain_5yr = income_improvement * (5 - transition_time_years) if (5 - transition_time_years) > 0 else 0
        
        roi_3yr = (income_gain_3yr - total_cost) / total_cost * 100 if total_cost > 0 else 0
        roi_5yr = (income_gain_5yr - total_cost) / total_cost * 100 if total_cost > 0 else 0
        
        return {
            'current_occupation': current_occupation,
            'suggested_occupation': suggested_occupation,
            'transition_difficulty': transition_difficulty,
            'transition_time_months': transition_time_months,
            'transition_cost': transition_cost,
            'opportunity_cost': opportunity_cost,
            'total_cost': total_cost,
            'current_income': current_income,
            'new_expected_income': new_expected_income,
            'annual_income_improvement': income_improvement,
            'payback_years': payback_years,
            'roi_3yr': roi_3yr,
            'roi_5yr': roi_5yr
        }
    
    def calculate_comprehensive_roi(self, person_data, top_n=3):
        """
        Calculate ROI for comprehensive optimization suggestions.
        
        Parameters:
        -----------
        person_data : dict
            Dictionary containing the person's feature values.
        top_n : int
            Number of top suggestions to analyze.
            
        Returns:
        --------
        dict
            Dictionary containing ROI calculations for different optimization strategies.
        """
        # Initialize return dictionary
        roi_results = {
            'education_roi': [],
            'occupation_roi': [],
            'combined_roi': []
        }
        
        try:
            # Get optimization suggestions
            optimization_results = self.optimizer.optimize_career_path(person_data)
            
            # Calculate ROI for education suggestions
            education_roi = []
            for sugg in optimization_results.get('education_suggestions', [])[:top_n]:
                try:
                    roi = self.calculate_education_roi(person_data, sugg.get('education'))
                    if roi:
                        roi['probability_improvement'] = sugg.get('improvement', 0)
                        education_roi.append(roi)
                except Exception as e:
                    print(f"Error calculating education ROI: {str(e)}")
            roi_results['education_roi'] = education_roi
            
            # Calculate ROI for occupation suggestions
            occupation_roi = []
            for sugg in optimization_results.get('occupation_suggestions', [])[:top_n]:
                try:
                    roi = self.calculate_occupation_roi(person_data, sugg.get('occupation'))
                    if roi:
                        roi['probability_improvement'] = sugg.get('improvement', 0)
                        occupation_roi.append(roi)
                except Exception as e:
                    print(f"Error calculating occupation ROI: {str(e)}")
            roi_results['occupation_roi'] = occupation_roi
            
            # Calculate ROI for combined suggestions
            combined_roi = []
            for sugg in optimization_results.get('combined_suggestions', [])[:top_n]:
                try:
                    # First calculate education ROI
                    edu_roi = self.calculate_education_roi(person_data, sugg.get('education'))
                    
                    # Then calculate occupation ROI with improved education
                    modified_data = copy.deepcopy(person_data)
                    modified_data['education'] = sugg.get('education')
                    if 'education_num' in person_data:
                        edu_num = self.optimizer.df[self.optimizer.df['education'] == sugg.get('education')]['education_num'].mean()
                        modified_data['education_num'] = edu_num
                    
                    occ_roi = self.calculate_occupation_roi(modified_data, sugg.get('occupation'))
                    
                    if edu_roi and occ_roi:
                        try:
                            # Calculate combined metrics safely
                            total_time_years = edu_roi.get('time_years', 0) + occ_roi.get('transition_time_months', 0) / 12
                            total_direct_cost = edu_roi.get('direct_cost', 0) + occ_roi.get('transition_cost', 0)
                            
                            # Opportunity cost calculation with safe fallbacks
                            transition_time_months = occ_roi.get('transition_time_months', 0)
                            if transition_time_months > 0:
                                time_factor = 1 + edu_roi.get('time_years', 0) / (transition_time_months / 12)
                                opportunity_factor = occ_roi.get('opportunity_cost', 0) * time_factor
                            else:
                                opportunity_factor = 0
                                
                            total_opportunity_cost = edu_roi.get('opportunity_cost', 0) + opportunity_factor
                            total_cost = total_direct_cost + total_opportunity_cost
                            
                            # Income improvement calculation
                            current_income = self.income_estimates.get(person_data.get('occupation', ''), {}).get('expected_income', 35000)
                            new_income = occ_roi.get('new_expected_income', current_income)
                            income_improvement = new_income - current_income
                            
                            # Payback period
                            if income_improvement > 0 and total_cost > 0:
                                payback_years = total_cost / income_improvement
                            else:
                                payback_years = float('inf')
                            
                            # 10-year ROI calculation
                            if total_time_years < 10 and total_cost > 0:
                                income_gain_10yr = income_improvement * (10 - total_time_years)
                                roi_10yr = (income_gain_10yr - total_cost) / total_cost * 100
                            else:
                                roi_10yr = 0
                            
                            combined_roi.append({
                                'education': sugg.get('education', 'Unknown'),
                                'occupation': sugg.get('occupation', 'Unknown'),
                                'probability': sugg.get('probability', 0),
                                'probability_improvement': sugg.get('improvement', 0),
                                'total_time_years': total_time_years,
                                'total_direct_cost': total_direct_cost,
                                'total_opportunity_cost': total_opportunity_cost,
                                'total_cost': total_cost,
                                'current_income': current_income,
                                'new_income': new_income,
                                'annual_income_improvement': income_improvement,
                                'payback_years': payback_years,
                                'roi_10yr': roi_10yr
                            })
                        except Exception as e:
                            print(f"Error calculating combined ROI metrics: {str(e)}")
                except Exception as e:
                    print(f"Error calculating ROI for combined suggestion: {str(e)}")
            
            roi_results['combined_roi'] = combined_roi
        except Exception as e:
            print(f"Error in calculate_comprehensive_roi: {str(e)}")
        
        return roi_results
    
    def plot_roi_comparison(self, roi_results, save_path=None):
        """
        Create a visualization comparing ROI of different career path options.
        
        Parameters:
        -----------
        roi_results : dict
            Results from calculate_comprehensive_roi method.
        save_path : str
            Path to save the plot. If None, the plot is displayed.
        """
        # Extract ROI data for comparison
        comparison_data = []
        
        # Education ROI
        for roi in roi_results.get('education_roi', []):
            comparison_data.append({
                'Category': 'Education',
                'Description': f"Education: {roi['suggested_education']}",
                'Time (Years)': roi['time_years'],
                'Total Cost': roi['total_cost'],
                'Annual Benefit': roi['annual_income_improvement'],
                'Payback (Years)': min(roi['payback_years'], 15),  # Cap at 15 years for visualization
                '5-Year ROI': roi['roi_5yr']
            })
        
        # Occupation ROI
        for roi in roi_results.get('occupation_roi', []):
            comparison_data.append({
                'Category': 'Occupation',
                'Description': f"Occupation: {roi['suggested_occupation']}",
                'Time (Years)': roi['transition_time_months'] / 12,
                'Total Cost': roi['total_cost'],
                'Annual Benefit': roi['annual_income_improvement'],
                'Payback (Years)': min(roi['payback_years'], 15),  # Cap at 15 years for visualization
                '5-Year ROI': roi['roi_5yr']
            })
        
        # Combined ROI
        for roi in roi_results.get('combined_roi', []):
            comparison_data.append({
                'Category': 'Combined',
                'Description': f"Edu: {roi['education']} + Occ: {roi['occupation']}",
                'Time (Years)': roi['total_time_years'],
                'Total Cost': roi['total_cost'],
                'Annual Benefit': roi['annual_income_improvement'],
                'Payback (Years)': min(roi['payback_years'], 15),  # Cap at 15 years for visualization
                '5-Year ROI': roi.get('roi_5yr', 0)  # Use 10-year if 5-year not available
            })
        
        # Create DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        if not comparison_df.empty:
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Career Path ROI Comparison', fontsize=16)
            
            # Plot 1: Time Investment
            sns.barplot(x='Time (Years)', y='Description', hue='Category', data=comparison_df, ax=axes[0, 0])
            axes[0, 0].set_title('Time Investment (Years)')
            axes[0, 0].set_xlabel('Years')
            axes[0, 0].set_ylabel('')
            
            # Plot 2: Total Cost
            sns.barplot(x='Total Cost', y='Description', hue='Category', data=comparison_df, ax=axes[0, 1])
            axes[0, 1].set_title('Total Cost (USD)')
            axes[0, 1].set_xlabel('Cost ($)')
            axes[0, 1].set_ylabel('')
            
            # Format y-axis with $ using formatter
            import matplotlib.ticker as mtick
            axes[0, 1].xaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
            
            # Plot 3: Annual Benefit
            sns.barplot(x='Annual Benefit', y='Description', hue='Category', data=comparison_df, ax=axes[1, 0])
            axes[1, 0].set_title('Annual Income Improvement (USD)')
            axes[1, 0].set_xlabel('Annual Benefit ($)')
            axes[1, 0].set_ylabel('')
            axes[1, 0].xaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
            
            # Plot 4: 5-Year ROI
            sns.barplot(x='5-Year ROI', y='Description', hue='Category', data=comparison_df, ax=axes[1, 1])
            axes[1, 1].set_title('5-Year Return on Investment (%)')
            axes[1, 1].set_xlabel('ROI (%)')
            axes[1, 1].set_ylabel('')
            axes[1, 1].xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:.0f}%'))
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()


def example_roi_calculation(model_path='../models/saved/random_forest', data_path='../data/adult.csv'):
    """
    Run an example ROI calculation.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model.
    data_path : str
        Path to the data file.
    """
    # Create optimizer and ROI calculator
    optimizer = CareerPathOptimizer(model_path, data_path)
    roi_calculator = ROICalculator(optimizer)
    
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
    print("Getting optimization suggestions...")
    results = optimizer.optimize_career_path(person_data)
    
    # Example education ROI calculation
    if results['education_suggestions']:
        suggested_education = results['education_suggestions'][0]['education']
        print(f"\n=== ROI for Education Improvement to {suggested_education} ===")
        education_roi = roi_calculator.calculate_education_roi(person_data, suggested_education)
        
        if education_roi:
            print(f"Direct Cost: ${education_roi['direct_cost']:,.2f}")
            print(f"Time Investment: {education_roi['time_years']} years")
            print(f"Opportunity Cost: ${education_roi['opportunity_cost']:,.2f}")
            print(f"Total Cost: ${education_roi['total_cost']:,.2f}")
            print(f"Annual Income Improvement: ${education_roi['annual_income_improvement']:,.2f}")
            print(f"Payback Period: {education_roi['payback_years']:.2f} years")
            print(f"5-Year ROI: {education_roi['roi_5yr']:.2f}%")
            print(f"10-Year ROI: {education_roi['roi_10yr']:.2f}%")
    
    # Example occupation ROI calculation
    if results['occupation_suggestions']:
        suggested_occupation = results['occupation_suggestions'][0]['occupation']
        print(f"\n=== ROI for Occupation Change to {suggested_occupation} ===")
        occupation_roi = roi_calculator.calculate_occupation_roi(person_data, suggested_occupation)
        
        if occupation_roi:
            print(f"Transition Difficulty: {occupation_roi['transition_difficulty']} out of 10")
            print(f"Transition Time: {occupation_roi['transition_time_months']:.1f} months")
            print(f"Transition Cost: ${occupation_roi['transition_cost']:,.2f}")
            print(f"Opportunity Cost: ${occupation_roi['opportunity_cost']:,.2f}")
            print(f"Total Cost: ${occupation_roi['total_cost']:,.2f}")
            print(f"Annual Income Improvement: ${occupation_roi['annual_income_improvement']:,.2f}")
            print(f"Payback Period: {occupation_roi['payback_years']:.2f} years")
            print(f"3-Year ROI: {occupation_roi['roi_3yr']:.2f}%")
            print(f"5-Year ROI: {occupation_roi['roi_5yr']:.2f}%")
    
    # Comprehensive ROI calculation
    print("\n=== Comprehensive ROI Analysis ===")
    comprehensive_roi = roi_calculator.calculate_comprehensive_roi(person_data)
    
    # Plot ROI comparison
    roi_calculator.plot_roi_comparison(comprehensive_roi, save_path='../plots/roi_comparison.png')
    print("ROI comparison plot saved to plots/roi_comparison.png")


if __name__ == "__main__":
    # Run example ROI calculation
    example_roi_calculation() 