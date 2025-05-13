import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from trade_fusion_simulation import TradeFusionSimulation, TraderType, SubscriptionTier

# Create output directory for visualizations
os.makedirs('output', exist_ok=True)

# Define base case parameters
base_params = {
    # Simulation parameters
    'simulation_months': 24,
    
    # Initial state parameters
    'initial_traders': 1000,  # 70% casual, 20% active, 8% professional, 2% whale
    'initial_projects': 10,   # 60% standard, 30% premium, 10% enterprise
    
    # Growth parameters
    'monthly_user_growth_rate': 0.15,
    'monthly_project_growth_rate': 0.10,
    'user_churn_rate': 0.05,
    'project_churn_rate': 0.08,
    'referral_conversion_rate': 0.10,
    
    # Challenge parameters
    'average_challenges_per_project': 1.5,
    
    # Financial parameters
    'development_cost': 225000,
    'monthly_operational_cost': 30000,
    'marketing_cost_percentage': 0.15,
    'reward_pool_allocation': 0.4,
    'referral_commission_percentage': 0.1
}

def run_base_scenario():
    """Run the base scenario and generate all visualizations"""
    print("Running base scenario simulation...")
    simulation = TradeFusionSimulation(base_params)
    simulation.run_simulation()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    simulation.visualize_financial_metrics(save_path='output')
    simulation.visualize_user_growth(save_path='output')
    simulation.visualize_project_growth(save_path='output')
    simulation.visualize_trader_earnings(save_path='output')
    
    # Generate revenue and reward breakdowns for months 3, 6, 12, and 24
    for month in [3, 6, 12, 24]:
        if month <= simulation.max_months:
            simulation.visualize_revenue_breakdown(month=month, save_path='output')
            simulation.visualize_reward_distribution(month=month, save_path='output')
    
    # Generate summary tables
    financial_summary = simulation.create_monthly_financial_summary(save_path='output')
    trader_economics = simulation.create_trader_economics_summary(save_path='output')
    project_economics = simulation.create_project_economics_summary(save_path='output')
    
    # Calculate break-even point
    break_even_month = simulation.calculate_break_even_point()
    if break_even_month:
        print(f"\nBreak-even point: Month {break_even_month}")
    else:
        print("\nThe platform does not break even within the simulation period.")
    
    # Calculate lifetime values
    ltv_data = simulation.calculate_lifetime_values()
    print("\nTrader Lifetime Values:")
    for trader_type, ltv in ltv_data['trader_ltv'].items():
        print(f"  {trader_type.value}: ${ltv:.2f}")
    
    print("\nProject Lifetime Values:")
    for tier, ltv in ltv_data['project_ltv'].items():
        print(f"  {tier.value}: ${ltv:.2f}")
    
    return simulation

def run_sensitivity_analysis(base_simulation):
    """Run sensitivity analysis on key parameters"""
    print("\nRunning sensitivity analysis...")
    
    # Define parameter ranges to test
    parameter_ranges = {
        'monthly_user_growth_rate': [0.05, 0.10, 0.15, 0.20, 0.25],
        'user_churn_rate': [0.02, 0.05, 0.08, 0.10, 0.15],
        'reward_pool_allocation': [0.2, 0.3, 0.4, 0.5, 0.6],
        'marketing_cost_percentage': [0.05, 0.10, 0.15, 0.20, 0.25]
    }
    
    # Run sensitivity analysis
    sensitivity_results = base_simulation.perform_sensitivity_analysis(parameter_ranges)
    
    # Save results to CSV
    sensitivity_results.to_csv('output/sensitivity_analysis.csv', index=False)
    
    # Visualize sensitivity analysis results
    for param in parameter_ranges.keys():
        param_data = sensitivity_results[sensitivity_results['parameter'] == param]
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(param_data['value'], param_data['monthly_revenue'], 'b-o')
        plt.title(f'Impact of {param} on Monthly Revenue')
        plt.xlabel(param)
        plt.ylabel('Monthly Revenue ($)')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(param_data['value'], param_data['monthly_profit'], 'g-o')
        plt.title(f'Impact of {param} on Monthly Profit')
        plt.xlabel(param)
        plt.ylabel('Monthly Profit ($)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'output/sensitivity_{param}.png')
        plt.show()
    
    print("Sensitivity analysis complete. Results saved to output/sensitivity_analysis.csv")
    return sensitivity_results

def run_shock_event_scenario():
    """Run a scenario with a market shock event"""
    print("\nRunning market shock event scenario...")
    
    # Copy base parameters and add shock event
    shock_params = base_params.copy()
    
    # Run simulation
    simulation = TradeFusionSimulation(shock_params)
    
    # Run first 12 months normally
    for month in range(1, 13):
        simulation.current_month = month
        
        # Update existing traders and projects
        simulation._update_traders()
        simulation._update_projects()
        
        # Generate new traders and projects
        simulation._generate_new_traders()
        simulation._generate_new_projects()
        
        # Create new challenges
        for project_id, project in simulation.projects.items():
            if project.is_active and random.random() < 0.3:
                simulation._create_challenge(project_id)
        
        # Reset challenge participants
        for challenge in simulation.challenges.values():
            challenge.participants = {}
            challenge.total_volume = 0
        
        # Assign traders to challenges
        simulation._assign_traders_to_challenges()
        
        # Distribute rewards
        simulation._distribute_challenge_rewards()
        
        # Collect subscription fees
        monthly_revenue = simulation._collect_monthly_subscription_fees()
        
        # Calculate financials and metrics
        financials = simulation._calculate_monthly_financials(monthly_revenue)
        metrics = simulation._calculate_monthly_metrics()
        
        # Record data
        simulation._record_monthly_data(financials, metrics)
    
    # Simulate market crash at month 13
    # Increase churn rates and decrease growth rates
    original_user_churn = shock_params['user_churn_rate']
    original_project_churn = shock_params['project_churn_rate']
    original_user_growth = shock_params['monthly_user_growth_rate']
    original_project_growth = shock_params['monthly_project_growth_rate']
    
    # Apply shock for 3 months
    for month in range(13, 16):
        simulation.current_month = month
        
        # Triple churn rates and halve growth rates during shock
        shock_params['user_churn_rate'] = original_user_churn * 3
        shock_params['project_churn_rate'] = original_project_churn * 3
        shock_params['monthly_user_growth_rate'] = original_user_growth / 2
        shock_params['monthly_project_growth_rate'] = original_project_growth / 2
        
        # Update with shock parameters
        simulation.params = shock_params
        
        # Run month with shock parameters
        simulation._update_traders()
        simulation._update_projects()
        simulation._generate_new_traders()
        simulation._generate_new_projects()
        
        for project_id, project in simulation.projects.items():
            if project.is_active and random.random() < 0.3:
                simulation._create_challenge(project_id)
        
        for challenge in simulation.challenges.values():
            challenge.participants = {}
            challenge.total_volume = 0
        
        simulation._assign_traders_to_challenges()
        simulation._distribute_challenge_rewards()
        monthly_revenue = simulation._collect_monthly_subscription_fees()
        financials = simulation._calculate_monthly_financials(monthly_revenue)
        metrics = simulation._calculate_monthly_metrics()
        simulation._record_monthly_data(financials, metrics)
    
    # Return to normal for remaining months
    shock_params['user_churn_rate'] = original_user_churn
    shock_params['project_churn_rate'] = original_project_churn
    shock_params['monthly_user_growth_rate'] = original_user_growth
    shock_params['monthly_project_growth_rate'] = original_project_growth
    
    simulation.params = shock_params
    
    for month in range(16, 25):
        simulation.current_month = month
        simulation._update_traders()
        simulation._update_projects()
        simulation._generate_new_traders()
        simulation._generate_new_projects()
        
        for project_id, project in simulation.projects.items():
            if project.is_active and random.random() < 0.3:
                simulation._create_challenge(project_id)
        
        for challenge in simulation.challenges.values():
            challenge.participants = {}
            challenge.total_volume = 0
        
        simulation._assign_traders_to_challenges()
        simulation._distribute_challenge_rewards()
        monthly_revenue = simulation._collect_monthly_subscription_fees()
        financials = simulation._calculate_monthly_financials(monthly_revenue)
        metrics = simulation._calculate_monthly_metrics()
        simulation._record_monthly_data(financials, metrics)
    
    # Visualize the impact of the shock
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(simulation.monthly_metrics['month'], simulation.monthly_metrics['monthly_revenue'])
    plt.axvspan(13, 15, alpha=0.3, color='red')
    plt.title('Monthly Revenue During Market Shock')
    plt.xlabel('Month')
    plt.ylabel('Revenue ($)')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(simulation.monthly_metrics['month'], simulation.monthly_metrics['active_traders'])
    plt.axvspan(13, 15, alpha=0.3, color='red')
    plt.title('Active Traders During Market Shock')
    plt.xlabel('Month')
    plt.ylabel('Number of Traders')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(simulation.monthly_metrics['month'], simulation.monthly_metrics['active_projects'])
    plt.axvspan(13, 15, alpha=0.3, color='red')
    plt.title('Active Projects During Market Shock')
    plt.xlabel('Month')
    plt.ylabel('Number of Projects')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(simulation.monthly_metrics['month'], simulation.monthly_metrics['monthly_profit'])
    plt.axvspan(13, 15, alpha=0.3, color='red')
    plt.title('Monthly Profit During Market Shock')
    plt.xlabel('Month')
    plt.ylabel('Profit ($)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('output/market_shock_impact.png')
    plt.show()
    
    print("Market shock scenario complete. Results saved to output/market_shock_impact.png")
    return simulation

def run_alternative_business_model():
    """Run a scenario with an alternative business model"""
    print("\nRunning alternative business model scenario...")
    
    # Copy base parameters and modify for alternative model
    alt_params = base_params.copy()
    
    # Alternative model: Higher subscription fees but lower reward pool allocation
    alt_params['reward_pool_allocation'] = 0.3  # Reduced from 0.4
    
    # Run simulation with alternative model
    alt_simulation = TradeFusionSimulation(alt_params)
    alt_simulation.run_simulation()
    
    # Run base model for comparison
    base_simulation = TradeFusionSimulation(base_params)
    base_simulation.run_simulation()
    
    # Compare the two models
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(base_simulation.monthly_metrics['month'], base_simulation.monthly_metrics['monthly_revenue'], 'b-', label='Base Model')
    plt.plot(alt_simulation.monthly_metrics['month'], alt_simulation.monthly_metrics['monthly_revenue'], 'r-', label='Alternative Model')
    plt.title('Monthly Revenue Comparison')
    plt.xlabel('Month')
    plt.ylabel('Revenue ($)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(base_simulation.monthly_metrics['month'], base_simulation.monthly_metrics['active_traders'], 'b-', label='Base Model')
    plt.plot(alt_simulation.monthly_metrics['month'], alt_simulation.monthly_metrics['active_traders'], 'r-', label='Alternative Model')
    plt.title('Active Traders Comparison')
    plt.xlabel('Month')
    plt.ylabel('Number of Traders')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(base_simulation.monthly_metrics['month'], base_simulation.monthly_metrics['active_projects'], 'b-', label='Base Model')
    plt.plot(alt_simulation.monthly_metrics['month'], alt_simulation.monthly_metrics['active_projects'], 'r-', label='Alternative Model')
    plt.title('Active Projects Comparison')
    plt.xlabel('Month')
    plt.ylabel('Number of Projects')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(base_simulation.monthly_metrics['month'], base_simulation.monthly_metrics['cumulative_profit'], 'b-', label='Base Model')
    plt.plot(alt_simulation.monthly_metrics['month'], alt_simulation.monthly_metrics['cumulative_profit'], 'r-', label='Alternative Model')
    plt.title('Cumulative Profit Comparison')
    plt.xlabel('Month')
    plt.ylabel('Profit ($)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('output/business_model_comparison.png')
    plt.show()
    
    # Compare break-even points
    base_break_even = base_simulation.calculate_break_even_point()
    alt_break_even = alt_simulation.calculate_break_even_point()
    
    print(f"Base model break-even point: {base_break_even if base_break_even else 'Not within simulation period'}")
    print(f"Alternative model break-even point: {alt_break_even if alt_break_even else 'Not within simulation period'}")
    
    print("Alternative business model comparison complete. Results saved to output/business_model_comparison.png")
    return base_simulation, alt_simulation

if __name__ == "__main__":
    print("TradeFusion Simulation")
    print("======================\n")
    
    # Run base scenario
    base_simulation = run_base_scenario()
    
    # Run sensitivity analysis
    sensitivity_results = run_sensitivity_analysis(base_simulation)
    
    # Run market shock scenario
    shock_simulation = run_shock_event_scenario()
    
    # Run alternative business model comparison
    base_model, alt_model = run_alternative_business_model()
    
    print("\nAll simulations complete. Results saved to the 'output' directory.")