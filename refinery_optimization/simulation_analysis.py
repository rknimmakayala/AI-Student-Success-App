"""
Refinery Optimization: Simulation and What-If Analysis
======================================================
This module performs comprehensive simulation and scenario analysis
for the refinery optimization model.

Author: Analytics Team
Date: 2025-11-21
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from refinery_lp_model import RefineryOptimizer, print_optimization_results
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class RefinerySimulator:
    """
    Comprehensive simulation and what-if analysis for refinery optimization.
    """

    def __init__(self):
        """Initialize the refinery simulator."""
        self.optimizer = RefineryOptimizer(
            capacity=100000,
            gasoline_yield=0.45,
            diesel_yield=0.35
        )

        # Baseline parameters
        self.baseline_params = {
            'crude_price': 60,
            'gasoline_price': 90,
            'diesel_price': 85,
            'var_op_cost': 5,
            'fixed_op_cost': 50000,
            'max_gasoline_demand': 50000,
            'max_diesel_demand': 40000,
            'min_gasoline': 10000,
            'min_diesel': 5000
        }

    def run_baseline_simulation(self):
        """Run baseline optimization and display results."""
        print("\n" + "="*80)
        print("BASELINE SIMULATION")
        print("="*80)

        result = self.optimizer.optimize(**self.baseline_params)
        print_optimization_results(result, "BASELINE SCENARIO RESULTS")

        return result

    def price_sensitivity_analysis(self):
        """
        Analyze sensitivity to price changes for crude oil, gasoline, and diesel.
        """
        print("\n" + "="*80)
        print("PRICE SENSITIVITY ANALYSIS")
        print("="*80)

        # Crude oil price sensitivity
        print("\n>>> Analyzing Crude Oil Price Sensitivity...")
        crude_prices = np.linspace(40, 80, 20)
        crude_sensitivity = self.optimizer.sensitivity_analysis(
            self.baseline_params,
            'crude_price',
            crude_prices
        )

        # Gasoline price sensitivity
        print(">>> Analyzing Gasoline Price Sensitivity...")
        gasoline_prices = np.linspace(70, 110, 20)
        gasoline_sensitivity = self.optimizer.sensitivity_analysis(
            self.baseline_params,
            'gasoline_price',
            gasoline_prices
        )

        # Diesel price sensitivity
        print(">>> Analyzing Diesel Price Sensitivity...")
        diesel_prices = np.linspace(70, 100, 20)
        diesel_sensitivity = self.optimizer.sensitivity_analysis(
            self.baseline_params,
            'diesel_price',
            diesel_prices
        )

        return {
            'crude': crude_sensitivity,
            'gasoline': gasoline_sensitivity,
            'diesel': diesel_sensitivity
        }

    def capacity_scenario_analysis(self):
        """
        Analyze different capacity scenarios.
        """
        print("\n" + "="*80)
        print("CAPACITY SCENARIO ANALYSIS")
        print("="*80)

        scenarios = {
            'Constrained (50K)': {
                **self.baseline_params,
                'max_gasoline_demand': 25000,
                'max_diesel_demand': 17500
            },
            'Maintenance (70K)': {
                **self.baseline_params,
                'max_gasoline_demand': 35000,
                'max_diesel_demand': 28000
            },
            'Baseline (100K)': self.baseline_params.copy(),
            'Expansion (120K)': {
                **self.baseline_params,
                'max_gasoline_demand': 60000,
                'max_diesel_demand': 48000
            }
        }

        # Adjust optimizer capacity for each scenario
        results = []
        for scenario_name, params in scenarios.items():
            if '50K' in scenario_name:
                self.optimizer.capacity = 50000
            elif '70K' in scenario_name:
                self.optimizer.capacity = 70000
            elif '120K' in scenario_name:
                self.optimizer.capacity = 120000
            else:
                self.optimizer.capacity = 100000

            result = self.optimizer.optimize(**params)
            result['scenario'] = scenario_name
            results.append(result)

        # Reset capacity to baseline
        self.optimizer.capacity = 100000

        df = pd.DataFrame(results)
        print("\nCapacity Scenario Results:")
        print(df[['scenario', 'crude_to_process', 'gasoline_produced',
                  'diesel_produced', 'profit', 'capacity_utilization']].to_string(index=False))

        return df

    def demand_scenario_analysis(self):
        """
        Analyze different demand scenarios.
        """
        print("\n" + "="*80)
        print("DEMAND SCENARIO ANALYSIS")
        print("="*80)

        scenarios = {
            'Low Demand': {
                **self.baseline_params,
                'max_gasoline_demand': 30000,
                'max_diesel_demand': 25000,
                'min_gasoline': 8000,
                'min_diesel': 4000
            },
            'Normal Demand': self.baseline_params.copy(),
            'High Demand': {
                **self.baseline_params,
                'max_gasoline_demand': 60000,
                'max_diesel_demand': 50000,
                'min_gasoline': 15000,
                'min_diesel': 8000
            },
            'Gasoline Heavy': {
                **self.baseline_params,
                'max_gasoline_demand': 60000,
                'max_diesel_demand': 30000,
                'min_gasoline': 15000,
                'min_diesel': 5000
            },
            'Diesel Heavy': {
                **self.baseline_params,
                'max_gasoline_demand': 40000,
                'max_diesel_demand': 50000,
                'min_gasoline': 10000,
                'min_diesel': 10000
            }
        }

        df = self.optimizer.scenario_analysis(scenarios)
        print("\nDemand Scenario Results:")
        print(df.to_string(index=False))

        return df

    def multi_factor_scenario_analysis(self):
        """
        Analyze combined scenarios (best case, worst case, most likely).
        """
        print("\n" + "="*80)
        print("MULTI-FACTOR SCENARIO ANALYSIS")
        print("="*80)

        scenarios = {
            'Best Case': {
                'crude_price': 50,  # Low crude cost
                'gasoline_price': 100,  # High product prices
                'diesel_price': 95,
                'var_op_cost': 4,  # Low operating cost
                'fixed_op_cost': 45000,
                'max_gasoline_demand': 60000,  # High demand
                'max_diesel_demand': 50000,
                'min_gasoline': 10000,
                'min_diesel': 5000
            },
            'Most Likely': self.baseline_params.copy(),
            'Worst Case': {
                'crude_price': 75,  # High crude cost
                'gasoline_price': 80,  # Low product prices
                'diesel_price': 75,
                'var_op_cost': 6,  # High operating cost
                'fixed_op_cost': 55000,
                'max_gasoline_demand': 35000,  # Low demand
                'max_diesel_demand': 25000,
                'min_gasoline': 10000,
                'min_diesel': 5000
            },
            'High Volatility': {
                'crude_price': 70,
                'gasoline_price': 95,
                'diesel_price': 80,
                'var_op_cost': 5,
                'fixed_op_cost': 50000,
                'max_gasoline_demand': 45000,
                'max_diesel_demand': 35000,
                'min_gasoline': 10000,
                'min_diesel': 5000
            }
        }

        # Adjust capacity for worst case
        original_capacity = self.optimizer.capacity
        for scenario_name, params in scenarios.items():
            if scenario_name == 'Worst Case':
                self.optimizer.capacity = 70000  # Maintenance scenario
            else:
                self.optimizer.capacity = original_capacity

            result = self.optimizer.optimize(**params)
            print_optimization_results(result, f"{scenario_name.upper()} SCENARIO")

        self.optimizer.capacity = original_capacity

        df = self.optimizer.scenario_analysis(scenarios)
        return df

    def breakeven_analysis(self):
        """
        Calculate breakeven points for various parameters.
        """
        print("\n" + "="*80)
        print("BREAKEVEN ANALYSIS")
        print("="*80)

        # Crude price breakeven
        crude_prices = np.linspace(50, 85, 50)
        breakeven_results = []

        for crude_price in crude_prices:
            params = self.baseline_params.copy()
            params['crude_price'] = crude_price
            result = self.optimizer.optimize(**params)
            breakeven_results.append({
                'crude_price': crude_price,
                'profit': result['profit']
            })

        df = pd.DataFrame(breakeven_results)

        # Find breakeven point (profit closest to 0)
        breakeven_idx = (df['profit'] - 0).abs().idxmin()
        breakeven_price = df.loc[breakeven_idx, 'crude_price']
        breakeven_profit = df.loc[breakeven_idx, 'profit']

        print(f"\nCrude Oil Breakeven Analysis:")
        print(f"  Breakeven Price: ${breakeven_price:.2f}/barrel")
        print(f"  Profit at Breakeven: ${breakeven_profit:,.2f}/day")
        print(f"  Current Price: ${self.baseline_params['crude_price']:.2f}/barrel")
        print(f"  Margin to Breakeven: ${self.baseline_params['crude_price'] - breakeven_price:.2f}/barrel")

        return df

    def optimize_product_mix(self):
        """
        Analyze optimal product mix under different price scenarios.
        """
        print("\n" + "="*80)
        print("PRODUCT MIX OPTIMIZATION ANALYSIS")
        print("="*80)

        scenarios = {
            'Equal Prices': {
                **self.baseline_params,
                'gasoline_price': 85,
                'diesel_price': 85
            },
            'Gasoline Premium': {
                **self.baseline_params,
                'gasoline_price': 95,
                'diesel_price': 80
            },
            'Diesel Premium': {
                **self.baseline_params,
                'gasoline_price': 85,
                'diesel_price': 95
            },
            'High Margin': {
                **self.baseline_params,
                'gasoline_price': 100,
                'diesel_price': 95
            }
        }

        results = []
        for scenario_name, params in scenarios.items():
            result = self.optimizer.optimize(**params)
            results.append({
                'Scenario': scenario_name,
                'Gasoline Price': params['gasoline_price'],
                'Diesel Price': params['diesel_price'],
                'Gasoline Produced': result['gasoline_produced'],
                'Diesel Produced': result['diesel_produced'],
                'Mix Ratio (G:D)': result['product_mix_ratio'],
                'Profit': result['profit']
            })

        df = pd.DataFrame(results)
        print("\nProduct Mix Analysis:")
        print(df.to_string(index=False))

        return df

    def monte_carlo_simulation(self, n_simulations=1000):
        """
        Run Monte Carlo simulation with random price variations.

        Args:
            n_simulations: Number of simulation runs
        """
        print("\n" + "="*80)
        print(f"MONTE CARLO SIMULATION ({n_simulations:,} runs)")
        print("="*80)

        np.random.seed(42)

        # Define distributions (normal with 10% std dev)
        crude_std = self.baseline_params['crude_price'] * 0.10
        gas_std = self.baseline_params['gasoline_price'] * 0.10
        diesel_std = self.baseline_params['diesel_price'] * 0.10

        results = []
        for i in range(n_simulations):
            params = self.baseline_params.copy()
            params['crude_price'] = max(30, np.random.normal(60, crude_std))
            params['gasoline_price'] = max(50, np.random.normal(90, gas_std))
            params['diesel_price'] = max(50, np.random.normal(85, diesel_std))

            result = self.optimizer.optimize(**params)
            results.append({
                'run': i + 1,
                'crude_price': params['crude_price'],
                'gasoline_price': params['gasoline_price'],
                'diesel_price': params['diesel_price'],
                'profit': result['profit'],
                'crude_processed': result['crude_to_process']
            })

        df = pd.DataFrame(results)

        print(f"\nMonte Carlo Results Summary:")
        print(f"  Mean Profit:          ${df['profit'].mean():>15,.2f}/day")
        print(f"  Median Profit:        ${df['profit'].median():>15,.2f}/day")
        print(f"  Std Deviation:        ${df['profit'].std():>15,.2f}")
        print(f"  Min Profit:           ${df['profit'].min():>15,.2f}/day")
        print(f"  Max Profit:           ${df['profit'].max():>15,.2f}/day")
        print(f"  5th Percentile:       ${df['profit'].quantile(0.05):>15,.2f}/day")
        print(f"  95th Percentile:      ${df['profit'].quantile(0.95):>15,.2f}/day")
        print(f"\nValue at Risk (VaR):")
        print(f"  5% VaR (daily):       ${df['profit'].quantile(0.05):>15,.2f}")
        print(f"  Annual VaR (5%):      ${df['profit'].quantile(0.05) * 365:>15,.2f}")

        return df


def create_visualizations(simulator):
    """
    Create comprehensive visualizations for all analyses.

    Args:
        simulator: RefinerySimulator instance
    """
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    # 1. Price Sensitivity Analysis
    print("\n>>> Creating price sensitivity charts...")
    sensitivity_data = simulator.price_sensitivity_analysis()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Price Sensitivity Analysis', fontsize=16, fontweight='bold')

    # Crude price sensitivity
    ax = axes[0, 0]
    ax.plot(sensitivity_data['crude']['crude_price'],
            sensitivity_data['crude']['profit'] / 1000, 'b-', linewidth=2)
    ax.axvline(x=60, color='r', linestyle='--', label='Baseline')
    ax.set_xlabel('Crude Oil Price ($/barrel)')
    ax.set_ylabel('Daily Profit ($1000s)')
    ax.set_title('Crude Oil Price Sensitivity')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Gasoline price sensitivity
    ax = axes[0, 1]
    ax.plot(sensitivity_data['gasoline']['gasoline_price'],
            sensitivity_data['gasoline']['profit'] / 1000, 'g-', linewidth=2)
    ax.axvline(x=90, color='r', linestyle='--', label='Baseline')
    ax.set_xlabel('Gasoline Price ($/barrel)')
    ax.set_ylabel('Daily Profit ($1000s)')
    ax.set_title('Gasoline Price Sensitivity')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Diesel price sensitivity
    ax = axes[1, 0]
    ax.plot(sensitivity_data['diesel']['diesel_price'],
            sensitivity_data['diesel']['profit'] / 1000, 'orange', linewidth=2)
    ax.axvline(x=85, color='r', linestyle='--', label='Baseline')
    ax.set_xlabel('Diesel Price ($/barrel)')
    ax.set_ylabel('Daily Profit ($1000s)')
    ax.set_title('Diesel Price Sensitivity')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Capacity utilization vs crude price
    ax = axes[1, 1]
    ax.plot(sensitivity_data['crude']['crude_price'],
            sensitivity_data['crude']['capacity_utilization'], 'purple', linewidth=2)
    ax.axvline(x=60, color='r', linestyle='--', label='Baseline')
    ax.set_xlabel('Crude Oil Price ($/barrel)')
    ax.set_ylabel('Capacity Utilization (%)')
    ax.set_title('Capacity Utilization vs Crude Price')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig('/home/user/AI-Student-Success-App/refinery_optimization/price_sensitivity_analysis.png',
                dpi=300, bbox_inches='tight')
    print("    Saved: price_sensitivity_analysis.png")

    # 2. Scenario Comparison
    print(">>> Creating scenario comparison charts...")
    multi_factor_data = simulator.multi_factor_scenario_analysis()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Multi-Factor Scenario Analysis', fontsize=16, fontweight='bold')

    # Profit comparison
    ax = axes[0, 0]
    scenarios = multi_factor_data['scenario']
    profits = multi_factor_data['profit'] / 1000
    colors = ['green' if p > 1000 else 'orange' if p > 500 else 'red' for p in profits]
    ax.barh(scenarios, profits, color=colors)
    ax.set_xlabel('Daily Profit ($1000s)')
    ax.set_title('Profit by Scenario')
    ax.grid(True, alpha=0.3, axis='x')

    # Production comparison
    ax = axes[0, 1]
    x = np.arange(len(scenarios))
    width = 0.35
    ax.bar(x - width/2, multi_factor_data['gasoline_produced'] / 1000,
           width, label='Gasoline', color='lightblue')
    ax.bar(x + width/2, multi_factor_data['diesel_produced'] / 1000,
           width, label='Diesel', color='lightcoral')
    ax.set_xlabel('Scenario')
    ax.set_ylabel('Production (1000 barrels/day)')
    ax.set_title('Production by Scenario')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Capacity utilization
    ax = axes[1, 0]
    ax.bar(scenarios, multi_factor_data['capacity_utilization'],
           color='steelblue')
    ax.set_ylabel('Capacity Utilization (%)')
    ax.set_title('Capacity Utilization by Scenario')
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=100, color='r', linestyle='--', label='Max Capacity')
    ax.legend()

    # Revenue vs Cost
    ax = axes[1, 1]
    x = np.arange(len(scenarios))
    width = 0.35
    ax.bar(x - width/2, multi_factor_data['total_revenue'] / 1000,
           width, label='Revenue', color='green', alpha=0.7)
    ax.bar(x + width/2, multi_factor_data['total_cost'] / 1000,
           width, label='Cost', color='red', alpha=0.7)
    ax.set_xlabel('Scenario')
    ax.set_ylabel('Amount ($1000s)')
    ax.set_title('Revenue vs Cost by Scenario')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('/home/user/AI-Student-Success-App/refinery_optimization/scenario_comparison.png',
                dpi=300, bbox_inches='tight')
    print("    Saved: scenario_comparison.png")

    # 3. Monte Carlo Results
    print(">>> Creating Monte Carlo simulation charts...")
    mc_data = simulator.monte_carlo_simulation(n_simulations=1000)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Monte Carlo Simulation Results (1,000 runs)', fontsize=16, fontweight='bold')

    # Profit distribution
    ax = axes[0, 0]
    ax.hist(mc_data['profit'] / 1000, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(mc_data['profit'].mean() / 1000, color='red', linestyle='--',
               linewidth=2, label=f'Mean: ${mc_data["profit"].mean()/1000:.0f}K')
    ax.axvline(mc_data['profit'].median() / 1000, color='green', linestyle='--',
               linewidth=2, label=f'Median: ${mc_data["profit"].median()/1000:.0f}K')
    ax.set_xlabel('Daily Profit ($1000s)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Daily Profit')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Profit vs crude price
    ax = axes[0, 1]
    scatter = ax.scatter(mc_data['crude_price'], mc_data['profit'] / 1000,
                        c=mc_data['profit'], cmap='RdYlGn', alpha=0.5)
    ax.set_xlabel('Crude Oil Price ($/barrel)')
    ax.set_ylabel('Daily Profit ($1000s)')
    ax.set_title('Profit vs Crude Oil Price')
    plt.colorbar(scatter, ax=ax, label='Profit ($)')
    ax.grid(True, alpha=0.3)

    # Profit vs gasoline price
    ax = axes[1, 0]
    scatter = ax.scatter(mc_data['gasoline_price'], mc_data['profit'] / 1000,
                        c=mc_data['profit'], cmap='RdYlGn', alpha=0.5)
    ax.set_xlabel('Gasoline Price ($/barrel)')
    ax.set_ylabel('Daily Profit ($1000s)')
    ax.set_title('Profit vs Gasoline Price')
    plt.colorbar(scatter, ax=ax, label='Profit ($)')
    ax.grid(True, alpha=0.3)

    # Box plot of profit percentiles
    ax = axes[1, 1]
    bp = ax.boxplot(mc_data['profit'] / 1000, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    ax.set_ylabel('Daily Profit ($1000s)')
    ax.set_title('Profit Distribution (Box Plot)')
    ax.grid(True, alpha=0.3, axis='y')

    # Add percentile annotations
    percentiles = [5, 25, 50, 75, 95]
    for p in percentiles:
        val = mc_data['profit'].quantile(p/100) / 1000
        ax.text(1.15, val, f'{p}th: ${val:.0f}K', fontsize=9)

    plt.tight_layout()
    plt.savefig('/home/user/AI-Student-Success-App/refinery_optimization/monte_carlo_simulation.png',
                dpi=300, bbox_inches='tight')
    print("    Saved: monte_carlo_simulation.png")

    # 4. Demand Scenarios
    print(">>> Creating demand scenario charts...")
    demand_data = simulator.demand_scenario_analysis()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Demand Scenario Analysis', fontsize=16, fontweight='bold')

    scenarios = demand_data['scenario']

    # Profit by demand scenario
    ax = axes[0, 0]
    profits = demand_data['profit'] / 1000
    colors = plt.cm.viridis(np.linspace(0, 1, len(scenarios)))
    ax.bar(scenarios, profits, color=colors)
    ax.set_ylabel('Daily Profit ($1000s)')
    ax.set_title('Profit by Demand Scenario')
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Production volumes
    ax = axes[0, 1]
    x = np.arange(len(scenarios))
    width = 0.35
    ax.bar(x - width/2, demand_data['gasoline_produced'] / 1000,
           width, label='Gasoline', color='gold')
    ax.bar(x + width/2, demand_data['diesel_produced'] / 1000,
           width, label='Diesel', color='silver')
    ax.set_ylabel('Production (1000 barrels/day)')
    ax.set_title('Production Volumes by Demand Scenario')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Capacity utilization
    ax = axes[1, 0]
    ax.plot(scenarios, demand_data['capacity_utilization'],
            marker='o', linewidth=2, markersize=8, color='darkorange')
    ax.set_ylabel('Capacity Utilization (%)')
    ax.set_title('Capacity Utilization by Demand Scenario')
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=100, color='r', linestyle='--', alpha=0.5)

    # Profit per barrel
    ax = axes[1, 1]
    ax.bar(scenarios, demand_data['profit_per_barrel'],
           color='teal')
    ax.set_ylabel('Profit per Barrel ($/barrel)')
    ax.set_title('Profit Efficiency by Demand Scenario')
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('/home/user/AI-Student-Success-App/refinery_optimization/demand_scenarios.png',
                dpi=300, bbox_inches='tight')
    print("    Saved: demand_scenarios.png")

    # 5. Tornado Diagram for Sensitivity
    print(">>> Creating tornado diagram...")
    fig, ax = plt.subplots(figsize=(12, 8))

    # Calculate impact of ±20% change in each parameter
    baseline = simulator.optimizer.optimize(**simulator.baseline_params)
    baseline_profit = baseline['profit']

    params_to_test = {
        'Crude Oil Price': ('crude_price', 60),
        'Gasoline Price': ('gasoline_price', 90),
        'Diesel Price': ('diesel_price', 85),
        'Variable Op Cost': ('var_op_cost', 5),
        'Fixed Op Cost': ('fixed_op_cost', 50000)
    }

    impacts = []
    for param_name, (param_key, base_value) in params_to_test.items():
        # Test -20%
        params_low = simulator.baseline_params.copy()
        params_low[param_key] = base_value * 0.8
        result_low = simulator.optimizer.optimize(**params_low)

        # Test +20%
        params_high = simulator.baseline_params.copy()
        params_high[param_key] = base_value * 1.2
        result_high = simulator.optimizer.optimize(**params_high)

        low_impact = result_low['profit'] - baseline_profit
        high_impact = result_high['profit'] - baseline_profit

        impacts.append({
            'parameter': param_name,
            'low_impact': low_impact / 1000,
            'high_impact': high_impact / 1000
        })

    # Sort by total impact
    impacts_df = pd.DataFrame(impacts)
    impacts_df['total_impact'] = impacts_df['low_impact'].abs() + impacts_df['high_impact'].abs()
    impacts_df = impacts_df.sort_values('total_impact', ascending=True)

    # Create tornado diagram
    y_pos = np.arange(len(impacts_df))
    ax.barh(y_pos, impacts_df['low_impact'], height=0.4,
            color='red', alpha=0.7, label='-20%')
    ax.barh(y_pos, impacts_df['high_impact'], height=0.4,
            color='green', alpha=0.7, label='+20%')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(impacts_df['parameter'])
    ax.set_xlabel('Impact on Daily Profit ($1000s)')
    ax.set_title('Tornado Diagram: Sensitivity of Profit to Parameter Changes (±20%)',
                 fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('/home/user/AI-Student-Success-App/refinery_optimization/tornado_diagram.png',
                dpi=300, bbox_inches='tight')
    print("    Saved: tornado_diagram.png")

    print("\n" + "="*80)
    print("All visualizations created successfully!")
    print("="*80)


def main():
    """
    Main execution function for comprehensive simulation and analysis.
    """
    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + "  REFINERY OPTIMIZATION: COMPREHENSIVE SIMULATION & WHAT-IF ANALYSIS".center(78) + "#")
    print("#" + " "*78 + "#")
    print("#"*80)

    # Initialize simulator
    simulator = RefinerySimulator()

    # 1. Baseline Simulation
    baseline_result = simulator.run_baseline_simulation()

    # 2. Price Sensitivity Analysis
    price_sensitivity = simulator.price_sensitivity_analysis()

    # 3. Capacity Scenarios
    capacity_scenarios = simulator.capacity_scenario_analysis()

    # 4. Demand Scenarios
    demand_scenarios = simulator.demand_scenario_analysis()

    # 5. Multi-Factor Scenarios
    multi_factor_scenarios = simulator.multi_factor_scenario_analysis()

    # 6. Breakeven Analysis
    breakeven_data = simulator.breakeven_analysis()

    # 7. Product Mix Optimization
    product_mix_data = simulator.optimize_product_mix()

    # 8. Monte Carlo Simulation
    mc_data = simulator.monte_carlo_simulation(n_simulations=1000)

    # 9. Create All Visualizations
    create_visualizations(simulator)

    # 10. Export Results
    print("\n" + "="*80)
    print("EXPORTING RESULTS TO CSV")
    print("="*80)

    capacity_scenarios.to_csv(
        '/home/user/AI-Student-Success-App/refinery_optimization/capacity_scenarios.csv',
        index=False
    )
    print("  Exported: capacity_scenarios.csv")

    demand_scenarios.to_csv(
        '/home/user/AI-Student-Success-App/refinery_optimization/demand_scenarios.csv',
        index=False
    )
    print("  Exported: demand_scenarios.csv")

    multi_factor_scenarios.to_csv(
        '/home/user/AI-Student-Success-App/refinery_optimization/multi_factor_scenarios.csv',
        index=False
    )
    print("  Exported: multi_factor_scenarios.csv")

    mc_data.to_csv(
        '/home/user/AI-Student-Success-App/refinery_optimization/monte_carlo_results.csv',
        index=False
    )
    print("  Exported: monte_carlo_results.csv")

    product_mix_data.to_csv(
        '/home/user/AI-Student-Success-App/refinery_optimization/product_mix_analysis.csv',
        index=False
    )
    print("  Exported: product_mix_analysis.csv")

    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + "  SIMULATION COMPLETE!".center(78) + "#")
    print("#" + " "*78 + "#")
    print("#"*80)

    return {
        'baseline': baseline_result,
        'price_sensitivity': price_sensitivity,
        'capacity_scenarios': capacity_scenarios,
        'demand_scenarios': demand_scenarios,
        'multi_factor_scenarios': multi_factor_scenarios,
        'breakeven_data': breakeven_data,
        'product_mix_data': product_mix_data,
        'monte_carlo_data': mc_data
    }


if __name__ == "__main__":
    results = main()
