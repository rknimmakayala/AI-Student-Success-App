"""
Refinery Linear Programming Optimization Model
===============================================
This module implements a linear programming model to optimize refinery production
to maximize profit from gasoline and diesel production.

Author: Analytics Team
Date: 2025-11-21
"""

import numpy as np
from scipy.optimize import linprog
import pandas as pd
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class RefineryOptimizer:
    """
    Linear Programming optimizer for refinery operations.

    Attributes:
        capacity: Maximum crude oil processing capacity (barrels/day)
        gasoline_yield: Conversion rate from crude to gasoline
        diesel_yield: Conversion rate from crude to diesel
    """

    def __init__(
        self,
        capacity: float = 100000,
        gasoline_yield: float = 0.45,
        diesel_yield: float = 0.35
    ):
        """
        Initialize the refinery optimizer.

        Args:
            capacity: Maximum crude oil processing capacity (barrels/day)
            gasoline_yield: Gasoline yield coefficient (0-1)
            diesel_yield: Diesel yield coefficient (0-1)
        """
        self.capacity = capacity
        self.gasoline_yield = gasoline_yield
        self.diesel_yield = diesel_yield

        # Validate inputs
        if gasoline_yield + diesel_yield > 1.0:
            raise ValueError("Sum of yield coefficients cannot exceed 1.0")

    def optimize(
        self,
        crude_price: float = 60,
        gasoline_price: float = 90,
        diesel_price: float = 85,
        var_op_cost: float = 5,
        fixed_op_cost: float = 50000,
        max_gasoline_demand: float = 50000,
        max_diesel_demand: float = 40000,
        min_gasoline: float = 10000,
        min_diesel: float = 5000
    ) -> Dict:
        """
        Optimize refinery production to maximize profit.

        Args:
            crude_price: Cost per barrel of crude oil ($)
            gasoline_price: Selling price per barrel of gasoline ($)
            diesel_price: Selling price per barrel of diesel ($)
            var_op_cost: Variable operating cost per barrel of crude ($)
            fixed_op_cost: Fixed daily operating cost ($)
            max_gasoline_demand: Maximum daily gasoline demand (barrels)
            max_diesel_demand: Maximum daily diesel demand (barrels)
            min_gasoline: Minimum daily gasoline production (barrels)
            min_diesel: Minimum daily diesel production (barrels)

        Returns:
            Dictionary containing optimization results
        """

        # Decision variable: barrels of crude oil to process
        # We only need one variable since gasoline and diesel are fixed proportions

        # Objective: Minimize negative profit (to maximize profit)
        # Profit = Revenue - Costs
        # Revenue = gasoline_price * gasoline_yield * crude + diesel_price * diesel_yield * crude
        # Costs = crude_price * crude + var_op_cost * crude + fixed_op_cost

        revenue_per_barrel = (gasoline_price * self.gasoline_yield +
                             diesel_price * self.diesel_yield)
        cost_per_barrel = crude_price + var_op_cost
        profit_per_barrel = revenue_per_barrel - cost_per_barrel

        # Objective coefficient (negative because linprog minimizes)
        c = [-profit_per_barrel]

        # Inequality constraints (A_ub @ x <= b_ub)
        A_ub = [
            [1],  # Capacity constraint: crude <= capacity
            [self.gasoline_yield],  # Gasoline demand: gasoline <= max_gasoline_demand
            [self.diesel_yield],  # Diesel demand: diesel <= max_diesel_demand
            [-self.gasoline_yield],  # Min gasoline: -gasoline <= -min_gasoline
            [-self.diesel_yield]  # Min diesel: -diesel <= -min_diesel
        ]

        b_ub = [
            self.capacity,
            max_gasoline_demand,
            max_diesel_demand,
            -min_gasoline,
            -min_diesel
        ]

        # Bounds for decision variable
        bounds = [(0, None)]

        # Solve the linear program
        result = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method='highs'
        )

        if not result.success:
            return {
                'status': 'infeasible',
                'message': result.message,
                'crude_to_process': 0,
                'gasoline_produced': 0,
                'diesel_produced': 0,
                'total_revenue': 0,
                'total_cost': 0,
                'profit': 0
            }

        # Extract results
        crude_to_process = result.x[0]
        gasoline_produced = crude_to_process * self.gasoline_yield
        diesel_produced = crude_to_process * self.diesel_yield

        # Calculate financial metrics
        total_revenue = (gasoline_produced * gasoline_price +
                        diesel_produced * diesel_price)
        total_cost = (crude_to_process * crude_price +
                     crude_to_process * var_op_cost +
                     fixed_op_cost)
        profit = total_revenue - total_cost

        # Calculate capacity utilization
        capacity_utilization = (crude_to_process / self.capacity) * 100

        # Calculate margins
        gasoline_margin = gasoline_price - (crude_price + var_op_cost) * (1 / self.gasoline_yield)
        diesel_margin = diesel_price - (crude_price + var_op_cost) * (1 / self.diesel_yield)

        # Check which constraints are binding
        binding_constraints = []
        tolerance = 1.0  # 1 barrel tolerance

        if abs(crude_to_process - self.capacity) < tolerance:
            binding_constraints.append('capacity')
        if abs(gasoline_produced - max_gasoline_demand) < tolerance:
            binding_constraints.append('gasoline_demand')
        if abs(diesel_produced - max_diesel_demand) < tolerance:
            binding_constraints.append('diesel_demand')
        if abs(gasoline_produced - min_gasoline) < tolerance:
            binding_constraints.append('min_gasoline')
        if abs(diesel_produced - min_diesel) < tolerance:
            binding_constraints.append('min_diesel')

        return {
            'status': 'optimal',
            'message': 'Optimization successful',
            'crude_to_process': round(crude_to_process, 2),
            'gasoline_produced': round(gasoline_produced, 2),
            'diesel_produced': round(diesel_produced, 2),
            'total_revenue': round(total_revenue, 2),
            'total_cost': round(total_cost, 2),
            'profit': round(profit, 2),
            'capacity_utilization': round(capacity_utilization, 2),
            'profit_per_barrel': round(profit / crude_to_process, 2) if crude_to_process > 0 else 0,
            'product_mix_ratio': round(gasoline_produced / diesel_produced, 2) if diesel_produced > 0 else 0,
            'gasoline_margin': round(gasoline_margin, 2),
            'diesel_margin': round(diesel_margin, 2),
            'binding_constraints': binding_constraints,
            'parameters': {
                'crude_price': crude_price,
                'gasoline_price': gasoline_price,
                'diesel_price': diesel_price,
                'var_op_cost': var_op_cost,
                'fixed_op_cost': fixed_op_cost
            }
        }

    def sensitivity_analysis(
        self,
        base_params: Dict,
        parameter: str,
        variation_range: np.ndarray
    ) -> pd.DataFrame:
        """
        Perform sensitivity analysis on a single parameter.

        Args:
            base_params: Dictionary of baseline parameters
            parameter: Name of parameter to vary
            variation_range: Array of values to test

        Returns:
            DataFrame with sensitivity analysis results
        """
        results = []

        for value in variation_range:
            params = base_params.copy()
            params[parameter] = value
            result = self.optimize(**params)

            results.append({
                parameter: value,
                'crude_to_process': result['crude_to_process'],
                'gasoline_produced': result['gasoline_produced'],
                'diesel_produced': result['diesel_produced'],
                'profit': result['profit'],
                'capacity_utilization': result['capacity_utilization'],
                'status': result['status']
            })

        return pd.DataFrame(results)

    def scenario_analysis(self, scenarios: Dict[str, Dict]) -> pd.DataFrame:
        """
        Run multiple scenarios and compare results.

        Args:
            scenarios: Dictionary of scenario names and their parameters

        Returns:
            DataFrame comparing scenario results
        """
        results = []

        for scenario_name, params in scenarios.items():
            result = self.optimize(**params)
            result['scenario'] = scenario_name
            results.append(result)

        df = pd.DataFrame(results)

        # Select key columns for comparison
        columns = [
            'scenario', 'status', 'crude_to_process', 'gasoline_produced',
            'diesel_produced', 'total_revenue', 'total_cost', 'profit',
            'capacity_utilization', 'profit_per_barrel'
        ]

        return df[columns]


def print_optimization_results(result: Dict, title: str = "Optimization Results"):
    """
    Print formatted optimization results.

    Args:
        result: Dictionary containing optimization results
        title: Title for the results display
    """
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}\n")

    print(f"Status: {result['status'].upper()}")
    if result['status'] != 'optimal':
        print(f"Message: {result['message']}")
        return

    print(f"\n{'PRODUCTION PLAN':^70}")
    print(f"{'-'*70}")
    print(f"  Crude Oil to Process:     {result['crude_to_process']:>15,.2f} barrels/day")
    print(f"  Gasoline Produced:        {result['gasoline_produced']:>15,.2f} barrels/day")
    print(f"  Diesel Produced:          {result['diesel_produced']:>15,.2f} barrels/day")
    print(f"  Capacity Utilization:     {result['capacity_utilization']:>15,.2f}%")

    print(f"\n{'FINANCIAL RESULTS':^70}")
    print(f"{'-'*70}")
    print(f"  Total Revenue:            ${result['total_revenue']:>15,.2f}")
    print(f"  Total Cost:               ${result['total_cost']:>15,.2f}")
    print(f"  Daily Profit:             ${result['profit']:>15,.2f}")
    print(f"  Profit per Barrel:        ${result['profit_per_barrel']:>15,.2f}")

    print(f"\n{'KEY METRICS':^70}")
    print(f"{'-'*70}")
    print(f"  Product Mix (Gas:Diesel): {result['product_mix_ratio']:>15,.2f}:1")
    print(f"  Gasoline Margin:          ${result['gasoline_margin']:>15,.2f}/barrel")
    print(f"  Diesel Margin:            ${result['diesel_margin']:>15,.2f}/barrel")

    if result['binding_constraints']:
        print(f"\n{'BINDING CONSTRAINTS':^70}")
        print(f"{'-'*70}")
        for constraint in result['binding_constraints']:
            print(f"  • {constraint}")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    # Example usage
    print("Refinery Linear Programming Optimization Model")
    print("=" * 70)

    # Initialize optimizer
    optimizer = RefineryOptimizer(
        capacity=100000,
        gasoline_yield=0.45,
        diesel_yield=0.35
    )

    # Run baseline optimization
    print("\n>>> Running Baseline Optimization...")
    baseline_result = optimizer.optimize(
        crude_price=60,
        gasoline_price=90,
        diesel_price=85,
        var_op_cost=5,
        fixed_op_cost=50000,
        max_gasoline_demand=50000,
        max_diesel_demand=40000,
        min_gasoline=10000,
        min_diesel=5000
    )

    print_optimization_results(baseline_result, "BASELINE SCENARIO")
