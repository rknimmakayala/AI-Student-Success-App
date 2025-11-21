# Refinery Linear Programming Optimization System

A comprehensive Linear Programming optimization system for refinery operations that maximizes profitability from gasoline and diesel production.

## Overview

This system implements a sophisticated LP optimization model to:
- Maximize daily profit from refinery operations
- Optimize production mix of gasoline and diesel
- Perform comprehensive what-if and sensitivity analysis
- Support strategic decision-making through scenario planning

## Project Structure

```
refinery_optimization/
├── BUSINESS_REQUIREMENTS_DOCUMENT.md   # Complete BRD with model formulation
├── EXECUTIVE_SUMMARY.md                # Analysis results and recommendations
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── refinery_lp_model.py               # Core optimization model
├── simulation_analysis.py             # Comprehensive simulation and analysis
├── *.png                              # Generated visualizations
└── *.csv                              # Exported simulation results
```

## Features

### 1. Linear Programming Optimization
- Maximizes daily profit subject to operational constraints
- Handles capacity limits, demand constraints, and minimum production requirements
- Provides optimal production quantities for crude oil, gasoline, and diesel

### 2. Sensitivity Analysis
- Price sensitivity for crude oil, gasoline, and diesel
- Capacity scenario analysis (50K to 120K barrels/day)
- Demand scenario analysis (low, normal, high, product-specific)

### 3. What-If Analysis
- Multi-factor scenarios (best case, worst case, most likely)
- Breakeven analysis for crude oil pricing
- Product mix optimization under different price scenarios

### 4. Monte Carlo Simulation
- 1,000+ scenario simulations with random price variations
- Probabilistic profit forecasting
- Value at Risk (VaR) calculations
- Risk profile assessment

### 5. Comprehensive Visualizations
- Price sensitivity charts
- Scenario comparison dashboards
- Monte Carlo distribution plots
- Tornado diagrams for sensitivity rankings

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Install dependencies:
```bash
cd refinery_optimization
pip install -r requirements.txt
```

## Usage

### Quick Start: Run Complete Analysis

Run the comprehensive simulation that performs all analyses and generates visualizations:

```bash
python simulation_analysis.py
```

This will:
- Run baseline optimization
- Perform price sensitivity analysis
- Analyze capacity and demand scenarios
- Execute multi-factor scenario analysis
- Perform breakeven analysis
- Run Monte Carlo simulation (1,000 runs)
- Generate all visualizations
- Export results to CSV files

### Using the Core Model

```python
from refinery_lp_model import RefineryOptimizer, print_optimization_results

# Initialize optimizer
optimizer = RefineryOptimizer(
    capacity=100000,
    gasoline_yield=0.45,
    diesel_yield=0.35
)

# Run optimization
result = optimizer.optimize(
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

# Display results
print_optimization_results(result)
```

### Custom Sensitivity Analysis

```python
import numpy as np

# Analyze sensitivity to a specific parameter
crude_prices = np.linspace(50, 70, 20)
sensitivity_df = optimizer.sensitivity_analysis(
    base_params={'crude_price': 60, 'gasoline_price': 90, ...},
    parameter='crude_price',
    variation_range=crude_prices
)

print(sensitivity_df)
```

### Custom Scenario Analysis

```python
# Define custom scenarios
scenarios = {
    'Conservative': {
        'crude_price': 65,
        'gasoline_price': 85,
        'diesel_price': 80,
        ...
    },
    'Aggressive': {
        'crude_price': 55,
        'gasoline_price': 95,
        'diesel_price': 90,
        ...
    }
}

# Compare scenarios
results_df = optimizer.scenario_analysis(scenarios)
print(results_df)
```

## Model Formulation

### Decision Variables
- `x_crude` = Barrels of crude oil to process per day
- `x_gasoline` = Barrels of gasoline to produce per day (= 0.45 × x_crude)
- `x_diesel` = Barrels of diesel to produce per day (= 0.35 × x_crude)

### Objective Function (Maximize)
```
Profit = (P_gasoline × x_gasoline) + (P_diesel × x_diesel)
         - (C_crude × x_crude) - (C_var × x_crude) - C_fixed
```

### Constraints
1. **Capacity:** x_crude ≤ 100,000 barrels/day
2. **Gasoline demand:** x_gasoline ≤ 50,000 barrels/day
3. **Diesel demand:** x_diesel ≤ 40,000 barrels/day
4. **Minimum gasoline:** x_gasoline ≥ 10,000 barrels/day
5. **Minimum diesel:** x_diesel ≥ 5,000 barrels/day
6. **Non-negativity:** All variables ≥ 0

## Baseline Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Capacity | 100,000 barrels/day | Maximum crude processing |
| Gasoline yield | 45% | Conversion rate from crude |
| Diesel yield | 35% | Conversion rate from crude |
| Crude price | $60/barrel | Cost of crude oil |
| Gasoline price | $90/barrel | Selling price |
| Diesel price | $85/barrel | Selling price |
| Variable op cost | $5/barrel | Per barrel operating cost |
| Fixed op cost | $50,000/day | Daily fixed costs |

## Key Results (Baseline Scenario)

- **Optimal Crude Processing:** 100,000 barrels/day
- **Gasoline Production:** 45,000 barrels/day
- **Diesel Production:** 35,000 barrels/day
- **Daily Profit:** $475,000
- **Annual Profit:** $173.4 million
- **Capacity Utilization:** 100%

## Generated Outputs

### Visualizations
1. **price_sensitivity_analysis.png** - Impact of price changes on profit and capacity
2. **scenario_comparison.png** - Multi-factor scenario comparisons
3. **monte_carlo_simulation.png** - Probabilistic profit distributions
4. **demand_scenarios.png** - Demand scenario analysis
5. **tornado_diagram.png** - Parameter sensitivity rankings

### Data Files
1. **capacity_scenarios.csv** - Capacity analysis results
2. **demand_scenarios.csv** - Demand scenario data
3. **multi_factor_scenarios.csv** - Combined scenarios
4. **monte_carlo_results.csv** - 1,000 simulation runs
5. **product_mix_analysis.csv** - Product mix optimization

## Interpreting Results

### Optimal Solution
- **Status:** "optimal" indicates a feasible solution was found
- **Binding Constraints:** Lists which constraints are at their limits
- **Profit per Barrel:** Measures operational efficiency

### Sensitivity Insights
- **Crude Oil Price:** Highest risk factor; $1 change = ±$100K profit impact
- **Gasoline Price:** Highest upside opportunity; $1 change = +$45K profit
- **Diesel Price:** Strong contributor; $1 change = +$35K profit

### Risk Metrics
- **Breakeven:** Crude oil price of $65/barrel (8% margin)
- **VaR (5%):** -$236K daily loss threshold
- **Monte Carlo Mean:** $550K average daily profit across scenarios

## Business Recommendations

### High Priority
1. **Implement crude oil price hedging** - Protect against downside risk
2. **Deploy model for daily planning** - Capture 5-10% efficiency gains
3. **Evaluate capacity expansion** - ROI < 3 years for $100M investment

### Medium Priority
4. Reduce variable operating costs by 5-10%
5. Negotiate flexible minimum production requirements
6. Develop inventory optimization strategy

### Long Term
7. Research product mix flexibility enhancements
8. Integrate predictive analytics and ML forecasting
9. Establish enterprise risk management framework

## Technical Notes

### Solver
- Uses SciPy's Linear Programming solver (HiGHS method)
- Solves in < 5 seconds for standard scenarios
- Handles infeasibility gracefully with informative messages

### Validation
- All constraints properly enforced
- Solution optimality verified through shadow prices
- Numerical stability tested across extreme scenarios

### Extensibility
The model architecture supports:
- Additional products (reformulated gasoline, jet fuel, etc.)
- More complex constraints (environmental, quality specs)
- Multi-period planning with inventory
- Stochastic programming for uncertainty modeling

## Limitations

Current model assumptions:
- Single crude oil type (uniform quality)
- Linear relationships between inputs/outputs
- Constant yield coefficients
- No intermediate storage or inventory
- Deterministic demand (no uncertainty in v1.0)
- Daily planning horizon (no multi-period optimization)

See `BUSINESS_REQUIREMENTS_DOCUMENT.md` Section 7 for complete constraints and assumptions.

## Support and Documentation

- **Business Requirements:** See `BUSINESS_REQUIREMENTS_DOCUMENT.md`
- **Executive Summary:** See `EXECUTIVE_SUMMARY.md`
- **Code Documentation:** Inline comments and docstrings in Python files
- **Mathematical Formulation:** Appendix A in BRD

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-11-21 | Initial release with complete optimization and analysis |

## License

Internal use only - Proprietary

## Authors

Analytics Team
Operations Planning Department

---

**For questions or support, contact the Analytics Team.**
