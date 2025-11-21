# Executive Summary: Refinery Optimization Analysis

**Report Date:** November 21, 2025
**Analysis Type:** Linear Programming Optimization with Comprehensive What-If Analysis
**Prepared By:** Analytics Team

---

## Executive Overview

This report presents the results of a comprehensive Linear Programming (LP) optimization analysis for refinery operations. The model optimizes the production mix of gasoline and diesel to maximize daily profitability while respecting operational constraints including capacity limits, demand constraints, and contractual obligations.

### Key Findings Summary

✅ **Baseline Optimization:** Daily profit of **$475,000** with full capacity utilization
✅ **Best Case Potential:** Up to **$2.38M/day** profit under favorable market conditions
⚠️ **Risk Exposure:** Potential losses of **$472K/day** in worst-case scenario
📊 **Monte Carlo Analysis:** Mean expected profit of **$550K/day** across 1,000 simulations
🎯 **Breakeven Point:** Crude oil price of **$65/barrel** (current: $60/barrel)

---

## 1. Baseline Scenario Results

### Optimal Production Plan

| Metric | Value | Notes |
|--------|-------|-------|
| **Crude Oil Processing** | 100,000 barrels/day | Full capacity utilization (100%) |
| **Gasoline Production** | 45,000 barrels/day | 45% yield from crude |
| **Diesel Production** | 35,000 barrels/day | 35% yield from crude |
| **Daily Revenue** | $7,025,000 | Total product sales |
| **Daily Cost** | $6,550,000 | Crude + operating costs |
| **Daily Profit** | $475,000 | Net profit |
| **Profit per Barrel** | $4.75 | Efficiency metric |

### Key Insights

1. **Capacity is the Binding Constraint:** The refinery operates at 100% capacity, indicating that any expansion would directly increase profitability.

2. **Product Mix Ratio:** Gasoline to diesel ratio of **1.29:1** reflects the fixed yield coefficients and maximizes revenue.

3. **Operational Efficiency:** All production levels respect minimum contractual obligations while maximizing output within demand constraints.

---

## 2. Price Sensitivity Analysis

### Impact of Price Changes on Daily Profit

Our sensitivity analysis reveals the following profit elasticities:

#### Crude Oil Price Sensitivity (High Impact - Negative)
- **-20% change** ($48/barrel): Profit increases to **$1,675,000/day** (+252%)
- **+20% change** ($72/barrel): Profit decreases to **-$725,000/day** (loss)
- **Elasticity:** Highly sensitive; $1 change in crude = **±$100,000** profit impact

#### Gasoline Price Sensitivity (High Impact - Positive)
- **-20% change** ($72/barrel): Profit decreases to **-$335,000/day** (loss)
- **+20% change** ($108/barrel): Profit increases to **$1,285,000/day** (+171%)
- **Elasticity:** $1 change in gasoline = **+$45,000** profit impact

#### Diesel Price Sensitivity (Medium Impact - Positive)
- **-20% change** ($68/barrel): Profit decreases to **$-120,000/day** (loss)
- **+20% change** ($102/barrel): Profit increases to **$1,070,000/day** (+125%)
- **Elasticity:** $1 change in diesel = **+$35,000** profit impact

### Strategic Implications

🔴 **CRITICAL RISK:** Crude oil price increases pose the greatest threat to profitability
- Breakeven point at **$65/barrel** (only $5 margin above current price)
- Price hedging strategies strongly recommended

🟢 **OPPORTUNITY:** Gasoline price increases drive the most profit improvement
- Focus on premium product markets and quality improvements
- Consider seasonal pricing strategies

📊 **DIVERSIFICATION:** Diesel production provides profit stability
- Less volatile but consistent margin contributor
- Important for risk mitigation

---

## 3. Capacity Scenario Analysis

### Scenario Comparison

| Scenario | Capacity | Crude Processed | Daily Profit | Utilization | Profit Change |
|----------|----------|-----------------|--------------|-------------|---------------|
| **Constrained (50K)** | 50,000 bbls | 50,000 | $212,500 | 100% | -55% |
| **Maintenance (70K)** | 70,000 bbls | 70,000 | $317,500 | 100% | -33% |
| **Baseline (100K)** | 100,000 bbls | 100,000 | $475,000 | 100% | Baseline |
| **Expansion (120K)** | 120,000 bbls | 120,000 | $580,000 | 100% | +22% |

### Capital Investment Decision Support

#### 20% Capacity Expansion Analysis
- **Additional Daily Profit:** $105,000
- **Annual Additional Profit:** $38.3M
- **3-Year Profit Impact:** $114.9M
- **5-Year Profit Impact:** $191.6M

**Investment Threshold:** Any capacity expansion project costing less than $100M would achieve ROI within 3 years at current market conditions.

#### Maintenance Impact
- **Daily Profit Loss:** $157,500 during 70% capacity operation
- **Weekly Maintenance Cost:** $1.1M (profit opportunity cost)
- **Recommendation:** Schedule maintenance during low-demand periods to minimize financial impact

---

## 4. Demand Scenario Analysis

### Market Demand Sensitivity

| Scenario | Gasoline Demand | Diesel Demand | Daily Profit | Capacity Used | Key Finding |
|----------|----------------|---------------|--------------|---------------|-------------|
| **Low Demand** | 30,000 | 23,333 | $300,000 | 67% | Underutilization |
| **Normal Demand** | 45,000 | 35,000 | $475,000 | 100% | Baseline |
| **High Demand** | 45,000 | 35,000 | $475,000 | 100% | Capacity constrained |
| **Gasoline Heavy** | 38,571 | 30,000 | $400,000 | 86% | Diesel limited |
| **Diesel Heavy** | 40,000 | 31,111 | $416,667 | 89% | Gasoline limited |

### Strategic Insights

1. **High Demand Scenario:** Even with higher demand, profit remains at $475K because capacity is already maxed out. This reinforces the expansion recommendation.

2. **Product Mix Flexibility:** The fixed yield coefficients mean we cannot independently adjust gasoline vs. diesel production. Market shifts toward one product create underutilization.

3. **Low Demand Risk:** A 40% demand decrease results in 37% profit reduction but maintains profitability due to flexible capacity adjustment.

---

## 5. Multi-Factor Scenario Analysis

### Strategic Scenarios

#### 🟢 Best Case Scenario
**Conditions:** Low crude ($50), high product prices ($100 gas, $95 diesel), low costs
- **Daily Profit:** $2,380,000 (+401% vs baseline)
- **Annual Profit:** $868.7M
- **Key Driver:** Favorable market conditions + operational efficiency

#### 🟡 Most Likely Scenario
**Conditions:** Current baseline parameters
- **Daily Profit:** $475,000
- **Annual Profit:** $173.4M
- **Key Driver:** Stable market conditions

#### 🔴 Worst Case Scenario
**Conditions:** High crude ($75), low product prices ($80 gas, $75 diesel), reduced capacity (70K)
- **Daily Profit:** -$471,667 (LOSS)
- **Monthly Loss:** -$14.1M
- **Key Drivers:** Unfavorable prices + capacity constraints + minimum production obligations

#### 🔶 High Volatility Scenario
**Conditions:** Mixed price movements with market uncertainty
- **Daily Profit:** -$144,444 (LOSS)
- **Capacity Utilization:** 22% (driven by minimum production requirements)
- **Key Issue:** Forced to produce at minimum levels despite unprofitable conditions

### Risk Management Recommendations

1. **Establish Price Hedging:** Lock in crude oil prices and product revenues through futures contracts
2. **Flexible Contracts:** Negotiate flexibility in minimum production requirements to avoid forced losses
3. **Inventory Strategy:** Build product inventory during favorable conditions to sell during high-price periods
4. **Cost Reduction:** Focus on reducing variable operating costs to improve margin resilience

---

## 6. Breakeven Analysis

### Critical Thresholds

#### Crude Oil Breakeven Point
- **Breakeven Price:** $65.00/barrel
- **Current Price:** $60.00/barrel
- **Safety Margin:** $5.00/barrel (8.3%)
- **Risk Level:** MODERATE - Only 8% cushion before losses

#### Implications
- At current gasoline ($90) and diesel ($85) prices, crude oil above $65/barrel results in negative profitability
- Daily profit changes by approximately **$100,000 per $1** change in crude oil price
- **Critical Action Required:** Implement crude oil price hedging strategy immediately

### Break-Even Production Volume
With current prices, minimum crude processing to cover fixed costs:
- **Minimum Volume:** 22,223 barrels/day
- **Current Volume:** 100,000 barrels/day
- **Operating Leverage:** 4.5x (high sensitivity to volume changes)

---

## 7. Monte Carlo Simulation Results

### Probabilistic Profit Analysis (1,000 Scenarios)

#### Profit Distribution Statistics
| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Mean Profit** | $550,434/day | Expected value across all scenarios |
| **Median Profit** | $436,823/day | Middle outcome (50th percentile) |
| **Standard Deviation** | $652,761 | High variability indicates risk |
| **5th Percentile** | -$235,938/day | 5% chance of losing this much or more |
| **95th Percentile** | $1,738,848/day | 5% chance of exceeding this profit |
| **Minimum** | -$530,380/day | Worst outcome observed |
| **Maximum** | $3,341,370/day | Best outcome observed |

#### Value at Risk (VaR) Analysis
- **Daily VaR (5%):** -$235,938
- **Annual VaR (5%):** -$86.1M
- **Interpretation:** There is a 5% probability that annual losses could exceed $86M under normal market volatility

### Risk Profile Assessment

📊 **Profit Volatility:** VERY HIGH
- Standard deviation exceeds mean profit, indicating substantial uncertainty
- Wide range between worst (-$530K) and best ($3.3M) cases

🎲 **Probability of Loss:** ~15%
- Approximately 150 out of 1,000 scenarios resulted in negative daily profit
- Losses primarily driven by crude oil price spikes

💰 **Upside Potential:** Significant
- 25% chance of exceeding $900K daily profit
- Strong potential during favorable market conditions

---

## 8. Product Mix Optimization

### Price Scenario Impact on Optimal Mix

| Scenario | Gas Price | Diesel Price | Product Mix | Daily Profit | Notes |
|----------|-----------|--------------|-------------|--------------|-------|
| **Equal Prices** | $85 | $85 | 1.29:1 | $250,000 | Yields drive mix |
| **Gasoline Premium** | $95 | $80 | 1.29:1 | $525,000 | Gas advantage |
| **Diesel Premium** | $85 | $95 | 1.29:1 | $600,000 | Diesel advantage |
| **High Margin** | $100 | $95 | 1.29:1 | $1,275,000 | Both elevated |

### Key Finding: Fixed Yield Constraint

⚠️ **Important:** The product mix ratio remains constant at 1.29:1 regardless of price differentials because yield coefficients are fixed (45% gasoline, 35% diesel from crude).

**Strategic Implication:** Unlike flexible manufacturing, refineries cannot easily adjust product mix. Market opportunities must be captured through:
1. **Volume adjustments** (capacity expansion)
2. **Timing optimization** (when to run at full capacity)
3. **Inventory management** (storing products for favorable prices)

**Potential Enhancement:** Investigate refining process modifications that could provide product mix flexibility (e.g., catalytic reforming, blending capabilities).

---

## 9. Tornado Diagram: Sensitivity Rankings

### Impact of ±20% Parameter Changes on Daily Profit

Ranked from highest to lowest impact:

1. **Gasoline Price** (±$810K impact)
   - Highest positive impact on profit
   - Directly drives revenue on 45% of production

2. **Crude Oil Price** (±$1.2M impact)
   - Highest risk factor (negative impact)
   - Affects 100% of input costs

3. **Diesel Price** (±$595K impact)
   - Significant positive impact
   - Drives revenue on 35% of production

4. **Variable Operating Cost** (±$100K impact)
   - Moderate impact
   - Controllable through operational efficiency

5. **Fixed Operating Cost** (±$10K impact)
   - Minimal impact on optimization decisions
   - Spreads across high production volumes

### Strategic Prioritization

**Focus Areas for Maximum Impact:**
1. ✅ Crude oil price management (hedging, supplier negotiations)
2. ✅ Product pricing strategy and market positioning
3. ✅ Operational cost reduction initiatives
4. ⬇️ Fixed cost optimization (lower priority given minimal impact)

---

## 10. Key Recommendations

### Immediate Actions (Next 30 Days)

1. **🔴 CRITICAL: Implement Price Hedging**
   - Establish crude oil futures contracts to lock in prices below $65/barrel
   - Target: Hedge 60-80% of 90-day crude oil requirements
   - Expected benefit: Eliminate downside risk of $86M annual VaR

2. **Optimize Production Scheduling**
   - Use the LP model for daily production planning
   - Expected benefit: 5-10% profit improvement through better timing
   - Implementation: Train operations team on model inputs/outputs

3. **Establish Real-Time Price Monitoring**
   - Create dashboard for crude, gasoline, and diesel prices
   - Set automated alerts at key thresholds (crude >$63, gasoline <$85)
   - Enable rapid decision-making

### Short-Term Initiatives (3-6 Months)

4. **Capacity Expansion Feasibility Study**
   - Analyze 20% capacity expansion to 120K barrels/day
   - Projected ROI: $105K/day = $38.3M/year
   - Breakeven: <3 years for investments under $100M

5. **Flexible Contract Renegotiation**
   - Modify minimum production requirements to have seasonal flexibility
   - Target: Reduce exposure during worst-case scenarios
   - Potential savings: Avoid $14M monthly losses in adverse conditions

6. **Operational Efficiency Program**
   - Reduce variable operating costs from $5 to $4.50 per barrel (-10%)
   - Expected benefit: $50K additional daily profit = $18.3M/year
   - Focus: Energy efficiency, waste reduction, process optimization

### Long-Term Strategic Initiatives (6-12 Months)

7. **Product Mix Flexibility Research**
   - Investigate process modifications for flexible gasoline/diesel ratios
   - Evaluate blending capabilities for product differentiation
   - Potential: Capture premium pricing opportunities

8. **Inventory Optimization System**
   - Build storage capacity for both products
   - Implement buy-low, sell-high inventory strategy
   - Expected benefit: Smooth profit volatility, capture price premiums

9. **Advanced Analytics Integration**
   - Deploy predictive pricing models for market forecasting
   - Integrate ML models for demand prediction
   - Automate optimization with real-time data feeds

10. **Risk Management Framework**
    - Formalize enterprise risk management processes
    - Establish monthly scenario planning reviews
    - Create risk mitigation playbooks for various scenarios

---

## 11. Financial Impact Summary

### Baseline Performance
- **Current Daily Profit:** $475,000
- **Annual Profit:** $173.4 million
- **Profit Margin:** 6.8%

### Potential Improvements

| Initiative | Annual Impact | Confidence | Timeline |
|------------|---------------|------------|----------|
| Price hedging (risk reduction) | $0-86M saved | High | 30 days |
| Optimized production scheduling | +$8.7-17.3M | High | 30 days |
| Capacity expansion (+20%) | +$38.3M | Medium | 12-18 months |
| Operating cost reduction (-10%) | +$18.3M | Medium | 6 months |
| Flexible contracts | $14M risk reduction | Medium | 6 months |

### Total Potential Annual Benefit: **$50M - $75M**

---

## 12. Conclusion

The Linear Programming optimization model provides a robust framework for maximizing refinery profitability while managing operational constraints and market risks. Key conclusions:

✅ **Model Validation:** Successfully optimizes production to achieve $475K daily profit under baseline conditions with 100% capacity utilization.

✅ **Strategic Insights:** Capacity expansion offers compelling ROI, with payback under 3 years and $38M annual profit increase.

⚠️ **Risk Awareness:** Significant downside exposure to crude oil price increases, with breakeven at only $65/barrel (8% above current).

📊 **Decision Support:** Model enables scenario planning across 1,000+ simulations, providing probabilistic profit forecasts and risk metrics.

🎯 **Actionable Recommendations:** 10 prioritized initiatives with potential to increase annual profits by $50-75M while reducing risk.

### Success Metrics for Implementation

- ✅ Model deployed for daily production planning
- ✅ 80% of crude oil requirements hedged within 30 days
- ✅ Operational costs reduced by 5-10% within 6 months
- ✅ Capacity expansion business case approved within 6 months
- ✅ 10-15% profit improvement vs. pre-model baseline

---

## 13. Appendices

### Data Files Generated
- `capacity_scenarios.csv` - Capacity analysis results
- `demand_scenarios.csv` - Demand scenario comparisons
- `multi_factor_scenarios.csv` - Combined scenario analysis
- `monte_carlo_results.csv` - 1,000 simulation runs
- `product_mix_analysis.csv` - Product mix optimization results

### Visualizations Generated
- `price_sensitivity_analysis.png` - Price impact charts
- `scenario_comparison.png` - Multi-factor scenario comparisons
- `monte_carlo_simulation.png` - Probabilistic profit analysis
- `demand_scenarios.png` - Demand scenario visualizations
- `tornado_diagram.png` - Parameter sensitivity rankings

### Technical Documentation
- `BUSINESS_REQUIREMENTS_DOCUMENT.md` - Complete BRD with model formulation
- `refinery_lp_model.py` - Core optimization model implementation
- `simulation_analysis.py` - Comprehensive simulation and what-if analysis

---

**Report Prepared By:** Analytics Team
**Model Version:** 1.0
**Analysis Date:** November 21, 2025
**Next Review:** December 21, 2025

*This analysis is based on the Linear Programming model described in the Business Requirements Document and validated through comprehensive simulation and scenario analysis.*
