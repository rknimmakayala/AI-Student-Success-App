# Business Requirements Document: Refinery Linear Programming Optimization Model

**Document Version:** 1.0
**Date:** November 21, 2025
**Project:** Refinery Production Optimization System
**Document Owner:** Operations Planning Team

---

## Executive Summary

This document outlines the business requirements for developing a Linear Programming (LP) optimization model to maximize profitability of refinery operations. The model will optimize the production mix of gasoline and diesel from crude oil processing, subject to operational constraints and market conditions.

### Strategic Objectives
- Maximize daily profit from refinery operations
- Optimize product mix (gasoline vs. diesel) based on market prices
- Ensure efficient utilization of refinery capacity
- Enable scenario planning and what-if analysis for strategic decision-making

---

## 1. Business Context

### 1.1 Background
Our refinery processes crude oil to produce gasoline and diesel. Current production decisions are made based on historical patterns and manual analysis, leading to suboptimal profit realization. An LP optimization model will enable data-driven, optimal production planning.

### 1.2 Business Problem
- **Challenge:** Determining the optimal daily production quantities for gasoline and diesel to maximize profit
- **Impact:** Suboptimal production mix results in estimated 5-15% profit leakage
- **Opportunity:** Systematic optimization can increase annual profits by $5M-$15M

### 1.3 Stakeholders
- **Primary:** Operations Planning Team, Plant Manager
- **Secondary:** Finance Department, Sales & Marketing, Supply Chain
- **Executive Sponsor:** Chief Operating Officer

---

## 2. Functional Requirements

### 2.1 Optimization Model Requirements

#### FR-1: Crude Oil Processing Capacity
- **Description:** Model must enforce daily crude oil processing capacity limit
- **Specification:** Maximum 100,000 barrels per day
- **Priority:** Critical
- **Rationale:** Physical constraint of refinery infrastructure

#### FR-2: Product Yield Coefficients
- **Description:** Model must account for conversion yields from crude to products
- **Specifications:**
  - Gasoline yield: 0.45 barrels per barrel of crude (45% conversion)
  - Diesel yield: 0.35 barrels per barrel of crude (35% conversion)
  - Residual/Loss: 0.20 (20% other products and losses)
- **Priority:** Critical
- **Rationale:** Based on refinery technical specifications and historical data

#### FR-3: Product Demand Constraints
- **Description:** Model must respect maximum daily demand for each product
- **Specifications:**
  - Maximum gasoline demand: 50,000 barrels per day
  - Maximum diesel demand: 40,000 barrels per day
- **Priority:** High
- **Rationale:** Market demand ceiling; overproduction leads to storage costs

#### FR-4: Minimum Production Requirements
- **Description:** Model must ensure minimum production to meet contractual obligations
- **Specifications:**
  - Minimum gasoline production: 10,000 barrels per day
  - Minimum diesel production: 5,000 barrels per day
- **Priority:** High
- **Rationale:** Long-term supply contracts with key customers

#### FR-5: Profit Maximization Objective
- **Description:** Model must maximize total daily profit
- **Formula:** Profit = (Gasoline Revenue + Diesel Revenue) - (Crude Cost + Operating Costs)
- **Priority:** Critical
- **Rationale:** Primary business objective

### 2.2 Input Parameters

#### FR-6: Market Pricing Inputs
- **Description:** Model must accept current market prices as inputs
- **Parameters:**
  - Crude oil cost per barrel (baseline: $60/barrel)
  - Gasoline selling price per barrel (baseline: $90/barrel)
  - Diesel selling price per barrel (baseline: $85/barrel)
- **Priority:** Critical
- **Update Frequency:** Daily

#### FR-7: Operating Cost Parameters
- **Description:** Model must include variable and fixed operating costs
- **Parameters:**
  - Variable operating cost: $5 per barrel of crude processed
  - Fixed daily operating cost: $50,000
- **Priority:** High
- **Rationale:** Complete cost accounting for profitability

### 2.3 Output Requirements

#### FR-8: Optimal Production Plan
- **Description:** Model must output optimal daily production quantities
- **Outputs:**
  - Crude oil to process (barrels)
  - Gasoline to produce (barrels)
  - Diesel to produce (barrels)
  - Expected daily profit ($)
- **Priority:** Critical

#### FR-9: Solution Validation
- **Description:** Model must validate solution feasibility
- **Validations:**
  - All constraints satisfied
  - Solution is optimal (not just feasible)
  - Shadow prices for constraints
- **Priority:** High

---

## 3. Scenario Analysis Requirements

### 3.1 What-If Analysis Capabilities

#### FR-10: Price Sensitivity Analysis
- **Description:** Model must support sensitivity analysis for price variations
- **Scenarios:**
  - Crude oil price fluctuations (±20%)
  - Gasoline price changes (±15%)
  - Diesel price changes (±15%)
- **Priority:** High
- **Rationale:** Market volatility requires scenario planning

#### FR-11: Capacity Scenario Analysis
- **Description:** Model must evaluate impact of capacity changes
- **Scenarios:**
  - Capacity expansion: +20% (120,000 barrels/day)
  - Capacity reduction during maintenance: -30% (70,000 barrels/day)
  - Capacity constraints: -50% (50,000 barrels/day)
- **Priority:** High
- **Rationale:** Capital investment and maintenance planning

#### FR-12: Demand Scenario Analysis
- **Description:** Model must handle demand variation scenarios
- **Scenarios:**
  - High demand period (seasonal peak)
  - Low demand period (seasonal trough)
  - Demand shifts (gasoline vs. diesel preference changes)
- **Priority:** Medium
- **Rationale:** Seasonal demand patterns and market trends

#### FR-13: Multi-Factor Scenario Analysis
- **Description:** Model must evaluate combined scenario impacts
- **Scenarios:**
  - Best case: High prices + high demand
  - Worst case: Low prices + low demand + capacity constraints
  - Most likely case: Baseline parameters
- **Priority:** High
- **Rationale:** Comprehensive risk assessment

---

## 4. Reporting and Visualization Requirements

### 4.1 Dashboard Requirements

#### FR-14: Optimization Results Dashboard
- **Description:** Visual display of optimal production plan
- **Components:**
  - Production quantities (bar charts)
  - Profit breakdown (pie/waterfall chart)
  - Capacity utilization metrics
  - Constraint status indicators
- **Priority:** High

#### FR-15: Sensitivity Analysis Reports
- **Description:** Visual representation of scenario analysis results
- **Components:**
  - Profit sensitivity to price changes (line charts)
  - Capacity impact analysis (comparison charts)
  - Multi-scenario comparison tables
  - Tornado diagrams for sensitivity ranking
- **Priority:** High

#### FR-16: Decision Support Metrics
- **Description:** Key performance indicators for decision-making
- **Metrics:**
  - Profit per barrel of crude processed
  - Capacity utilization percentage
  - Product mix ratio (gasoline:diesel)
  - Margin per product
- **Priority:** Medium

---

## 5. Technical Requirements

### 5.1 Model Implementation

#### TR-1: Optimization Solver
- **Requirement:** Use industry-standard LP solver
- **Specification:** Python-based (PuLP, SciPy, or equivalent)
- **Priority:** Critical
- **Rationale:** Maintainability and accessibility

#### TR-2: Solution Time
- **Requirement:** Model must solve within acceptable time
- **Specification:** < 5 seconds for single optimization
- **Priority:** High
- **Rationale:** Interactive decision-making support

#### TR-3: Numerical Stability
- **Requirement:** Model must handle numerical edge cases
- **Specification:** Proper handling of infeasibility and unboundedness
- **Priority:** High

### 5.2 Data Requirements

#### TR-4: Input Data Validation
- **Requirement:** Validate all input parameters
- **Validations:**
  - Positive prices and costs
  - Capacity constraints > 0
  - Yield coefficients sum to ≤ 1.0
- **Priority:** High

#### TR-5: Historical Data Storage
- **Requirement:** Store optimization results for trending
- **Specification:** CSV/database format with timestamp
- **Priority:** Medium
- **Rationale:** Performance tracking and model validation

---

## 6. Non-Functional Requirements

### 6.1 Performance

#### NFR-1: Scalability
- **Requirement:** Model must handle additional products/constraints
- **Specification:** Extensible architecture for 5+ products
- **Priority:** Medium

#### NFR-2: Reliability
- **Requirement:** Model must produce consistent results
- **Specification:** Deterministic optimization (same inputs = same outputs)
- **Priority:** Critical

### 6.2 Usability

#### NFR-3: User Interface
- **Requirement:** Intuitive interface for non-technical users
- **Specification:** Clear parameter inputs, visual outputs
- **Priority:** High

#### NFR-4: Documentation
- **Requirement:** Comprehensive user and technical documentation
- **Components:**
  - User guide for parameter inputs
  - Technical documentation of model formulation
  - Interpretation guide for results
- **Priority:** High

### 6.3 Maintainability

#### NFR-5: Code Quality
- **Requirement:** Well-structured, documented code
- **Standards:** PEP 8 compliance, inline comments
- **Priority:** Medium

---

## 7. Constraints and Assumptions

### 7.1 Business Constraints
1. Single crude oil type (uniform quality)
2. No intermediate product storage (produce to order)
3. No product blending or quality variations
4. Deterministic demand and prices (no uncertainty modeling in v1.0)
5. Daily planning horizon (no multi-period optimization)

### 7.2 Technical Assumptions
1. Linear relationships between inputs and outputs
2. Constant yield coefficients (no quality variations)
3. No setup or changeover costs
4. Instantaneous production (no time delays)
5. Perfect information (all parameters known with certainty)

### 7.3 Limitations
1. Model does not account for:
   - Crude oil quality variations
   - Product quality specifications
   - Inventory management
   - Supply chain logistics
   - Environmental constraints
   - Workforce scheduling

---

## 8. Success Criteria

### 8.1 Model Validation
- [ ] Model solves successfully for baseline scenario
- [ ] All constraints are properly enforced
- [ ] Optimal solution verified against manual calculations
- [ ] Sensitivity analysis produces reasonable results

### 8.2 Business Value
- [ ] Model recommendations increase profit vs. current operations
- [ ] Decision-makers can interpret results without assistance
- [ ] What-if analysis supports strategic planning discussions
- [ ] Model adopted for daily production planning

### 8.3 Performance Metrics
- **Target:** 10-15% profit improvement over current planning method
- **Baseline Comparison:** Compare model results vs. last 90 days actual performance
- **Acceptance Criteria:** Model consistently recommends more profitable solutions

---

## 9. Implementation Phases

### Phase 1: Model Development (Week 1-2)
- Develop core LP optimization model
- Implement baseline scenario
- Validate against test cases

### Phase 2: Scenario Analysis (Week 3)
- Implement what-if analysis scenarios
- Develop sensitivity analysis capabilities
- Create visualization dashboards

### Phase 3: Testing & Validation (Week 4)
- User acceptance testing with operations team
- Validation against historical data
- Documentation completion

### Phase 4: Deployment (Week 5)
- Production deployment
- User training
- Monitoring and support setup

---

## 10. Risks and Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Model produces infeasible solutions | High | Low | Comprehensive input validation and constraint checking |
| Users don't trust model results | High | Medium | Transparent model explanation, validation against known scenarios |
| Price data unavailable | Medium | Low | Default to last known prices, manual override capability |
| Model too complex for users | Medium | Medium | Simplified interface, training, documentation |
| Market conditions outside model assumptions | Medium | High | Regular model review, parameter updates, disclaimer on limitations |

---

## 11. Dependencies

1. **Python environment** with optimization libraries (PuLP, NumPy, Pandas)
2. **Historical data** on crude oil prices, product prices, and production volumes
3. **Operations team** availability for requirements validation and testing
4. **IT infrastructure** for model deployment and data access

---

## 12. Approval

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Business Owner | Operations Planning Manager | _____________ | _____ |
| Technical Lead | Analytics Team Lead | _____________ | _____ |
| Executive Sponsor | Chief Operating Officer | _____________ | _____ |

---

## 13. Revision History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | 2025-11-21 | Claude | Initial BRD creation |

---

## Appendix A: Mathematical Model Formulation

### Decision Variables
- `x_crude` = Barrels of crude oil to process per day
- `x_gasoline` = Barrels of gasoline to produce per day
- `x_diesel` = Barrels of diesel to produce per day

### Objective Function (Maximize)
```
Profit = (P_gasoline × x_gasoline) + (P_diesel × x_diesel)
         - (C_crude × x_crude) - (C_var × x_crude) - C_fixed
```

Where:
- `P_gasoline` = Gasoline selling price ($/barrel)
- `P_diesel` = Diesel selling price ($/barrel)
- `C_crude` = Crude oil cost ($/barrel)
- `C_var` = Variable operating cost ($/barrel)
- `C_fixed` = Fixed daily operating cost ($)

### Constraints

1. **Production relationship constraints:**
   - `x_gasoline = 0.45 × x_crude`
   - `x_diesel = 0.35 × x_crude`

2. **Capacity constraint:**
   - `x_crude ≤ 100,000`

3. **Demand constraints:**
   - `x_gasoline ≤ 50,000`
   - `x_diesel ≤ 40,000`

4. **Minimum production constraints:**
   - `x_gasoline ≥ 10,000`
   - `x_diesel ≥ 5,000`

5. **Non-negativity:**
   - `x_crude ≥ 0`
   - `x_gasoline ≥ 0`
   - `x_diesel ≥ 0`

---

## Appendix B: Glossary

- **LP (Linear Programming):** Mathematical optimization technique for maximizing/minimizing a linear objective function subject to linear constraints
- **Yield Coefficient:** Proportion of input converted to output product
- **Shadow Price:** Marginal value of relaxing a constraint by one unit
- **Sensitivity Analysis:** Study of how output varies with changes in input parameters
- **What-If Analysis:** Evaluation of different scenarios by changing input assumptions
- **Capacity Utilization:** Percentage of maximum capacity actually used
- **Product Mix:** Proportion of different products in total production
