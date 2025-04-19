This project began from a personal question: How could I better allocate my limited capital — from part-time work and internship pay — between ETFs and individual stocks to balance long-term safety and short-term growth? 

I came up with a simple LP model inspired by Math482 for single-day portfolio optimization and progressively evolves into a multi-period, data-driven strategy guided by historical trends and market state modeling.

### Phase 1: Linear Programming Allocation
- Select one ETF and one individual stock from candidate sets
- Use LP to determine how to invest under budget constraints

    - ### LP Model Formulation
    Objective:   Maximize returns r_1 x_1 + r_2 x_2

    Constraints:  
      x_1 + x_2 <= 3400 (my budget)  
      x_1, x_2 >= 0 (long-only, no short-selling)
      risk(x1, x2) <= RiskTolerance (future work)


### Phase 2: Intelligent Selection
- Use past data to determine which ETF/stock pair to pick
- Incorporate simple rules or classifiers

### Phase 3: State Modeling (Markov)
- Model market as state transitions (bull, bear, volatile)
- Select assets based on predicted market state

### Phase 4: Multi-period Optimization (Reinforcement Learning)
- Treat asset allocation as a sequential decision process
- Use MDP/Q-learning to learn adaptive strategy



### Status & Roadmap
This project is in its early design/documentation phase.

- [x] Defined LP model structure
- [x] Implement data loading + ETF/stock selection interface
        _Utilized AKShare to fetch historical prices of selected assets._
- [ ] Add performance logging + result plots
- [ ] (Future) Implement Markov-based state modeling
- [ ] (Future) Reinforcement Learning simulator


### 📊 Data Source

This project uses [AKShare](https://github.com/akfamily/akshare) as the data provider for financial market data.  
AKShare is an open-source Python package designed for academic and research use.

[![Data: akshare](https://img.shields.io/badge/Data%20Science-AKShare-green)](https://github.com/akfamily/akshare)
