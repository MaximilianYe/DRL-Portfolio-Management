# Hierarchical Multi-Agent RL Trading System

A reinforcement learning-based portfolio optimization system built on a hierarchical multi-agent architecture. Multiple regime-specialized worker agents each learn distinct trading policies via PPO, while a softmax manager dynamically aggregates their actions based on inferred market state — enabling adaptive multi-asset allocation across varying market conditions.

## Architecture

```
                    ┌─────────────────────┐
                    │   Market Data        │
                    │   (yfinance)         │
                    └─────────┬───────────┘
                              ▼
                    ┌─────────────────────┐
                    │ Feature Engineering  │
                    │ 30-day return seqs   │
                    └─────────┬───────────┘
                              ▼
                    ┌─────────────────────┐
                    │    GRU Encoder       │
                    │ Temporal embedding   │
                    └─────────┬───────────┘
                              ▼
               ┌─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┐
               │  Worker Agents (N TBD)     │
               │ ┌────────┐     ┌────────┐ │
               │ │Agent 1 │ ... │Agent N │ │
               │ │ (PPO)  │     │ (PPO)  │ │
               │ └───┬────┘     └───┬────┘ │
               └─ ─ ─┼─ ─ ─ ─ ─ ─ ─┼─ ─ ─┘
                      ▼             ▼
                    ┌─────────────────────┐
                    │   Softmax Manager   │
                    │ Weighted aggregation │
                    └─────────┬───────────┘
                              ▼
                    ┌─────────────────────┐
                    │ Portfolio Allocation │
                    └─────────────────────┘
```

**Key design decisions:**

- **GRU over MLP** — Sequential 30-day return windows capture temporal dependencies that flat feature vectors miss. The GRU encoder produces a compact state embedding shared across all worker agents.
- **Hierarchical multi-agent** — Instead of a single monolithic policy, multiple worker agents specialize under different market conditions. The number and specialization strategy of workers is an active design question.
- **Softmax manager** — Learns a soft weighting over worker outputs conditioned on market state, avoiding hard regime switches and enabling smooth policy transitions.


## Tech Stack

| Component | Tool |
|---|---|
| RL framework | Stable Baselines3 (PPO) |
| Temporal model | PyTorch GRU |
| Data source | yfinance |
| Training env | Google Colab (GPU) |
| Reproducibility | cuDNN deterministic mode, global seeding |

## Project Structure

```
.
├── IA0/                    # Archived: MLP + RL initial 
├── IA1/                    # Current system (GRU + hierarchical MARL)
│   ├── agents/             # Worker agent definitions
│   ├── envs.py             # Custom Gym trading environment
│   └── model.py            # Training
└── README.md
```

> **`IA0/`** archives the first iteration of this project — an MLP-based RL trading system. That phase focused on foundational problems: designing a realistic trading environment, handling dependent action spaces (portfolio weights must sum to 1), and getting PPO to converge on basic allocation tasks. Lessons from IA0 directly informed the current GRU-based architecture.

## Evolution

| Phase | Approach | Key challenges addressed | Status |
|---|---|---|---|
| IA0 | MLP + PPO | Env design, dependent action space, basic convergence | ✅ Archived |
| Current | GRU + hierarchical MARL | Temporal modeling, regime specialization, manager design | 🔧 In progress |

## Open Design Questions

- **How many worker agents?** Exploring whether regime-based specialization (bull/bear/sideways) or other decomposition strategies (asset-class, volatility-level) yield better results.
- **Manager architecture** — Softmax weighting is the current approach; alternatives include attention-based gating or learned mixture-of-experts.
- **Reward shaping** — Balancing raw returns against risk-adjusted metrics (Sharpe, max drawdown penalties).

## Roadmap

- [ ] Complete GRU encoder integration with SB3 custom policy
- [ ] Ablation: single-agent vs multi-agent baseline comparison
- [ ] Transaction cost modeling in reward function
- [ ] Walk-forward validation framework
- [ ] Live paper trading evaluation

## License

MIT
