import os
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class StockEnv(gym.Env):
  def __init__(self, data_matrix, symbols_list, lookback_window=29, max_stocks=200, initial_capital=2000, prev_weights=None, verbose=False, training=True):
    
    self.lookback_window = lookback_window
    self.symbols = symbols_list + ['CASH'] 
    self.n_stocks = len(symbols_list)
    self.stock_names = symbols_list

    print(f"Env Setting:")
    print(f"num of assets: {self.n_stocks}")
    print(f"lookback window: {self.lookback_window}天")
    
    self.observation_space = spaces.Box(
      low=-np.inf,
      high=np.inf,
      shape=(self.n_stocks, self.lookback_window), 
      dtype=np.float32
    )
    
    self.action_space = spaces.Box(
      low=-3.0, 
      high=3.0, 
      shape=(self.n_stocks + 1,), # including cash
      dtype=np.float32
    )

    # status var
    self.current_step = 0
    self.turnover_history = []
    self.total_turnover = 0.0
    self.daily_returns = []
    self.positive_days = 0
    self.total_trading_days = 0
    self.data_matrix = data_matrix
    
    # record portfolio adjustment
    self.traded_stocks = set()
    self.stock_operation_history = {}
    self.daily_operations = []

    self.prev_weights = prev_weights
    
    # if initialization, set weights to zero.
    if prev_weights is None:
      self.portfolio_weights = np.zeros(self.n_stocks + 1)
      self.portfolio_weights[-1] = 1.0  # set cash
    else:
      self.portfolio_weights = prev_weights
      self.prev_weights = prev_weights

    self.initial_capital = initial_capital
    self.current_capital = self.initial_capital

    # train hyperparameters
    self.asset_visit_counts = np.zeros(self.n_stocks + 1)
    self.curiosity_scale = 0.1
    self.lambda_turnover = 0.12
    self.training = training

    self.verbose = verbose  

    
  def reset(self, seed=None, options=None):
    super().reset(seed=seed)

    self.current_step = 0

    if self.prev_weights is None:
      self.portfolio_weights = np.zeros(self.n_stocks + 1)
      self.portfolio_weights[-1] = 1.0  # 100% cash
    else: 
      self.portfolio_weights = self.prev_weights
    
    current_sample = self.data_matrix[self.current_step]
    observation = current_sample[:, :self.lookback_window]
        
    self.current_capital = self.initial_capital
    
    # reset stat
    self.turnover_history = []
    self.total_turnover = 0.0
    self.daily_returns = []
    self.positive_days = 0
    self.total_trading_days = 0

    self.traded_stocks = set()
    self.stock_operation_history = {}
    self.daily_operations = []

    return self.get_observation(), {}

  def get_observation(self):
    current_sample = self.data_matrix[self.current_step]
    observation = current_sample[:, :self.lookback_window]

    return observation


  def step(self, action):
    # Record the weights before the portfolio adjustment
    pre_trade_weights = self.portfolio_weights.copy()
    
    # Execute adjustment
    action_probs = np.exp(action)
    action_probs = action_probs / np.sum(action_probs)

    # Choose top k
    k = min(7, self.n_stocks + 1)
    top_k_indices = np.argsort(action_probs)[-k:]
    
    new_weights = np.zeros(self.n_stocks + 1) 
    new_weights[top_k_indices] = action_probs[top_k_indices]
    new_weights = new_weights / np.sum(new_weights)

    new_stock_weights = new_weights[:-1] 
    new_cash_weight = new_weights[-1]   

    # Fetch current weights (excluding cash)
    current_stock_weights = pre_trade_weights[:-1].copy()

    proposed_turnover = np.sum(np.abs(new_stock_weights - current_stock_weights)) / 2
    MIN_TURNOVER_THRESHOLD = 0.02
    
    if proposed_turnover < MIN_TURNOVER_THRESHOLD:
      post_trade_weights = pre_trade_weights.copy() # remain prev weights
      step_turnover = 0.0
      today_operations = []
      transaction_cost = 0.0
        
    else:
      post_trade_weights = new_weights.copy() # using targeted weights
      step_turnover = proposed_turnover
      
      # record operations
      today_operations = self._record_stock_operations(pre_trade_weights, post_trade_weights)
      
      transaction_cost = 0 #self._calculate_transaction_cost(pre_trade_weights, post_trade_weights)
        
    self.current_step += 1
    current_returns = self._get_current_returns()

    # Calculate the daily earnings based on the weights after portfolio adjustment
    portfolio_return = np.sum(post_trade_weights * current_returns)
    
    after_return_values = post_trade_weights * (1 + current_returns)
    total_value = np.sum(after_return_values)
    
    # update capital
    self.portfolio_weights = after_return_values / total_value
    self.current_capital *= total_value

    
    self.turnover_history.append(step_turnover)
    self.total_turnover += step_turnover

    
    
    # record stat
    self.daily_returns.append(portfolio_return)
    self.return_std = np.std(self.daily_returns) if len(self.daily_returns) > 1 else 1.0
    self.total_trading_days += 1
    if portfolio_return > 0:
        self.positive_days += 1
    
    current_win_rate = self.positive_days / self.total_trading_days if self.total_trading_days > 0 else 0
    
    total_weight = np.sum(self.portfolio_weights)
    investment_ratio = 1 - (self.portfolio_weights[-1] / total_weight) if total_weight > 0 else 0
    
    # reward design
    reward = 0
    
    reward += self._calculate_reward(portfolio_return, step_turnover, new_weights=self.portfolio_weights)

    # end condition
    max_steps = len(self.data_matrix) - 1
    done = (self.current_step >= max_steps or 
            self.current_capital <= 0.1 * self.initial_capital)
    
    
    info = {
      'capital': self.current_capital,
      'return': (self.current_capital - self.initial_capital) / self.initial_capital,
      'portfolio_weights': self.portfolio_weights.copy(),
      'current_returns': current_returns,
      'transaction_cost': transaction_cost,
      'old_weights': pre_trade_weights,
      'step': self.current_step,
      'step_turnover': step_turnover,
      'total_turnover': self.total_turnover,
      'avg_turnover': self.total_turnover / max(1, self.current_step),
      'daily_return': portfolio_return,
      'win_rate': current_win_rate,
      'positive_days': self.positive_days,
      'total_trading_days': self.total_trading_days,
      'consecutive_wins': self._get_consecutive_wins(),
      'max_consecutive_wins': self._get_max_consecutive_wins(),
      'max_consecutive_losses': self._get_max_consecutive_losses(),
      'traded_stocks_count': len(self.traded_stocks),
      'today_operations': today_operations,
      'total_operations': sum(len(ops) for ops in self.stock_operation_history.values()),
      'cash_ratio': self.portfolio_weights[-1],
      'investment_ratio': 1 - self.portfolio_weights[-1],
    }

    # verbose输出
    if self.verbose:
      rebalancing_changes = post_trade_weights - pre_trade_weights
      has_active_rebalancing = np.any(np.abs(rebalancing_changes) > 1e-6)
          
      if has_active_rebalancing:
        print(f"\n=== Step {self.current_step}: Active portfolio adjustment ===")
        
        if today_operations:
            print("📋 Trading stocks today:")
            for stock_name, op_type, weight_change in today_operations:
                print(f"  📈 {stock_name}: {op_type} ({weight_change:+.6f})")
        
        
        cash_change = rebalancing_changes[-1]
        if abs(cash_change) > 0.05:
            print(f"💰 Cash adjustment: {pre_trade_weights[-1]:.1%} → {post_trade_weights[-1]:.1%} ({cash_change:+.1%})")
        
        print(f"\n💼 调仓后持仓:")
        for i, weight in enumerate(self.portfolio_weights[:-1]):
            if weight > 1e-4:
                stock_name = self.stock_names[i]
                return_today = current_returns[i]
                print(f"  📈 {stock_name}: {weight:.4f} (return rate today: {return_today:+.6f})")
        print(f"  💰 cash: {self.portfolio_weights[-1]:.4f}")
        
        print(f"\n⚡ trading stat:")
        print(f"  📊 turnover: {step_turnover:.4f}")
        print(f"  💸 transaction cost: {transaction_cost:.6f}")
        print(f"  📈 portfolio return: {portfolio_return:+.6f}")
        print(f"  🎯 win rate: {current_win_rate:.2%}")
        print(f"  🏢 Number of assets operated: {len(self.traded_stocks)}/{self.n_stocks + 1}")
        
        print("=" * 70)
      elif self.current_step % 50 == 0:
        print(f"\n--- Step {self.current_step}: market volatility ---")
        holding_count = sum(1 for weight in self.portfolio_weights[:-1] if weight > 1e-4)
        print(f"  Holding counts: {holding_count}/{self.n_stocks + 1}")
        print(f"  investment ratio: {investment_ratio:.1%}")
        print(f"  portfolio return: {portfolio_return:+.6f}")
        print(f"  cumulative return: {(self.current_capital/self.initial_capital-1):.4f}")

    return self.get_observation(), reward, done, False, info
  
  def _calculate_reward(self, portfolio_return, step_turnover, new_weights):
    
    reward = portfolio_return / (self.return_std + 1e-8)
    
    reward -= self.lambda_turnover * step_turnover
    
    if self.training:
      r_curiosity = self._asset_curiosity_reward(new_weights)
      reward += self.curiosity_scale * r_curiosity
    
    return reward

  def _asset_curiosity_reward(self, new_weights, threshold=0.01):
    
    reward = 0.0
    for i, w in enumerate(new_weights):
        if w > threshold:
            n = self.asset_visit_counts[i]
            reward += 1.0 / (np.sqrt(n) + 1)
            self.asset_visit_counts[i] += 1
    return reward

  def _record_stock_operations(self, old_weights, new_weights):
    today_operations = []
    
    for i, (old_weight, new_weight) in enumerate(zip(old_weights[:-1], new_weights[:-1])):
      weight_change = new_weight - old_weight
      
      if abs(weight_change) > 1e-6: 
        stock_name = self.stock_names[i]
        
        if old_weight == 0 and new_weight > 0:
            operation_type = "Buy"
        elif old_weight > 0 and new_weight == 0:
            operation_type = "Close Position"
        elif old_weight > 0 and new_weight > old_weight:
            operation_type = "Add Position"
        elif old_weight > 0 and new_weight < old_weight:
            operation_type = "Reduce Position"
        else:
            operation_type = "Adjust"
        
        # record info
        operation_info = {
            'step': self.current_step,
            'type': operation_type,
            'old_weight': old_weight,
            'new_weight': new_weight,
            'change': weight_change
        }
        
        if stock_name not in self.stock_operation_history:
            self.stock_operation_history[stock_name] = []
        self.stock_operation_history[stock_name].append(operation_info)
        
        self.traded_stocks.add(stock_name)
        today_operations.append((stock_name, operation_type, weight_change))
    
    
    if today_operations:
      self.daily_operations.append({
          'step': self.current_step,
          'operations': today_operations
      })
      
      return today_operations
  
  def _get_current_returns(self):

    # data_matrix[step] shape: (num_stocks, 30)
    current_sample = self.data_matrix[self.current_step]
    stock_returns = current_sample[:, -1]  # (num_stocks,)

    # cash return rate 0.0
    current_returns = np.append(stock_returns, 0.0)
    return current_returns
      

  def _get_consecutive_wins(self):
    if not self.daily_returns:
        return 0
    
    consecutive = 0
    for ret in reversed(self.daily_returns):
        if ret > 0:
            consecutive += 1
        else:
            break
    return consecutive

  def _get_max_consecutive_wins(self):
    if not self.daily_returns:
        return 0
    
    max_wins = 0
    current_wins = 0
    
    for ret in self.daily_returns:
        if ret > 0:
            current_wins += 1
            max_wins = max(max_wins, current_wins)
        else:
            current_wins = 0
    
    return max_wins

  def _get_max_consecutive_losses(self):
    if not self.daily_returns:
        return 0
    
    max_losses = 0
    current_losses = 0
    
    for ret in self.daily_returns:
        if ret < 0:
            current_losses += 1
            max_losses = max(max_losses, current_losses)
        else:
            current_losses = 0
    
    return max_losses

  def _calculate_benchmark_return(self, current_returns, top_k=5):
    stock_returns = current_returns[:-1]
    
    actual_k = min(top_k, len(stock_returns))
    
    if actual_k == 0:
        return 0.0
    
    top_k_returns = np.partition(stock_returns, -actual_k)[-actual_k:]
  
    benchmark_return = np.mean(top_k_returns)
    
    return benchmark_return
  