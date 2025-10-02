import os
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


def prepare_stocks_from_list(data_folder_path, only_stocks=True):
    """
    Read all CSVs, keeping all original features
    """
    stock_data = {}
    all_dataframes = {}
    min_length = float('inf')
    csv_files = [f for f in os.listdir(data_folder_path) if f.endswith('.csv')]
    
    for csv_file in csv_files:
        file_path = os.path.join(data_folder_path, csv_file)
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        stock_name = csv_file.replace('.csv', '').replace('_stock_processed', '').replace('_processed', '')
        
        all_dataframes[stock_name] = df
        min_length = min(min_length, len(df))
    
    # Truncate to the same length
    for name, df in all_dataframes.items():
        stock_data[name] = df.tail(min_length).copy()

    return stock_data


class SimpleUSStockEnv(gym.Env): 
    def __init__(self, stock_data_dict, stock_features=['market_uptrend', 'market_volatility', 'momentum_10d_percentile', 'RSI', 'volatility_22d', 'standard_price', 'momentum_5d'], #涨跌幅 , 'vc_5d_mean', 
                max_stocks=200, initial_capital=2000, prev_weights=None, verbose=False): # ['standard_price', 'rsi', 'rsi_50', 'rsi_5d_trend', 'overbought', 'oversold', 'momentum_5d']
        super().__init__()
        
        self.stock_data_dict = stock_data_dict
        self.stock_features = stock_features  # Unified feature configuration
        
        print(f"📊 Loading assets: {len(stock_data_dict)}")
        
        # Limit number of stocks
        stock_names = list(self.stock_data_dict.keys())
        if len(stock_names) > max_stocks:
            print(f"Stock count {len(stock_names)} exceeds limit {max_stocks}. Truncating to first {max_stocks}.")
            stock_names = stock_names[:max_stocks]
        
        self.stock_names = stock_names
        self.n_stocks = len(stock_names)
        
        # Calculate total features
        total_features = len(self.stock_features) * self.n_stocks
        
        # Preprocess features
        self.processed_data = self._preprocess_features()
        
       # Observation dimension = total features + position weights
        obs_dim = total_features + self.n_stocks + 1
        
        # Get data length
        data_length = len(list(self.processed_data.values())[0])
        
        print(f"🎯 US Stock Environment Setup:")
        print(f"  Asset count: {self.n_stocks}")
        print(f"  Features per asset: {len(self.stock_features)}")
        print(f"  Total features: {total_features}")
        print(f"  Observation dimension: {obs_dim}")
        print(f"  Effective data length: {data_length} days")
        print(f"  Using features: {self.stock_features}")
        
        self.observation_space = spaces.Box(
            low=-5.0,  
            high=5.0, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-3.0, 
            high=3.0, 
            shape=(self.n_stocks + 1,)
        )
        
        # State variables
        self.current_step = 0
        self.turnover_history = []
        self.total_turnover = 0.0
        self.daily_returns = []
        self.positive_days = 0
        self.total_trading_days = 0
        self.benchmark_beats = 0

        # Stock operation statistics
        self.traded_stocks = set()
        self.stock_operation_history = {}
        self.daily_operations = []

        self.prev_weights = prev_weights
        if prev_weights is None:
            self.portfolio_weights = np.zeros(self.n_stocks + 1)
            self.portfolio_weights[-1] = 1.0  # 100% cash
        else:
            self.portfolio_weights = prev_weights
            self.prev_weights = prev_weights

        self.initial_capital = initial_capital
        self.current_capital = self.initial_capital
        self.verbose = verbose  

    def _preprocess_features(self):
        """Extract unified features"""
        feature_data = {}
        
        for stock in self.stock_names:
            stock_df = self.stock_data_dict[stock]
            stock_features = []
            
            for feature_name in self.stock_features:
                if feature_name in stock_df.columns:
                    # Check and handle NaN values
                    feature = stock_df[feature_name].fillna(0).values
                    stock_features.append(feature)
                else:
                    # If feature does not exist, fill with zeros
                    print(f"⚠️ Warning: Asset {stock} missing feature {feature_name}. Filling with zeros.")
                    stock_features.append(np.zeros(len(stock_df)))
            
            # Stack all features for this stock
            if stock_features:
                feature_data[stock] = np.column_stack(stock_features)
            else:
                # If no features, provide zero features
                feature_data[stock] = np.zeros((len(stock_df), len(self.stock_features)))
        
        return feature_data

    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
    
        self.current_step = 0
        if self.prev_weights is None:
            self.portfolio_weights = np.zeros(self.n_stocks + 1)
            self.portfolio_weights[-1] = 1.0  # 100% cash
        else: 
            self.portfolio_weights = self.prev_weights
            
        self.current_capital = self.initial_capital
        
        # Reset all statistics
        self.turnover_history = []
        self.total_turnover = 0.0
        self.daily_returns = []
        self.positive_days = 0
        self.total_trading_days = 0
        self.benchmark_beats = 0
        self.traded_stocks = set()
        self.stock_operation_history = {}
        self.daily_operations = []
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Get current observation (features + position info)"""
        observations = []
        
        # Collect current features for all stocks
        for stock in self.stock_names:
            stock_current_feature = self.processed_data[stock][self.current_step]
            observations.append(stock_current_feature)
        
        # Add current position (weights) information
        observations.append(self.portfolio_weights)
        
        # Concatenate all observations
        full_observation = np.concatenate(observations)
        
        return full_observation.astype(np.float32)
    
    def _record_stock_operations(self, old_weights, new_weights):
        """Record stock operations"""
        today_operations = []
        
        for i, (old_weight, new_weight) in enumerate(zip(old_weights[:-1], new_weights[:-1])):
            weight_change = new_weight - old_weight
            
            if abs(weight_change) > 1e-6: # Actual weight change occurred
                stock_name = self.stock_names[i]
                
                # Record operation type
                if old_weight == 0 and new_weight > 0:
                    operation_type = "Buy"
                elif old_weight > 0 and new_weight == 0:
                    operation_type = "Sell-All"
                elif old_weight > 0 and new_weight > old_weight:
                    operation_type = "Increase"
                elif old_weight > 0 and new_weight < old_weight:
                    operation_type = "Decrease"
                else:
                    operation_type = "Adjust"
                
                # Record operation info
                operation_info = {
                    'step': self.current_step,
                    'type': operation_type,
                    'old_weight': old_weight,
                    'new_weight': new_weight,
                    'change': weight_change
                }
                
                # Add to history
                if stock_name not in self.stock_operation_history:
                    self.stock_operation_history[stock_name] = []
                self.stock_operation_history[stock_name].append(operation_info)
                
                # Add to traded stocks set
                self.traded_stocks.add(stock_name)
                today_operations.append((stock_name, operation_type, weight_change))
        
        # Record today's operations
        if today_operations:
            self.daily_operations.append({
                'step': self.current_step,
                'operations': today_operations
            })
        
        return today_operations
    
    def step(self, action):
        # Record weights before rebalancing
        pre_trade_weights = self.portfolio_weights.copy()
        
        
        action_probs = np.exp(action)
        action_probs = action_probs / np.sum(action_probs)

        # Select top k from stocks + cash
        k = min(5, self.n_stocks + 1)
        top_k_indices = np.argsort(action_probs)[-k:]
        
        new_weights = np.zeros(self.n_stocks + 1) 
        new_weights[top_k_indices] = action_probs[top_k_indices]
        new_weights = new_weights / np.sum(new_weights)

        new_stock_weights = new_weights[:-1] 
        new_cash_weight = new_weights[-1]   

         # Get current stock weights (excluding cash)
        current_stock_weights = pre_trade_weights[:-1].copy()

        proposed_turnover = np.sum(np.abs(new_stock_weights - current_stock_weights)) / 2
        MIN_TURNOVER_THRESHOLD = 0.02
        
        if proposed_turnover < MIN_TURNOVER_THRESHOLD:
            # no rebalancing
            post_trade_weights = pre_trade_weights.copy() 
            step_turnover = 0.0
            today_operations = []
            transaction_cost = 0.0
            
        else:
            # Execute rebalancing
            post_trade_weights = new_weights.copy()
            step_turnover = proposed_turnover
            
            # Record stock operations
            today_operations = self._record_stock_operations(pre_trade_weights, post_trade_weights)
            # Calculate transaction cost
            transaction_cost = 0 #self._calculate_transaction_cost(pre_trade_weights, post_trade_weights)
            


        # Advance time, get current day's returns
        self.current_step += 1
        current_returns = self._get_current_returns()

        # Calculate portfolio return based on post-trade weights
        portfolio_return = np.sum(post_trade_weights * current_returns)
        
        # After market close, asset values update based on returns
        after_return_values = post_trade_weights * (1 + current_returns)
        total_value = np.sum(after_return_values)
        
        # Normalize weights and update capital
        self.portfolio_weights = after_return_values / total_value
        self.current_capital *= total_value

        
        self.turnover_history.append(step_turnover)
        self.total_turnover += step_turnover

        
        
        # Record win rate data
        self.daily_returns.append(portfolio_return)
        self.total_trading_days += 1
        if portfolio_return > 0:
            self.positive_days += 1
        
        current_win_rate = self.positive_days / self.total_trading_days if self.total_trading_days > 0 else 0
        
        total_weight = np.sum(self.portfolio_weights)
        investment_ratio = 1 - (self.portfolio_weights[-1] / total_weight) if total_weight > 0 else 0
        
        # Reward design
        reward = 0
        
        # 1. Base reward: encourage stable positive returns
        reward += 200 * portfolio_return
        
        # 2. Penalize excessive turnover
        if step_turnover > 0.1:
            reward -= step_turnover * 8

        # 3. Encourage investment (being in the market)
        if portfolio_return > 0.0:
            reward += investment_ratio * 1.5

        benchmark_return = self._calculate_benchmark_return(current_returns, top_k=5)
        
        # Calculate "regret": difference between portfolio return and optimal benchmark
        regret = portfolio_return - benchmark_return
        
        # Add "regret" (multiplied by a factor) to the total reward.
        # If portfolio underperforms benchmark (negative regret), it's a penalty.
        reward += regret * 10.0

        # Check termination condition
        max_steps = len(list(self.processed_data.values())[0]) - 1
        done = (self.current_step >= max_steps or 
                self.current_capital <= 0.1 * self.initial_capital)
        
        # Additional info
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

        # Verbose output
        if self.verbose:
            rebalancing_changes = post_trade_weights - pre_trade_weights
            has_active_rebalancing = np.any(np.abs(rebalancing_changes) > 1e-6)
                
            if has_active_rebalancing:
                print(f"\n=== Step {self.current_step}: Active Rebalancing ===")
                
                if today_operations:
                    print("📋 Today's Operations:")
                    for stock_name, op_type, weight_change in today_operations:
                        print(f"  📈 {stock_name}: {op_type} ({weight_change:+.6f})")
                
                # Cash position change
                cash_change = rebalancing_changes[-1]
                if abs(cash_change) > 0.05:
                    print(f"💰 Cash Position Change: {pre_trade_weights[-1]:.1%} → {post_trade_weights[-1]:.1%} ({cash_change:+.1%})")
                
                print(f"\n💼 Holdings After Rebalance:")
                for i, weight in enumerate(self.portfolio_weights[:-1]):
                    if weight > 1e-4:
                        stock_name = self.stock_names[i]
                        return_today = current_returns[i]
                        print(f"  📈 {stock_name}: {weight:.4f} (Today's Return: {return_today:+.6f})")
                print(f"  💰 Cash: {self.portfolio_weights[-1]:.4f}")
                
                print(f"\n⚡ Trading Stats:")
                print(f"  📊 Turnover: {step_turnover:.4f}")
                print(f"  💸 Transaction Cost: {transaction_cost:.6f}")
                print(f"  📈 Portfolio Return: {portfolio_return:+.6f}")
                print(f"  🎯 Win Rate: {current_win_rate:.2%}")
                print(f"  🏢 Traded Assets: {len(self.traded_stocks)}/{self.n_stocks}")
                
                print("=" * 70)
            elif self.current_step % 50 == 0:
                print(f"\n--- Step {self.current_step}: Market Fluctuation ---")
                holding_count = sum(1 for weight in self.portfolio_weights[:-1] if weight > 1e-4)
                print(f"  Holding Count:{holding_count}/{self.n_stocks}")
                print(f"  Investment Ratio: {investment_ratio:.1%}")
                print(f"  Today's Return: {portfolio_return:+.6f}")
                print(f"  Cumulative Return: {(self.current_capital/self.initial_capital-1):.4f}")

        return self._get_observation(), reward, done, False, info
    

    def _calculate_transaction_cost(self, old_weights, new_weights, fee_rate=0.001):
        """Calculate transaction cost (fee)"""
        turnover = np.sum(np.abs(new_weights[:-1] - old_weights[:-1]))
        transaction_cost = turnover * fee_rate
        return transaction_cost
    
    def _get_current_returns(self):
        """Get returns for all stocks at the current step"""
        current_returns = []
        
        for stock in self.stock_names:
            stock_df = self.stock_data_dict[stock]
            stock_return = stock_df['daily_return'].iloc[self.current_step]
            current_returns.append(stock_return)
        
        # Cash return is 0
        cash_return = 0.0
        current_returns.append(cash_return)
        
        return np.array(current_returns)
        

    def _get_consecutive_wins(self):
        """Calculate current consecutive winning days"""
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
        """Calculate max consecutive winning days"""
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
        """Calculate max consecutive losing days"""
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
        """
        Calculate the average return of the top_k best-performing assets as a benchmark.
        """
        # Exclude the last return (cash, which is always 0) from today's returns
        stock_returns = current_returns[:-1]
        
        # Handle cases where asset count is less than top_k
        actual_k = min(top_k, len(stock_returns))
        
        if actual_k == 0:
            return 0.0
        
        # Find the k highest returns. np.partition is faster than a full sort.
        top_k_returns = np.partition(stock_returns, -actual_k)[-actual_k:]
        
        # Calculate the mean of these top_k returns
        benchmark_return = np.mean(top_k_returns)
        
        return benchmark_return