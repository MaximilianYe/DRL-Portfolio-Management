import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import torch
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import copy
import torch
from envs.env_v1_cont_act_sp import SimpleUSStockEnv

def set_seed(seed):
    """
    Set all relevant random seeds to ensure experiment reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"✅ All random seeds have been set to: {seed}")


def incremental_learning(model, new_stock_data, features_config, timesteps=10000, period_idx=None, prev_weights=None, prev_capital=None):
    """Incremental learning function - simple save version"""
    print(f"🎓 Starting incremental learning with {len(new_stock_data)} assets for {timesteps} steps")
    
    # Create a new environment (with new data)
    learn_env = DummyVecEnv([lambda: SimpleUSStockEnv(
        stock_data_dict=new_stock_data,
        stock_features=features_config,  # Now using simplified config
        max_stocks=200
    )])

    n_envs = model.n_envs
    def make_env():
        return SimpleUSStockEnv(
            stock_data_dict=new_stock_data,
            stock_features=features_config,
            max_stocks=200,
            initial_capital=prev_capital,
            prev_weights=prev_weights,
            verbose=False
        )
    
    model.verbose = 0

    # Create n_envs environments
    learn_env = DummyVecEnv([make_env for _ in range(n_envs)])
    
    # Set the new environment
    model.set_env(learn_env)
    
    # Adjust learning rate (use a smaller Lr for incremental learning)
    # original_lr = model.learning_rate
    original_lr_schedule = model.lr_schedule
    new_lr = 3e-4
    
    # Create a constant learning rate schedule function
    def constant_lr_schedule(progress_remaining):
        return new_lr
    
    # Set the new learning rate schedule
    model.lr_schedule = constant_lr_schedule
    model.learning_rate = new_lr
    
    
    # Continue training
    model.learn(total_timesteps=timesteps, progress_bar=False)
    
    # Restore original learning rate
    # model.learning_rate = original_lr
    
    learn_env.close()
    
    if period_idx is not None:
        save_dir = "models/us_stock_online_models"
        os.makedirs(save_dir, exist_ok=True)
        
        model_path = os.path.join(save_dir, f"period_{period_idx+1}_model.zip")
        model.save(model_path)
        print(f"  Period {period_idx+1} model saved: {model_path}")
    
    print(f"✅ Incremental learning complete")
    
    return model

def prepare_stocks_excluding_benchmark(data_folder_path, exclude_files=['510300_benchmark.csv'], max_stocks=200):
    """Load stock data, exclude benchmark files, and limit stock count"""
    stock_data = {}
    all_dataframes = {}
    min_length = float('inf')
    
    if not os.path.exists(data_folder_path):
        raise FileNotFoundError(f"Data folder not found: {data_folder_path}")
    
    csv_files = [f for f in os.listdir(data_folder_path) 
                 if f.endswith('.csv') and f not in exclude_files]
    
    print(f"Found {len(csv_files)} asset files")
    
    # Asset filtering logic
    filtered_csv_files = []
    for csv_file in csv_files:
        stock_name = csv_file.replace('.csv', '')
        asset_type = _determine_single_asset_type(stock_name)
        # if asset_type == 'futures':
        filtered_csv_files.append(csv_file)
        # else:
        #     print(f"skip non-future assets: {stock_name} (type: {asset_type})")
    
    csv_files = filtered_csv_files
    
    
    if len(csv_files) > max_stocks:
        print(f"Asset count exceeds limit {max_stocks}, selecting first {max_stocks} assets")
        csv_files = csv_files[:max_stocks]
    
    print(f"Actually using {len(csv_files)} assets for evaluation")
    
    for csv_file in csv_files:
        file_path = os.path.join(data_folder_path, csv_file)
        try:
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
            stock_name = csv_file.replace('.csv', '')
            all_dataframes[stock_name] = df
            min_length = min(min_length, len(df))
        except Exception as e:
            print(f"❌ Failed to read file {csv_file}: {e}")
            continue
    
    print(f"Shortest data length: {min_length} days")
    
    # Truncate to the same length
    for name, df in all_dataframes.items():
        stock_data[name] = df.tail(min_length).copy()

    return stock_data

def _determine_single_asset_type(stock_name):
    """Determine the type of a single asset"""
    if '.OF' in stock_name:
        return 'bond_fund'
    elif 'CB' in stock_name:
        return 'convertible_bond'
    elif 'C' in stock_name or 'M' in stock_name:
        return 'stock'  
    elif 'F' in stock_name:
        return 'futures'
    elif 'O' in stock_name:
        return 'option'
    else:
        return 'stock'

def evaluate_model_in_period(model, eval_data, start_idx, end_idx, features_config, prev_weights=None, initial_capital=3400):
    """Evaluate model performance over a specified period"""
    print(f"  Evaluating period: {start_idx} to {end_idx-1} (Total {end_idx-start_idx} days)")
    
    daily_returns_list = []
    period_length = end_idx - start_idx
    if period_length < 2:
        print(f"  Skipping short period: Day {start_idx+1}-{end_idx} (Only {period_length} days, cannot calculate return)")
        
        # Return default results
        return {
            'period_return': 0.0,
            'final_capital': initial_capital,
            'final_weights': prev_weights if prev_weights is not None else np.zeros(len(eval_data) + 1),
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'daily_returns': [],
            'capital_curve': [initial_capital],
            'action_stats': {'active_trades': 0, 'total_periods': 0, 'total_stock_trades': 0, 'avg_positions': 0},
            'profit_loss_ratio': 0.0,
            'step_count': 0,
            'annualized_turnover': 0.0,
            'trading_stats': {'traded_assets': set(), 'traded_count': 0, 'total_operations': 0, 'operation_history': {}}
        }
    
    #  Prepare data for this period
    period_data = {}
    sample_dates = None
    
    for asset_name, asset_df in eval_data.items():
        period_df = asset_df.iloc[start_idx:end_idx].copy()
        period_data[asset_name] = period_df
        
        if sample_dates is None:
            sample_dates = period_df.index
    
    print(f"  Trading dates: {sample_dates[0]} to {sample_dates[-1]}")
    
    # Create environment
    eval_env = SimpleUSStockEnv(
        stock_data_dict=period_data,
        stock_features=features_config,
        max_stocks=200,
        initial_capital=initial_capital,
        prev_weights=prev_weights,
        verbose=True
    )
    
    obs, _ = eval_env.reset()

    
    # Record data
    capital_list = [eval_env.current_capital]
    daily_returns_list = []
    turnover_list = []
    
    action_stats = {
        'active_trades': 0,
        'total_periods': 0,
        'total_stock_trades': 0,
        'avg_positions': 0
    }
    
    step_count = 0
    prev_weights_step = eval_env.portfolio_weights.copy()
    
    # Start trading loop
    while True:
        # Predict action
        action, _ = model.predict(obs, deterministic=True)
        
        # Record weights before rebalancing
        weights_before = eval_env.portfolio_weights.copy()
        
        # Execute action
        obs, reward, done, truncated, info = eval_env.step(action)
        
        # Record weights after rebalancing
        weights_after = info['portfolio_weights']
        
        # Analyze weight changes
        if prev_weights_step is not None:
            weight_changes = np.abs(weights_after[:-1] - prev_weights_step[:-1])
            significant_changes = weight_changes > 0.01
            n_trades = np.sum(significant_changes)
            
            if n_trades > 0:
                action_stats['active_trades'] += 1
                action_stats['total_stock_trades'] += n_trades
            
            action_stats['total_periods'] += 1
            
            # Count active positions
            active_positions = np.sum(weights_after[:-1] > 0.01)
            action_stats['avg_positions'] = (
                action_stats.get('avg_positions', 0) * step_count + active_positions
            ) / (step_count + 1)
        
        prev_weights_step = weights_after.copy()
        
        # Record data
        capital_list.append(info['capital'])
        daily_returns_list.append(info['daily_return'])
        turnover_list.append(info['step_turnover'])
        print(f"daily return {info['daily_return']}")
        step_count += 1
        
        if done or truncated:
            break

    single_day_max_loss = abs(min(daily_returns_list)) if daily_returns_list else 0

    # Calculate period statistics
    final_capital = capital_list[-1]
    period_return = (final_capital / initial_capital) - 1
    
    # Win Rate
    positive_days = sum(1 for ret in daily_returns_list if ret > 0)
    total_days = len(daily_returns_list)
    win_rate = positive_days / total_days if total_days > 0 else 0
    
    # Sharpe Ratio
    if len(daily_returns_list) > 1:
        returns_mean = np.mean(daily_returns_list)
        returns_std = np.std(daily_returns_list, ddof=1)
        if returns_std > 0:
            risk_free_rate = 0.03 / 250
            daily_sharpe = (returns_mean - risk_free_rate) / returns_std
            annualized_sharpe = daily_sharpe * np.sqrt(250)
        else:
            annualized_sharpe = 0
    else:
        annualized_sharpe = 0
    
    # Turnover Rate
    avg_turnover = np.mean(turnover_list) if turnover_list else 0
    annualized_turnover = avg_turnover * 250
    

    
    
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    current_wins = 0
    current_losses = 0
    
    # Profit/Loss Ratio
    for ret in daily_returns_list:
        if ret > 0:
            current_wins += 1
            current_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, current_wins)
        elif ret < 0:
            current_losses += 1
            current_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, current_losses)
        else:
            current_wins = 0
            current_losses = 0
        
    # Calculate average win and average loss     
    winning_days = [ret for ret in daily_returns_list if ret > 0]
    losing_days = [ret for ret in daily_returns_list if ret < 0]
    
    avg_win = np.mean(winning_days) if winning_days else 0
    avg_loss = np.mean(losing_days) if losing_days else 0
    profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')


    traded_assets = eval_env.traded_stocks  # Assets traded this period
    trading_summary = {
        'traded_assets': traded_assets,
        'traded_count': len(traded_assets),
        'total_operations': sum(len(ops) for ops in eval_env.stock_operation_history.values()),
        'operation_history': eval_env.stock_operation_history
    }


    print(f" Period performance: Return {period_return*100:.2f}%, Win Rate {win_rate:.1f}, Sharpe {annualized_sharpe :.2f}, Annualized Turnover {annualized_turnover*100:.0f}%")
    
    eval_env.close()
    
    return {
        'period_return': period_return,
        'final_capital': final_capital,
        'final_weights': weights_after,
        'win_rate': win_rate,
        'sharpe_ratio': annualized_sharpe,
        'daily_returns': daily_returns_list,
        'capital_curve': capital_list,
        'action_stats': action_stats,
        'profit_loss_ratio': profit_loss_ratio,
        'step_count': step_count,
        'annualized_turnover': annualized_turnover,
        'trading_stats': trading_summary,
        'single_day_max_loss': single_day_max_loss
    }

def split_data_by_periods(eval_data, period_length=22):
    """Split data into periods"""
    sample_stock = list(eval_data.values())[0]
    total_days = len(sample_stock)
    
    periods = []
    start_idx = 0
    
    while start_idx < total_days:
        end_idx = min(start_idx + period_length, total_days)
        periods.append((start_idx, end_idx))
        start_idx = end_idx
    
    return periods

def calculate_individual_asset_returns(eval_data, test_days):
    """Calculate returns for each asset - enhanced version"""
    asset_returns = {}
    
    for asset_name, asset_df in eval_data.items():
        try:
            if 'daily_return' in asset_df.columns: 
                returns_data = asset_df['daily_return'].head(test_days)
            elif 'daily_returns' in asset_df.columns:
                returns_data = asset_df['daily_returns'].head(test_days)
            elif 'returns' in asset_df.columns:
                returns_data = asset_df['returns'].head(test_days)
            else:
                continue
            
            # Basic statistics
            cumulative_return = (1 + returns_data).cumprod().iloc[-1] - 1
            positive_days = (returns_data > 0).sum()
            total_days = len(returns_data)
            win_rate = positive_days / total_days if total_days > 0 else 0
            
            #  Sharpe Ratio calculation
            if len(returns_data) > 1:
                returns_mean = returns_data.mean()
                returns_std = returns_data.std(ddof=1)
                if returns_std > 0:
                    risk_free_rate = 0.03 / 250  # Daily risk-free rate
                    daily_sharpe = (returns_mean - risk_free_rate) / returns_std
                    annualized_sharpe = daily_sharpe * np.sqrt(250)
                else:
                    annualized_sharpe = 0.0
            else:
                annualized_sharpe = 0.0
            
            #  Max Drawdown and Calmar Ratio calculation
            cumulative_values = (1 + returns_data).cumprod()
            peak = cumulative_values.expanding().max()
            drawdowns = (peak - cumulative_values) / peak
            max_drawdown = drawdowns.max()
            

            single_day_max_loss = abs(returns_data.min()) if len(returns_data) > 0 else 0
            single_day_max_gain = returns_data.max() if len(returns_data) > 0 else 0
            
            
            # Annualized return
            if total_days > 0:
                annualized_return = (1 + cumulative_return) ** (250 / total_days) - 1
            else:
                annualized_return = 0.0
                
            # Calmar Ratio
            if max_drawdown > 0:
                calmar_ratio = annualized_return / max_drawdown
            else:
                calmar_ratio = float('inf') if annualized_return > 0 else 0.0
            
            # Win rate after filtering out zero-return days
            non_zero_returns = returns_data[returns_data != 0.0]
            if len(non_zero_returns) > 0:
                non_zero_positive = (non_zero_returns > 0).sum()
                win_rate_no_zero = non_zero_positive / len(non_zero_returns)
            else:
                win_rate_no_zero = 0.0
            
            asset_returns[asset_name] = {
                'cumulative_return': cumulative_return * 100,
                'annualized_return': annualized_return * 100,
                'win_rate': win_rate * 100,
                'win_rate_no_zero': win_rate_no_zero * 100,
                'sharpe_ratio': annualized_sharpe,
                'max_drawdown': max_drawdown * 100,
                'calmar_ratio': calmar_ratio,
                'total_days': total_days,
                'positive_days': positive_days,
                'zero_days': (returns_data == 0.0).sum(),
                'volatility': returns_data.std() * np.sqrt(250) * 100,  # Annualized volatility
                'single_day_max_loss': single_day_max_loss * 100,  # Convert to percentage
                'single_day_max_gain': single_day_max_gain * 100,
            }
            
        except Exception as e:
            print(f" Error calculating returns for asset {asset_name}: {e}")
            continue
    
    return asset_returns

def print_asset_performance_summary(asset_returns, top_n=30):
    """Print asset performance summary - fixed format version"""
    if not asset_returns:
        print(" No asset return data")
        return
    
    print("\n" + "="*120)
    print(" Individual Asset Detailed Analysis (vs. Model Performance)")
    print("="*120)
    
    sorted_assets = sorted(asset_returns.items(), key=lambda x: x[1]['cumulative_return'], reverse=True)
    
    # Count valid assets (excluding anomalies)
    valid_assets = []
    for name, stats in sorted_assets:
        # Exclude assets with 0 return and anomalous Calmar ratio
        if not (stats['cumulative_return'] == 0.0 and stats['calmar_ratio'] in [0.0, float('inf')]):
            valid_assets.append((name, stats))
    
    print(f"  Asset Quality Overview:")
    print(f"  Total Assets: {len(sorted_assets)}")
    print(f"  Valid Assets: {len(valid_assets)} (excluding anomalies)")
    print(f"  Anomalous Assets: {len(sorted_assets) - len(valid_assets)}")
    
    # Calculate asset pool statistics
    if valid_assets:
        all_returns = [stats['cumulative_return'] for _, stats in valid_assets]
        all_sharpe = [stats['sharpe_ratio'] for _, stats in valid_assets if stats['sharpe_ratio'] != 0.0]
        all_calmar = [stats['calmar_ratio'] for _, stats in valid_assets 
                     if stats['calmar_ratio'] not in [0.0, float('inf')]]
        all_win_rates = [stats['win_rate_no_zero'] for _, stats in valid_assets]
        all_drawdowns = [stats['max_drawdown'] for _, stats in valid_assets]
        
        positive_assets = sum(1 for ret in all_returns if ret > 0)
        
        print(f"\n  Asset Pool Statistics (Valid Assets):")
        print(f"  Profitable Asset Ratio: {positive_assets}/{len(valid_assets)} ({positive_assets/len(valid_assets)*100:.1f}%)")

        # Return Statistics
        print(f"    Return Statistics:")
        print(f"    Average Cumulative Return: {np.mean(all_returns):.2f}%")
        print(f"    Median Return: {np.median(all_returns):.2f}%")
        print(f"    Best Return: {max(all_returns):.2f}%")
        print(f"    Worst Return: {min(all_returns):.2f}%")
        print(f"    Return Standard Deviation: {np.std(all_returns):.2f}%")
        
        # Sharpe Ratio Statistics
        if all_sharpe:
            positive_sharpe = sum(1 for s in all_sharpe if s > 0)
            print(f"    Sharpe Ratio Statistics:")
            print(f"    Average Sharpe Ratio: {np.mean(all_sharpe):.3f}")
            print(f"    Median Sharpe Ratio: {np.median(all_sharpe):.3f}")
            print(f"    Best Sharpe Ratio: {max(all_sharpe):.3f}")
            print(f"    Worst Sharpe Ratio: {min(all_sharpe):.3f}")
            print(f"    Assets with Positive Sharpe: {positive_sharpe}/{len(all_sharpe)} ({positive_sharpe/len(all_sharpe)*100:.1f}%)")
        
        if all_calmar:
            larger_than_one_calmar = sum(1 for c in all_calmar if c > 1.0)
            positive_calmar = sum(1 for c in all_calmar if c > 0.0)
            print(f"    Calmar Ratio Statistics:")
            print(f"    Average Calmar Ratio: {np.mean(all_calmar):.3f}")
            print(f"    Median Calmar Ratio: {np.median(all_calmar):.3f}")
            print(f"    Best Calmar Ratio: {max(all_calmar):.3f}")
            print(f"    Worst Calmar Ratio: {min(all_calmar):.3f}")
            print(f"    Calmar Ratio > 1.0: {larger_than_one_calmar}/{positive_calmar} ({(larger_than_one_calmar/positive_calmar)*100:.1f}%)")
        
        # Win Rate Statistics
        if all_win_rates:
            print(f"    Win Rate Statistics:")
            print(f"    Average Win Rate: {np.mean(all_win_rates):.1f}%")
            print(f"    Median Win Rate: {np.median(all_win_rates):.1f}%")
            print(f"    Best Win Rate: {max(all_win_rates):.1f}%")
            print(f"    Worst Win Rate: {min(all_win_rates):.1f}%")
        
        # Drawdown Statistics
        if all_drawdowns:
            print(f"    Max Drawdown Statistics:")
            print(f"    Average Max Drawdown: {np.mean(all_drawdowns):.2f}%")
            print(f"    Median Max Drawdown: {np.median(all_drawdowns):.2f}%")
            print(f"    Smallest Max Drawdown: {min(all_drawdowns):.2f}%")
            print(f"    Largest Max Drawdown: {max(all_drawdowns):.2f}%")
    
    # Detailed Table Display - Fixed columns
    print(f"\n  Top {min(top_n, len(valid_assets))} Best Performing Assets:")
    print("-" * 130)  # Increase width
    print(f"{'Rank':<4} {'Asset Name':<40} {'Cum. Ret':<8} {'Ann. Ret':<8} {'WR':<6} {'WR*':<6} {'Sharpe':<6} {'MDD':<6} {'Calmar':<8}")  # Adjust column widths
    print("-" * 130)
    
    for i, (asset_name, stats) in enumerate(valid_assets[:top_n]):
        calmar_str = f"{stats['calmar_ratio']:.2f}" if stats['calmar_ratio'] != float('inf') else "∞"
        if len(calmar_str) > 8:  # Adjust truncation length
            calmar_str = f"{stats['calmar_ratio']:.1f}"
        
        print(f"{i+1:<4} {asset_name:<40} {stats['cumulative_return']:>6.2f}% "  # Increase asset name column width
              f"{stats['annualized_return']:>6.2f}% {stats['win_rate']:>5.1f}% "
              f"{stats['win_rate_no_zero']:>5.1f}% {stats['sharpe_ratio']:>5.2f} "
              f"{stats['max_drawdown']:>5.1f}% {calmar_str:>8}")  # Adjust Calmar column width
    
    print(f"\n  Top {min(top_n, len(valid_assets))} Worst Performing Assets:")
    print("-" * 120)
    print(f"{'Rank':<4} {'Asset Name':<35} {'Cum. Ret':<8} {'Ann. Ret':<8} {'WR':<6} {'WR*':<6} {'Sharpe':<6} {'MDD':<6} {'Calmar':<6}")
    print("-" * 120)
    
    worst_assets = valid_assets[-top_n:] if len(valid_assets) > top_n else valid_assets
    for i, (asset_name, stats) in enumerate(worst_assets):
        calmar_str = f"{stats['calmar_ratio']:.2f}" if stats['calmar_ratio'] not in [float('inf'), 0.0] else "N/A"
        if stats['calmar_ratio'] == float('inf'):
            calmar_str = "∞"
        
        print(f"{i+1:<4} {asset_name:<35} {stats['cumulative_return']:>6.2f}% "
              f"{stats['annualized_return']:>6.2f}% {stats['win_rate']:>5.1f}% "
              f"{stats['win_rate_no_zero']:>5.1f}% {stats['sharpe_ratio']:>5.2f} "
              f"{stats['max_drawdown']:>5.1f}% {calmar_str:>6}")
    
    return valid_assets  # Return valid assets for later use


def compare_model_vs_assets(model_stats, asset_returns):
    """Model vs. Asset Pool Comparison - Fixed Version"""
    print(f"\n" + "="*80)
    print("  Model vs. Asset Pool Detailed Comparison")
    print("="*80)
    
    # Calculate asset pool statistics
    valid_assets = [(name, stats) for name, stats in asset_returns.items() 
                   if not (stats['cumulative_return'] == 0.0 and stats['calmar_ratio'] in [0.0, float('inf')])]
    
    if not valid_assets:
        print("  No valid asset data for comparison")
        return
    
    # Asset pool statistics
    all_returns = [stats['cumulative_return'] for _, stats in valid_assets]
    all_sharpe = [stats['sharpe_ratio'] for _, stats in valid_assets if stats['sharpe_ratio'] != 0.0]
    all_calmar = [stats['calmar_ratio'] for _, stats in valid_assets 
                 if stats['calmar_ratio'] not in [0.0, float('inf')]]
    all_win_rates = [stats['win_rate_no_zero'] for _, stats in valid_assets]  
    all_drawdowns = [stats['max_drawdown'] for _, stats in valid_assets]      
    
    
    all_single_day_losses = [stats['single_day_max_loss'] for _, stats in valid_assets]
    
    pool_stats = {
        'avg_return': np.mean(all_returns),
        'median_return': np.median(all_returns),
        'best_return': max(all_returns),
        'worst_return': min(all_returns),
        'positive_ratio': sum(1 for ret in all_returns if ret > 0) / len(all_returns),
        'avg_sharpe': np.mean(all_sharpe) if all_sharpe else 0,
        'best_sharpe': max(all_sharpe) if all_sharpe else 0,
        'worst_sharpe': min(all_sharpe) if all_sharpe else 0,
        'avg_calmar': np.mean(all_calmar) if all_calmar else 0,
        'best_calmar': max(all_calmar) if all_calmar else 0,
        'worst_calmar': min(all_calmar) if all_calmar else 0,
        'avg_winrate': np.mean(all_win_rates),
        'best_winrate': max(all_win_rates),
        'worst_winrate': min(all_win_rates),
        'avg_drawdown': np.mean(all_drawdowns),
        'best_drawdown': min(all_drawdowns),  # Smallest drawdown is best
        'worst_drawdown': max(all_drawdowns),
        'avg_single_day_loss': np.mean(all_single_day_losses),
        'best_single_day_loss': min(all_single_day_losses),  # Smallest loss is best
        'worst_single_day_loss': max(all_single_day_losses),
    }
    
    print(f"  Comparison Results:")
    print(f"{'Metric':<20} {'Model':<15} {'Pool Avg':<15} {'Pool Best':<15} {'Pool Worst':<15} {'Model Rank':<15}")
    print("-" * 120)
    
    # === Cumulative Return Comparison ===
    model_return = model_stats['cumulative_return']
    better_than_count = sum(1 for ret in all_returns if model_return > ret)
    model_rank = f"{better_than_count}/{len(all_returns)}"
    rank_pct = f"({better_than_count/len(all_returns)*100:.1f}%)"
    
    
    print(f"{'Cumulative Return':<20} {model_return:>13.2f}% {pool_stats['avg_return']:>13.2f}% "
          f"{pool_stats['best_return']:>13.2f}% {pool_stats['worst_return']:>13.2f}% "
          f"{model_rank:>7} {rank_pct}")
    
    # Sharpe Ratio Comparison
    if all_sharpe:
        model_sharpe = model_stats['sharpe_ratio']
        better_sharpe = sum(1 for s in all_sharpe if model_sharpe > s)
        sharpe_rank = f"{better_sharpe}/{len(all_sharpe)}"
        sharpe_pct = f"({better_sharpe/len(all_sharpe)*100:.1f}%)"
        
        print(f"{'Sharpe Ratio':<20} {model_sharpe:>13.3f} {pool_stats['avg_sharpe']:>13.3f} "
              f"{pool_stats['best_sharpe']:>13.3f} {pool_stats['worst_sharpe']:>13.3f} "
              f"{sharpe_rank:>7} {sharpe_pct}")
    
    # Calmar Ratio Comparison
    if all_calmar:
        model_calmar = model_stats['calmar_ratio']
        better_calmar = sum(1 for c in all_calmar if model_calmar > c)
        calmar_rank = f"{better_calmar}/{len(all_calmar)}"
        calmar_pct = f"({better_calmar/len(all_calmar)*100:.1f}%)"
        
        print(f"{'Calmar Ratio':<20} {model_calmar:>13.3f} {pool_stats['avg_calmar']:>13.3f} "
              f"{pool_stats['best_calmar']:>13.3f} {pool_stats['worst_calmar']:>13.3f} "
              f"{calmar_rank:>7} {calmar_pct}")
    
    # Win Rate Comparison
    if 'win_rate' in model_stats:
        model_winrate = model_stats['win_rate']
        better_winrate = sum(1 for wr in all_win_rates if model_winrate > wr)
        winrate_rank = f"{better_winrate}/{len(all_win_rates)}"
        winrate_pct = f"({better_winrate/len(all_win_rates)*100:.1f}%)"
        
        print(f"{'Win Rate':<20} {model_winrate:>12.1f}% {pool_stats['avg_winrate']:>12.1f}% "
              f"{pool_stats['best_winrate']:>12.1f}% {pool_stats['worst_winrate']:>12.1f}% "
              f"{winrate_rank:>7} {winrate_pct}")
    else:
        print(f"{'Win Rate (Best/Worst)':<20} {'N/A':<15} {pool_stats['avg_winrate']:>12.1f}% "
              f"{pool_stats['best_winrate']:>12.1f}% {pool_stats['worst_winrate']:>12.1f}% {'N/A':<15}")
    
    # Drawdown Comparison
    if 'max_drawdown' in model_stats:
        model_drawdown = model_stats['max_drawdown']
        better_drawdown = sum(1 for dd in all_drawdowns if model_drawdown < dd)
        drawdown_rank = f"{better_drawdown}/{len(all_drawdowns)}"
        drawdown_pct = f"({better_drawdown/len(all_drawdowns)*100:.1f}%)"
        
        print(f"{'Max Drawdown':<20} {model_drawdown:>12.1f}% {pool_stats['avg_drawdown']:>12.1f}% "
              f"{pool_stats['best_drawdown']:>12.1f}% {pool_stats['worst_drawdown']:>12.1f}% "
              f"{drawdown_rank:>7} {drawdown_pct}")
    else:
        print(f"{'Drawdown (Min/Max)':<20} {'N/A':<15} {pool_stats['avg_drawdown']:>12.1f}% "
              f"{pool_stats['best_drawdown']:>12.1f}% {pool_stats['worst_drawdown']:>12.1f}% {'N/A':<15}")
    
    if 'single_day_max_loss' in model_stats:
        model_single_day_loss = model_stats['single_day_max_loss']
        better_single_day_loss = sum(1 for sdl in all_single_day_losses if model_single_day_loss < sdl)
        single_day_loss_rank = f"{better_single_day_loss}/{len(all_single_day_losses)}"
        single_day_loss_pct = f"({better_single_day_loss/len(all_single_day_losses)*100:.1f}%)"
        
        print(f"{'Max Single Day Loss':<20} {model_single_day_loss:>12.1f}% {pool_stats['avg_single_day_loss']:>12.1f}% "
              f"{pool_stats['best_single_day_loss']:>12.1f}% {pool_stats['worst_single_day_loss']:>12.1f}% "
              f"{single_day_loss_rank:>7} {single_day_loss_pct}")
    else:
        print(f"{'Single Day Loss (Min/Max)':<20} {'N/A':<15} {pool_stats['avg_single_day_loss']:>12.1f}% "
              f"{pool_stats['best_single_day_loss']:>12.1f}% {pool_stats['worst_single_day_loss']:>12.1f}% {'N/A':<15}")
    
   # Summary Section
    print(f"\n  Comparison Summary:")
    print(f"    Asset Pool Quality: {pool_stats['positive_ratio']:.1%} of assets had positive returns")
    print(f"    Model Relative Rank: Return beat {better_than_count/len(all_returns)*100:.1f}% of individual assets")
    
    if model_return > pool_stats['avg_return']:
        print(f"  ✅ Model Return ({model_return:.2f}%) > Pool Average ({pool_stats['avg_return']:.2f}%)")
    else:
        print(f"  ⚠️ Model Return ({model_return:.2f}%) < Pool Average ({pool_stats['avg_return']:.2f}%)")
    
    if model_return > pool_stats['median_return']:
        print(f"  ✅ Model Return ({model_return:.2f}%) > Pool Median ({pool_stats['median_return']:.2f}%)")
    else:
        print(f"  ⚠️ Model Return ({model_return:.2f}%) < Pool Median ({pool_stats['median_return']:.2f}%)")
    
    # Multi-dimensional Performance Assessment
    print(f"\n Multi-dimensional Performance Assessment:")
    print(f"   Return Performance: Ranked #{model_rank} out of {len(all_returns) - better_than_count + 1} assets")
    if all_sharpe:
        print(f"    Risk-Adjusted: Sharpe Ratio beat {better_sharpe/len(all_sharpe)*100:.1f}% of assets")
    if all_calmar:
        print(f"    Calmar Ratio: Beat {better_calmar/len(all_calmar)*100:.1f}% of assets")
    
    if 'win_rate' in model_stats:
        print(f"    Win Rate: Beat {better_winrate/len(all_win_rates)*100:.1f}% of assets")
    if 'max_drawdown' in model_stats:
        print(f"    Risk Control: Drawdown control beat {better_drawdown/len(all_drawdowns)*100:.1f}% of assets")
    
    print(f"    Asset Pool Win Rate Range: {pool_stats['worst_winrate']:.1f}% - {pool_stats['best_winrate']:.1f}%")
    print(f"    Asset Pool Drawdown Range: {pool_stats['best_drawdown']:.1f}% - {pool_stats['worst_drawdown']:.1f}%")
    return pool_stats


# === Check model_performance dictionary creation ===
# Ensure model_performance passed to compare_model_vs_assets contains cumulative_return
def debug_model_performance_creation():
    """Debug model performance dictionary creation"""
    # Your original code's issue might be here:
    
    # Check if final_result contains cumulative_return
    print(" Debugging model_performance creation:")
    print(f"final_result keys: {list(final_result.keys()) if 'final_result' in locals() else 'final_result is undefined'}")

    model_performance = {
        'cumulative_return': final_result['cumulative_return'], 
        'sharpe_ratio': annualized_sharpe,
        'calmar_ratio': calmar_ratio if 'calmar_ratio' in locals() else 0,
        'win_rate': overall_win_rate * 100,  
        'max_drawdown': max_drawdown * 100  
    }
    
    print(f"model_performance: {model_performance}")
    return model_performance


def set_seed(seed):
    """
    Set all relevant random seeds to ensure experiment reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"✅ All random seeds have been set to: {seed}")

# ==================== Main Program ====================

SEED = 42 
set_seed(SEED)


print("Loading models...")
model_paths = [
    # "models/back_up/best_model.zip"
    #"models/back_up/COIN_GE_TSLA_7_features_20230701.zip"
    # "models/us_stock_enhanced_base/test_learned_model.zip"
    # "models/us_stock_enhanced_base/AVGO_XLE_main.zip"
    "models/us_stock_enhanced_base/enhanced_base_2_model.zip",
    # "models/us_stock_enhanced_base/800k_800k.zip"
]

model_loaded = False
model = None

for model_path in model_paths:
    try:
        if os.path.exists(model_path):
            model = PPO.load(model_path, seed=SEED)
            print(f"✅ Successfully loaded continuous model from {model_path}")
            model_loaded = True
            break
    except Exception as e:
        print(f"❌ Failed to load model from {model_path}: {e}")
        continue

if not model_loaded:
    print("❌ Could not load any model, exiting")
    sys.exit(1)

print("Loading data...")
val_data_folder = 'data/processed/val' # 'data/processed/train' # 

try:
    full_eval_data = prepare_stocks_excluding_benchmark(val_data_folder, exclude_files=[], max_stocks=200)
    print(f"Asset count: {len(full_eval_data)}")
    print(f"Asset list sample: {list(full_eval_data.keys())[:5]}...")
    
except FileNotFoundError as e:
    print(f"❌ Data loading failed: {e}")
    sys.exit(1)

# Get data length
stock_length = len(list(full_eval_data.values())[0])

actual_length = stock_length

# Configuration parameters
features_config = ['market_uptrend', 'market_volatility', 'momentum_10d_percentile', 'RSI', 'volatility_22d', 'standard_price', 'momentum_5d'] # 'standard_price', 'RSI', 'rsi_50', 'rsi_5d_trend', 'overbought', 'oversold', 'momentum_5d'

#Modify
period_length = 136   # Days per period 376 #
initial_capital = 2000

# Split time periods
periods = split_data_by_periods(full_eval_data, period_length)
print(f"\n  Split into {len(periods)} periods:")

sample_stock = list(full_eval_data.values())[0]
for i, (start, end) in enumerate(periods):
    start_date = sample_stock.index[start] if start < len(sample_stock) else "Out of range"
    end_date = sample_stock.index[end-1] if (end-1) < len(sample_stock) else "Out of range"
    print(f"  Period {i+1}: Day {start+1}-{end} ({start_date} to {end_date})")

print("\n  Starting online learning backtest...")
print("="*80)

# 在线学习主循环
all_period_results = []
all_period_results_detailed = []
cumulative_capital = initial_capital
prev_weights = None

all_traded_assets = set()  # Record all traded assets
period_trading_stats = {}  # Record trading stats for each period

working_model = model  # Start with the base model

# Modify
save_dir = "models/us_stock_online_models" # "models/us_stock_enhanced_base" #

STOP_AT_PERIOD = 12 #12
for period_idx, (start_idx, end_idx) in enumerate(periods):
    print(f"\n  === Period {period_idx+1}/{len(periods)}: Day {start_idx+1}-{end_idx} ===")

    if period_idx + 1 > STOP_AT_PERIOD:
        print(f"  Reached specified stop period {STOP_AT_PERIOD}, stopping execution")
        break

    # 1. Evaluate current period
    print(f"  Current capital: {cumulative_capital:.2f}")
    if prev_weights is not None:
        cash_ratio = prev_weights[-1]
        print(f"  Inherited weights - Cash ratio: {cash_ratio:.2%}")
    
    period_result = evaluate_model_in_period(
        model=working_model,
        eval_data=full_eval_data,
        start_idx=start_idx,
        end_idx=end_idx,
        features_config=features_config,
        prev_weights=prev_weights,
        initial_capital=cumulative_capital
    )

    all_period_results_detailed.append(period_result)
    
    period_traded = period_result['trading_stats']['traded_assets']
    all_traded_assets.update(period_traded)  
    
    period_trading_stats[period_idx + 1] = {
        'traded_assets': period_traded,
        'traded_count': len(period_traded),
        'total_operations': period_result['trading_stats']['total_operations']
    }
    
    print(f"  Period {period_idx+1} Trading Stats:")
    print(f"  Assets traded this period: {len(period_traded)}")
    print(f"  Cumulative assets traded: {len(all_traded_assets)}")

    # 2. Update cumulative capital and weights
    cumulative_capital_copy = cumulative_capital
    cumulative_capital = period_result['final_capital']
    prev_weights_copy = prev_weights
    prev_weights = period_result['final_weights']
    cumulative_return = (cumulative_capital / initial_capital - 1) * 100
    
    # 3. Record results
    period_summary = {
        'period': period_idx + 1,
        'start_day': start_idx + 1,
        'end_day': end_idx,
        'period_return': period_result['period_return'] * 100,
        'cumulative_capital': cumulative_capital,
        'cumulative_return': cumulative_return,
        'win_rate': period_result['win_rate'],
        'sharpe_ratio': period_result['sharpe_ratio'],
        'annualized_turnover': period_result['annualized_turnover'] * 100
    }
    
    all_period_results.append(period_summary)

    print(f"  Period {period_idx+1} Summary:")
    print(f"  Period return: {period_result['period_return']*100:.2f}%")
    print(f"  Cumulative capital: {cumulative_capital:.2f}")
    print(f"  Cumulative return: {cumulative_return:.2f}%")
    print(f"  Win Rate: {period_result['win_rate']:.5f}")

    
    # 4. Incremental learning (except for the last period)
    if period_idx < len(periods) - 1:
        print(f"\n  Executing incremental learning...")
        
        # Prepare incremental learning data (current period + previous data)
        learn_start = max(0, start_idx)  # Include previous 10 days of data
        learn_end = end_idx
        
        incremental_data = {}
        for stock_name, stock_df in full_eval_data.items():
            incremental_data[stock_name] = stock_df.iloc[learn_start:learn_end].copy()
        
        # Execute incremental learning
        learning_steps = 250000 # min(5000, 1000 * (period_idx + 1))  # Gradually increase learning steps
        working_model = incremental_learning(
            model=working_model,
            new_stock_data=incremental_data,
            features_config=features_config,
            timesteps=learning_steps,
            period_idx=period_idx,
            prev_weights=prev_weights_copy,
            prev_capital=cumulative_capital_copy
        )
        
        print(f"✅ Incremental learning complete (Steps: {learning_steps})")

# Final Summary
print("\n" + "="*100)
print("  Online Learning Backtest Complete - Final Summary")
print("="*100)


# Modify 
final_model_path = os.path.join(save_dir, "test_learned_model.zip") #    "230101_0430_230701_240101_240401_241015.zip"     )  "enhanced_base_2_model.zip"  "enhanced_base_2_model.zip"
working_model.save(final_model_path)
print(f"  Final model saved: {final_model_path}")

# Last day
if all_period_results:
    # Use the env state from the last period
    last_period_idx = len(periods) - 1
    start_idx, end_idx = periods[last_period_idx]
    
    # Rebuild the last period's environment
    period_data = {}
    for asset_name, asset_df in full_eval_data.items():
        period_data[asset_name] = asset_df.iloc[start_idx:end_idx].copy()
    
    eval_env = SimpleUSStockEnv(
        stock_data_dict=period_data,
        stock_features=features_config,
        max_stocks=200,
        initial_capital=cumulative_capital,
        prev_weights=prev_weights,
        verbose=False
    )
    
    # Fast-forward to the last day
    obs, _ = eval_env.reset()
    max_steps = len(list(period_data.values())[0]) - 1
    for step in range(max_steps):
        action, _ = working_model.predict(obs, deterministic=True)
        obs, _, done, _, _ = eval_env.step(action)
        if done:
            break
    
    # Display current holdings
    print(f"\n  Current Holdings (After last day's close):")
    current_weights = eval_env.portfolio_weights
    stock_names = eval_env.stock_names
    
    print(f"  Stock Holdings:")
    for i, (stock_name, weight) in enumerate(zip(stock_names, current_weights[:-1])):
        if weight > 1e-4:  # Only show positions with actual holdings
            print(f"  📈 {stock_name}: {weight:.4f} ({weight*100:.2f}%)")
    
    print(f"  Cash: {current_weights[-1]:.4f} ({current_weights[-1]*100:.2f}%)")
    
    # Predict for tomorrow
    tomorrow_action, _ = working_model.predict(obs, deterministic=True)
    
    # Convert to weights
    action_probs = np.exp(tomorrow_action)
    action_probs = action_probs / np.sum(action_probs)
    
    k = min(5, len(action_probs))
    top_k_indices = np.argsort(action_probs)[-k:]
    
    tomorrow_weights = np.zeros_like(action_probs)
    tomorrow_weights[top_k_indices] = action_probs[top_k_indices]
    tomorrow_weights = tomorrow_weights / np.sum(tomorrow_weights)
    
    print(f"\n  Tomorrow's Weight Prediction:")
    print(f"  Recommended Stock Holdings:")
    for i, (stock_name, weight) in enumerate(zip(stock_names, tomorrow_weights[:-1])):
        if weight > 1e-4:  # Only show assets with actual allocation
            change = weight - current_weights[i]
            change_str = f"({change:+.4f})" if abs(change) > 1e-4 else ""
            print(f"  📈 {stock_name}: {weight:.4f} ({weight*100:.2f}%) {change_str}")
    
    print(f"  Recommended Cash: {tomorrow_weights[-1]:.4f} ({tomorrow_weights[-1]*100:.2f}%)")
    


if all_period_results:
    final_result = all_period_results[-1]
    
    print(f"\n  Overall Performance:")
    print(f"  Total periods: {len(all_period_results)}")
    print(f"  Final cumulative return: {final_result['cumulative_return']:.2f}%")
    print(f"  Final capital: {final_result['cumulative_capital']:.2f}")
    
   
    # Collect daily returns from all periods
    all_daily_returns = []
    all_period_returns = []
    all_sharpe_ratios = []
    all_turnover_rates = []
    
    for result in all_period_results:
        all_period_returns.append(result['period_return'])
        all_sharpe_ratios.append(result['sharpe_ratio'])
        all_turnover_rates.append(result['annualized_turnover'])
    
    # Collect daily returns from period_result (if available)
    # Note: This needs to be fetched from the last period's detailed data, or recalculated
    
    # Calculate overall Sharpe ratio (based on period returns)
    if len(all_period_returns) > 1:
        period_returns_array = np.array(all_period_returns) / 100  # Convert to decimal
        period_mean = np.mean(period_returns_array)
        period_std = np.std(period_returns_array, ddof=1)
        
        if period_std > 0:
            # Assuming 3% annualized risk-free rate, 22 days per period
            risk_free_period = (0.03 / 250) * 22  # Risk-free rate per period
            period_sharpe = (period_mean - risk_free_period) / period_std
            # Annualized Sharpe ratio
            periods_per_year = 250 / 22
            annualized_sharpe = period_sharpe * np.sqrt(periods_per_year)
        else:
            annualized_sharpe = 0
    else:
        annualized_sharpe = 0
    
    # Calculate overall profit/loss ratio (based on period returns)
    winning_periods = [ret for ret in all_period_returns if ret > 0]
    losing_periods = [ret for ret in all_period_returns if ret < 0]
    
    avg_winning_period = np.mean(winning_periods) if winning_periods else 0
    avg_losing_period = np.mean(losing_periods) if losing_periods else 0
    overall_profit_loss_ratio = abs(avg_winning_period / avg_losing_period) if avg_losing_period != 0 else float('inf')
    
    # Period win rate
    period_win_rate = len(winning_periods) / len(all_period_returns) if all_period_returns else 0
    
    # Max drawdown calculation
    cumulative_capitals = [initial_capital]
    for result in all_period_results:
        cumulative_capitals.append(result['cumulative_capital'])
    
    peak = cumulative_capitals[0]
    max_drawdown = 0
    for capital in cumulative_capitals:
        if capital > peak:
            peak = capital
        drawdown = (peak - capital) / peak
        max_drawdown = max(max_drawdown, drawdown)
    
    # print("DEBUG - Capital series:", cumulative_capitals)

    # negative_periods = [r for r in all_period_results if r['period_return'] < 0]
    # print(f"DEBUG - Negative return periods: {len(negative_periods)}/{len(all_period_results)}")
    # Average turnover rate
    avg_turnover = np.mean(all_turnover_rates) if all_turnover_rates else 0
    
    print(f"\n  Key Performance Indicators (KPIs):")
    print(f"    Overall Sharpe Ratio: {annualized_sharpe:.3f}")
    print(f"    Overall Profit/Loss Ratio: {overall_profit_loss_ratio:.3f}")
    print(f"    Period Win Rate: {period_win_rate:.1%} ({len(winning_periods)}/{len(all_period_returns)} periods)")
    print(f"    Max Drawdown: {max_drawdown:.2%}")
    print(f"    Average Annualized Turnover: {avg_turnover:.1f}%")
    
    # Annualized Return
    total_days = sum(result['end_day'] - result['start_day'] + 1 for result in all_period_results)
    total_return = final_result['cumulative_return'] / 100
    if total_days > 0:
        annualized_return = (1 + total_return) ** (250 / total_days) - 1
        print(f"    Annualized Return: {annualized_return:.2%}")
    
    

    
    all_daily_returns = []
    for period_result in all_period_results_detailed:
        if 'daily_returns' in period_result:
            all_daily_returns.extend(period_result['daily_returns'])
    overall_single_day_max_loss = abs(min(all_daily_returns)) if all_daily_returns else 0.0

    max_drawdown = max_drawdown if max_drawdown > 0.0001 else overall_single_day_max_loss
    # Risk-Adjusted Return
    if max_drawdown > 0.0001:  # Use a small threshold to avoid division by a tiny number
        calmar_ratio = annualized_return / max_drawdown
    elif overall_single_day_max_loss > 0:
        calmar_ratio = annualized_return / overall_single_day_max_loss
    else:
        calmar_ratio = float('inf') if annualized_return > 0 else 0.0

    if all_daily_returns:
        overall_single_day_max_loss = abs(min(all_daily_returns))
        overall_single_day_max_gain = max(all_daily_returns)
        print(f"  📉 Max Single Day Loss: {overall_single_day_max_loss:.2%}")
        print(f"  📈 Max Single Day Gain: {overall_single_day_max_gain:.2%}")
    else:
        overall_single_day_max_loss = 0
        print("  📉 Max Single Day Loss: No data")


    
    print(f"\n  Asset Trading Overview:")
    print(f"  Total Asset Pool Size: {len(full_eval_data)}")
    print(f"  Assets Traded: {len(all_traded_assets)}")
    print(f"  Trading Coverage: {len(all_traded_assets)/len(full_eval_data)*100:.1f}%")

    # Display list of traded assets
    if len(all_traded_assets) <= 50:  # If not too many, show all
        print(f"\n  All Traded Assets:")
        for i, asset in enumerate(sorted(all_traded_assets), 1):
            print(f"  {i:2d}. {asset}")
    else:  # If too many, show a subset
        print(f"\n  Traded Assets (Top 30):")
        for i, asset in enumerate(sorted(list(all_traded_assets)[:30]), 1):
            print(f"  {i:2d}. {asset}")
        print(f"  ... and {len(all_traded_assets)-30} more assets")

    total_positive_days = 0
    total_trading_days = 0
    valid_periods = 0  # Count valid periods
    
    for result in all_period_results:
        # Filter condition: Exclude anomalous periods with 0 return and 0 win rate
        if result['period_return'] == 0.0 and result['win_rate'] == 0.0:
            print(f"  Skipping anomalous Period {result['period']}: Return {result['period_return']:.4f}%, Win Rate {result['win_rate']:.5f}")
            continue
            
        # Calculate specific days from each period's win rate and length
        period_days = result['end_day'] - result['start_day'] + 1
        period_positive_days = int(result['win_rate'] * period_days)
        
        total_positive_days += period_positive_days
        total_trading_days += period_days
        valid_periods += 1
    
    overall_win_rate = total_positive_days / total_trading_days if total_trading_days > 0 else 0
    
    print(f"\n  Trading Statistics:")
    print(f"    Overall Daily Win Rate: {overall_win_rate:.1%} ({total_positive_days}/{total_trading_days} days)")
    print(f"    Valid Periods: {valid_periods}/{len(all_period_results)}")
    print(f"    Total Trading Days: {total_trading_days} days")

    print(f"\n  Performance by Period:")
    print("-" * 95)
    print(f"{'Period':<6} {'Day Range':<15} {'Period Ret':<10} {'Cum. Ret':<10} {'Win Rate':<8} {'Sharpe':<8} {'Turnover':<8}")
    print("-" * 95)
    
    for result in all_period_results:
        day_range = f"{result['start_day']}-{result['end_day']}"
        print(f"{result['period']:<6} {day_range:<15} {result['period_return']:>8.2f}% "
              f"{result['cumulative_return']:>8.2f}% {result['win_rate']:>6.3f} "
              f"{result['sharpe_ratio']:>6.2f} {result['annualized_turnover']:>6.1f}%")
    
    # Performance Analysis
    if len(all_period_results) > 1:
        early_returns = [r['period_return'] for r in all_period_results[:len(all_period_results)//2]]
        later_returns = [r['period_return'] for r in all_period_results[len(all_period_results)//2:]]
        
        early_avg = np.mean(early_returns)
        later_avg = np.mean(later_returns)
        
        print(f"\n🎓 Online Learning Effect:")
        print(f"  First Half Avg Return: {early_avg:.2f}%")
        print(f"  Second Half Avg Return: {later_avg:.2f}%")
        print(f"  Improvement: {later_avg - early_avg:+.2f}%")
        
        if later_avg > early_avg:
            print("✅ Online learning is effective! Model performance improved.")
        else:
            print("⚠️ Online learning effect is not significant.")
        
        # Stability Analysis
        period_volatility = np.std(all_period_returns)
        print(f"  Period Return Volatility: {period_volatility:.2f}%")
        
        # Consecutive Losing Periods Analysis
        max_consecutive_losses = 0
        current_losses = 0
        for ret in all_period_returns:
            if ret < 0:
                current_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
            else:
                current_losses = 0
        print(f"  Max Consecutive Losing Periods: {max_consecutive_losses}")

model_performance = {
    'cumulative_return': final_result['cumulative_return'],
    'sharpe_ratio': annualized_sharpe,
    'calmar_ratio': calmar_ratio if 'calmar_ratio' in locals() else 0,
    'win_rate': overall_win_rate * 100,  
    'max_drawdown': max_drawdown * 100,
    'single_day_max_loss': overall_single_day_max_loss * 100 if 'overall_single_day_max_loss' in locals() else 0
}

# Enhanced Asset Analysis
asset_returns = calculate_individual_asset_returns(full_eval_data, actual_length)
print_asset_performance_summary(asset_returns, top_n=30)
compare_model_vs_assets(model_performance, asset_returns)
print("\n  Online Learning Summary:")
print("  • One trading period every 22 days")
print("  • After the period, use its data for incremental learning")
print("  • Next period inherits the final portfolio weights from the last period")
print("  • Model gradually learns to discover high-yield assets")