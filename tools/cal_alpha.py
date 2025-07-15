import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from envs.portfolio_val_env_v1 import SimpleStockEnv, prepare_stocks_from_list

def calculate_benchmark_return():
    
    # 1. fetching data address
    benchmark_data = prepare_stocks_from_list('data/processed/val')
    
    # check if 510300 exists
    if '510300_etf_processed' not in benchmark_data:
        print("Error：can't find 510300_etf_processed")
        return None, None, None
        
    benchmark_df = benchmark_data['510300_etf_processed']
    
    print(f"test.csv time range:")
    print(f"  start: {benchmark_df.index[0]}")
    print(f"  end: {benchmark_df.index[-1]}")
    print(f"  total num of day: {len(benchmark_df)}")
    
    # 2. Ignore first 30 days（lookback window），fetching 
    if len(benchmark_df) <= 30:
        print("Err：less than 30 days，lookback window can't be skipped")
        return None, None, None
        
    benchmark_trading_data = benchmark_df.iloc[30:]  # starting from 31
    actual_trading_days = len(benchmark_trading_data)
    
    print(f"\nActual trading data:")
    print(f"  total days: {len(benchmark_df)}")
    print(f"  Lookback window: 30")
    print(f"  actual trading days: {actual_trading_days}")
    
    # 3. num → pct
    benchmark_returns = benchmark_trading_data['涨跌幅'].fillna(0) / 100
    
    print(f"\nCorresponding time:")
    print(f"  start date: {benchmark_trading_data.index[0]}")
    print(f"  end date: {benchmark_trading_data.index[-1]}")
    print(f"  num of trading days: {len(benchmark_trading_data)}")
    
    # 4. calculating benchmark rate of return 
    benchmark_cumulative_return = (1 + benchmark_returns).cumprod() - 1
    benchmark_total_return = benchmark_cumulative_return.iloc[-1]
    
    print(f"\nbaseline performance:")
    print(f"  total rate of return: {benchmark_total_return:.4f} ({benchmark_total_return*100:.2f}%)")
    
    # 5. Calculate α
    model_return = 0.0092  # rate of return of your model
    alpha = model_return - benchmark_total_return
    
    print(f"\nCalculating Alpha:")
    print(f"  rate of return of Model: {model_return*100:.2f}%")
    print(f"  benchmark rate of return: {benchmark_total_return*100:.2f}%")
    print(f"  Alpha: {alpha*100:.2f}%")
    
    if alpha > 0:
        print(f"  🎉 Model wins {alpha*100:.2f}%!")
    else:
        print(f"  📉 Model loses {abs(alpha)*100:.2f}%")
    
    return alpha, benchmark_total_return, actual_trading_days

if __name__ == "__main__":
    calculate_benchmark_return()