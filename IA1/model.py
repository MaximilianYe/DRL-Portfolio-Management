import os
import sys
sys.path.append("/content/drive/MyDrive/Investment-Reinforcement-Engine-for-Neural-Evaluation/IA1/")

import torch as th
import numpy as np
import pandas as pd

import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from torch.utils.data import DataLoader

from envs import StockEnv

import random
random.seed(42)
np.random.seed(42)
th.manual_seed(42)
th.cuda.manual_seed_all(42)
th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = False
  

class GRUModel(BaseFeaturesExtractor):
  def __init__(self, observation_space: gym.Space, features_dim: int = 128, num_layers: int = 2):
    super().__init__(observation_space, features_dim)

    num_assets = observation_space.shape[0]
    hidden_size = 256

    self.gru = nn.GRU(input_size=num_assets, 
      hidden_size=hidden_size, 
      num_layers=num_layers, 
      batch_first=True, 
      dropout=0.3 if num_layers > 1 else 0.0  
    )
    
    self.head = nn.Sequential(
      nn.Linear(hidden_size, 128),
      nn.ReLU(),
    )

  def forward(self, x):
    x = x.transpose(1, 2)       # x: (batch, 121, 29) → (batch, 29, 121)
    _, h_n = self.gru(x)       # h_n: (2, batch, 256)
    out = h_n[-1]                 # (batch, 256)
    features = self.head(out)     # (batch, 128)
    return features

  
policy_kwargs = dict(
  features_extractor_class=GRUModel,
  features_extractor_kwargs=dict(features_dim=128, num_layers=2),
  net_arch=dict(pi=[], vf=[])
)

def train(train_mat, val_mat, symbols, SAVE_DIR):

  # n_envs = 4
  # def make_env():
  #   def _init():
  #     return StockEnv(
  #       data_matrix=train_mat,
  #       symbols_list=symbols,
  #       lookback_window=29,
  #       initial_capital=2000,
  #     )
  #   return _init

  # train_env = SubprocVecEnv([make_env() for _ in range(n_envs)])
  train_env = StockEnv(
    data_matrix=train_mat,
    symbols_list=symbols,
    lookback_window=29,
    initial_capital=2000,
  )

  val_env = StockEnv(         
    data_matrix=val_mat,
    symbols_list=symbols,
    lookback_window=29,
    initial_capital=2000,
  )

  model = PPO(
    "MlpPolicy", 
    env = train_env, 
    policy_kwargs=policy_kwargs, 
    learning_rate=3e-5,  
    # n_steps=512 // n_envs,
    # batch_size=128,
    # n_epochs=10,
    gamma=0.99,
    # gae_lambda=0.95,
    clip_range=0.2,
    # ent_coef=0.01,  # entropy
    device="cuda",
    verbose=1,
    seed=42
  )

  eval_callback = EvalCallback(
    val_env,
    best_model_save_path=SAVE_DIR,
    log_path=os.path.join(SAVE_DIR, "eval_logs"),
    eval_freq=2048, #  // n_envs
    n_eval_episodes=1,
    deterministic=True,
  )

  model.learn(total_timesteps=300000, 
    callback=eval_callback,
    progress_bar=True,
  )
 
  model.save(os.path.join(SAVE_DIR, "full_agent"))
  print(f"✅ save to {SAVE_DIR}/")


def evaluate(test_mat, symbols, model_path, training=False):
  # class StockEnv(gym.Env):
  #   def __init__(self, data_matrix, symbols_list, lookback_window=29, max_stocks=200, initial_capital=2000, prev_weights=None, verbose=False, training=True):
  
  test_env = StockEnv(
    data_matrix=test_mat,
    symbols_list=symbols,
    lookback_window=29,
    initial_capital=2000,
    training=training,
    verbose=True,
  )

  model = PPO.load(model_path)

  obs, info = test_env.reset()
  done = False

  step_count = 0
  month_start_capital = test_env.current_capital 
  monthly_returns = []
  
  while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = test_env.step(action)
    
    step_count += 1

    # settled every 22 trading days
    if step_count % 22 == 0:
      month_ret = (info['capital'] - month_start_capital) / month_start_capital
      monthly_returns.append(month_ret)
      print(f"📅 The {len(monthly_returns)}-th month (Step {step_count}): Monthly rate of return {month_ret:+.4%}  capital {info['capital']:.2f}")
      month_start_capital = info['capital'] 
    
    if truncated:
      break

  if step_count % 22 != 0:
    month_ret = (info['capital'] - month_start_capital) / month_start_capital
    monthly_returns.append(month_ret)
    print(f"📅 The {len(monthly_returns)}-th month (tail, {step_count % 22} days): Monthly rate of return {month_ret:+.4%}")
  
  # ===== cal Sharpe / Max Drawdown / Calmar =====
  daily_rets = np.array(test_env.daily_returns)
  n_days = len(daily_rets)
  
  # Sharpe: Annualized by actual days
  if n_days > 1 and np.std(daily_rets) > 1e-10:
    sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)
  else:
    sharpe = 0.0
  
  # max drawdown: Based on the capital curve
  cumulative = np.cumprod(1 + daily_rets)
  running_max = np.maximum.accumulate(cumulative)
  drawdowns = (cumulative - running_max) / running_max
  max_drawdown = np.min(drawdowns)  # neg
  
  # Calmar: Annualized return / | maximum drawdown |
  total_return = cumulative[-1] - 1
  annualized_return = (1 + total_return) ** (252 / n_days) - 1
  calmar = annualized_return / abs(max_drawdown) if abs(max_drawdown) > 1e-10 else 0.0

  # stat
  print(f"\n{'='*50}")
  print(f"Final captial: {info['capital']:.2f}")
  print(f"Rate of Return: {info['return']:.4%}")
  print(f"Win Rate: {info['win_rate']:.2%}")
  print(f"Total turnover rate: {info['total_turnover']:.4f}")
  print(f"Max consecutive_wins: {info['max_consecutive_wins']}")
  print(f"Max_consecutive_losses: {info['max_consecutive_losses']}")
  print(f"\n📊 Indicators (based on {n_days} trading days):")
  print(f"  Sharpe Ratio:  {sharpe:.4f}")
  print(f"  Max Drawdown:  {max_drawdown:.4%}")
  print(f"  Annualized Return: {annualized_return:.4%}")
  print(f"  Calmar Ratio:  {calmar:.4f}")
  
  print(f"\n{'='*50}")
  print(f"Monthly yield summary:")
  for i, r in enumerate(monthly_returns):
    print(f"  Month {i+1}: {r:+.4%}")
  if monthly_returns:
    print(f"  Monthly rate of return: {np.mean(monthly_returns):+.4%}")
    print(f"  Standard deviation of monthly returns: {np.std(monthly_returns):.4%}")
  
  info['monthly_returns'] = monthly_returns
  return info
  


if __name__ == "__main__":
  base = os.path.dirname(os.path.abspath(__file__))
  train_dir = os.path.join(base, "data/train/")
  train_mat = np.load(os.path.join(train_dir, "samples.npy"))  # (T, N, 30)
  symbols = pd.read_csv(os.path.join(train_dir,"symbols.csv"))["symbol"].tolist()
  
  val_dir = os.path.join(base, "data/val")
  val_mat = np.load(os.path.join(val_dir, "samples.npy"))

  test_dir = os.path.join(base, "data/test")
  test_mat = np.load(os.path.join(test_dir, "samples.npy"))

  SAVE_DIR = os.path.join(base, "full")
  os.makedirs(SAVE_DIR, exist_ok=True)


  # train 
  # train(train_mat, val_mat, symbols, SAVE_DIR)
  
  # eval
  # evaluate(val_mat, symbols, model_path = os.path.join(SAVE_DIR, "full_agent"), training=False)

  # evaluate(val_mat, symbols, model_path = os.path.join(SAVE_DIR, "best_model"), training=False)

  # test
  # evaluate(test_mat, symbols, model_path = os.path.join(SAVE_DIR, "full_agent"), training=False)
  evaluate(test_mat, symbols, model_path = os.path.join(SAVE_DIR, "best_model"), training=False)

