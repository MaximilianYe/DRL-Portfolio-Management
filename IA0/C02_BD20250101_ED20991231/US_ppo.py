import os
import sys
# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch as th
import numpy as np
import time # <-- Added for final time tracking
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import gymnasium as gym

# Modify environment import
from env_v1 import SimpleUSStockEnv, prepare_stocks_from_list

def make_env(stock_data, stock_features, rank=0):
    """Factory function to create environment - supports parallel envs"""
    def _init():
        env = SimpleUSStockEnv(
            stock_data_dict=stock_data, 
            stock_features=stock_features,  # Use simplified feature configuration
            max_stocks=200,
            verbose=False  # Disable verbose output in parallel env
        )   
        return env
    return _init

class DetailedDebugCallback(BaseCallback):
    """Detailed Debugging Callback (For Entropy)"""
    def __init__(self, log_freq=1000):
        super().__init__()
        self.log_freq = log_freq
        self.step_count = 0
        self.reward_history = []
        self.entropy_history = []
        self.std_history = []
        self.policy_loss_history = []
        self.value_loss_history = []
        
    def _on_step(self) -> bool:
        self.step_count += 1
        
        # Collect info from the current step
        if hasattr(self, 'locals') and 'infos' in self.locals:
            infos = self.locals['infos']
            if len(infos) > 0 and isinstance(infos[0], dict):
                # Collect rewards
                rewards = [info.get('daily_return', 0) for info in infos if isinstance(info, dict)]
                if rewards:
                    self.reward_history.extend(rewards)
        
        # Periodically print detailed debug info
        if self.step_count % self.log_freq == 0:
            print(f"\n{'='*60}")
            print(f"🔍 Detailed Debug Report - Step {self.step_count}")
            print(f"{'='*60}")
            
            # 1. Reward Analysis
            if self.reward_history:
                recent_rewards = self.reward_history[-1000:]  # Last 1000 rewards
                print(f"  Reward Analysis:")
                print(f"  Avg reward (last 1000 steps): {np.mean(recent_rewards):.6f}")
                print(f"  Reward std: {np.std(recent_rewards):.6f}")
                print(f"  Reward range: [{np.min(recent_rewards):.6f}, {np.max(recent_rewards):.6f}]")
                print(f"  Positive reward ratio: {np.mean(np.array(recent_rewards) > 0):.2%}")
            
            # 2. Policy Network Analysis
            policy = self.model.policy
            print(f"\n Policy Network Analysis:")
            
            for name, param in policy.named_parameters():
                if 'log_std' in name.lower():
                    log_std = param.data.clone()
                    std = th.exp(log_std)
                    
                    print(f"    {name}:")
                    print(f"    log_std range: [{log_std.min().item():.6f}, {log_std.max().item():.6f}]")
                    print(f"    log_std mean: {log_std.mean().item():.6f}")
                    print(f"    Actual std range: [{std.min().item():.6f}, {std.max().item():.6f}]")
                    print(f"    Actual std mean: {std.mean().item():.6f}")
                    
                    # Gradient analysis
                    if param.grad is not None:
                        grad = param.grad
                        print(f"    Gradient range: [{grad.min().item():.8f}, {grad.max().item():.8f}]")
                        print(f"    Gradient mean: {grad.mean().item():.8f}")
                        print(f"    Gradient L2 norm: {grad.norm().item():.8f}")
                        
                        # Check for gradient anomalies
                        if grad.norm().item() > 1.0:
                            print(f"    Gradient might be too large!")
                        elif grad.norm().item() < 1e-8:
                            print(f"    Gradient is almost zero!")
                    else:
                        print(f"    Gradient is None!")
                    
                    # Record history
                    self.std_history.append(std.mean().item())
                    
                    # Calculate theoretical entropy
                    action_dim = len(log_std)
                    theoretical_entropy = 0.5 * action_dim * (1 + np.log(2 * np.pi)) + log_std.sum().item()
                    self.entropy_history.append(theoretical_entropy)
                    print(f"    Theoretical entropy: {theoretical_entropy:.4f}")
            
            # 3. Optimizer State
            print(f"\nOptimizer State:")
            optimizer = policy.optimizer
            for i, param_group in enumerate(optimizer.param_groups):
                print(f"  Param group {i}:")
                print(f"    Learning rate: {param_group['lr']}")
                if 'momentum' in param_group:
                    print(f"    Momentum: {param_group['momentum']}")
            
            # 4. Trend Analysis
            if len(self.entropy_history) >= 2:
                print(f"\nTrend Analysis:")
                recent_entropy_change = self.entropy_history[-1] - self.entropy_history[-2]
                print(f"  Recent entropy change: {recent_entropy_change:+.6f}")
                
                if len(self.entropy_history) >= 5:
                    avg_change = np.mean(np.diff(self.entropy_history[-5:]))
                    print(f"  Avg change (last 5): {avg_change:+.6f}")
                    
                    if avg_change > 0.1:
                        print(f"Entropy is growing rapidly!")
                        self._diagnose_entropy_growth()
            
            # 5. Environment Info
            if hasattr(self, 'locals') and 'infos' in self.locals and len(self.locals['infos']) > 0:
                info = self.locals['infos'][0]
                if isinstance(info, dict):
                    print(f"\n  Environment State:")
                    print(f"  Capital: {info.get('capital', 'N/A')}")
                    print(f"  Return: {info.get('return', 'N/A')}")
                    print(f"  Win Rate: {info.get('win_rate', 'N/A')}")
                    print(f"  Turnover: {info.get('step_turnover', 'N/A')}")
                    print(f"  Traded Assets: {info.get('traded_stocks_count', 'N/A')}")
                    print(f"  Cash Ratio: {info.get('cash_ratio', 'N/A')}")
            
            print(f"{'='*60}\n")
        
        return True
    
    def _diagnose_entropy_growth(self):
        """Diagnose the cause of entropy growth"""
        print(f"\nDiagnosing abnormal entropy growth:")
        
        # Check reward signal
        if self.reward_history:
            recent_rewards = self.reward_history[-1000:]
            if np.mean(recent_rewards) < -0.1:
                print(f"Average reward is too low, may cause policy divergence!")
            elif np.std(recent_rewards) > 1.0:
                print(f"Reward variance is too high, signal is too noisy!")
        
        # Check learning rate
        lr = self.model.learning_rate
        if isinstance(lr, float) and lr > 1e-3:
            print(f"Learning rate might be too high: {lr}")
        
        # Check entropy coefficient
        ent_coef = self.model.ent_coef
        if ent_coef > 0.1:
            print(f"Entropy coefficient might be too high: {ent_coef}")
        
def start_ppo_training():
    """Main function to run the PPO training"""
    th.set_num_threads(4)
    
    print("Loading US Stock Data...")
    stock_data = prepare_stocks_from_list('data/processed/train')
    
    # Change to US stock feature configuration
    stock_features = ['price_vs_ma252', 'momentum_10d_percentile', 'price_percentile_60', 'RSI', 'volatility_22d', 'volume_momentum_5d', 'momentum_5d']
    print(f"Using features: {stock_features}")
    
    print("Creating training environment...")
    
    try:
        n_envs = 4
        env = SubprocVecEnv([make_env(stock_data, stock_features, i) for i in range(n_envs)])
        print(f"Successfully created {n_envs} parallel environments")
    except Exception as e:
        print(f"Parallel env creation failed: {e}")
        print("Falling back to single-thread environment")
        n_envs = 1
        env = DummyVecEnv([make_env(stock_data, stock_features, 0)])
    
    eval_env = DummyVecEnv([make_env(stock_data, stock_features)])
    
    print("Environment Info:")
    print(f"  Env count: {n_envs}")
    print(f"  Observation Space: {env.observation_space}")
    print(f"  Action Space: {env.action_space}")
    
    obs = env.reset()
    print(f"  Observation Shape: {obs.shape}")
    
    # Use more conservative hyperparameters
    print(f"\nUsing conservative hyperparameter settings:")
    model = PPO(
        policy='MlpPolicy', 
        env=env,
        policy_kwargs=dict(
            activation_fn=th.nn.ReLU,
            net_arch=[dict(pi=[1024, 512], vf=[1024, 512])],
        ),
        device='cpu',
        learning_rate=1e-5,  # Lower learning rate
        n_steps=512 // n_envs,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Lower entropy coefficient
        max_grad_norm=0.1,  # Strengthen gradient clipping
        verbose=1,
        seed=42
    )
    
    print(f"  Learning Rate: {model.learning_rate}")
    print(f"  Entropy Coef: {model.ent_coef}")
    print(f"  Max Grad Norm: {model.max_grad_norm}")
    
    # Create debug callbacks
    debug_callback = DetailedDebugCallback(log_freq=2000)
    
    # Modify model save path
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/us_stock_model/',
        log_path='./logs/us_stock_logs/',
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    # Removed PortfolioInfoCallback
    callbacks = CallbackList([debug_callback, eval_callback])
    
    print("  Starting US Stock investment training...")
    print("  Focus on: entropy trend and cash management strategy")
    
    # Train
    total_timesteps = 300000
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks
    )
    
    print("  Training complete!")
    
    # Final analysis
    print(f"\n  Final Analysis:")
    if debug_callback.entropy_history:
        print(f"  Entropy change: {debug_callback.entropy_history[0]:.2f} → {debug_callback.entropy_history[-1]:.2f}")
        entropy_trend = np.diff(debug_callback.entropy_history)
        print(f"  Average change rate: {np.mean(entropy_trend):+.6f}")
        if np.mean(entropy_trend) > 0.01:
            print(f"Confirmed: Abnormal continuous entropy growth!")
        else:
            print(f"Entropy change is within reasonable limits")
    
    if debug_callback.reward_history:
        print(f"  Average Reward: {np.mean(debug_callback.reward_history):.6f}")
        print(f"  Reward Std: {np.std(debug_callback.reward_history):.6f}")
    
    env.close()
    eval_env.close()
    
    return model

if __name__ == "__main__":
    # Create model and log directories
    os.makedirs('./models/us_stock_model/', exist_ok=True)
    os.makedirs('./logs/us_stock_logs/', exist_ok=True)
    
    start_time = time.time()
    
    print("🇺🇸 US Stock Investment Reinforcement Learning Training")
    
    try:
        model = start_ppo_training()
        
        if model is not None:
            # Save the model
            model.save("./models/us_stock_model/best_model")
            print(f"Model saved to: ./models/us_stock_model/best_model")
    
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Print final runtime
        total_time = time.time() - start_time
        print(f"\nTotal runtime: {total_time:.1f} seconds")