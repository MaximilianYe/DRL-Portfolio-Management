import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch as th
print(f"CUDA avail: {th.cuda.is_available()}")
print(f"num of CUDA device: {th.cuda.device_count()}")
if th.cuda.is_available():
    print(f"curr device: {th.cuda.get_device_name()}")
    print(f"using device: {th.cuda.current_device()}")

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from envs.training_env.portfolio_env_v2 import SimpleStockEnv, prepare_stocks_from_list

def complete_ppo_pipeline():
    

    print("create folders...")
    os.makedirs('./best_model/', exist_ok=True)    # new folder
    os.makedirs('./logs/', exist_ok=True)          # new log
    os.makedirs('./models/', exist_ok=True)        # new model

    stock_data = prepare_stocks_from_list('data/processed/train')
    env = SimpleStockEnv(stock_data_dict=stock_data)

    # model 1
    policy_kwargs_1 = dict(
        activation_fn = th.nn.ReLU,  # Tanh,  
        net_arch = [
            dict(pi=[128, 128], vf=[128, 128])
        ]
    )

        # hyperparameters
    model1 = PPO(
        policy='MlpPolicy', 
        env = env,
        policy_kwargs=policy_kwargs_1,
        device='cpu',

        learning_rate=3e-4,           
        n_steps=2048,                 # collecting num of steps before update
        batch_size=64,                
        n_epochs=10,                  
        gamma=0.99,                   # discount 
        gae_lambda=0.95,              # GAE para
        clip_range=0.2,               # PPO clip para
        
        verbose=1  
    )

    # model 2
    # policy_kwargs_2 = dict(

    # )

    model2 = PPO(
        policy='MlpPolicy', 
        env = env,
        policy_kwargs=policy_kwargs_1,
        device='cpu',

        learning_rate=3e-4,           
        n_steps=2048,                 
        batch_size=64,                
        n_epochs=10,                  
        gamma=0.99,                   
        gae_lambda=0.95,              
        clip_range=0.2,               
        
        verbose=1  
    )

    model3 = PPO(
        policy='MlpPolicy', 
        env = env,
        policy_kwargs=policy_kwargs_1,
        device='cpu',

        learning_rate=3e-4,           
        n_steps=4096,                 
        batch_size=64,                
        n_epochs=10,                  
        gamma=0.99,                   
        gae_lambda=0.95,              
        clip_range=0.2,               
        
        verbose=1  
    )

    model4 = PPO(
        policy='MlpPolicy', 
        env = env,
        policy_kwargs=policy_kwargs_1,
        device='cpu',

        learning_rate=3e-4,           
        n_steps=4096,                 
        batch_size=64,                
        n_epochs=10,                  
        gamma=0.99,                   
        gae_lambda=0.95,              
        clip_range=0.2,               
        
        verbose=1  
    )
    
    # 3. set monitor
    print("setting training monitor...")
    eval_env = SimpleStockEnv(stock_data_dict=stock_data)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./best_model/model1/',
        log_path='./logs/model1/',
        eval_freq=10000
    )


    print("start training...")
    model1.learn(
        total_timesteps=100000,
        callback=eval_callback
    )

    eval_callback_2 = EvalCallback(
        eval_env,
        best_model_save_path='./best_model/model2/',  
        log_path='./logs/model2/',                  
        eval_freq=10000
    )

    model2.learn(
        total_timesteps=50000,
        callback=eval_callback_2
    )

    eval_callback_3 = EvalCallback(
        eval_env,
        best_model_save_path='./best_model/model3/',  
        log_path='./logs/model3/',                    
        eval_freq=10000
    )

    model3.learn(
        total_timesteps=50000,
        callback=eval_callback_3
    )

    eval_callback_4 = EvalCallback(
        eval_env,
        best_model_save_path='./best_model/model4/',  
        log_path='./logs/model4/',                    
        eval_freq=10000
    )

    model4.learn(
        total_timesteps=100000,
        callback=eval_callback_4
    )

    

    # 5. save model
    print("save model...")
    model1.save("./models/zip/model1")
    model2.save("./models/zip/model2") 
    model3.save("./models/zip/model3")
    model4.save("./models/zip/model4")

    return {
        'model1': model1,
        'model2': model2, 
        'model3': model3,
        'model4': model4
    }

if __name__ == "__main__":
    model = complete_ppo_pipeline()