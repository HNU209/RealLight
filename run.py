from env import Env
from algos import *
from utils import *

import pandas as pd
import numpy as np
import random
import torch
import yaml
import time
import os

def train(args, env, agents):
    print(f'ENVIRONMENT TYPE : {args["env_type"]}')
    print(f'ENVIRONMENT NAME : {args["env_name"]}')
    print(f'EPISODE : {args["episode"]}\n\n')
    print('training start!\n\n')
    
    start_time = time.time()
    
    eps_average_travel_time = []
    for episode in range(args['episode']):
        for agent in agents:
            agent.init_agent()

        if args['replay_save'] and ((episode == 0) or ((episode + 1) % args['save_interval'] == 0)):
            env.set_save_replay(True)
            env.set_replay_path(f'replay/{episode + 1}')
        else:
            env.set_save_replay(False)
    
        state = env.reset()
        done = False
        
        if args['algo'] == 'fixed':
            while not done:
                phase_lst = get_action_fn(agents)
                _, _, done = env.step(phase_lst)
                if done: break
        
        elif args['algo'] == 'rl':
            while not done:
                phase_lst = get_action_fn(agents, state)
                s_prime, reward, done = env.step(phase_lst)
                
                for idx, agent in enumerate(agents):
                    if agent.count == 0 and agent.yellow['status'] != True:
                        agent.put((agent.state, agent.action, agent.prob, reward[idx], s_prime[idx], done))
                
                state = s_prime
                
                if done: break
            
            for agent in agents:
                agent.train()
        
        else:
            NotImplementedError()
        
        att = env.get_average_travel_time()
        eps_average_travel_time.append(att)
        min_att = min(eps_average_travel_time)
        
        end_time = time.time()
        print(f'epoch : [ {episode + 1: >3d} / {args["episode"]} ]\taverage-travel-time : {att:.2f}\tmin-average-travel-time : {min_att:.2f}\telapsed-time : {end_time - start_time:.2f}s')
    
    if args['algo'] == 'rl' and args['model_save']:
        model_save_base_folder = f'{env.data_dir}/{env.env_type}/{env.env_name}/models'
        model_save_folder = f'models_{args["cctv_dist"]}'
        if model_save_folder not in os.listdir(model_save_base_folder):
            os.makedirs(f'{model_save_base_folder}/{model_save_folder}')
        
        for agent in agents:
            ### save model for evaluation in data folder
            torch.save(agent.actor, f'{model_save_base_folder}/{model_save_folder}/actor_{agent.args["name"]}')
            torch.save(agent.critic, f'{model_save_base_folder}/{model_save_folder}/critic_{agent.args["name"]}')
            
            ### save model in results
            torch.save(agent.actor, f'{env.config_dir}/models/actor_{agent.args["name"]}')
            torch.save(agent.critic, f'{env.config_dir}/models/critic_{agent.args["name"]}')
    
    with open(f'{env.config_dir}/conf.yaml', 'w') as f:
        yaml.safe_dump(args, f, indent=4)
    
    print(f'\n\nmin-average-travel-time : {min_att:.2f}\n\n')

def eval(args, env, agents):
    print(f'ENVIRONMENT TYPE : {args["env_type"]}')
    print(f'ENVIRONMENT NAME : {args["env_name"]}')
    print(f'EVALUEATION EPISODE : {args["eval_episode"]}\n\n')
    print('evaluating start!\n\n')
    
    start_time = time.time()
    
    if args['algo'] == 'rl':
        saved_model_folder = f"{env.data_dir}/{env.env_type}/{env.env_name}/models"
        for agent in agents:
            agent.actor = torch.load(f'{saved_model_folder}/models_{args["cctv_dist"]}/actor_{agent.args["name"]}')
            agent.critic = torch.load(f'{saved_model_folder}/models_{args["cctv_dist"]}/critic_{agent.args["name"]}')
            agent.actor.eval()
            agent.critic.eval()

    total_phase_count = []
    eps_average_travel_time = []
    for episode in range(args['eval_episode']):
        for agent in agents:
            agent.init_agent()

        if args['replay_save'] and ((episode == 0) or ((episode + 1) % args['save_interval'] == 0)):
            env.set_save_replay(True)
            env.set_replay_path(f'replay/{episode + 1}')
        else:
            env.set_save_replay(False)
    
        state = env.reset()
        done = False
        
        if args['algo'] == 'fixed':
            while not done:
                phase_lst = get_action_fn(agents)
                _, _, done = env.step(phase_lst)
                if done: break
        
        elif args['algo'] == 'rl':
            while not done:
                phase_lst = get_action_fn(agents, state)
                s_prime, reward, done = env.step(phase_lst)
                
                state = s_prime
                
                total_phase_count.append(phase_lst)
            
                if done: break
            
        else:
            NotImplementedError()
        
        att = env.get_average_travel_time()
        eps_average_travel_time.append(att)
        min_att = min(eps_average_travel_time)
        
        end_time = time.time()
        print(f'epoch : [ {episode + 1: >3d} / {args["eval_episode"]} ]\taverage-travel-time : {att:.2f}\tmin-average-travel-time : {min_att:.2f}\telapsed-time : {end_time - start_time:.2f}s')
    
    df = pd.DataFrame(total_phase_count)
    df.to_csv(f'{env.config_dir}/phase_count_{args["env_name"]}.csv', index=None)
    
    with open(f'{env.config_dir}/conf.yaml', 'w') as f:
        yaml.safe_dump(args, f, indent=4)
    
    mean_att = np.mean(eps_average_travel_time)
    print(f'\n\nmean-average-travel-time : {mean_att:.2f}\n\n')

if __name__ == '__main__':
    with open('conf.yaml', 'r') as f:
        args = yaml.safe_load(f)
    
    args['seed'] = args['seed'] + 1 if args['eval'] else args['seed']
    
    args['file_loc'] = os.path.dirname(os.path.abspath(__file__))
    args['data_folder_loc'] = os.path.join(args['file_loc'], args['data_folder'])
    args['env_folder_loc'] = os.path.join(args['data_folder_loc'], args['env_type'], args['env_name'])
    
    env = Env(args)
    args['n_agent'] = len(env.intersection_name)
    
    ## fixed seed
    env.seed(args['seed'])
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed'])
    
    if args['algo'] == 'fixed':
        agents = generator_fixed_agents(args, env)
    elif args['algo'] == 'rl':
        agents = generator_agents(args, env)
    else:
        NotImplementedError()
    
    if args['env_name'] == 'daejeon_daeduck':
        if args['algo'] == 'fixed':
            if args['yellow_phase']:
                get_action_fn = get_fixed_agent_actions_yellow
            else:
                get_action_fn = get_fixed_agent_actions
        else:
            if args['yellow_phase']:
                get_action_fn = get_actions_yellow
            else:
                get_action_fn = get_actions
    else:
        if args['algo'] == 'fixed':
            get_action_fn = get_fixed_agent_actions
        else:
            get_action_fn = get_actions
    
    if args['eval']:
        eval(args, env, agents)
    else:
        train(args, env, agents)