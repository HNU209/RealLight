from algos import FixedTime, RealLight
from copy import copy

def generator_agents(args, env):
    agents = []
    for index in range(args['n_agent']):
        args_ = copy(args)
        args_['name'] = env.intersection_name[index]
        args_['in_dim'] = env.observation_space[index]
        
        if 'daejeon_daeduck' == args['env_name']:
            n_action = env.action_space[index] - 1
        else:
            n_action = 4
        
        args_['out_dim'] = n_action * 5
        args_['phase_dim'] = n_action
        agents.append(RealLight(args_))
    return agents

def generator_fixed_agents(args, env):
    agents = []
    for index in range(args['n_agent']):
        args_ = copy(args)
        args_['name'] = env.intersection_name[index]
        agents.append(FixedTime(args_))
    return agents

## not use yellow phase
def get_actions(agents, state_lst):
    phase_lst = []
    
    for index, agent in enumerate(agents):
        if agent.count == 0:
            agent.state = state_lst[index]
            action, prob, phase, timing = agent.get_action(agent.state)
            agent.action = action
            agent.prob = prob
            agent.phase = phase
            agent.timing = timing
            agent.count = timing
        else:
            phase = agent.phase
        agent.count -= 1
        
        phase_lst.append({
            f'{agent.args["name"]}': phase + 1,
            'yellow_status': agent.yellow['status']
        })
    return phase_lst

## use yellow phase
def get_actions_yellow(agents, state_lst):
    phase_lst = []
    
    for index, agent in enumerate(agents):
        if agent.yellow['status'] == None or (agent.count == 0 and agent.yellow['time'] == 0):
            agent.state = state_lst[index]
            action, prob, phase, timing = agent.get_action(agent.state)
            agent.action = action
            agent.prob = prob
            agent.phase = phase
            agent.timing = timing
            agent.count = timing
            agent.count -= 1
            agent.yellow['status'] = False
        
        elif agent.count == 0 and agent.yellow['status']:
            phase = agent.phase
            agent.yellow['time'] -= 1
            
        else:
            phase = agent.phase
            agent.count -= 1
            
        phase_lst.append({
            f'{agent.args["name"]}': phase + 1,
            'yellow_status': agent.yellow['status']
        })
        
        if agent.count == 0 and agent.yellow['status'] == False:
            agent.yellow['status'] = True
            agent.yellow['time'] = 3
        elif agent.yellow['time'] == 0 and agent.yellow['status'] == True:
            agent.yellow['status'] = None
    return phase_lst

def get_fixed_agent_actions(agents):
    phase_lst = []
    
    for _, agent in enumerate(agents):
        phase = agent.get_action()
        phase_lst.append({
            f'{agent.args["name"]}': phase + 1,
            'yellow_status': agent.yellow['status']
        })
    return phase_lst

def get_fixed_agent_actions_yellow(agents):
    phase_lst = []
    
    for _, agent in enumerate(agents):
        phase = agent.get_action_yellow()
        phase_lst.append({
            f'{agent.args["name"]}': phase + 1,
            'yellow_status': agent.yellow['status']
        })
    return phase_lst