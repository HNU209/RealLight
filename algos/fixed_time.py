import json

class FixedTime:
    def __init__(self, args):
        self.args = args
        self.time_type = self.args['time_type']
        self.intersection_name = self.args['name']
        
        self.tod = self.load_TOD()
        self.real_intersection_name = self.tod['real_name']
        self.total_duration = self.tod['total_duration']
        self.phase_dict = self.tod['action']
        self.phase_lst = sorted([k for k in self.phase_dict.keys()])
        self.init_agent()
        
    def load_TOD(self):
        with open(f'./data/{self.args["env_type"]}/{self.args["env_name"]}/tod_data/{self.time_type}_tod.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        tod = list(filter(lambda x: x['env_name'] == self.intersection_name, data))[0]
        return tod
    
    def init_agent(self):
        self.curr_phase = 0
        self.init_action_count()
        self.yellow = {
            'status': None,
            'phase': 0,
            'time': 0,
        }
    
    def init_action_count(self):
        self.action_count = {a:s for a, s in self.phase_dict.items()}
    
    def get_action(self):
        if not any(self.action_count.values()):
            self.init_action_count()

        action = int(self.curr_phase)
        self.action_count[str(self.curr_phase)] -= 1
        
        if self.action_count[str(self.curr_phase)] == 0:
            self.curr_phase += 1
            self.curr_phase %= len(self.phase_lst)
        return action
    
    def get_action_yellow(self):
        if self.yellow['status'] == None and self.yellow['time'] == 0:
            if not any(self.action_count.values()):
                self.init_action_count()
            action = int(self.curr_phase)
            self.action_count[str(self.curr_phase)] -= 1
            self.yellow['status'] = False
        
        elif self.action_count[str(self.curr_phase)] == 0 and self.yellow['status']:
            action = int(self.curr_phase)
            self.yellow['time'] -= 1
        else:
            action = int(self.curr_phase)
            self.action_count[str(self.curr_phase)] -= 1
            
        if self.action_count[str(self.curr_phase)] == 0 and self.yellow['status'] == False:
            self.yellow['status'] = True
            self.yellow['time'] = 3
            
        elif self.yellow['time'] == 0 and self.yellow['status'] == True:
            self.yellow['status'] = None
            
            if self.action_count[str(self.curr_phase)] == 0:
                self.curr_phase += 1
                self.curr_phase %= len(self.phase_lst)
        return action