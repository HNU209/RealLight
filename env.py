from collections import defaultdict
from datetime import datetime
import numpy as np
import cityflow
import shutil
import json
import os

class Env:
    def __init__(self, args):
        self.args = args
        self.algo = self.args['algo']
        self.env_name = self.args['env_name']
        self.env_type = self.args['env_type']
        self.dir_name = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.dir_name, 'data')
        
        ### environment parameters
        self.is_done = False
        self.current_step = 0
        self.cctv_dist = self.args['cctv_dist']
        self.steps_per_episode = self.args['steps_per_episode']
        
        ### default settings
        self.t = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.c = f'{self.env_name}_{self.algo}'
        self.result_folder = f"{self.dir_name}/results/{'_'.join([self.t, self.c])}/"
        
        self.set_config()
        
        ### create cityflow engine
        self.cityflow = cityflow.Engine(f'{self.config_dir}/config.json', thread_num=2)
        
        ### road roadnet file
        self.roadnet = self.load_roadnet()
        
        ### road intersection
        self.intersection, self.intersection_name = self.get_intersections()
        
        ### get start lanes & end lanes
        self.se = self.get_start_end_lanes()
        
        ### lane length
        self.lane_length = self.get_lane_length()
        
        ### action - phase
        self.phase_lane = self.get_phase_lane()
        
        ### observation space & action space
        self.observation_space, self.action_space = self.get_observation_space(), self.get_action_space()
        
        ### variables to reflect the real route
        self.direction_dict = {
            '3-0': 'ne',
            '3-3': 'ns',
            '3-2': 'nw',
            '0-1': 'wn',
            '0-0': 'we',
            '0-3': 'ws',
            '1-2': 'sw',
            '1-1': 'sn',
            '1-0': 'se',
            '2-3': 'es',
            '2-2': 'ew',
            '2-1': 'en'
        }
        
        if self.env_name == 'daejeon_daeduck':
            self.time_type = self.args['time_type']
            self.intersection_ratio = self.get_intersection_ratio()
            self.total_road = self.get_total_road()
        
    def get_observation_space(self):
        state = self.reset()
        observation_space = [len(s) for s in state]
        return observation_space
    
    def get_action_space(self):
        action_space = []
        for intersection in self.intersection:
            action_space.append(len(intersection['trafficLight']['lightphases']))
        return action_space
    
    def reset(self):
        self.cityflow.reset()
        self.current_step = 0
        self.is_done = False
        state = self.get_state()
        return state
    
    def get_state(self):
        lane_vehs = self.cityflow.get_lane_vehicles()
        
        cctv_dist_veh_intersection = {}
        for intersection_id, se_dict in self.se.items():
            start_lane = se_dict['start']
            density_veh_lane = []
            
            for lane in start_lane:
                vehs = lane_vehs[lane]
                lane_length = self.lane_length[lane.rsplit('_', 1)[0]]
                cctv_dist = lane_length if self.cctv_dist == 'max' else self.cctv_dist
                volume = 0
                
                for veh_id in vehs:
                    veh_info = self.cityflow.get_vehicle_info(veh_id)
                    veh_dist = float(veh_info['distance'])
                    
                    if (lane_length - veh_dist) <= cctv_dist:
                        volume += 1

                ### density
                mean_veh_length = 5
                min_veh_gap = 2.5
                max_vehicle_num = int(cctv_dist // (mean_veh_length + min_veh_gap))
                density = volume / max_vehicle_num
                density_veh_lane.append(density)
            
            substate = self.get_substate(intersection_id)
            cctv_dist_veh_intersection[intersection_id] = density_veh_lane + substate
        return [veh_indensity for veh_indensity in cctv_dist_veh_intersection.values()]
    
    def get_substate(self, intersection_name):
        substate = []
        lane_vehs = self.cityflow.get_lane_vehicles()
        
        _, init_f, init_b = intersection_name.split('_')
        for i in [-1, 1]:
            for j in [-1, 1]:
                f = int(init_f) + i
                b = int(init_b) + j
                neighbor_intersection = f'intersection_{f}_{b}'
                
                if neighbor_intersection not in self.se:
                    substate.append(0)
                    continue
                
                se_dict = self.se[neighbor_intersection]
                end_lane = se_dict['end']
                density_veh_lane = []
                    
                for lane in end_lane:
                    vehs = lane_vehs[lane]
                    lane_length = self.lane_length[lane.rsplit('_', 1)[0]]
                    cctv_dist = lane_length if self.cctv_dist == 'max' else self.cctv_dist
                    volume = 0
                    
                    for veh_id in vehs:
                        veh_info = self.cityflow.get_vehicle_info(veh_id)
                        veh_dist = float(veh_info['distance'])
                    
                        if (lane_length - veh_dist) <= cctv_dist:
                            volume += 1
            
                    ### density
                    mean_vehicle_length = 5
                    mean_vehicle_gap = 2.5
                    max_vehicle_num = int(cctv_dist // (mean_vehicle_length + mean_vehicle_gap))
                    density = volume / max_vehicle_num
                    density_veh_lane.append(density)
        
                mean_density = np.mean(density_veh_lane)
                substate.append(mean_density)
        return substate
    
    def get_reward(self):
        reward = []
        lane_vehicles = self.cityflow.get_lane_vehicles()
        
        for lanes in self.se.values():
            r = 0
            start = lanes['start']
            end = lanes['end']
            
            for s in start:
                r -= len(lane_vehicles[s])
            for e in end:
                r += len(lane_vehicles[e])
            reward.append(r)
        return reward
    
    def step(self, phase_lst):
        if self.env_name == 'daejeon_daeduck':
            self.append_vehicle_route()
        
        visual_light = defaultdict(lambda : defaultdict(lambda : {}))
        
        for intersection_name, start_end_lane in self.se.items():
            for start_lane in start_end_lane['start']:
                road, lane = start_lane.rsplit('_', 1)
                visual_light[intersection_name][road][int(lane)] = 'r'
                
        for intersection_name, phase_dict in zip(self.intersection_name, phase_lst):
            assert intersection_name in phase_dict
            phase, yellow_status = phase_dict[intersection_name], phase_dict['yellow_status']
            for lane in self.phase_lane[intersection_name][phase]:
                road, lane = lane.rsplit('_', 1)
                
                if yellow_status:
                    visual_light[intersection_name][road][int(lane)] = 'y'
                else:
                    visual_light[intersection_name][road][int(lane)] = 'g'
            
            if yellow_status:
                for lane in self.phase_lane[intersection_name][0]:
                    road, lane = lane.rsplit('_', 1)
                    visual_light[intersection_name][road][int(lane)] = 'g'

        visual_light_text = ''
        for intersection_name, visual_light_data in visual_light.items():
            for road, lane_to_visual_light in visual_light_data.items():
                sorted_values = sorted(lane_to_visual_light.items(), key=lambda x: x[0], reverse=False)
                sorted_values_to_text = ' '.join(list(map(lambda x: x[1], sorted_values)))
                visual_light_text += f'{road} {sorted_values_to_text},'
                
        for inter_id, phase_dict in zip(self.intersection_name, phase_lst):
            assert inter_id in phase_dict
            # self.cityflow.set_tl_phase(inter_id, phase_dict[inter_id])
            phase, yellow_status = phase_dict[inter_id], phase_dict['yellow_status']
            if yellow_status:
                self.cityflow.set_tl_phase(inter_id, 0)
            else:
                self.cityflow.set_tl_phase(inter_id, phase_dict[inter_id])
        
        self.cityflow.next_step(visual_light_text)
        self.current_step += 1
        
        state = self.get_state()
        reward = self.get_reward()
        
        if self.current_step == self.steps_per_episode:
            self.is_done = True
        return state, reward, self.is_done
    
    def set_config(self):
        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)
        
        if not os.path.exists(f'{self.result_folder}/replay'):
            os.makedirs(f'{self.result_folder}/replay')
        if not os.path.exists(f'{self.result_folder}/models'):
            os.makedirs(f'{self.result_folder}/models')
        
        ### config 파일 복사
        shutil.copy(f'{self.dir_name}/conf.yaml', f'{self.result_folder}/conf.yaml')
        shutil.copy(f'{self.data_dir}/config.json', f'{self.result_folder}/config.json')
        shutil.copy(f'{self.data_dir}/{self.env_type}/{self.env_name}/roadnet.json', f'{self.result_folder}/roadnet.json')
        if self.env_name == 'daejeon_daeduck':
            shutil.copy(f'{self.data_dir}/{self.env_type}/{self.env_name}/{self.args["time_type"]}_flow.json', f'{self.result_folder}/flow.json')
        else:
            shutil.copy(f'{self.data_dir}/{self.env_type}/{self.env_name}/flow.json', f'{self.result_folder}/flow.json')
            
        self.config_dir = self.result_folder

        with open(f'{self.config_dir}/config.json', 'r') as f:
            config = json.load(f)
            config['dir'] = self.config_dir

        with open(f'{self.config_dir}/config.json', 'w') as f:
            json.dump(config, f, indent=4)
    
    def load_roadnet(self):
        with open(f'{self.config_dir}/roadnet.json', 'r') as f:
            roadnet = json.load(f)
        return roadnet
    
    def get_intersections(self):
        intersection_dict = {inter['id']:inter for inter in self.roadnet['intersections'] if not inter['virtual']}
        intersection_name = list(intersection_dict.keys())
        intersection = list(intersection_dict.values())
        return intersection, intersection_name

    def get_start_end_lanes(self):
        se = {}
        for intersection in self.intersection:
            s_lanes, e_lanes = [], []
            intersection_id = intersection['id']
            road_links = intersection['roadLinks']
            
            for road in road_links:
                start_road = road['startRoad']
                end_road = road['endRoad']
                lane_links = road['laneLinks']
                for lane in lane_links:
                    start_lane = lane['startLaneIndex']
                    end_lane = lane['endLaneIndex']
                    s_lanes.append(f'{start_road}_{start_lane}')
                    e_lanes.append(f'{end_road}_{end_lane}')
            
            s_lanes = sorted(list(set(s_lanes)))
            e_lanes = sorted(list(set(e_lanes)))
            se[intersection_id] = {
                'start': s_lanes,
                'end': e_lanes
            }
        return se

    def get_lane_length(self):
        length_dict = {}
        roads = self.roadnet['roads']
        for lane in roads:
            id = lane['id']
            x1, y1 = lane['points'][0]['x'], lane['points'][0]['y']
            x2, y2 = lane['points'][1]['x'], lane['points'][1]['y']
            length = round(((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5, 2)
            length_dict[id] = length
        return length_dict

    def get_phase_lane(self):
        phase_dict = {}
        for intersection in self.intersection:
            intersection_id = intersection['id']
            phase_lane = {}
            
            for idx, light_phase in enumerate(intersection['trafficLight']['lightphases']):
                phase_index = light_phase['availableRoadLinks']
                lane_lst = []
                for index in phase_index:
                    road = intersection['roadLinks'][index]
                    for lane in road['laneLinks']:
                        l = f"{road['startRoad']}_{lane['startLaneIndex']}"
                        if l not in lane_lst:
                            lane_lst.append(l)
                phase_lane[idx] = lane_lst
            phase_dict[intersection_id] = phase_lane
        return phase_dict

    ### dynamic vehicle control for daejeon daeduck environments
    def get_intersection_ratio(self):
        ratio_file = f'{self.env_type}/{self.env_name}/ratio_data/{self.time_type}_ratio.json'
        total_ratio_file = os.path.join(self.data_dir, ratio_file)
        with open(total_ratio_file, 'r') as f:
            intersection_ratio = json.load(f)
        return intersection_ratio
    
    def get_total_road(self):
        total_road = sum([intersection['roadLinks'] for intersection in self.intersection if not intersection['virtual']], [])
        return total_road
    
    def append_vehicle_route(self):
        lane2vehicles = self.cityflow.get_lane_vehicles()
        for _, vehicles_lst in lane2vehicles.items():
            for vehicle_id in vehicles_lst:
                vehicle_info = self.cityflow.get_vehicle_info(vehicle_id)
                route = vehicle_info['route'].strip().split()
                if len(route) != 1: continue
                else:
                    road = vehicle_info['road']
                    next_route = self.get_next_route(road)
                    route.append(next_route)
                    # route_ = ' '.join(route)
                    self.cityflow.set_vehicle_route(vehicle_id, [next_route])
                
    def get_next_route(self, road):
        available_road = list(filter(lambda x: x['startRoad'] == road, self.total_road))
        directions = {}
        
        for road_ in available_road:
            start_road = road_['startRoad']
            end_road = road_['endRoad']
            
            start_direct = start_road.rsplit('_', 1)[-1]
            intersection, end_direct = end_road.rsplit('_', 1)
            direct = self.direction_dict[f'{start_direct}-{end_direct}']
            directions[direct] = road_['endRoad']
            
        if len(directions.values()) > 0:
            intersection = intersection.split('_', 1)[-1]
            if intersection in self.intersection_ratio.keys():
                direct_rate = self.intersection_ratio[intersection]
                mean_vehicle_n = np.array([direct_rate[i] for i in directions.keys()])
                prob = mean_vehicle_n / sum(mean_vehicle_n)
                next_direct = np.random.choice(list(directions.keys()), 1, p=prob)
            else:
                next_direct = np.random.choice(list(directions.keys()), 1)
            next_road = directions[next_direct[0]]
        else:
            next_road = ''
        return next_road
    
    def get_average_travel_time(self):
        return self.cityflow.get_average_travel_time()
    
    def render(self, mode='human'):
        print("Current time: " + self.cityflow.get_current_time())
    
    def set_replay_path(self, file_name):
        self.cityflow.set_replay_file(f'{file_name}.txt')

    def seed(self, seed=None):
        self.cityflow.set_random_seed(seed)

    def get_path_to_config(self):
        return self.config_dir

    def set_save_replay(self, save_replay):
        self.cityflow.set_save_replay(save_replay)