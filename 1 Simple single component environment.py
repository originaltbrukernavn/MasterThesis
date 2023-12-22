import numpy as np
import random
import gymnasium
from gymnasium import spaces

class Simple_Single_Component(gymnasium.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self):
        # Action space: (0) no action, (1) inspection
        self.action_space = spaces.Discrete(2)

        # Observation space:
        # | Num | 0                                             | Min | Max |
        # |-----|-----------------------------------------------|-----|-----|
        # | 0   | (Blade) Time since maintenance action (weeks) | 0   | inf |

        COMPONENTS = 1
        low = np.zeros((COMPONENTS),dtype=np.float32)
        high = np.ones((COMPONENTS),dtype=np.float32)
        high *= 10**6
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.time_since_maintenance = None

    def step(self, action):
        COMPONENTS = 1
        MC_LOOPS = 10
        POWER_CAPACITY = 5
        CAPACITY_FACTOR = 0.49
        VALUE_OF_MWH = 40
        INSPECTION_COST = 2000
        CONVERSION_RATE = 1.14

        # Cost of repair, cost of replacement
        COST_OF_MAINTENANCE = np.array([
            [4000, 200000]], dtype=np.float32)
        COST_OF_MAINTENANCE *= CONVERSION_RATE

        # Duration of repair, duration of replacement
        DURATION_OF_MAINTENANCE = np.array([
            [3, 70]], dtype=np.float32)

        # Weibull parameters for the markov transition to degradated, critical, failure states
        SCALE_PARAM = np.array([
            [23.02, 2.88, 2.88]], dtype=np.float32)
        SCALE_PARAM = 1/(SCALE_PARAM*52)
        SHAPE_PARAM = np.array([
            [1.2, 1.2, 1.2]], dtype=np.float32)
        z_normal = np.zeros(COMPONENTS, dtype=np.float32)
        z_degraded = np.zeros(COMPONENTS, dtype=np.float32)
        z_critical = np.zeros(COMPONENTS, dtype=np.float32)
        reward = 0
        done = False
        production_hours = 7*24
        reliability = [1] * COMPONENTS
        system_reliability = 1
        state_dist = np.zeros((COMPONENTS, 4), dtype=np.float32)
        for component in range (0, COMPONENTS):
            for mc_loop in range (0, MC_LOOPS):
                state = 0
                time_in_state = 0
                for t in range (0, round(self.time_since_maintenance[component])+1):
                    if state == 0:
                        z_normal[component] = SHAPE_PARAM[component,0]*SCALE_PARAM[component,0]*((SCALE_PARAM[component,0]*time_in_state)**(SHAPE_PARAM[component,0]-1))
                        if z_normal[component] >= random.randint(1,10**6)/10**6:
                            state = 1
                            time_in_state = 0
                        else:
                            time_in_state += 1
                    elif state == 1:
                        z_degraded[component] = SHAPE_PARAM[component,1]*SCALE_PARAM[component,1]*((SCALE_PARAM[component,1]*time_in_state)**(SHAPE_PARAM[component,1]-1))
                        if z_degraded[component] >= random.randint(1,10**6)/10**6:
                            state = 2
                            time_in_state = 0
                        else:
                            time_in_state += 1
                    elif state == 2:
                        z_critical[component] = SHAPE_PARAM[component,2]*SCALE_PARAM[component,2]*((SCALE_PARAM[component,2]*time_in_state)**(SHAPE_PARAM[component,2]-1))
                        if z_critical[component] >= random.randint(1,10**6)/10**6:
                            state = 3
                            time_in_state = 0
                        else:
                            time_in_state += 1
                state_dist[component, state] += 1/MC_LOOPS
            reliability[component] = 1 - state_dist[component, 3]
            system_reliability *= reliability[component]
            self.time_since_maintenance[component] += 1
        if action == 1:
            reward -= INSPECTION_COST
            production_hours -= 1
            for component in range (0, COMPONENTS):
                random_number = random.randint(1,10**6)/10**6
                if state_dist[component, 3] >= random_number:
                    # System component has failed
                    reward -= COST_OF_MAINTENANCE[component,1]
                    self.time_since_maintenance[component] = 0
                    production_hours -= DURATION_OF_MAINTENANCE[component,1]
                elif (state_dist[component, 3] + state_dist[component, 2]) >= random_number: 
                    # System component is in critical state
                    reward -= COST_OF_MAINTENANCE[component,1]
                    self.time_since_maintenance[component] = 0
                    production_hours -= DURATION_OF_MAINTENANCE[component,1]
                elif (state_dist[component, 3] + state_dist[component, 2] + state_dist[component, 1]) >= random_number:
                    # System component is in degraded state
                    reward -= COST_OF_MAINTENANCE[component,0]
                    self.time_since_maintenance[component] = 0
                    production_hours -= DURATION_OF_MAINTENANCE[component,0] 

        reward += system_reliability * production_hours * POWER_CAPACITY * CAPACITY_FACTOR * VALUE_OF_MWH
        reward /= 10**6
        self.runtime += 1
        if self.runtime >= 3000000:
            done=True
        truncated = False
        info = {}
        obs = self.time_since_maintenance
        return np.array(obs, dtype=np.float32), reward, done, truncated, info

    def render(self):
        pass

    def reset(self, seed = None, options = None):
        super().reset(seed = seed)
        COMPONENTS = 1
        MAX = [2*52]    # Adds some variance in start age to encourage exploration
        self.runtime = 0
        self.number_of_repairs_since_replacement = np.zeros(COMPONENTS,dtype=np.float32)
        self.time_since_maintenance = np.zeros(COMPONENTS,dtype=np.float32)
        if options == None:
            self.time_since_maintenance[0] = random.randint(0, MAX[0])
        obs = self.time_since_maintenance
        info = {}
        return np.array(obs, dtype=np.float32), info