import traci
import numpy as np
import random
import timeit
import os

# phase codes based on environment.net.xml
PHASE_NS_GREEN = 0  # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 code 11
PHASE_EWL_YELLOW = 7

# ignore above shits
# yellow has phase % 2 == 1
NS_GREEN = 0
NS_YELLOW = 1
EW_GREEN = 2
EW_YELLOW = 3

class Simulation:
    def __init__(self, models, mems, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_states, num_actions, training_epochs, light_ids, incoming_roads):
        self._models = models # dict of models
        self._mems = mems # dict of mem arrays
        self._TrafficGen = TrafficGen
        self._sumo_cmd = sumo_cmd
        self._gamma = gamma
        self._step = 0
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_store = [] # this has to be transferred to the Model
        self._cumulative_wait_store = [] # could be transferred to model?
        self._avg_queue_length_store = [] # ""
        self._training_epochs = training_epochs
        self._light_ids = light_ids # list of light ids
        self._incoming_roads = incoming_roads # dict, with traffic_light_idx: [incoming roads]
        self._incoming_lanes = {} # lanes with underscores
        self._remaining_steps = {} # remaining steps per intersection
        for id in self._light_ids:
            self._remaining_steps[id] = green_duration

    def run(self, episode, epsilon):
        """
        Runs an episode of simulation, then starts a training session
        """
        #print('gay1')
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        self._incoming_lanes = self._get_incoming_lanes()

        # inits at the start of an episode run
        self._step = 0
        self._waiting_times = {}
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0

        # below are changed for multi agent
        old_action = {}
        old_state = {}
        old_total_wait = {}
        yellow_countdown = {}
        action = {}

        # multi agent: define as dictionaries
        for id in self._light_ids:
            old_total_wait[id] = 0
            old_state[id] = -1
            old_action[id] = 0
            yellow_countdown[id] = self._yellow_duration # variable to count down the number steps light remains yellow for

        store_next_action = {} # stores the next action to take
        if_state_change = {} # False if no change, True if change
        for id in self._light_ids:
            if_state_change[id] = False

        reward = {}

        while self._step < self._max_steps:

            # get current state of intersection
            current_state = self._get_state() # COMPLETE

            # calculate reward
            # calculate accumulated total waiting time
            current_total_wait = self._collect_waiting_times() # COMPLETE
            
            for id in self._light_ids:
                if if_state_change[id] or self._step < 1:
                    reward[id] = old_total_wait[id] - current_total_wait[id]

            # save the data
            if self._step != 0:
                for id in self._light_ids:
                    if if_state_change[id]: # if there is a state change, then store
                        # self._Memory.add_sample((old_state, old_action, reward, current_state))
                        self._mems[id].add_sample((old_state[id], old_action[id], reward[id], current_state[id]))

            for id in self._light_ids:
                if self._remaining_steps[id] <= 0 and traci.trafficlight.getPhase(id) % 2 == 0: # take new action, since prev phase is green
                    #print(f'{id} 1', end=" ")
                    action[id] = self._choose_action(current_state[id], epsilon, id)
                    if_state_change[id] = True
                    if old_action[id] != action[id]:
                        self._set_yellow_phase(old_action[id], id)
                        self._remaining_steps[id] = self._yellow_duration
                        store_next_action[id] = action[id]
                    else: 
                        self._set_green_phase(action[id], id)
                        self._remaining_steps[id] = self._green_duration
                elif self._remaining_steps[id] <= 0 and traci.trafficlight.getPhase(id) % 2 == 1: # take saved action
                    #print(f'{id} 2', end=' ')
                    if_state_change[id] = False
                    self._set_green_phase(store_next_action[id], id)
                elif self._remaining_steps != 0:
                    #print(f'{id} 3', end=' ')
                    if_state_change[id] = False

                self._remaining_steps[id] -= 1
            
            #print(' ')

            
            #print(self._remaining_steps)
            traci.simulationStep()
            self._step += 1

            # save variables for later & accumulate reward
            for id in self._light_ids:
                if if_state_change[id] == True:
                    old_state[id] = current_state[id]
                    old_action[id] = action[id]
                    old_total_wait[id] = current_total_wait[id]

            # saving only the meaningful reward to better see if the agent is behaving correctly
            for id in self._light_ids:
                if if_state_change[id]:
                    if reward[id] < 0:
                        self._sum_neg_reward += reward[id]

        self._save_episode_stats()
        print("Total reward:", self._sum_neg_reward, "- Epsilon:", round(epsilon, 2))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        print("Training...")
        start_time = timeit.default_timer()
        for id in self._light_ids:
            for _ in range(self._training_epochs):
                self._replay(id)
        training_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time, training_time
    
    def _simulate(self, steps_todo):
        """
        Execute steps in sumo while gathering statistics
        """
        if (self._step + steps_todo) >= self._max_steps: # do not do more steps than the max allowed no of steps
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep() # step forwards 1
            self._step += 1 # update the step counter 
            steps_todo -= 1
            queue_length = self._get_queue_length()
            self._sum_queue_length += queue_length
            self._sum_waiting_time += queue_length # 1 step while waiting in queue means 1 second waited, for each car, therefore queue_light == waited_seconds

    def _collect_waiting_times(self):
        """
        Retrieve the waiting time of every car in the incoming roads.
        Return dictionary of waiting time of all cars in road in each intersection
        """
        # incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        # all_waiting_times = {}
        # car_list = traci.vehicle.getIDList()
        total_waiting_times = {}

        for light in self._light_ids:
            car_list = []
            tuple_list = [traci.edge.getLastStepVehicleIDs(edge) for edge in self._incoming_roads[light]] 
            for tup in tuple_list:
                car_list.extend(tup) # car_list has all cars for that intersection
            
            waiting_time = 0

            for car_id in car_list:
                wait_time = traci.vehicle.getWaitingTime(car_id) # time car has been waiting for
                road_id = traci.vehicle.getRoadID(car_id) # id of road
                if road_id in self._incoming_roads[light]:
                    waiting_time += wait_time
            total_waiting_times[light] = waiting_time
        return total_waiting_times

    def _choose_action(self, state, epsilon, id):
        """
        Decide whether to perform an explorative or exploitative action, according to an epsilon-greedy policy.
        Returns an action for a specific traffic light (id)
        """ 
        if random.random() < epsilon:
            return random.randint(0, self._num_actions - 1) # random action
        else:
            return np.argmax(self._models[id].predict_one(state)) # the best action given the current state
    
    def _set_yellow_phase(self, old_action, id):
        """
        Activate the correct yellow light combination in sumo for a specific traffic id
        """
        yellow_phase_code = old_action * 2 + 1 # obtain the yellow phase code, based on the old action
        #print(yellow_phase_code)
        traci.trafficlight.setPhase(id, yellow_phase_code)

    def _set_green_phase(self, action_number, id):
        """
        Activate the correct green light combination in sumo
        """
        if action_number == 0:
            traci.trafficlight.setPhase(id, NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase(id, EW_GREEN)
        """ elif action_number == 2:
            traci.trafficlight.setPhase(id, PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase(id, PHASE_EWL_GREEN) """

    # CHANGE
    def _get_queue_length(self):
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        queue_length = halt_N + halt_E + halt_S + halt_W
        return queue_length
    
    def _get_state(self):
        """
        Retrieve the states of the intersections from sumo, in the form of cell occupancy.
        Return list of states per intersection in the form of a dictionary.
        """
        list_states = {}

        # iterate over all traffic lights, put cars in list_states dictionary
        for light in self._light_ids:
            state = np.zeros(self._num_states) # to add to the list_states
            #car_list = traci.vehicle.getIDList() # all cars in THAT SPECIFIC SIGNAL
            car_list = []
            tuple_list = [traci.edge.getLastStepVehicleIDs(edge) for edge in self._incoming_roads[light]] 
            for tup in tuple_list:
                car_list.extend(tup)
            
            # at this point, car_list has all the cars on the lane going into the traffic light 'light'

            for car_id in car_list:
                lane_pos = traci.vehicle.getLanePosition(car_id) # v v important!!!
                lane_id = traci.vehicle.getLaneID(car_id) # this also v v important!!!
                lane_pos = 150 - lane_pos

                # distance in meters from the traffic light -> mapping into cells
                if lane_pos < 7:
                    lane_cell = 0
                elif lane_pos < 14:
                    lane_cell = 1
                elif lane_pos < 21:
                    lane_cell = 2
                elif lane_pos < 28:
                    lane_cell = 3
                elif lane_pos < 40:
                    lane_cell = 4
                elif lane_pos < 60:
                    lane_cell = 5
                elif lane_pos < 100:
                    lane_cell = 6
                elif lane_pos < 160:
                    lane_cell = 7
                elif lane_pos < 400:
                    lane_cell = 8
                elif lane_pos <= 750:
                    lane_cell = 9

                if lane_id in self._incoming_lanes[light]:
                    lane_group = self._incoming_lanes[light].index(lane_id) # index of the lane group 
                else:
                    lane_group = -1

                if lane_group >= 1 and lane_group <= 7:
                    car_position = int(str(lane_group) + str(lane_cell))  # composition of the two postion ID to create a number in interval 0-79
                    valid_car = True
                elif lane_group == 0:
                    car_position = lane_cell
                    valid_car = True
                else:
                    valid_car = False  # flag for not detecting cars crossing the intersection or driving away from it

                if valid_car:
                    state[car_position] = 1  # write the position of the car car_id in the state array in the form of "cell occupied"
            
            list_states[light] = state

        return list_states

    # CHANGE
    def _replay(self, id):
        """
        Retrieve a group of samples from the memory and for each of them update the learning equation, then train
        """
        batch = self._mems[id].get_samples(self._models[id].batch_size)

        if len(batch) > 0: # if the memory is full enough
            states = np.array([val[0] for val in batch])
            next_states = np.array([val[3] for val in batch]) 

            # prediction
            q_s_a = self._models[id].predict_batch(states)
            q_s_a_d = self._models[id].predict_batch(next_states)

            # setup training arrays
            x = np.zeros((len(batch), self._num_states))
            y = np.zeros((len(batch), self._num_actions))

            for i, b in enumerate(batch):
                state, action, reward, _ = b[0], b[1], b[2], b[3] # extract data from one sample
                current_q = q_s_a[i] # get the predicted value of state
                current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i]) # update Q(state, action)
                x[i] = state
                y[i] = current_q # Q(state) that includes the updated action value

            self._models[id].train_batch(x, y) # train the NN


    def _save_episode_stats(self):
        """
        Save the stats of the episode to plot the graphs at the end of the session
        """
        self._reward_store.append(self._sum_neg_reward) # how much negative reward in this episode
        self._cumulative_wait_store.append(self._sum_waiting_time) # total number of seconds waited by cars in this episode
        self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps) # average number of queued cars per step, in the episode

    @property
    def reward_store(self):
        return self._reward_store
    
    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store
    
    @property
    def avg_queue_length_store(self):
        return self._avg_queue_length_store
    
    def _get_incoming_lanes(self):
        '''
        Works only if there is no unique left and right lanes
        '''
        incoming_lanes = {}
        for light in self._light_ids:
            incoming_lanes[light] = sorted(list(set(traci.trafficlight.getControlledLanes(light))))
        
        return incoming_lanes
            


        


        






