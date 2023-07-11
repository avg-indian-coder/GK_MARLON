from __future__ import absolute_import, print_function

import os
import sys
import datetime

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci
import sumolib
from model import TrainModel
from memory import Memory
from generator import TrafficGen
from training_simulation import Simulation
from shutil import copyfile
import utils

""" incoming_lanes = {
    '1': ['-v11', 'h12', 'v12', '-h11'],
    '2': ['-v21', 'h13', 'v22', '-h12'],
    '5': ['-v12', 'h22', 'v13', '-h21'],
    '6': ['-v22', 'h23', 'v23', '-h22']
} """ # how to automate

incoming_lanes = {
    'A1': ['left1A1', 'B1A1', 'top0A1', 'A0A1'],
    'B1': ['right1B1', 'top1B1', 'A1B1', 'B0B1'],
    'A0': ['B0A0', 'A1A0', 'left0A0', 'bottom0A0'],
    'B0': ['right0B0', 'B1B0', 'A0B0', 'bottom1B0'],
}

""" NSR_GREEN = 0
NSR_YELLOW = 1
NSL_GREEN = 2
NSL_YELLOW = 3
EWR_GREEN = 4
EWR_YELLOW = 5
EWL_GREEN = 6
EWL_YELLOW = 7 """

NS_GREEN = 0
NS_YELLOW = 1
EW_GREEN = 2
EW_YELLOW = 3

if __name__ == '__main__':
    config = utils.import_train_configuration(config_file='train.ini')
    #sumoBinary = sumolib.checkBinary('sumo-gui')
    #sumo_cmd = [sumoBinary, "-c", 'network/2x2.sumocfg', "--no-step-log", "true", "--waiting-time-memory", str(max_steps)]
    path = utils.set_train_path(config['models_path_name'])
    sumo_cmd = utils.set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'], path)

    # print(config['network'])
    # automate opening of sumo, getting ids, and closing it to create models

    light_ids = ['A1', 'B1', 'A0', 'B0']

    models = []
    mem_arr = []

    for ids in light_ids:
        Model = TrainModel(
            id=ids,
            num_layers=config['num_layers'],
            width=config['width_layers'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            input_dim=config['num_states'],
            output_dim=config['num_actions']
        )
        models.append(Model)

        memory = Memory(
        config['memory_size_max'],
        config['memory_size_min']
        )
        mem_arr.append(memory)

    models = dict(zip(light_ids, models))
    mems = dict(zip(light_ids, mem_arr))

    # refine traffic gen to include onlyt the config file without the network + route
    TrafficGen = TrafficGen(config['network'], config['route'], config['max_steps'], config['n_cars_generated'])

    # define visualization classes here
    
    Simulation = Simulation(
        models,
        mems,
        TrafficGen,
        sumo_cmd,
        config['gamma'],
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states'],
        config['num_actions'],
        config['training_epochs'],
        light_ids,
        incoming_lanes
    )

    episode = 0
    timestamp_start = datetime.datetime.now()

    while episode < config['total_episodes']:
        print('\n----- Episode', str(episode + 1), 'of', str(config['total_episodes']))
        epsilon = 1.0 - (episode / 100)
        simulation_time, training_time = Simulation.run(episode, epsilon)
        print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:', round(simulation_time + training_time))
        episode += 1

    print('\n---------- Start time:', timestamp_start)
    print('---------- End time:', datetime.datetime.now())
    print('---------- Session info saved at:', path)

    Model.save_model(path)

    copyfile(src='train.ini', dst=os.path.join(path, 'train.ini'))

    #visualize at the end


if __name__ == '__notmain__':
    sumoBinary = sumolib.checkBinary('sumo-gui')
    sumo_cmd = [sumoBinary, "-c", 'network/anshul_2x2.sumocfg', "--no-step-log", "true", "--waiting-time-memory", str(5400)]
    traci.start(sumo_cmd)

    print(traci.trafficlight.getIDList())
    print(sorted(list(set(traci.trafficlight.getControlledLanes('A0')))))

    while True:
        #print('lane position of 0:', 150 - traci.vehicle.getLanePosition('0'))
        #print('lane position of 1:', 150 - traci.vehicle.getLanePosition('1'))
        print('lane id:', traci.vehicle.getLaneID('0'))
        print('Cars on A0A1:', traci.edge.getLastStepVehicleIDs('A0A1'))
        traci.simulationStep()
        #phase = int(input('Enter phase:'))
        #traci.trafficlight.setPhase('A0', phase)
        # traci.simulation.getTime()