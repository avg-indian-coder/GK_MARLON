import os
import sys

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import sumolib
import traci

NSR_GREEN = 0
NSR_YELLOW = 1
NSL_GREEN = 2
NSL_YELLOW = 3
EWR_GREEN = 4
EWR_YELLOW = 5
EWL_GREEN = 6
EWR_GREEN = 7

if __name__ == '__main__':
    max_steps = 5400
    sumoBinary = sumolib.checkBinary('sumo-gui')
    sumo_cmd = [sumoBinary, "-c", 'network/2x2.sumocfg', "--no-step-log", "true", "--waiting-time-memory", str(max_steps)]

    traci.start(sumo_cmd)
    traci.simulationStep()
    print(traci.trafficlight.getIDList())

    traci.trafficlight.setPhase('1', EWR_GREEN)

    while True:
        traci.simulationStep()


    #define the model
    #define memory for replay
    #define traffic generator details
    #define visualization details
    ###define simulation class ***MOST IMPORTANT***

    #start the simulation here in a while loop
    #visualize at the end
