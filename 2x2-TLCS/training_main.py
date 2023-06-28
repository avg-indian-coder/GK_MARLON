import os
import sys

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import sumolib
import traci

if __name__ == '__main__':
    max_steps = 5400
    sumoBinary = sumolib.checkBinary('sumo-gui')
    sumo_cmd = [sumoBinary, "-c", 'nets\2way-single-intersection\single-intersection-train-uniform.sumocfg', "--no-step-log", "true", "--waiting-time-memory", str(max_steps)]

    #define the model
    #define memory for replay
    #define traffic generator details
    #define visualization details
    ###define simulation class ***MOST IMPORTANT***

    #start the simulation here in a while loop
    #visualize at the end
