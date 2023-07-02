import numpy as np
import math
import os
import platform

def generator(net, route, end_time, no_vehicles):
    '''
    Generate the route file based on above the parameters
    '''
    if platform.system() == 'Windows':
        cmd_code = f'python3 "%SUMO_HOME%/tools/randomTrips.py" --validate -r {route} --end {no_vehicles} -n {net}'
    else:
        cmd_code = f'SUMO_HOME/tools/randomTrips.py --validate -r {route} --end {no_vehicles} -n {net}'
    os.system(cmd_code)

    f = open(route, "r+")
    l = f.readlines()

    for i in range(len(l)):
        if "vehicle" in l[i]:
            line_idx = i
            break

    vehicle_count = len(l[line_idx + 1:]) / 3 # count of the vehicles in the route

    # get a weibull distribution too now, assume the simulation starts at 0 and ends at end_time
    timings = np.random.weibull(2, int(vehicle_count))
    timings = np.sort(timings)

    car_gen_steps = []
    min_old = math.floor(timings[1])
    max_old = math.ceil(timings[-1])
    min_new = 0
    max_new = end_time
    for value in timings:
        car_gen_steps = np.append(car_gen_steps, ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)

    car_gen_steps = np.rint(car_gen_steps)  # round every value to int -> effective steps when a car will be generated

    # car_gen_steps now contains the sorted times according to the Weibull distribution
    # now the job is to replace the time-steps in the xml file with the ones in car_gen_steps

    for step in car_gen_steps:
        l[line_idx] = l[line_idx][:l[line_idx].find('depart')] + f'depart="{step}"' + '>\n'
        line_idx += 3

    f.close()

    f = open(route, "w")
    f.writelines(l)

    return car_gen_steps


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    car_gen_steps = generator('network/2x2.net.xml', 'network/routes.rou.xml', 5400, 2000)

    if False:
        plt.title("Weibull distribution (2000 cars generated)")
        plt.xlabel("Time steps")
        plt.ylabel("Number of cars generated")
        plt.hist(car_gen_steps, bins = 54)
        plt.show()