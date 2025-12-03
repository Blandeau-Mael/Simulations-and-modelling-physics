#!/bin/python3

# Template for traffic simulation
# BH, MP 2021-11-15, latest version 2024-11-08.

"""
    This template is used as backbone for the traffic simulations.
    Its structure resembles the one of the pendulum project, that is you have:
    (a) a class containing the state of the system and it's parameters
    (b) a class storing the observables that you want then to plot
    (c) a class that propagates the state in time (which in this case is discrete), and
    (d) a class that encapsulates the aforementioned ones and performs the actual simulation
    You are asked to implement the propagation rule(s) corresponding to the traffic model(s) of the project.
"""

import math
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy.random as rng
import numpy as np
import random

import matplotlib


class Cars:

    """ Class for the state of a number of cars """

    def __init__(self, numCars=5, roadLength=50, v0=1):
        self.numCars    = numCars
        self.roadLength = roadLength
        self.t  = 0
        self.x  = []
        self.v  = []
        self.c  = []
        self.f  = []
        for i in range(numCars):
            # TODO: Set the initial position for each car.
            # Note that the ordering of the cars on the road needs to match
            # the order in which you compute the distances between cars
            self.x.append(i * self.roadLength // numCars)# the position of the cars on the road
            #self.x.append(i) 
            self.v.append(v0)       # the speed of the cars
            self.c.append(i)        # the color of the cars (for drawing)
            self.f.append(1+v0+0.2*v0**3)

    # NOTE: you can, but don't have to use this function for computing distances
    def distance(self, i):
        # TODO: Implement the function returning the PERIODIC distance 
        # between car i and the one in front 
        j = (i + 1) % self.numCars
        d = (self.x[j] - self.x[i]) % self.roadLength
        return d


class Observables:

    """ Class for storing observables """

    def __init__(self):
        self.time = []          # list to store time
        self.flowrate = []      # list to store the flow rate
        self.avg_fuel = []        

class BasePropagator:

    def __init__(self):
        return
        
    def propagate(self, cars, obs):

        """ Perform a single integration step """
        
        fr = self.timestep(cars, obs)
        avg_fuel = sum(cars.f) #/ cars.numCars

        # Append observables to their lists
        obs.time.append(cars.t)
        obs.flowrate.append(fr)
        obs.avg_fuel.append(avg_fuel)
              
    def timestep(self, cars, obs):

        """ Virtual method: implemented by the child classes """
        
        pass
      
        
class ConstantPropagator(BasePropagator) :
    
    """ 
        Cars do not interact: each position is just 
        updated using the corresponding velocity 
    """
    
    def timestep(self, cars, obs):
        for i in range(cars.numCars):
            cars.x[i] += cars.v[i]
        cars.t += 1
        flow_road = 0
        for i in range(cars.numCars):
            flow_road += cars.v[i]
        flow = flow_road / cars.roadLength        
        for i in range(cars.numCars):
            cars.f[i] = 1 + cars.v[i] + 0.2*cars.v[i]**3
        
        return flow

# TODO
# HERE YOU SHOULD IMPLEMENT THE DIFFERENT CAR BEHAVIOR RULES
# Define you own class which inherits from BasePropagator (e.g. MyPropagator(BasePropagator))
# and implement timestep according to the rule described in the project

class MyPropagator(BasePropagator) :

    def __init__(self, vmax, p):
        BasePropagator.__init__(self)
        self.vmax = vmax
        self.p = p

    def timestep(self, cars, obs):
        # TODO Here you should implement the car behaviour rules
        for i in range(cars.numCars):
            if cars.v[i] < self.vmax:
                cars.v[i] += 1
        for i in range(cars.numCars):
            dist = cars.distance(i)
            if cars.v[i] >= dist:
                cars.v[i] = dist - 1
        for i in range(cars.numCars):
            if cars.v[i] > 0:
                if random.random() < self.p:
                    cars.v[i] -= 1
        for i in range(cars.numCars):
            cars.x[i] += cars.v[i]
        cars.t += 1

        flow_road = 0
        for i in range(cars.numCars):
            flow_road += cars.v[i]
        flow = flow_road / cars.roadLength
        
        for i in range(cars.numCars):
            cars.f[i] = 1 + cars.v[i] + 0.2*cars.v[i]**3
            
        return flow
            

############################################################################################

def draw_cars(cars, cars_drawing):

    """ Used later on to generate the animation """
    theta = []
    r     = []

    for position in cars.x:
        # Convert to radians for plotting  only (do not use radians for the simulation!)
        theta.append(position * 2 * math.pi / cars.roadLength)
        r.append(1)

    return cars_drawing.scatter(theta, r, c=cars.c, cmap='hsv')


def animate(framenr, cars, obs, propagator, road_drawing, stepsperframe):

    """ Animation function which integrates a few steps and return a drawing """

    for it in range(stepsperframe):
        propagator.propagate(cars, obs)

    return draw_cars(cars, road_drawing),


class Simulation:

    def reset(self, cars=Cars()) :
        self.cars = cars
        self.obs = Observables()

    def __init__(self, cars=Cars()) :
        self.reset(cars)

    def plot_observables(self, title="simulation"):
        plt.clf()
        plt.title(title)
        plt.plot(self.obs.time, self.obs.flowrate)
        plt.xlabel('time')
        plt.ylabel('flow rate')
        plt.savefig(title + ".pdf")
        plt.show()

    # Run without displaying any animation (fast)
    def run(self,
            propagator,
            numsteps=200,           # final time
            title="simulation",     # Name of output file and title shown at the top
            ):

        for it in range(numsteps):
            propagator.propagate(self.cars, self.obs)

        self.plot_observables(title)

    # Run while displaying the animation of bunch of cars going in circe (slow-ish)
    def run_animate(self,
            propagator,
            numsteps=200,           # Final time
            stepsperframe=1,        # How many integration steps between visualising frames
            title="simulation",     # Name of output file and title shown at the top
            ):

        numframes = int(numsteps / stepsperframe)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
        ax.axis('off')
        # Call the animator, blit=False means re-draw everything
        anim = animation.FuncAnimation(plt.gcf(), animate,  # init_func=init,
                                       fargs=[self.cars,self.obs,propagator,ax,stepsperframe],
                                       frames=numframes, interval=50, blit=True, repeat=False)
        plt.show()

        # If you experience problems visualizing the animation and/or
        # the following figures comment out the next line 
        # plt.waitforbuttonpress(30)

        self.plot_observables(title)
    

# It's good practice to encapsulate the script execution in 
# a main() function (e.g. for profiling reasons)
def main() :

    # Here you can define one or more instances of cars, with possibly different parameters, 
    # and pass them to the simulator 

    # Be sure you are passing the correct initial conditions!
    cars = Cars(numCars = 20, roadLength=50)

    # Create the simulation object for your cars instance:
    simulation = Simulation(cars)

    #simulation.run_animate(propagator=ConstantPropagator())
    simulation.run_animate(propagator=MyPropagator(vmax=2, p=0.5))
    
    #simulation.run(propagator=ConstantPropagator(), numsteps=1500)
    #simulation.run(propagator=MyPropagator(vmax=2, p=0.5), numsteps=1500)

def exercise_22a_spacetime():
    cars = Cars(numCars = 5, roadLength=50)
    simulation = Simulation(cars)

    #propagator = MyPropagator(vmax=2, p=0.5)
    propagator = ConstantPropagator()

    numsteps = 200
    positions_over_time = []

    # Run the simulation manually, storing positions:
    for _ in range(numsteps):
        propagator.propagate(simulation.cars, simulation.obs)
        positions_over_time.append(simulation.cars.x.copy())

    # Convert to array for easier plotting:
    positions_over_time = np.array(positions_over_time)

    # Plot the space-time diagram:
    plt.figure()
    for car_index in range(cars.numCars):
        plt.scatter(positions_over_time[:, car_index], np.arange(numsteps), s=8, label=f"Car {car_index}")

    plt.xlabel("Position on road")
    plt.ylabel("Time")
    plt.title("Space–Time Diagram of Traffic, Constant Propagator")
    #plt.show()
    plt.savefig("Space–Time Diagram of Traffic, Constant Propagator.pdf")
    
def exercise_22a_flowrate():
    roadLength = 50
    vmax = 2
    p = 0.5

    numsCars = np.linspace(2, 50, 49, dtype=int)   # 20 density points from sparse to packed
    avg_flow_rates = []

    for numCars in numsCars:

        cars = Cars(numCars=numCars, roadLength=roadLength)
        simulation = Simulation(cars)
        #propagator = MyPropagator(vmax=vmax, p=p)
        propagator = ConstantPropagator()
        

        measurement_steps = 300

        total_flow = 0
        for _ in range(measurement_steps):
            propagator.propagate(simulation.cars, simulation.obs)
            total_flow += propagator.timestep(simulation.cars, simulation.obs)

        avg_flow = total_flow / measurement_steps
        avg_flow_rates.append(avg_flow)

    plt.figure()
    plt.plot(numsCars, avg_flow_rates, '-o')
    plt.xlabel("Amount of cars for a road of length 50")
    plt.ylabel("Average Flow Rate (cars per timestep)")
    plt.title("Fundamental Diagram of Traffic Flow, Constant Propagator")
    #plt.show() 
    plt.savefig("Fundamental Diagram of Traffic Flow, Constant Propagator.pdf") 

def exercise_22b(num_simulations=2):
    avg_flows = []

    for _ in range(num_simulations):
        cars = Cars(numCars=25, roadLength=50)
        simulation = Simulation(cars)
        propagator = MyPropagator(vmax=2, p=0.5)

        measurement_steps = 1450
        flows = []

        for _ in range(measurement_steps):
            propagator.propagate(simulation.cars, simulation.obs)
            flow = propagator.timestep(simulation.cars, simulation.obs)
            flows.append(flow)

        avg_flows.append(np.mean(flows))

    avg_flows = np.array(avg_flows, dtype=float)

    overall_average = np.mean(avg_flows)
    std_dev = np.std(avg_flows, ddof=1)
    std_error = std_dev / np.sqrt(len(avg_flows))

    print("Average flow across simulations:", overall_average)
    print("Standard error:", std_error)

def exercise_22c():
    vmax = 2
    p = 0.2
    density = 0.3
    road_lengths = np.linspace(10, 500, 100, dtype=int) 
    equilibration_steps = 1500

    plt.figure()
    avg_flows = []

    for L in road_lengths:
        numCars = int(density * L)

        cars = Cars(numCars=numCars, roadLength=L)
        simulation = Simulation(cars)
        propagator = MyPropagator(vmax=vmax, p=p)

        # equilibration phase (let system settle)
        total_flow = 0
        for _ in range(equilibration_steps):
            total_flow += propagator.timestep(simulation.cars, simulation.obs)

        avg_flow = total_flow / equilibration_steps
        avg_flows.append(avg_flow)

    plt.plot(road_lengths, avg_flows)
    plt.xlabel("Road Length (L)")
    plt.ylabel("Average Flow Rate (cars per timestep)")
    plt.title("Flow vs System Size at Constant Density")
    plt.savefig("Flow_vs_System_Size_at_Constant_Density.pdf")

def exercise_22d():
    #vmax_values = np.linspace(1.1,5,100)
    vmax_values = np.linspace(1,5,5)
    
    mean_flows = []
    for vmax in vmax_values:
        cars = Cars(numCars=40, roadLength=100) #density of 0.4
        sim = Simulation(cars)
        sim.run(propagator=MyPropagator(vmax=vmax, p=0.25), numsteps=1500)
        
        flow = np.array(sim.obs.flowrate, dtype=float)
        mean_flow = np.mean(flow)
        mean_flows.append(mean_flow)
    plt.clf()
    plt.title("Flowrate vs vmax for density=0.4, p=0.25, 1500 timesteps")
    plt.plot(vmax_values, mean_flows, label=f"vmax={vmax}")
    plt.xlabel("time")
    plt.ylabel("Flowrate")
    plt.legend()
    plt.savefig("Flowrate vs vmax for density=0.4, p=0.25, 1500 timesteps.pdf")
    #plt.show()
    
    
    mean_fuels = []
    for vmax in vmax_values:
        cars = Cars(numCars=40, roadLength=100) #density of 0.4
        sim = Simulation(cars)
        sim.run(propagator=MyPropagator(vmax=vmax, p=0.25), numsteps=1500)
        
        fuel = np.array(sim.obs.avg_fuel, dtype=float)
        mean_fuel = np.mean(fuel)
        mean_fuels.append(mean_fuel)
    plt.clf()
    plt.title("Mean fuel vs vmax for density=0.4, p=0.25, 1500 timesteps")
    plt.plot(vmax_values, mean_fuels, label=f"vmax={vmax}")
    plt.xlabel("vmax")
    plt.ylabel("Avg Fuel")
    plt.legend()
    plt.savefig("Mean fuels per car vs vmax for density=0.4, p=0.25, 1500 timesteps.pdf")
    #plt.show()
    
    mean_ratios = []
    for vmax in vmax_values:
        cars = Cars(numCars=40, roadLength=100) #density of 0.4
        sim = Simulation(cars)
        sim.run(propagator=MyPropagator(vmax=vmax, p=0.25), numsteps=1500)
        
        fuel = np.array(sim.obs.avg_fuel, dtype=float)
        flow = np.array(sim.obs.flowrate, dtype=float)
        ratio = np.mean(fuel)/np.mean(flow)
        mean_ratio = np.mean(ratio)
        mean_ratios.append(mean_ratio)
    plt.clf()
    plt.title("Fuel divided by the flowrate vs vmax for density=0.4, p=0.25, 1500 time steps")
    plt.plot(vmax_values, mean_ratios, label=f"vmax={vmax}")
    plt.xlabel("vmax")
    plt.ylabel("Ratios")
    plt.legend()
    plt.savefig("Fuel per car divided by the flowrate vs vmax for density=0.4, p=0.25, 1500 time steps.pdf")
    #plt.show()
    
    
# Calling 'main()' if the script is executed.
# If the script is instead just imported, main is not called (this can be useful if you want to
# write another script importing and utilizing the functions and classes defined in this one)
if __name__ == "__main__" :
    main()
    #exercise_22a_spacetime()
    #exercise_22a_flowrate()
    #exercise_22b()
    #exercise_22c()
    #exercise_22d()

