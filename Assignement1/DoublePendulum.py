#!/bin/python3

# Python simulation of a double pendulum with real time animation.
# BH, MP, AJ, TS 2020-10-27, latest version 2022-10-25.

from matplotlib import animation
import matplotlib.pyplot as plt

# Using numba to speed up force calculation
# More info: https://numba.pydata.org/numba-doc/latest/user/5minguide.html
import numba as nmb

# Numba likes numpy
import numpy as np

"""
    This script simulates and animates a double pendulum.
    Classes are similar to the ones of pendolum_template.py. The main differences are:
    - coordinates are obtained from the total energy value E (look at what functions
        Oscillator.p2squaredFromH and Oscillator.__init__ do)
    - you are asked to implement the expression for the derivatives of the Hamiltonian 
        w.r.t. coordinates p1 and p2
    - you are asked to check when the conditions to produce the Poincare' map are
        satisfied and append the coordinates' values to some container to then plot
"""

# Global constants
G = 9.8  # gravitational acceleration

"""
    The state vector is defined as:
        x = [q1, q2, p1, p2]
    That is: q1=x[0], q2=x[1], p1=x[2], p2=x[3]
"""

# Kinetic energy
def Ekin(osc):
    return 1 / (2.0 * osc.m * osc.L * osc.L) * ( osc.x[2] * osc.x[2] + 2.0 * osc.x[3] * osc.x[3] - 2.0 * osc.x[2] * osc.x[3] * np.cos(
        osc.x[0] - osc.x[1])) / (1 + (np.sin(osc.x[0] - osc.x[1])) ** 2)

# Potential energy
def Epot(osc):
    return osc.m * G * osc.L * (3 - 2 * np.cos(osc.x[0]) - np.cos(osc.x[1]))


# Class that holds the parameter and state of a double pendulum
class Oscillator:

    def p2squaredFromH(self):
        return (self.E - Epot(self)) * (1 + (np.sin(self.x[0] - self.x[1])) ** 2) * self.m * self.L * self.L

    # Initial condition is [q1, q2, p1, p2]; p2 is however re-obtained based on the value of E
    # therefore you can use any value for init_cond[3]
    def __init__(self, m=1, L=1, t0=0, E=15, init_cond=[0.0, 0.0, 0.0, -1.0]) :
        self.m = m      # mass of the pendulum bob
        self.L = L      # arm length
        self.t = t0     # the initial time
        self.E = E      # total conserved energy
        self.x = np.zeros(4)
        self.x[0] = init_cond[0]
        self.x[1] = init_cond[1]
        self.x[2] = init_cond[2]
        self.x[3] = -1.0
        while (self.x[3] < 0):
            # Comment the two following lines in case you want to exactly prescribe values to q1 and q2
            # However, be sure that there exists a value of p2 compatible with the imposed total energy E!
            #self.x[0] = np.pi * (2 * np.random.random() - 1)
            #self.x[1] = np.pi * (2 * np.random.random() - 1)
            p2squared = self.p2squaredFromH()
            if (p2squared >= 0):
                self.x[3] = np.sqrt(p2squared)
        self.q2_prev = self.x[1]
        print("Initialization:")
        print("E  = "+str(self.E))
        print("q1 = "+str(self.x[0]))
        print("q2 = "+str(self.x[1]))
        print("p1 = "+str(self.x[2]))
        print("p2 = "+str(self.x[3]))


# Class for storing observables for an oscillator
class Observables:

    def __init__(self):
        self.time = []          # list to store time
        self.q1list = []        # list to store q1
        self.q2list = []        # list to store q2
        self.p1list = []        # list to store p1
        self.p2list = []        # list to store p2
        self.epot = []          # list to store potential energy
        self.ekin = []          # list to store kinetic energy
        self.etot = []          # list to store total energy
        self.poincare_q1 = []   # list to store q1 for Poincare plot
        self.poincare_p1 = []   # list to store p1 for Poincare plot


# Derivate of H with respect to p1
@nmb.jit(nopython=True)
def dHdp1(x, m, L):
    return 1 / (2 * m * L * L * (1 + (np.sin(x[0] - x[1])**2))) * (2 * x[2] - 2 * x[3] * np.cos(x[0] - x[1]))


# Derivate of H with respect to p2
@nmb.jit(nopython=True)
def dHdp2(x, m, L):
    return 1 / (2 * m * L *L * (1 + (np.sin(x[0] - x[1])**2))) * (4 * x[3] - 2 * x[2] * np.cos(x[0] - x[1]))


# Derivate of H with respect to q1
@nmb.jit(nopython=True)
def dHdq1(x, m, L):
    return 1 / (2.0 * m * L * L) * (
        -2 * (x[2] * x[2] + 2 * x[3] * x[3]) * np.cos(x[0] - x[1]) + x[2] * x[3] * (4 + 2 * (np.cos(x[0] - x[1])) ** 2)) * np.sin(
            x[0] - x[1]) / (1 + (np.sin(x[0] - x[1])) ** 2) ** 2 + m * G * L * 2.0 * np.sin(x[0])

# Derivate of H with respect to q2
@nmb.jit(nopython=True)
def dHdq2(x, m, L):
    return 1 / (2.0 * m * L * L) * (
        2 * (x[2] * x[2] + 2 * x[3] * x[3]) * np.cos(x[0] - x[1]) - x[2] * x[3] * (4 + 2 * (np.cos(x[0] - x[1])) ** 2)) * np.sin(x[0] - x[1]) / (
            1 + (np.sin(x[0] - x[1])) ** 2) ** 2 + m * G * L * np.sin(x[1])

def HamiltonEquations(x, m, L):
    return np.array([dHdp1(x, m, L), dHdp2(x, m, L), -dHdq1(x, m, L), -dHdq2(x, m, L)])

class RK4Integrator:

    def __init__(self, dt=0.01):
        self.dt = dt    # time step
        self.mult_vec = np.transpose(np.array([1,2,2,1]))

    def integrate(self,
                  osc,
                  obs, 
                  ):

        """ Perform a single integration step """
        self.timestep(osc, obs)

        """ Append observables to their lists """
        obs.time.append(osc.t)
        obs.q1list.append(osc.x[0])
        obs.q2list.append(osc.x[1])
        obs.p1list.append(osc.x[2])
        obs.p2list.append(osc.x[3])
        obs.epot.append(Epot(osc))
        obs.ekin.append(Ekin(osc))
        obs.etot.append(Epot(osc) + Ekin(osc))
        if (osc.q2_prev * osc.x[1] < 0) and (osc.x[2] < osc.x[3]):
            obs.poincare_q1.append(osc.x[0])
            obs.poincare_p1.append(osc.x[2])
        osc.q2_prev = osc.x[1]


    """
        Implementation of RK4 for a system of 4 variables
        It's much more compact when you write in in vector form
    """
    def timestep(self, osc, obs):
        dt = self.dt
        osc.t += dt

        # Initialization
        x = osc.x
        m = osc.m
        L = osc.L

        # RK4 coefficients (Butcher tableau)
        ab = np.zeros( (4,4), dtype=float )

        # First sub-step:
        ab[:,0] = dt*HamiltonEquations(x,m,L)

        # Second sub-step:
        ab[:,1] = dt*HamiltonEquations(x+0.5*ab[:,0],m,L)

        # Third sub-step:
        ab[:,2] = dt*HamiltonEquations(x+0.5*ab[:,1],m,L)

        # Fourth sub-step:
        ab[:,3] = dt*HamiltonEquations(x+ab[:,2],m,L)

        osc.x += np.matmul(ab,self.mult_vec) / 6.0


# Animation function which integrates a few steps and return a line for the pendulum
def animate(framenr, osc, obs, integrator, pendulum_lines, stepsperframe):
    for it in range(stepsperframe):
        integrator.integrate(osc, obs)

    x1 = np.sin(osc.x[0])
    y1 = -np.cos(osc.x[0])
    x2 = x1 + np.sin(osc.x[1])
    y2 = y1 - np.cos(osc.x[1])
    pendulum_lines.set_data([0, x1, x2], [0, y1, y2])
    return pendulum_lines,


class Simulation:

    def __init__(self, osc=None):
        self.reset(osc)

    def reset(self, osc=None):
        if osc is None:
            osc = Oscillator()
        self.oscillator = osc
        self.obs = Observables()

    def run(self,
            integrator,
            tmax=300.,   # final time
            outfile='energy1.pdf'
            ):

        n = int(tmax / integrator.dt)

        for it in range(n):
            integrator.integrate(self.oscillator, self.obs)

        #self.plot_observables(title="Energy="+str(self.oscillator.E))

    def run_animate(self,
            integrator,
            tmax=80.,           # final time
            stepsperframe=5,    # how many integration steps between visualising frames
            outfile='energy1.pdf'
            ):

        numframes = int(tmax / (stepsperframe * integrator.dt))

        plt.clf()

        # If you experience problems visualizing the animation try to comment/uncomment this line
        fig = plt.figure()

        ax = plt.subplot(xlim=(-2.2, 2.2), ylim=(-2.2, 2.2))
        plt.axhline(y=0)  # draw a default hline at y=1 that spans the xrange
        plt.axvline(x=0)  # draw a default vline at x=1 that spans the yrange
        pendulum_lines, = ax.plot([], [], lw=5)

        # Call the animator, blit=True means only re-draw parts that have changed
        anim = animation.FuncAnimation(fig, animate,  # init_func=init,
                                       fargs=[self.oscillator, self.obs, integrator, pendulum_lines, stepsperframe],
                                       frames=numframes, interval=25, blit=True, repeat=False)
        
        # If you experience problems visualizing the animation try to comment/uncomment this line 
        #plt.show()

        # If you experience problems visualizing the animation try to comment/uncomment this line 
        plt.waitforbuttonpress(10)

    # Plot coordinates and energies (to be called after running)
    def plot_observables(self, title="Double pendulum") :
        '''
        plt.figure()
        plt.title(title)
        plt.xlabel('q1')
        plt.ylabel('p1')
        plt.plot(self.obs.q1list, self.obs.p1list)
        plt.tight_layout()  # adapt the plot area tot the text with larger fonts 
        '''
        """
        plt.figure()
        plt.title(title)
        plt.xlabel('q2')
        plt.ylabel('p2')
        plt.plot(self.obs.q2list, self.obs.p2list)
        plt.plot([0.0, 0.0], [min(self.obs.p2list), max(self.obs.p2list)], 'k--')
        plt.tight_layout()  # adapt the plot area tot the text with larger fonts 
        """
        """
        plt.figure()
        plt.title(title)
        plt.xlabel('q1')
        plt.ylabel('p1')
        plt.plot(self.obs.poincare_q1, self.obs.poincare_p1, 'o')
        plt.tight_layout()  # adapt the plot area tot the text with larger fonts
        """
        """
        plt.figure()
        plt.title(title)
        plt.xlabel('time')
        plt.ylabel('energy')
        plt.plot(self.obs.time, self.obs.epot, self.obs.time, self.obs.ekin, self.obs.time, self.obs.etot)
        plt.legend(('Epot', 'Ekin', 'Etot'))
        plt.tight_layout()  # adapt the plot area tot the text with larger fonts 
        """
        #plt.show()

        #plt.savefig(title + ".pdf")

 
def exercise_15a() :
    E_list = [1, 15, 40]
    title_list = ["Energy=1 DP anim", "Energy=15 DP anim", "Energy=40 DP anim"]
    q1 = 0.0
    tmax = 30
    L = 1
    
    for E, title in zip(E_list, title_list):
        osc = Oscillator(E = E, init_cond=[0.0, 0.0, q1, -1.0])
        sim = Simulation(osc)
        sim.run_animate(integrator = RK4Integrator(0.05), tmax=tmax, title= title)
        
        # Convert angles to positions
        x1 = L * np.sin(sim.obs.q1list)
        y1 = -L * np.cos(sim.obs.q1list)

        x2 = x1 + L * np.sin(sim.obs.q2list)
        y2 = y1 - L* np.cos(sim.obs.q2list)

        plt.figure()
        plt.title(title + " (Real Space Trajectory)")
        plt.xlabel('x position')
        plt.ylabel('y position')

        # Trace (the path of the second mass)
        plt.plot(x2, y2, linewidth=1)

        # Optionally plot the rod + masses for the last frame
        plt.scatter([0, x1[-1], x2[-1]], [0, y1[-1], y2[-1]], s=30)
        plt.plot([0, x1[-1]], [0, y1[-1]], 'k-', linewidth=1)
        plt.plot([x1[-1], x2[-1]], [y1[-1], y2[-1]], 'k-', linewidth=1)

        plt.gca().set_aspect('equal', 'box')
        plt.tight_layout()
        plt.savefig(title + ".pdf")
        
def exercise_15b():
    E_list = [1, 15, 40]
    title_list = ["Energy=1 phase space", "Energy=15 phase space", "Energy=40 phase space"]
    q1 = 0.0
    tmax = 30
    L = 1
    
    for E, title in zip(E_list, title_list):
        osc = Oscillator(E = E, init_cond=[0.0, 0.0, q1, -1.0])
        sim = Simulation(osc)
        sim.run(integrator = RK4Integrator(0.05), tmax=tmax)
        
        plt.figure()
        plt.title(title)
        plt.xlabel('q1')
        plt.ylabel('p1')
        plt.plot(sim.obs.q1list, sim.obs.p1list)
        plt.tight_layout()  # adapt the plot area tot the text with larger fonts 
        plt.savefig(title + "1.pdf")
        
        plt.figure()
        plt.title(title)
        plt.xlabel('q2')
        plt.ylabel('p2')
        plt.plot(sim.obs.q2list, sim.obs.p2list)
        plt.plot([0.0, 0.0], [min(sim.obs.p2list), max(sim.obs.p2list)], 'k--')
        plt.tight_layout()  # adapt the plot area tot the text with larger fonts 
        plt.savefig(title + "2.pdf")
        
def exercise_15c():
    obslist = []
    for _ in range(5):
        sim = Simulation()
        sim.run(RK4Integrator())
        obslist.append(sim.obs.poincare_q1)
        obslist.append(sim.obs.poincare_p1)
    plt.figure()
    plt.title("Poincare map")
    plt.xlabel('q1')
    plt.ylabel('p1')
    plt.plot(obslist[0], obslist[1], 'or', markersize = 2)
    plt.plot(obslist[2], obslist[3], 'og', markersize = 2)
    plt.plot(obslist[4], obslist[5], 'ob', markersize = 2)
    plt.plot(obslist[6], obslist[7], 'ok', markersize = 2)
    plt.plot(obslist[8], obslist[9], 'oy', markersize = 2)
    plt.tight_layout() 
    plt.show()
    
    
    E = 15
    
    # First initial condition (0,0,0)
    osc1 = Oscillator(E=E, init_cond=[0.0, 0.0, 0.0, -1.0])
    sim1 = Simulation(osc1)
    sim1.run(RK4Integrator())

    # Second initial condition (1.1,0,0)
    osc2 = Oscillator(E=E, init_cond=[1.1, 0.0, 0.0, -1.0])
    sim2 = Simulation(osc2)
    sim2.run(RK4Integrator())

    plt.figure()
    plt.title("Poincaré Plot at E=15: Comparing Two Initial Conditions")
    plt.xlabel("q1")
    plt.ylabel("p1")

    plt.plot(sim1.obs.poincare_q1, sim1.obs.poincare_p1, 'o', markersize=2, label="IC: (0,0,0)")
    plt.plot(sim2.obs.poincare_q1, sim2.obs.poincare_p1, 'o', markersize=2, label="IC: (1.1,0,0)")

    plt.legend()
    plt.tight_layout()
    plt.show()

    

if __name__ == "__main__" :
    #exercise_15a()
    #exercise_15b()
    exercise_15c()