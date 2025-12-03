#!/bin/python3
# Python simulation of a simple planar pendulum with real time animation
# BH, OF, MP, AJ, TS 2020-10-20, latest version 2022-10-25.

from matplotlib import animation
import matplotlib.pyplot as plt
from pylab import *
from scipy.interpolate import interp1d

"""
    This script defines all the classes needed to simulate (and animate) a single pendulum.
    Hierarchy (somehow in order of encapsulation):
    - Oscillator: a struct that stores the parameters of an oscillator (harmonic or pendulum)
    - Observable: a struct that stores the oscillator's coordinates and energy values over time
    - BaseSystem: harmonic oscillators and pendolums are distinguished only by the expression of
                    the return force. This base class defines a virtual force method, which is
                    specified by its child classes
                    -> Harmonic: specifies the return force as -k*t (i.e. spring)
                    -> Pendulum: specifies the return force as -k*sin(t)
    - BaseIntegrator: parent class for all time-marching schemes; function integrate performs
                    a numerical integration steps and updates the quantity of the system provided
                    as input; function timestep wraps the numerical scheme itself and it's not
                    directly implemented by BaseIntegrator, you need to implement it in his child
                    classes (names are self-explanatory)
                    -> EulerCromerIntegrator: ...
                    -> VerletIntegrator: ...
                    -> RK4Integrator: ...
    - Simulation: this last class encapsulates the whole simulation procedure; functions are 
                    self-explanatory; you can decide whether to just run the simulation or to
                    run while also producing an animation: the latter option is slower
"""

# Global constants
G = 9.8  # gravitational acceleration

class Oscillator:

    """ Class for a general, simple oscillator """

    def __init__(self, m=1, c=4, t0=0, theta0=0, dtheta0=0, gamma=0):
        self.m = m              # mass of the pendulum bob
        self.c = c              # c = g/L
        self.L = G / c          # string length
        self.t = t0             # the time
        self.theta = theta0     # the position/angle
        self.dtheta = dtheta0   # the velocity
        self.gamma = gamma      # damping

class Observables:

    """ Class for storing observables for an oscillator """

    def __init__(self):
        self.time = []          # list to store time
        self.pos = []           # list to store positions
        self.vel = []           # list to store velocities
        self.energy = []        # list to store energy


class BaseSystem:
    
    def force(self, osc):

        """ Virtual method: implemented by the childc lasses  """

        pass


class Harmonic(BaseSystem):
    def force(self, osc):
        return - osc.m * ( osc.c*osc.theta + osc.gamma*osc.dtheta ) #With dampening


class Pendulum(BaseSystem):
    def force(self, osc):
        return - osc.m * ( osc.c*np.sin(osc.theta) + osc.gamma*osc.dtheta )
    


class BaseIntegrator:

    def __init__(self, _dt=0.01) :
        self.dt = _dt   # time step

    def integrate(self, simsystem, osc, obs):

        """ Perform a single integration step """
        
        self.timestep(simsystem, osc, obs)

        # Append observables to their lists
        obs.time.append(osc.t)
        obs.pos.append(osc.theta)
        obs.vel.append(osc.dtheta)
        # Function 'isinstance' is used to check if the instance of the system object is 'Harmonic' or 'Pendulum'
        if isinstance(simsystem, Harmonic) :
            # Harmonic oscillator energy
            obs.energy.append(0.5 * osc.m * osc.L ** 2 * osc.dtheta ** 2 + 0.5 * osc.m * G * osc.L * osc.theta ** 2)
        else :
            # Pendulum energy
            # TODO: Append the total energy for the pendulum (use the correct formula!)
            obs.energy.append(0.5 * osc.m * osc.L ** 2 * osc.dtheta ** 2 + osc.m * G * osc.L * (1-cos(osc.theta)))


    def timestep(self, simsystem, osc, obs):

        """ Virtual method: implemented by the child classes """
        
        pass


# HERE YOU ARE ASKED TO IMPLEMENT THE NUMERICAL TIME-MARCHING SCHEMES:

class EulerCromerIntegrator(BaseIntegrator):
    def timestep(self, simsystem, osc, obs):
        accel = simsystem.force(osc) / osc.m
        osc.t += self.dt
        # TODO: Implement the integration here, updating osc.theta and osc.dtheta
        #osc.dtheta -= osc.c * osc.theta * self.dt    
        osc.dtheta += accel * self.dt      
        osc.theta += osc.dtheta * self.dt  


class VerletIntegrator(BaseIntegrator):
    def timestep(self, simsystem, osc, obs):
        accel = simsystem.force(osc) / osc.m
        osc.t += self.dt
        # TODO: Implement the integration here, updating osc.theta and osc.dtheta
        osc.theta += osc.dtheta * self.dt + 0.5 * accel * self.dt**2
        accel_new = simsystem.force(osc) / osc.m
        osc.dtheta += 0.5 * (accel + accel_new) * self.dt


class RK4Integrator(BaseIntegrator):
    def timestep(self, simsystem, osc, obs):
        osc.t += self.dt
        # TODO: Implement the integration here, updating osc.theta and osc.dtheta
        
        def acceleration(theta, dtheta, t):
            osc_temp = type(osc)()     # temporary object of same class
            osc_temp.__dict__.update(osc.__dict__)
            osc_temp.theta = theta
            osc_temp.dtheta = dtheta
            return simsystem.force(osc_temp) / osc.m
        
        a1 = acceleration(osc.theta, osc.dtheta, osc.t) * self.dt
        b1 = osc.dtheta * self.dt
        
        a2 = acceleration(osc.theta + b1/2, osc.dtheta + a1/2, osc.t+self.dt/2 ) * self.dt
        b2 = (osc.dtheta + a1/2) * self.dt
        
        a3 = acceleration(osc.theta + b2/2, osc.dtheta + a2/2, osc.t+self.dt/2 ) * self.dt
        b3 = (osc.dtheta + a2/2) * self.dt
        
        a4 = acceleration(osc.theta+b3, osc.dtheta+a3, osc.t+self.dt) * self.dt
        b4 = (osc.dtheta+a3) * self.dt
        
        osc.dtheta += (1/6)*(a1 + 2*a2 + 2*a3 + a4)
        osc.theta += (1/6)*(b1 + 2*b2 + 2*b3 + b4)


# Animation function which integrates a few steps and return a line for the pendulum
def animate(framenr, simsystem, oscillator, obs, integrator, pendulum_line, stepsperframe):
    
    for it in range(stepsperframe):
        integrator.integrate(simsystem, oscillator, obs)

    x = np.array([0, np.sin(oscillator.theta)])
    y = np.array([0, -np.cos(oscillator.theta)])
    pendulum_line.set_data(x, y)
    return pendulum_line,


class Simulation:

    def reset(self, osc=Oscillator()) :
        self.oscillator = osc
        self.obs = Observables()

    def __init__(self, osc=Oscillator()) :
        self.reset(osc)

    # Run without displaying any animation (fast)
    def run(self,
            simsystem,
            integrator,
            tmax=30.,               # final time
            ):

        n = int(tmax / integrator.dt)

        for it in range(n):
            integrator.integrate(simsystem, self.oscillator, self.obs)

    # Run while displaying the animation of a pendulum swinging back and forth (slow-ish)
    # If too slow, try to increase stepsperframe
    def run_animate(self,
            simsystem,
            integrator,
            tmax=30.,               # final time
            stepsperframe=1         # how many integration steps between visualising frames
            ):

        numframes = int(tmax / (stepsperframe * integrator.dt))-2

        # WARNING! If you experience problems visualizing the animation try to comment/uncomment this line
        plt.clf()

        # If you experience problems visualizing the animation try to comment/uncomment this line
        # fig = plt.figure()

        ax = plt.subplot(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))
        plt.axhline(y=0)  # draw a default hline at y=1 that spans the xrange
        plt.axvline(x=0)  # draw a default vline at x=1 that spans the yrange
        pendulum_line, = ax.plot([], [], lw=5)
        plt.title(title)
        # Call the animator, blit=True means only re-draw parts that have changed
        anim = animation.FuncAnimation(plt.gcf(), animate,  # init_func=init,
                                       fargs=[simsystem,self.oscillator,self.obs,integrator,pendulum_line,stepsperframe],
                                       frames=numframes, interval=25, blit=True, repeat=False)

        # If you experience problems visualizing the animation try to comment/uncomment this line
        # plt.show()

        # If you experience problems visualizing the animation try to comment/uncomment this line
        #plt.waitforbuttonpress(10)
        anim.save("pendulum.mp4", fps=30, writer="ffmpeg")
        print("Animation saved as pendulum.mp4")

    # Plot coordinates and energies (to be called after running)
    def plot_observables(self, title="simulation", ref_E=None):

        plt.clf()
        plt.title(title)
        plt.plot(self.obs.time, self.obs.pos, 'b-', label="Position")
        plt.plot(self.obs.time, self.obs.vel, 'r-', label="Velocity")
        plt.plot(self.obs.time, self.obs.energy, 'g-', label="Energy")
        if ref_E != None :
            plt.plot([self.obs.time[0],self.obs.time[-1]] , [ref_E, ref_E], 'k--', label="Ref.")
        plt.xlabel('time')
        plt.ylabel('observables')
        plt.legend()
        plt.savefig(title + ".pdf")
        #plt.show()


# It's good practice to encapsulate the script execution in 
# a function (e.g. for profiling reasons)
def exercise_11() :
    c_value = 4
    m = 1.0
    dt_list = [0.01, 0.05, 0.1, 0.2]
    integrators = [EulerCromerIntegrator, VerletIntegrator, RK4Integrator]
    integrator_names = ['Euler-Cromer', 'Velocity-Verlet', 'RK4']
    thetas_pi = [0.1, 0.5]
    tmax = 30.0


    omega = np.sqrt(c_value)


    for theta_pi in thetas_pi:
        theta0 = theta_pi * np.pi
        for dt in dt_list:
            for integ_cls, name in zip(integrators, integrator_names):
                osc = Oscillator(m=m, c=c_value, theta0=theta0, dtheta0=0.0)
                sim = Simulation(osc)
                integ = integ_cls(_dt=dt)
                sim.run(Harmonic(), integ, tmax=tmax)
                ref_E = sim.obs.energy[0]
                sim.plot_observables(title=f"Harmonic {name}, dt={dt}", ref_E=ref_E)


    best_integrator = RK4Integrator(_dt=0.01)
    for theta_pi in thetas_pi:
        theta0 = theta_pi * np.pi
        osc = Oscillator(m=m, c=c_value, theta0=theta0, dtheta0=0.0)
        sim = Simulation(osc)
        sim.run(Pendulum(), best_integrator, tmax=tmax)
        ref_E = sim.obs.energy[0]
        sim.plot_observables(title=f"Pendulum RK4", ref_E=ref_E)
    
    dt = 0.2
    integ_cls = VerletIntegrator
    osc = Oscillator(m=m, c=c_value, theta0=0.1*np.pi , dtheta0=0.0)
    sim = Simulation(osc)
    integ = integ_cls(_dt=dt)
    sim.run(Harmonic(), integ, tmax=tmax)
    ref_E = sim.obs.energy[0]
    sim.plot_observables(title=f"Harmonic Verlet, dt=0.2", ref_E=ref_E)
    
def exercise_11_summary():
    c_value = 4
    m = 1.0
    dt_list = [0.2, 0.2, 0.2, 0.2]
    integrators = [EulerCromerIntegrator, VerletIntegrator, RK4Integrator]
    integrator_names = ['Euler-Cromer', 'Velocity-Verlet', 'RK4']
    thetas_pi = [0.1, 0.5]
    tmax = 100

    omega = np.sqrt(c_value)

    # --- 1) Harmonic oscillator comparison: all integrators, same plot ---
    for theta_pi in thetas_pi:
        theta0 = theta_pi * np.pi

        fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        fig.suptitle(f"Harmonic Oscillator Comparison, θ₀={theta0:.2f} rad")

        t_ref = np.linspace(0, tmax, 2000)
        theta_exact = theta0 * np.cos(omega * t_ref)
        dtheta_exact = -theta0 * omega * np.sin(omega * t_ref)

        axs[0].plot(t_ref, theta_exact, label="Exact", linewidth=2)
        axs[1].plot(t_ref, dtheta_exact, label="Exact", linewidth=2)

        for (integ_cls, name), dt in zip(zip(integrators, integrator_names), dt_list):
            osc = Oscillator(m=m, c=c_value, theta0=theta0, dtheta0=0.0)
            sim = Simulation(osc)
            integ = integ_cls(_dt=dt)
            sim.run(Harmonic(), integ, tmax=tmax)

            t = sim.obs.time
            axs[0].plot(t, sim.obs.pos, label=f"{name}, dt={dt}")
            axs[1].plot(t, sim.obs.vel, label=f"{name}, dt={dt}")
            axs[2].plot(t, sim.obs.energy, label=f"{name}, dt={dt}")

        axs[0].set_ylabel("θ(t)")
        axs[1].set_ylabel("ω(t)")
        axs[2].set_ylabel("Energy")
        axs[2].set_xlabel("Time")
        for ax in axs:
            ax.grid(True)
            ax.legend()

        plt.show()

    # --- 2) Pendulum with best integrator (RK4), compare small vs large angle ---
    best_integrator = RK4Integrator(_dt=0.01)

    # Create one figure for overlapping plots
    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    fig.suptitle("Pendulum (RK4): Small vs Large Initial Angle")

    for theta_pi in thetas_pi:
        theta0 = theta_pi * np.pi

        osc = Oscillator(m=m, c=c_value, theta0=theta0, dtheta0=0.0)
        sim = Simulation(osc)
        sim.run(Pendulum(), best_integrator, tmax=tmax)

        # Extract data once, no built-in plot call
        t = sim.obs.time
        theta = sim.obs.pos
        dtheta = sim.obs.vel
        energy = sim.obs.energy

        label = f"θ₀ = {theta0:.2f} rad"
        
        axs[0].plot(t, theta, label=label)
        axs[1].plot(t, dtheta, label=label)
        axs[2].plot(t, energy, label=label)

    axs[0].set_ylabel("θ(t)")
    axs[1].set_ylabel("ω(t)")
    axs[2].set_ylabel("Energy")
    axs[2].set_xlabel("Time")

    for ax in axs:
        ax.grid(True)
        ax.legend()

    plt.show()

    # --- 3) Highlight Velocity-Verlet dt stability for Harmonic ---
    osc = Oscillator(m=m, c=c_value, theta0=0.1*np.pi, dtheta0=0.0)
    sim = Simulation(osc)
    integ = VerletIntegrator(_dt=0.2)
    sim.run(Harmonic(), integ, tmax=tmax)
    sim.plot_observables(title="Harmonic with Verlet, dt=0.2")
    
def RK4_longtime():
    c_value = 4
    m = 1.0
    dt = 0.2
    integ = RK4Integrator()
    thetas_pi = [0.1, 0.5]
    tmax = 400
    for theta_pi in thetas_pi:
        theta0 = theta_pi * np.pi
        osc = Oscillator(m=m, c=c_value, theta0=theta0, dtheta0=0.0)
        sim = Simulation(osc)
        integ = RK4Integrator(dt)
        sim.run(Harmonic(), integ, tmax=tmax)
        ref_E = sim.obs.energy[0]
        sim.plot_observables(title=f"Harmonic Oscillator, dt={dt}", ref_E=ref_E)

def exercise_12():
    c_value = 4
    m = 1.0
    dt = 0.001
    tmax = 20
    integrator = RK4Integrator(_dt=dt)
    g = G
    L = g / c_value

    theta0_list = np.linspace(0.01*np.pi, 0.8*np.pi, 8)
    
    T_pendulum = []
    T_harmonic = []
    T_theory = []

    for theta0 in theta0_list:
        #Pendulum
        osc = Oscillator(m=m, c=c_value, theta0=theta0, dtheta0=0.0)
        sim = Simulation(osc)
        sim.run(Pendulum(), integrator, tmax=tmax)
        pos = np.array(sim.obs.pos)
        time = np.array(sim.obs.time)

        peaks = (np.diff(np.sign(np.diff(pos))) < 0).nonzero()[0] + 1
        if len(peaks) > 1:
            T_num = np.mean(np.diff(time[peaks]))
        else:
            T_num = np.nan
        T_pendulum.append(T_num)

        #Harmonic oscillator
        osc_h = Oscillator(m=m, c=c_value, theta0=theta0, dtheta0=0.0)
        sim_h = Simulation(osc_h)
        sim_h.run(Harmonic(), integrator, tmax=tmax)
        pos_h = np.array(sim_h.obs.pos)
        time_h = np.array(sim_h.obs.time)
        peaks_h = (np.diff(np.sign(np.diff(pos_h))) < 0).nonzero()[0] + 1
        if len(peaks_h) > 1:
            T_num_h = np.mean(np.diff(time_h[peaks_h]))
        else:
            T_num_h = np.nan
        T_harmonic.append(T_num_h)

        #Theory
        T0 = 2 * np.pi * np.sqrt(L / g)
        T_series = T0 * (1 + (1/16)*theta0**2 + (11/3072)*theta0**4 + (173/737280)*theta0**6)
        T_theory.append(T_series)

    # --- Plot results ---
    plt.figure()
    plt.plot(theta0_list, T_pendulum, 'ro-', label='Pendulum (numerical)')
    plt.plot(theta0_list, T_harmonic, 'b--', label='Harmonic oscillator')
    plt.plot(theta0_list, T_theory, 'g-', label='Pendulum (theory)')
    plt.xlabel(r'Initial angle $\theta_0$ [rad]')
    plt.ylabel('Period T [s]')
    plt.title('Period vs Initial Angle')
    plt.legend()
    plt.savefig("T(initial-angle).pdf")
    
def exercise_13():
    c_value = 4
    m = 1.0
    dt = 0.001
    tmax = 20
    integrator = RK4Integrator(_dt=dt)
    g = G
    L = g / c_value
    gamma = 0.5
    theta0 = 1
    
    #part a, plot nr1
    osc = Oscillator(m=m, c=c_value, theta0=theta0, dtheta0=0.0, gamma=gamma)
    sim = Simulation(osc)
    sim.run(Harmonic(), integrator, tmax=tmax)
    pos = np.array(sim.obs.pos)
    vel = np.array(sim.obs.vel)
    energy = np.array(sim.obs.energy)
    time = np.array(sim.obs.time)
    sim.plot_observables(title=f"Harmonic_plot")
    
    #part b, find tau
    deriv = np.diff(pos)
    sign_change = np.sign(deriv)
    maxima_idx = (np.diff(sign_change) < 0).nonzero()[0] + 1
    minima_idx = (np.diff(sign_change) > 0).nonzero()[0] + 1

    max_times = time[maxima_idx]
    min_times = time[minima_idx]
    max_vals = pos[maxima_idx]
    min_vals = pos[minima_idx]

    upper_env = interp1d(max_times, max_vals, kind='cubic', fill_value='extrapolate')
    lower_env = interp1d(min_times, min_vals, kind='cubic', fill_value='extrapolate')
    
    dense_t = np.linspace(max_times[0], max_times[-1], 2000)
    upper_vals = upper_env(dense_t)
    below = np.where(upper_vals <= 0.37)[0]
    if len(below) > 0:
        i = below[0]
        tau_t = dense_t[i]

    print(tau_t)

    theta_1e = np.max(np.abs(pos)) / np.e

    plt.figure(figsize=(8,4))
    plt.plot(time, pos, 'b-', label='θ(t)')
    plt.plot(max_times, max_vals, 'ro', label='maxima')
    plt.plot(min_times, min_vals, 'go', label='minima')
    plt.plot(time, upper_env(time), 'r--', label='upper envelope')
    plt.plot(time, lower_env(time), 'g--', label='lower envelope')
    
    plt.axhline(theta_1e, color='k', linestyle='--', linewidth=1.5, label='θ = max(θ)/e')
    
    plt.xlabel('Time [s]')
    plt.ylabel('θ (rad)')
    plt.title('Interpolated Upper and Lower Envelopes of θ(t)')
    plt.legend()
    plt.grid(True)
    plt.savefig("Interpolated Upper and Lower Envelopes of θ(t).pdf")
    
    #part c tau, beroende av gamma
    gamma_list = np.linspace(0, 3, 5) #50
    tau_list = []
    
    for gamma in gamma_list:
        osc = Oscillator(m=m, c=c_value, theta0=theta0, dtheta0=0.0, gamma=gamma)
        sim = Simulation(osc)
        sim.run(Harmonic(), integrator, tmax=tmax)
        pos = np.array(sim.obs.pos)
        vel = np.array(sim.obs.vel)
        energy = np.array(sim.obs.energy)
        time = np.array(sim.obs.time)
        sim.plot_observables(title=f"Harmonic_plot1")
        
        deriv = np.diff(pos)
        sign_change = np.sign(deriv)
        maxima_idx = (np.diff(sign_change) < 0).nonzero()[0] + 1
        minima_idx = (np.diff(sign_change) > 0).nonzero()[0] + 1

        max_times = time[maxima_idx]
        min_times = time[minima_idx]
        max_vals = pos[maxima_idx]
        min_vals = pos[minima_idx]

        upper_env = interp1d(max_times, max_vals, kind='cubic', fill_value='extrapolate')
        lower_env = interp1d(min_times, min_vals, kind='cubic', fill_value='extrapolate')
        
        dense_t = np.linspace(max_times[0], max_times[-1], 2000)
        upper_vals = upper_env(dense_t)
        below = np.where(upper_vals <= 0.37)[0]
        if len(below) > 0:
            i = below[0]
            tau_t = dense_t[i]
            #print(tau_t)
        tau_list.append(tau_t)
        
    plt.figure()
    plt.plot(gamma_list, tau_list, 'r-', label='Tau(gamma)')
    plt.xlabel('Gamma')
    plt.ylabel('Tau [s]')
    plt.title('Tau (relaxation time) vs Gamma (dampending)')
    plt.legend()
    plt.savefig("Tau(gamma).pdf")
    
    #part d: find gamma critical
    gamma = 3
    #gamma = 4
    
    osc = Oscillator(m=m, c=c_value, theta0=theta0, dtheta0=0.0, gamma=gamma)
    sim = Simulation(osc)
    sim.run(Harmonic(), integrator, tmax=tmax)
    pos = np.array(sim.obs.pos)
    vel = np.array(sim.obs.vel)
    energy = np.array(sim.obs.energy)
    time = np.array(sim.obs.time)
    sim.plot_observables(title=f"Harmonic_plot3")
    
    pos = np.array(sim.obs.pos)
    time = np.array(sim.obs.time)

    # Value we want to detect crossing
    threshold = 1/np.e

    # Compute difference from threshold
    diff = pos - threshold

    crossing_times = []

    for i in range(len(diff)-1):
        if diff[i] == 0:
            crossing_times.append(time[i])
        elif diff[i] * diff[i+1] < 0:  # sign change → crossing
            # Linear interpolation for better accuracy
            t0, t1 = time[i], time[i+1]
            y0, y1 = diff[i], diff[i+1]
            t_cross = t0 - y0 * (t1 - t0) / (y1 - y0)
            crossing_times.append(t_cross)

    print("Times where θ(t) = 1/e:", crossing_times)
    
def exercise_14():
    gamma = 1
    c_value = 4
    theta0 = np.pi/2
    dtheta0 = 0.0
    tmax = 30
    
    osc = Oscillator(c=c_value, theta0=theta0, dtheta0=dtheta0, gamma=gamma)
    sim = Simulation(osc)
    sim.run(Pendulum(), RK4Integrator(0.01), tmax=tmax)
    
    pos = np.array(sim.obs.pos)
    vel = np.array(sim.obs.vel)
    
    plt.figure(figsize=(6, 5))
    plt.plot(pos, vel, color='blue')
    plt.title("Phase Portrait of the Pendulum (dθ/dt vs θ)")
    plt.xlabel("Position (θ)")
    plt.ylabel("Velocity (dθ/dt)")
    plt.grid(True)
    plt.savefig("phase portrait.pdf")
    
"""
    This directive instructs Python to run what comes after ' if __name__ == "__main__" : '
    if the script pendulum_template.py is executed 
    (e.g. by running "python3 pendulum_template.py" in your favourite terminal).
    Otherwise, if pendulum_template.py is imported as a library 
    (e.g. by calling "import pendulum_template as dp" in another Python script),
    the following is ignored.
    In this way you can choose whether to code the solution to the exericises here in this script 
    or to have (a) separate script(s) that include pendulum_template.py as library.
"""

def test_f():
    gamma = 1
    c_value = 4
    theta0 = np.pi/2
    dtheta0 = 0.0
    tmax = 30
    
    osc = Oscillator(c=c_value, theta0=theta0, dtheta0=dtheta0, gamma=gamma)
    sim = Simulation(osc)
    sim.run_animate(Pendulum(), RK4Integrator(0.01), tmax=tmax)

if __name__ == "__main__" :
    #exercise_11()
    #exercise_11_summary()
    #exercise_12()
    exercise_13()
    #exercise_14()
    #test_f()
    #RK4_longtime()