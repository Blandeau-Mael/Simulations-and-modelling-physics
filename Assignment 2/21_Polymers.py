import random
import matplotlib.pyplot as plt
import numpy as np

def random_walk(steps):
    x, y = 0, 0
    xs = [x]
    ys = [y]
    for _ in range(steps):
        r = int(random.random() * 4)
        if r == 0:
            x += 1
        elif r == 1:
            x -= 1
        elif r == 2:
            y += 1
        else:
            y -= 1
        xs.append(x)
        ys.append(y)
    return xs, ys

def lcg_random_walk(n_steps, r0, a, c, m):
    """
    r_n = (a * r_{n-1} + c) % m
    r // (m//4).
    m has to be >0 4
    """
    r = r0
    x = y = 0
    xs = [x]
    ys = [y]

    for _ in range(n_steps):
        r = (a * r + c) % m
        direction = int(4*r/m)

        if direction == 0:
            x += 1
        elif direction == 1:
            x -= 1
        elif direction == 2:
            y += 1
        else:
            y -= 1

        xs.append(x)
        ys.append(y)

    return xs, ys

def self_avoiding_walk(steps):
    x, y = 0, 0
    xs = [x]
    ys = [y]
    visited = {(x, y)}
    for _ in range(steps):
        r = int(random.random() * 4)
        if r == 0:
            x += 1
        elif r == 1:
            x -= 1
        elif r == 2:
            y += 1
        else:
            y -= 1
        if (x, y) in visited:
            return None
        
        visited.add((x, y))
        xs.append(x)
        ys.append(y)
    #return True #för ex21d
    return xs, ys #för ex21e

def self_avoiding_walk_memory(steps):
    x, y = 0, 0
    visited = {(x, y)}
    last_step = None

    for _ in range(steps):
        directions = [0, 1, 2, 3]
        if last_step is not None:
            if last_step == 0:
                directions.remove(1)
            if last_step == 1:
                directions.remove(0)
            if last_step == 2:
                directions.remove(3)
            if last_step == 3:
                directions.remove(2)
        r = random.choice(directions)

        if r == 0:
            x += 1
        elif r == 1:
            x -= 1
        elif r == 2:
            y += 1
        else:
            y -= 1

        if (x, y) in visited:
            return None 

        visited.add((x, y))
        last_step = r 

    return True
 
def exercise_21a():
    for steps in [10, 100, 1000]:
        xs, ys = random_walk(steps)
        plt.figure()
        plt.plot(xs, ys)
        plt.title(f"Random Walk: {steps} Steps")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
        #plt.savefig(f"Random Walk 2: {steps} Steps.pdf")
        
def exercise_21b():
    n_steps = 1000

    test_params = [
        (1, 3, 4, 128),
        (1, 3, 4, 129),
        (1, 3, 4, 130),
        (1, 3, 5, 128),
        (1, 3, 4, 128),
        (1, 4, 4, 128),
        (1, 3, 4, 128),
        (2, 3, 4, 128)
    ]
    
    for params in test_params:
        xs, ys = lcg_random_walk(n_steps, *params)
        r0, a, c, m = params
        
        plt.figure()
        plt.plot(xs, ys)
        plt.title(f"LCG Walk: r0={r0}, a={a}, c={c}, m={m}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        #plt.show()
        plt.savefig(f"LCG Walk: r0={r0}, a={a}, c={c}, m={m}, 1000 Steps.pdf")

def exercise_21c():
    N_list = np.linspace(1.0, 1000, 50, dtype=int)
    random_walk_count = 500
    
    rms_values = []
    stderr_values = [] 

    for N in N_list:
        distances_squared = []
        for _ in range(random_walk_count):
            x, y = random_walk(N)
            R2 = x[-1]**2 + y[-1]**2
            distances_squared.append(R2)
        
        distances_squared = np.array(distances_squared)
        
        mean_R2 = np.mean(distances_squared)
        rms = np.sqrt(mean_R2)
        rms_values.append(rms)

        std_R2 = np.std(distances_squared, ddof=1)
        stderr_mean_R2 = std_R2 / np.sqrt(random_walk_count)
        stderr_rms = stderr_mean_R2 / (2 * rms)
        stderr_values.append(stderr_rms)

    rms_values = np.array(rms_values)
    stderr_values = np.array(stderr_values)

    plt.figure()
    plt.errorbar(N_list, rms_values, yerr=stderr_values, linewidth=1, label="Measured RMS (with error bars)")
    plt.plot(N_list, np.sqrt(N_list), linestyle='--', label="√N (theory)")
    plt.xlabel("Number of steps N")
    plt.ylabel("RMS end-to-end distance √<R²>")
    plt.title("RMS End-to-End Distance vs Step Count")
    plt.legend()
    plt.show()
    #plt.savefig("RMS_End_to_End_Distance_with_Error_Bars.pdf")

def exercise_21d():
    N_list = np.linspace(1.0, 100, 20, dtype=int)
    random_walk_count = 500           # number of walks to attempt per N
    success_fraction = []

    for N in N_list:
        successes = 0
        for _ in range(random_walk_count):
            result = self_avoiding_walk_memory(N)
            if result is not None:
                successes += 1
        success_fraction.append(successes / random_walk_count)

    plt.figure()
    plt.plot(N_list, success_fraction)
    plt.xlabel("Walk length N")
    plt.ylabel("Fraction of successful self-avoiding walks")
    plt.title("No-backmove Self-Avoiding Walk Survival Probability (3-directions)")
    #plt.show()
    plt.savefig("No-backmove Self-Avoiding Walk Survival Probability (3-directions).pdf")
    
def exercise_21e():
    N_list = np.linspace(1.0, 10, 100, dtype=int)
    random_walk_count = 10000
    
    rms_values = []
    stderr_values = [] 

    for N in N_list:
        distances_squared = []
        for _ in range(random_walk_count):
            results = self_avoiding_walk(N)
            if results == None:
                continue
            x, y = results
            R2 = x[-1]**2 + y[-1]**2
            distances_squared.append(R2)
            
        if len(distances_squared) == 0:
            rms_values.append(np.nan)
            stderr_values.append(np.nan)
            continue
        
        distances_squared = np.array(distances_squared)
        
        mean_R2 = np.mean(distances_squared)
        rms = np.sqrt(mean_R2)
        rms_values.append(rms)

        std_R2 = np.std(distances_squared, ddof=1)
        stderr_mean_R2 = std_R2 / np.sqrt(random_walk_count)
        stderr_rms = stderr_mean_R2 / (2 * rms)
        stderr_values.append(stderr_rms)

    rms_values = np.array(rms_values)
    stderr_values = np.array(stderr_values)
    
    plt.figure()
    plt.loglog(N_list, rms_values, linewidth=1, label="Measured RMS, self avoiding walk")
    plt.loglog(N_list, np.sqrt(N_list), '--', label="√N (random walk RMS theory)")
    plt.loglog(N_list, N_list**0.71, '--', label="N^0.71, Self avoiding walk curve fit")
    plt.xlabel("Number of steps N (log scale)")
    plt.ylabel("RMS end-to-end distance √<R²> (log scale)")
    plt.title("RMS End-to-End Distance for self avoiding walk(log-log scale)")
    plt.legend()
    #plt.show()
    plt.savefig("RMS End-to-End Distance for self avoiding walk(log-log scale).pdf")
    
if __name__ == "__main__" :
    #exercise_21a()
    exercise_21b()
    #exercise_21c()
    #exercise_21d()
    #exercise_21e()