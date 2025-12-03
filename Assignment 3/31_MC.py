import numpy as np
import matplotlib.pyplot as plt

def _P(x):
    return x * np.exp(-x)

def metropolis(N_total=200000, N_0=20000, delta=1.0):
    x = 1.0 
    accepts = 0
    total = 0
    samples = []

    for i in range(N_total):
        total += 1
        x_trial = x + np.random.uniform(-delta, delta)

        if x_trial < 0:
            x_trial = -x_trial

        A = min(1.0, _P(x_trial) / _P(x))

        if np.random.rand() < A:
            x = x_trial
            accepts += 1
        else: 
            x = x
        samples.append(x)

    return np.array(samples[N_0:]), accepts/total

def trial():
    samples, _ = metropolis()
    x_mean = np.mean(samples)
    print("Estimated <x> =", x_mean)


    exact = 2
    print("Exact value =", exact)
    
def errors():
    deltas = np.linspace(0.1, 10, 25)
    N = 10000
    N_0 = 1000
    runs = 50
    exact = 2.0

    se = []
    rms = []
    for d in deltas:
        run_avgs = []
        run_vars = []

        for i in range(runs):
            samples, _ = metropolis(N, N_0, d)
            run_avgs.append(np.mean(samples))
            run_vars.append(np.var(samples))

        run_avgs = np.array(run_avgs)
        run_vars = np.array(run_vars)

        se.append(np.sqrt(np.mean(run_vars) / N))
        rms.append(np.sqrt(np.mean((run_avgs - exact)**2)))

    plt.plot(deltas, se, 'o-', label="Naive SE (sqrt(var/N))")
    plt.plot(deltas, rms, 's-', label="Actual RMS error")
    plt.xlabel("Proposal step size delta")
    plt.ylabel("Error")
    plt.title("Error vs Proposal Step Size delta in Metropolis Sampling")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    #plt.show()
    plt.savefig("Error vs Proposal Step Size delta in Metropolis Sampling.pdf")
   
def acceptance_rates():
    deltas = np.linspace(0.1, 10, 25)
    rates = []

    for delta in deltas:
        _, rate = metropolis(N_total=50000, delta=delta)
        rates.append(rate)

    plt.plot(deltas, rates, 'o-')
    plt.xlabel("Proposal step size delta")
    plt.ylabel("Acceptance rate")
    plt.title("Metropolis Acceptance Rate vs delta")
    plt.grid(True)
    plt.tight_layout()
    #plt.show() 
    plt.savefig("Metropolis Acceptance Rate vs delta.pdf") 
   
if __name__ == "__main__" :
    #trial()
    #errors()
    acceptance_rates()