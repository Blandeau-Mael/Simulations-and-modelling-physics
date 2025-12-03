import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import random

L = 10
n = 10*2
nsteps = 200

v = np.zeros((n+1, n+1))
v = np.full((n+1, n+1), 7.0)

# Set boundary conditions
for i in range(1,n):
    v[0,i] = 10.0
    v[n,i] = 10.0
    v[i,0] = 5.0
    v[i,n] = 5.0

fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(v, cmap=None, interpolation='nearest')
fig.colorbar(im)

def relax_sequential(n, v):
    for x in range(1,n):
        for y in range(1,n):
            v[x,y] = 0.25 * (v[x-1,y] + v[x+1,y] + v[x,y-1] + v[x,y+1])
    return v 

def relax(v, steps=500):
    v = v.copy()
    for _ in range(steps):
        for x in range(1, n):
            for y in range(1, n):
                v[x, y] = 0.25 * (
                    v[x-1, y] + v[x+1, y] +
                    v[x, y-1] + v[x, y+1]
                )
    return v

v_relaxed = relax(v, steps=1000)
            
def random_walk_to_boundary(x, y):
    while True:
        r = random.randint(0,3)
        if r == 0:   x += 1
        elif r == 1: x -= 1
        elif r == 2: y += 1
        else:        y -= 1

        if x == 0 or x == n or y == 0 or y == n:
            return v[x, y]

def walkers(x, y, nwalk):
    total = 0.0
    for _ in range(nwalk):
        total += random_walk_to_boundary(x, y)
    return total / nwalk
            
def update(step):
    print(step)
    global n, v
    print(v[(n+1)//2,(n+1)//2])

    if step > 0:
        relax_sequential(n, v)

    im.set_array(v)
    return im,

def execise_43a():
    V100 = walkers(3, 3, 100)
    V1000 = walkers(3, 3, 1000)
    V10000 = walkers(3, 3, 10000)

    print("Monte Carlo (100 walkers)  =", V100)
    print("Monte Carlo (1000 walkers) =", V1000)
    print("Monte Carlo (10000 walkers) =", V10000)

    anim = animation.FuncAnimation(fig, update, frames=nsteps+1, interval=200, blit=True, repeat=False)
    plt.show()
    
def exercise_43points():
    points = [
    (3, 10),
    (8, 8),
    (10, 3),
    (12, 12)
    ]

    walkers_list = np.linspace(100, 10000, 10, dtype=int)

    results = {}

    for p in points:
        x, y = p
        results[p] = {"relax": v_relaxed[x, y]}
        for nw in walkers_list:
            results[p][nw] = walkers(x, y, nw)

    for p in points:
        print(f"\nPOINT {p}:")
        print(f"  Relaxation: {results[p]['relax']:.4f}")
        for nw in walkers_list:
            print(f"  MC {nw:4d} walkers: {results[p][nw]:.4f}")
            
def exercise_43plots():
    
    walkers_list = np.linspace(100, 5000, 15, dtype=int)

    meanerrors = []

    x, y = (10, 10)
    relaxed = v_relaxed[x, y]
    for nw in walkers_list:
        errors = []
        for _ in range(15):
            walker = walkers(x, y, nw)
            errors.append(abs(walker-relaxed))
        meanerrors.append(np.mean(errors))  
    
    ref_curve = 1 / np.sqrt(walkers_list)

    # Plotting
    plt.figure()
    plt.plot(walkers_list, meanerrors, marker='o', label="Mean Error")
    plt.plot(walkers_list, ref_curve, marker='s', linestyle='--', label="1/sqrt(N)")

    plt.xlabel("Number of Walkers (log scale)")
    plt.ylabel("Mean Error (log scale)")
    plt.title("Mean Error vs Number of Walkers (Log-Log)")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.show()  
            

#execise_43a()    
#exercise_43points()
exercise_43plots()