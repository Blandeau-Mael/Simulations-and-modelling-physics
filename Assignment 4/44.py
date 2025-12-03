import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import random

def random_walk_to_boundary_coords(x, y, n):
    """Performs a random walk starting at (x, y) until it hits the boundary."""
    # Use temporary copies for the walk
    current_x, current_y = x, y
    
    while True:
        # 0: x+1 (right), 1: x-1 (left), 2: y+1 (up/down depends on convention), 3: y-1
        r = random.randint(0, 3) 
        if r == 0: 
            current_x += 1
        elif r == 1: 
            current_x -= 1
        elif r == 2: 
            current_y += 1
        else: 
            current_y -= 1

        # Check if the boundary is hit
        if current_x == 0 or current_x == n or current_y == 0 or current_y == n:
            return current_x, current_y

def compute_G(start_x, start_y, n, nwalk=200):
    """Computes the Green's function G(start_x, start_y, x_b, y_b) 
    by running nwalk random walks."""
    
    # G is the probability distribution of hitting the boundary, 
    # initialized to zero probability for all points (including corners)
    G_row = np.zeros((n+1, n+1)) 
    
    for _ in range(nwalk):
        x_b, y_b = random_walk_to_boundary_coords(start_x, start_y, n)
        # Increment the count for the hit boundary point
        G_row[x_b, y_b] += 1
    
    # Normalize by the number of walkers to get the probability/Green's function
    G_row /= nwalk
    return G_row

# Parameters
L = 10
n = 10 # n = 10 * 2
nwalk = 1000 # number of walkers per interior site

# Initialize G storage (G(x, y, x_b, y_b))
# We only need G for interior points (1..n-1) and boundary points (0 or n)
G = np.zeros(((n+1), (n+1), (n+1), (n+1))) 

# Iterate over all interior points (x, y)
for x in range(1, n):
    for y in range(1, n):
        # Compute the probability distribution G(x, y, :, :) for this starting point
        G[x, y, :, :] = compute_G(x, y, n, nwalk)
        
print("Green's function G computed using the Random Walk method.")
#print(G)


# Initialize potential array V(x, y) for interior points
v= np.zeros((n+1, n+1))

for i in range(1, n):
    v[0, i] = 10.0 # x=0, y=i
    v[n, i] = 10.0 # x=20, y=i
    v[i, 0] = 5.0  # x=i, y=0
    v[i, n] = 5.0  # x=i, y=20
    
#the five boundary conditions to change for max (3,5)
v[3, 0] = 20
v[4, 0] = 20
v[5, 0] = 20
v[6, 0] = 20
v[7, 0] = 20

# Iterate over all interior points (x, y)
for x in range(1, n):
    for y in range(1, n):
        sum_potential = 0.0
        for i in range(1, n):
            sum_potential += G[x, y, 0, i] * v[0, i]
            sum_potential += G[x, y, n, i] * v[n, i]
            sum_potential += G[x, y, i, 0] * v[i, 0]
            sum_potential += G[x, y, i, n] * v[i, n]
            
        # Apply the scaling factor (1/n) from the problem's formula
        v[x, y] = sum_potential

# Print the potential at the center point (10, 10)
center_potential = v[n//2, n//2]
p35_potential = v[3,5] 
p53_potential = v[5,3]


print("\n--- Results ---")
print(f"Potential V(x, y) using the formula V = (1/n) * sum(G*V_b): {center_potential:.4f}")
print(f"pontential at (3,5): {p35_potential}")
print(f"pontential at (3,5): {p53_potential}")

# Display the potential V_rw (V = (1/n) * sum(G*V_b))
plt.figure()
plt.title(f"Potential V(x, y) using Green's Function (V = 1/n * Sum(G*Vb))\npotential at (5,3): {p53_potential:.4f}")
plt.imshow(v, cmap='viridis', origin='lower')
plt.colorbar(label='Potential V')
plt.xlabel('y')
plt.ylabel('x')
plt.savefig("Potential V(x, y) using Green's Function to maximize (5,3) test1.pdf")
plt.show()