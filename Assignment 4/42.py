import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

L = 10
n = 10*2
nsteps = 200

v = np.zeros((n+1, n+1))
v = np.full((n+1, n+1), 9.0)

# Set boundary conditions
for i in range(1,n):
    v[0,i] = 10.0
    v[n,i] = 10.0
    v[i,0] = 10.0
    v[i,n] = 10.0

fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(v, cmap=None, interpolation='nearest')
fig.colorbar(im)

def relax_sequential(n, v):
    for x in range(1,n):
        for y in range(1,n):
            v[x,y] = 0.25 * (v[x-1,y] + v[x+1,y] + v[x,y-1] + v[x,y+1])
            
def relax_red_black(n, v):

    for x in range(1, n):
        for y in range(1, n):
            if (x + y) % 2 == 0:      # red
                v[x,y] = 0.25 * (v[x-1,y] + v[x+1,y] +
                                 v[x,y-1] + v[x,y+1])

    for x in range(1, n):
        for y in range(1, n):
            if (x + y) % 2 == 1:      # black
                v[x,y] = 0.25 * (v[x-1,y] + v[x+1,y] +
                                 v[x,y-1] + v[x,y+1])

def update(step):
    print(step)
    global n, v
    print(v[(n+1)//2,(n+1)//2])

    if step > 0:
        #relax_sequential(n, v)
        relax_red_black(n, v)

    im.set_array(v)
    return im,

anim = animation.FuncAnimation(fig, update, frames=nsteps+1, interval=200, blit=True, repeat=False)
plt.show()
