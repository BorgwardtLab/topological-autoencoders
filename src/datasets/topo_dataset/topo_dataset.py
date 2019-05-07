'''
Trying out Tadasets library for generating topological synthetic datasets: 
Here an overlay of a torus, sphere and swiss roll
'''

import numpy as np 
#import tadasets 
import matplotlib.pyplot as plt
import custom_shapes as cs

roll, y_r = cs.swiss_roll(n=400, r=20)
sphere, y_s = cs.sphere(n=100, r=15)
torus, y_t = cs.torus(n=700, c=4, a=2)

#shift the shapes in space:
torus = torus*20
#sphere[:,2] = sphere[:,2] - 30 
roll = roll*2 

data = np.concatenate([torus, roll, sphere], axis=0)
labels = np.concatenate([y_t, y_r, y_s], axis=0)
#tadasets.plot3d(data)
cmaps = [plt.cm.winter, plt.cm.spring]
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
for data, color, cmap in zip([roll, torus],[y_r, y_t], cmaps):
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=color, cmap=cmap)

plt.show()
