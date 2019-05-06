'''
Trying out Tadasets library for generating topological synthetic datasets: 
Here an overlay of a torus, sphere and swiss roll
'''

import numpy as np 
import tadasets 
import matplotlib.pyplot as plt

roll = tadasets.swiss_roll(n=500, r=20)
sphere = tadasets.sphere(n=500, r=15)
torus = tadasets.torus(n=200, c=4, a=2)
data = np.concatenate([torus, roll, sphere], axis=0)
#tadasets.plot3d(data)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
for data, colors in zip([roll, sphere, torus],['black','green','red']):
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=[colors])

plt.show()
