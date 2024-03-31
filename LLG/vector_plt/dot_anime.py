import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import ArtistAnimation

t = np.linspace(0, np.pi * 4, 50)
z = np.linspace(0, 1, 50)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

frames = []
for i in np.linspace(0, np.pi * 2, 50):
    x = np.cos(t + i)
    y = np.sin(t + i)
    frame = ax.plot(x, y, z, marker='o', markersize=5, color='blue')
    frames.append(frame)

ani = ArtistAnimation(fig, frames, interval=100)

plt.show()