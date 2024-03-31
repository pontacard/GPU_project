import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import pandas as pd

print(np.linspace(0, 1500, 1500) )

f = open("testa.txt", "w")
for i in range(1,4):
    f.write(f'{i}/2\n')
f.close()
