#input

import numpy as np
import matplotlib.pyplot as plt

def coordinate_3d(axes, range_x, range_y, range_z, grid = True):
    axes.set_xlabel("x", fontsize = 14)
    axes.set_ylabel("y", fontsize = 14)
    axes.set_zlabel("z", fontsize = 14)
    axes.set_xlim(range_x[0], range_x[1])
    axes.set_ylim(range_y[0], range_y[1])
    axes.set_zlim(range_z[0], range_z[1])
    if grid == True:
        axes.grid()

def visual_vector_3d(axes, loc, vector, color = "red"):
    axes.quiver(loc[0], loc[1], loc[2],
              vector[0], vector[1], vector[2],
              color = color, lw=3)


fig = plt.figure(figsize = (6, 6))
ax = fig.add_subplot(111, projection='3d')

# 3D座標を設定
coordinate_3d(ax, [-10, 10], [-10, 10], [0, 10], grid = True)

# 始点を設定
o = [0, 0, 0]

# 3Dベクトルを定義
v1 = np.array([-7, -7, 7])
v2 = np.array([-7,  7, 7])
v3 = np.array([ 7, -7, 7])
v4 = np.array([ 7,  7, 7])

# 3Dベクトルを配置
visual_vector_3d(ax, o, v1, "red")
visual_vector_3d(ax, o, v2, "blue")
visual_vector_3d(ax, o, v3, "green")
visual_vector_3d(ax, o, v4, "magenta")
plt.show()