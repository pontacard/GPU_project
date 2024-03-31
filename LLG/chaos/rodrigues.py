import numpy as np

v1 = np.array([3/5,4/5,0])
v2 = np.array([0,3/5,4/5])
theta = np.pi/4

mcro = np.cross(v1,v2)
print(mcro)
nmcro = mcro/np.linalg.norm(mcro)
print(nmcro)
cos = np.cos(theta)
sin = np.sin(theta)
R =np.array([[(nmcro[0]**2) *(1 - cos) + cos                  , (nmcro[0] * nmcro[1]) * (1-cos) - nmcro[2] * sin, (nmcro[0] * nmcro[2]) * (1-cos) + nmcro[1] * sin],
             [(nmcro[0] * nmcro[1]) * (1-cos) + nmcro[2] * sin, (nmcro[1]**2) *(1 - cos) + cos,                   (nmcro[1] * nmcro[2]) * (1-cos) - nmcro[0] * sin],
             [(nmcro[0] * nmcro[2]) * (1-cos) - nmcro[1] * sin, (nmcro[1] * nmcro[2]) * (1-cos) + nmcro[0] * sin,  (nmcro[2]**2) *(1 - cos) + cos]                  ])

print(np.dot(R , v1))