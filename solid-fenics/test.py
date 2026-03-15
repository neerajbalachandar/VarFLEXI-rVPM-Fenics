from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

# Create mesh
mesh = UnitSquareMesh(10, 10)

# Function space
V = FunctionSpace(mesh, "Lagrange", 1)

# Tabulate DOF coordinates
dof_coords = V.tabulate_dof_coordinates()

# Access coordinate of first DOF
print(dof_coords)
# print((dof_coords))
plt.plot(dof_coords)
plt.show()