# An Aeroelastic Solver Integrating Reformulated-Vortex-Particle and Finite-Element Methods across Non-Conforming Interfaces
## VarFlExI: Variable Fidelity Unsteady Flow – FEniCS Exchange Interface

This codebase implements **VarFlExI**, a partitioned framework that facilitates information exchange between variable-fidelity non-conforming interface solvers ($\texttt{FLOWUnsteady}$) and $\texttt{FEniCS}$-based numerical models for unsteady aerodynamic simulations.

[solid_solver.pdf](https://github.com/user-attachments/files/31613787/solid_solver.pdf)
[fluid_solver.pdf](https://github.com/user-attachments/files/31613786/fluid_solver.pdf)


### Coupling code
python varflexi.py --config_path ./config

### Solid structural solver
python solid_solver.py --config_path ./config

### Fluid aerodynamic solver
add here
