# **VarFlExI**
## Variable Fidelity Unsteady Flow – FEniCS Exchange Interface

This codebase implements **VarFlExI**, an interface designed to facilitate the exchange of information between variable-fidelity flow solvers (FlowUnsteady) and FEniCS-based numerical models for unsteady aerodynamic simulations.

<img width="1672" height="742" alt="fluid_solver" src="https://github.com/user-attachments/assets/271dac28-132d-40d1-bd63-2e3e8820e267" />
<img width="2238" height="1016" alt="solid_solver" src="https://github.com/user-attachments/assets/b04b3d8f-cdae-4e87-8ea0-540af3b3a7df" />


### Coupling code
python varflexi.py --config_path ./config

### Solid structural solver
python solid_solver.py --config_path ./config

### Fluid aerodynamic solver
add here