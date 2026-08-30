# An Aeroelastic Solver Integrating Reformulated-Vortex-Particle and Finite-Element Methods across Non-Conforming Interfaces
## $\texttt{VarFlExI}$: Variable Fidelity Unsteady Flow – FEniCS Exchange Interface

This codebase implements **$\texttt{VarFlExI}$**, a partitioned framework that facilitates information exchange between variable-fidelity non-conforming interface solvers ($\texttt{FLOWUnsteady}$) and $\texttt{FEniCS}$ for unsteady bi-directionally coupled $\textit{FSI}$ simulations. The framework ensures strong work conservation via either of two methods: Wendland Kernels and Common Refinement Methods, both implemented using Transfer Operators.

[solid_solver.pdf](https://github.com/user-attachments/files/31613787/solid_solver.pdf)
[fluid_solver.pdf](https://github.com/user-attachments/files/31613786/fluid_solver.pdf)


### Run Coupling code
python coupling/varflexi.py --config_path "$CONFIG_DIR" &
COUPLING_PID=$!

### Run Solid solver
python solid/structural_solver.py --config_path "$CONFIG_DIR" &
SOLID_PID=$!

### Run Fluid solver
julia --project="$PROJECT_DIR" \
    fluid/fluid.jl \
    --fluid "$CONFIG_DIR/fluid_params.yaml" \
    --solid "$CONFIG_DIR/solid_params.yaml" \
    --coupling "$CONFIG_DIR/coupling_params.yaml"


Run the file ./run.sh

Arxiv publication: 
