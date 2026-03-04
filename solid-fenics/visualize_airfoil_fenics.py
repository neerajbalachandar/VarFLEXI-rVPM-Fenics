from dolfin import *
import numpy as np
import os

parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

# Geometry settings (must match elastodynamics_integration_v4.py mapping)
span = 1.0
root_chord = 0.12
tip_chord = 0.08
thickness_ratio = 0.12
leading_edge_sweep = 0.0

nx, ny, nz = 32, 200, 8
span_strips = 200


def chord_at(y_val):
    eta = min(max(y_val / span, 0.0), 1.0)
    return root_chord + (tip_chord - root_chord) * eta


def x_leading_edge_at(y_val):
    eta = min(max(y_val / span, 0.0), 1.0)
    return leading_edge_sweep * eta


def naca_half_thickness(xi):
    xi_clip = min(max(xi, 0.0), 1.0)
    return 5.0 * thickness_ratio * (
        0.2969 * np.sqrt(xi_clip)
        - 0.1260 * xi_clip
        - 0.3516 * xi_clip**2
        + 0.2843 * xi_clip**3
        - 0.1015 * xi_clip**4
    )


mesh = BoxMesh(Point(0.0, 0.0, -0.5), Point(1\section{Validation, Computation and Results}
￼
\begin{frame}{Results}
￼
\centering
\textbf{Tip Displacement}
￼
\vspace{0.1cm}
￼
\includegraphics[width=0.35\linewidth]{images/tip_disp_v3.png}
￼
\vspace{0.1cm}
￼
\textbf{Energy Evolution}
￼
\vspace{0.1cm}
￼
\includegraphics[width=0.85\linewidth]{images/energy_v3.png}
￼
\end{frame}.0, span, 0.5), nx, ny, nz)
coords = mesh.coordinates()
for i in range(coords.shape[0]):
    xi = coords[i, 0]
    y_val = coords[i, 1]
    z_ref = coords[i, 2]
    chord = chord_at(y_val)
    x_le = x_leading_edge_at(y_val)
    zeta = 2.0 * z_ref
    half_t = chord * naca_half_thickness(xi)
    coords[i, 0] = x_le + xi * chord
    coords[i, 2] = zeta * half_t

panel_tol_x = 0.75 * (root_chord / nx)
panel_tol_z = 1.25 * (root_chord * thickness_ratio / nz)

facet_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
facet_markers.set_all(0)


class AeroSurface(SubDomain):
    def inside(self, X, on_boundary):
        if not on_boundary:
            return False
        y_val = X[1]
        chord = chord_at(y_val)
        if chord <= 0.0:
            return False
        x_le = x_leading_edge_at(y_val)
        xi = (X[0] - x_le) / chord
        if xi < -0.02 or xi > 1.02:
            return False
        z_surf = chord * naca_half_thickness(xi)
        return abs(abs(X[2]) - z_surf) <= panel_tol_z


AeroSurface().mark(facet_markers, 5)

# Scalar field that highlights nodes used for spanwise strip loading at 3/4 chord.
V = FunctionSpace(mesh, "CG", 1)
cp_mask = Function(V, name="cp_strip_mask")
xi_field = Function(V, name="xi_local")
y_strip = Function(V, name="strip_index")

dof_coords = V.tabulate_dof_coordinates().reshape((-1, 3))
cp_vals = cp_mask.vector().get_local()
xi_vals = xi_field.vector().get_local()
strip_vals = y_strip.vector().get_local()

strip_hits = np.zeros(span_strips, dtype=np.int32)

for i, X in enumerate(dof_coords):
    y_val = X[1]
    chord = chord_at(y_val)
    x_le = x_leading_edge_at(y_val)
    xi = (X[0] - x_le) / max(chord, 1e-12)
    xi_vals[i] = xi

    eta = min(max(y_val / span, 0.0), 1.0)
    strip_idx = min(int(eta * span_strips), span_strips - 1)
    strip_vals[i] = float(strip_idx)

    z_surf = chord * naca_half_thickness(xi)
    x_cp = x_le + 0.75 * chord
    if abs(X[0] - x_cp) <= panel_tol_x and abs(abs(X[2]) - z_surf) <= panel_tol_z:
        cp_vals[i] = 1.0
        strip_hits[strip_idx] += 1
    else:
        cp_vals[i] = 0.0

cp_mask.vector().set_local(cp_vals)
cp_mask.vector().apply("insert")
xi_field.vector().set_local(xi_vals)
xi_field.vector().apply("insert")
y_strip.vector().set_local(strip_vals)
y_strip.vector().apply("insert")

out_dir = "solid-fenics/results/fenics-visualization"
os.makedirs(out_dir, exist_ok=True)

# Safer ParaView workflow:
# 1) Write legacy VTK/PVD for robust loading.
# 2) Write separate XDMF files per field (avoid mixed topology/attribute XDMF files).
File(os.path.join(out_dir, "mesh.pvd")) << mesh
File(os.path.join(out_dir, "facet_markers.pvd")) << facet_markers
File(os.path.join(out_dir, "cp_strip_mask.pvd")) << cp_mask
File(os.path.join(out_dir, "xi_local.pvd")) << xi_field
File(os.path.join(out_dir, "strip_index.pvd")) << y_strip

mesh_xdmf = XDMFFile(os.path.join(out_dir, "mesh_only.xdmf"))
mesh_xdmf.write(mesh)

cp_xdmf = XDMFFile(os.path.join(out_dir, "cp_strip_mask.xdmf"))
cp_xdmf.parameters["flush_output"] = True
cp_xdmf.parameters["functions_share_mesh"] = True
cp_xdmf.parameters["rewrite_function_mesh"] = False
cp_xdmf.write_checkpoint(cp_mask, "cp_strip_mask", 0.0, XDMFFile.Encoding.HDF5, False)

xi_xdmf = XDMFFile(os.path.join(out_dir, "xi_local.xdmf"))
xi_xdmf.parameters["flush_output"] = True
xi_xdmf.parameters["functions_share_mesh"] = True
xi_xdmf.parameters["rewrite_function_mesh"] = False
xi_xdmf.write_checkpoint(xi_field, "xi_local", 0.0, XDMFFile.Encoding.HDF5, False)

strip_xdmf = XDMFFile(os.path.join(out_dir, "strip_index.xdmf"))
strip_xdmf.parameters["flush_output"] = True
strip_xdmf.parameters["functions_share_mesh"] = True
strip_xdmf.parameters["rewrite_function_mesh"] = False
strip_xdmf.write_checkpoint(y_strip, "strip_index", 0.0, XDMFFile.Encoding.HDF5, False)

empty_strips = int(np.sum(strip_hits == 0))
print("Wrote FEniCS visualization files to:", out_dir)
print("Recommended ParaView start file:", os.path.join(out_dir, "cp_strip_mask.pvd"))
print("Total strips:", span_strips)
print("Strips without any selected 3/4-chord dof:", empty_strips)
print("Min/Max selected dofs per strip:", int(strip_hits.min()), int(strip_hits.max()))
