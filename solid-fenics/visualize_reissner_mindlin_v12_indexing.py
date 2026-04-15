import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

TMP_CACHE_DIR = os.path.join("/tmp", "varflexi-fenics-cache")
os.makedirs(TMP_CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(TMP_CACHE_DIR, "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(TMP_CACHE_DIR, "xdg"))
os.environ.setdefault("DIJITSO_CACHE_DIR", os.path.join(TMP_CACHE_DIR, "dijitso"))
os.environ.setdefault("INSTANT_CACHE_DIR", os.path.join(TMP_CACHE_DIR, "instant"))

INTERACTIVE_MODE = os.getenv("INDEX_VIZ_INTERACTIVE", "1") == "1"
SAVE_OUTPUTS = os.getenv("INDEX_VIZ_SAVE", "0") == "1"
SHOW_SEQUENTIALLY = os.getenv("INDEX_VIZ_SEQUENTIAL", "1") == "1"
MATPLOTLIB_BACKEND = os.getenv("INDEX_VIZ_BACKEND", "TkAgg")

import matplotlib

if INTERACTIVE_MODE:
    matplotlib.use(MATPLOTLIB_BACKEND)
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from dolfin import *
from scipy.spatial import cKDTree


parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True


span = 1.0
root_chord = 0.12
tip_chord = 0.12
thickness_ratio = 0.12
leading_edge_sweep = 0.0

nx = int(os.getenv("INDEX_VIZ_NX", "120"))
ny = int(os.getenv("INDEX_VIZ_NY", "12"))
n_span = int(os.getenv("INDEX_VIZ_NSPAN", "30"))
n_chord = int(os.getenv("INDEX_VIZ_NCHORD", "6"))
rbf_epsilon = float(os.getenv("INDEX_VIZ_RBF_EPS", "1.0"))
rbf_neighbors = int(os.getenv("INDEX_VIZ_RBF_NEIGHBORS", "1"))
label_stride = int(os.getenv("INDEX_VIZ_LABEL_STRIDE", "40"))
cp_labels_to_show = int(os.getenv("INDEX_VIZ_CP_LABELS", "30"))
rbf_samples_to_show = int(os.getenv("INDEX_VIZ_RBF_SAMPLES", "8"))

mesh = RectangleMesh(Point(0.0, 0.0), Point(1.0, span), nx, ny)


def chord_at(y_val):
    eta = min(max(y_val / span, 0.0), 1.0)
    return root_chord + (tip_chord - root_chord) * eta


def x_leading_edge_at(y_val):
    eta = min(max(y_val / span, 0.0), 1.0)
    return leading_edge_sweep * eta


coords = mesh.coordinates()
for i in range(coords.shape[0]):
    xi = coords[i, 0]
    y_val = coords[i, 1]
    coords[i, 0] = x_leading_edge_at(y_val) + xi * chord_at(y_val)


U_el = VectorElement("CG", mesh.ufl_cell(), 2, dim=2)
W_el = FiniteElement("CG", mesh.ufl_cell(), 2)
T_el = VectorElement("CG", mesh.ufl_cell(), 2, dim=2)
V = FunctionSpace(mesh, MixedElement([U_el, W_el, T_el]))
Vt = VectorFunctionSpace(mesh, "CG", 1, dim=3)


panel_node_cache = {}
eta_span_comm = (np.arange(n_span, dtype=float) + 0.5) / n_span
eta_chord_edges = np.linspace(0.0, 1.0, n_chord + 1)
eta_chord_comm = eta_chord_edges[:-1] + 0.75 * (
    eta_chord_edges[1:] - eta_chord_edges[:-1]
)


def as_eta_array(raw, n_expected):
    arr = np.asarray(raw, dtype=float).reshape(-1)
    if len(arr) != n_expected:
        raise RuntimeError(f"Expected {n_expected} eta values, got {len(arr)}")
    return arr


def get_scalar_space_coords(space):
    scalar_space = space.sub(0).collapse()
    xy = scalar_space.tabulate_dof_coordinates().reshape((-1, mesh.geometry().dim()))
    xyz = np.zeros((xy.shape[0], 3), dtype=float)
    xyz[:, :2] = xy
    return xyz


def get_panel_node_ids(n_span_in, n_chord_in, eta_chord):
    eta_chord = as_eta_array(eta_chord, n_chord_in)
    key = (n_span_in, n_chord_in, tuple(np.round(eta_chord, 8)))
    if key in panel_node_cache:
        return panel_node_cache[key]

    coords_xyz = get_scalar_space_coords(Vt)
    panel_node_ids = [[] for _ in range(n_span_in * n_chord_in)]

    for i_node, X in enumerate(coords_xyz):
        y_val = X[1]
        chord = chord_at(y_val)
        if chord <= 0.0:
            continue
        x_le = x_leading_edge_at(y_val)
        xi = (X[0] - x_le) / max(chord, 1.0e-12)
        if xi < -0.02 or xi > 1.02:
            continue
        eta_s = np.clip(y_val / span, 0.0, 1.0)
        i_span = min(int(eta_s * n_span_in), n_span_in - 1)
        j_chord = int(np.argmin(np.abs(eta_chord - xi)))
        panel_idx = i_span * n_chord_in + j_chord
        panel_node_ids[panel_idx].append(i_node)

    for i_span in range(n_span_in):
        y_target = (i_span + 0.5) * span / n_span_in
        chord = chord_at(y_target)
        x_le = x_leading_edge_at(y_target)
        for j_chord in range(n_chord_in):
            panel_idx = i_span * n_chord_in + j_chord
            if panel_node_ids[panel_idx]:
                continue
            xi_target = eta_chord[j_chord]
            x_target = x_le + xi_target * chord
            distances = np.sum(
                (coords_xyz[:, :2] - np.array([[x_target, y_target]])) ** 2, axis=1
            )
            panel_node_ids[panel_idx].append(int(np.argmin(distances)))

    panel_node_cache[key] = panel_node_ids
    return panel_node_ids


def extract_coupling_node_indices(n_span_in, n_chord_in, eta_chord, coords_xyz):
    eta_chord = as_eta_array(eta_chord, n_chord_in)
    tree = cKDTree(coords_xyz[:, :2])
    targets_xy = []
    for i_span in range(n_span_in):
        y_target = eta_span_comm[i_span] * span
        chord = chord_at(y_target)
        x_le = x_leading_edge_at(y_target)
        for j_chord in range(n_chord_in):
            xi_target = eta_chord[j_chord]
            x_target = x_le + xi_target * chord
            targets_xy.append([x_target, y_target])
    targets_xy = np.asarray(targets_xy, dtype=float)
    _, idx = tree.query(targets_xy, k=1)
    targets_xyz = np.zeros((targets_xy.shape[0], 3), dtype=float)
    targets_xyz[:, :2] = targets_xy
    return np.asarray(idx, dtype=np.int64), targets_xyz


def build_local_rbf_map(fluid_points, solid_points, epsilon, n_neighbors=32):
    fluid_points = np.asarray(fluid_points, dtype=float)
    solid_points = np.asarray(solid_points, dtype=float)
    eps2 = max(float(epsilon) ** 2, 1.0e-16)
    n_s = solid_points.shape[0]
    k = int(max(1, min(n_neighbors, n_s)))

    tree = cKDTree(solid_points)
    d, idx = tree.query(fluid_points, k=k)
    if k == 1:
        d = d.reshape(-1, 1)
        idx = idx.reshape(-1, 1)
    nbr_ids = idx.astype(np.int64)
    r2 = d * d
    nbr_w = np.exp(-r2 / eps2)
    row_sum = np.sum(nbr_w, axis=1, keepdims=True)
    bad = np.where(row_sum[:, 0] <= 1.0e-16)[0]
    for bi in bad:
        nbr_w[bi, :] = 0.0
        nbr_w[bi, 0] = 1.0
    row_sum = np.maximum(row_sum, 1.0e-16)
    nbr_w /= row_sum
    return nbr_ids, nbr_w


def save_csv(path, header, rows):
    with open(path, "w", encoding="ascii") as f:
        f.write(header + "\n")
        for row in rows:
            f.write(",".join(row) + "\n")


def label_points(ax, xy, labels, stride, fontsize=5):
    for i in range(0, len(labels), max(1, stride)):
        ax.text(xy[i, 0], xy[i, 1], str(labels[i]), fontsize=fontsize, color="black")


def base_axis_style(ax, title):
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)


def finalize_figure(fig, out_dir, filename):
    fig.tight_layout()
    if SAVE_OUTPUTS:
        fig.savefig(os.path.join(out_dir, filename), dpi=220)

    if INTERACTIVE_MODE and SHOW_SEQUENTIALLY:
        print(f"Showing {filename}")
        print("Use the toolbar to zoom/pan, then close the window to continue.")
        plt.show(block=True)
        plt.close(fig)
        return

    if not INTERACTIVE_MODE:
        plt.close(fig)


def save_mesh_vertex_plot(out_dir, mesh_xy):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(mesh_xy[:, 0], mesh_xy[:, 1], c=np.arange(len(mesh_xy)), s=10, cmap="viridis")
    label_points(ax, mesh_xy, np.arange(len(mesh_xy)), label_stride)
    base_axis_style(ax, "Mesh vertex indices")
    finalize_figure(fig, out_dir, "mesh_vertex_indices.png")


def save_cell_plot(out_dir, mesh_xy, cells):
    triang = mtri.Triangulation(mesh_xy[:, 0], mesh_xy[:, 1], cells)
    centroids = mesh_xy[cells].mean(axis=1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.triplot(triang, color="0.55", linewidth=0.4)
    ax.scatter(centroids[:, 0], centroids[:, 1], c=np.arange(len(cells)), s=8, cmap="plasma")
    label_points(ax, centroids, np.arange(len(cells)), max(1, label_stride // 2))
    base_axis_style(ax, "Cell/element indices")
    finalize_figure(fig, out_dir, "cell_indices.png")


def save_scalar_node_plot(out_dir, scalar_xyz, cp_node_ids):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(scalar_xyz[:, 0], scalar_xyz[:, 1], c=np.arange(len(scalar_xyz)), s=10, cmap="viridis")
    cp_xy = scalar_xyz[cp_node_ids, :2]
    ax.scatter(cp_xy[:, 0], cp_xy[:, 1], s=18, facecolors="none", edgecolors="red", linewidths=0.8)
    label_points(ax, scalar_xyz[:, :2], np.arange(len(scalar_xyz)), label_stride)
    base_axis_style(ax, "CG1/interface node indices with coupling nodes highlighted")
    finalize_figure(fig, out_dir, "scalar_interface_node_indices.png")


def save_panel_plot(out_dir, scalar_xyz, panel_node_ids, cp_targets, cp_node_ids):
    node_panel = np.full((scalar_xyz.shape[0],), -1, dtype=int)
    for panel_idx, ids in enumerate(panel_node_ids):
        for node_id in ids:
            node_panel[node_id] = panel_idx

    fig, ax = plt.subplots(figsize=(10, 4))
    scatter = ax.scatter(
        scalar_xyz[:, 0], scalar_xyz[:, 1], c=node_panel, s=10, cmap="tab20", vmin=-1
    )
    ax.scatter(cp_targets[:, 0], cp_targets[:, 1], c="black", s=7, marker="x")
    ax.scatter(
        scalar_xyz[cp_node_ids, 0],
        scalar_xyz[cp_node_ids, 1],
        s=18,
        facecolors="none",
        edgecolors="red",
        linewidths=0.8,
    )
    for i in range(min(cp_labels_to_show, len(cp_targets))):
        ax.text(cp_targets[i, 0], cp_targets[i, 1], f"cp{i}", fontsize=5, color="black")
    base_axis_style(ax, "Panel ownership and coupling/control-point targets")
    fig.colorbar(scatter, ax=ax, label="panel_idx")
    finalize_figure(fig, out_dir, "panel_and_control_point_map.png")


def save_mixed_dof_plot(out_dir, mixed_xy, dof_ids, title, filename):
    dof_xy = mixed_xy[dof_ids]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(dof_xy[:, 0], dof_xy[:, 1], c=dof_ids, s=8, cmap="viridis")
    label_points(ax, dof_xy, dof_ids, label_stride)
    base_axis_style(ax, title)
    finalize_figure(fig, out_dir, filename)


def save_rbf_plot(out_dir, scalar_xyz, cp_targets, nbr_ids, nbr_w):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(scalar_xyz[:, 0], scalar_xyz[:, 1], c="0.75", s=8, label="solid/interface nodes")
    ax.scatter(cp_targets[:, 0], cp_targets[:, 1], c="black", s=10, marker="x", label="fluid/control points")

    n_show = min(rbf_samples_to_show, len(cp_targets))
    for cp_idx in range(n_show):
        x0, y0 = cp_targets[cp_idx, :2]
        for q_idx in range(nbr_ids.shape[1]):
            solid_id = nbr_ids[cp_idx, q_idx]
            x1, y1 = scalar_xyz[solid_id, :2]
            alpha = 0.15 + 0.85 * nbr_w[cp_idx, q_idx]
            ax.plot([x0, x1], [y0, y1], color="tab:red", alpha=alpha, linewidth=0.7)
        ax.text(x0, y0, f"cp{cp_idx}", fontsize=5, color="black")

    base_axis_style(ax, "Sampled local RBF communication links")
    ax.legend(loc="upper right")
    finalize_figure(fig, out_dir, "rbf_neighbor_links.png")


out_dir = os.path.join(SCRIPT_DIR, "..", "results", "reissner_mindlin_v12_indexing")
os.makedirs(out_dir, exist_ok=True)

mesh_xy = mesh.coordinates().copy()
cells = mesh.cells().copy()
scalar_xyz = get_scalar_space_coords(Vt)
cp_node_ids, cp_targets = extract_coupling_node_indices(
    n_span, n_chord, eta_chord_comm, scalar_xyz
)
cp_node_set = set(cp_node_ids.tolist())
panel_node_ids = get_panel_node_ids(n_span, n_chord, eta_chord_comm)
nbr_ids, nbr_w = build_local_rbf_map(cp_targets, scalar_xyz, rbf_epsilon, rbf_neighbors)

mixed_xy = V.tabulate_dof_coordinates().reshape((-1, mesh.geometry().dim()))
dofs_u_x = np.asarray(V.sub(0).sub(0).dofmap().dofs(), dtype=np.int64)
dofs_u_y = np.asarray(V.sub(0).sub(1).dofmap().dofs(), dtype=np.int64)
dofs_w = np.asarray(V.sub(1).dofmap().dofs(), dtype=np.int64)
dofs_theta_x = np.asarray(V.sub(2).sub(0).dofmap().dofs(), dtype=np.int64)
dofs_theta_y = np.asarray(V.sub(2).sub(1).dofmap().dofs(), dtype=np.int64)

save_mesh_vertex_plot(out_dir, mesh_xy)
save_cell_plot(out_dir, mesh_xy, cells)
save_scalar_node_plot(out_dir, scalar_xyz, cp_node_ids)
save_panel_plot(out_dir, scalar_xyz, panel_node_ids, cp_targets, cp_node_ids)
save_mixed_dof_plot(out_dir, mixed_xy, dofs_u_x, "Mixed-space DOF indices: u_x", "dofs_u_x.png")
save_mixed_dof_plot(out_dir, mixed_xy, dofs_u_y, "Mixed-space DOF indices: u_y", "dofs_u_y.png")
save_mixed_dof_plot(out_dir, mixed_xy, dofs_w, "Mixed-space DOF indices: w", "dofs_w.png")
save_mixed_dof_plot(
    out_dir, mixed_xy, dofs_theta_x, "Mixed-space DOF indices: theta_x", "dofs_theta_x.png"
)
save_mixed_dof_plot(
    out_dir, mixed_xy, dofs_theta_y, "Mixed-space DOF indices: theta_y", "dofs_theta_y.png"
)
save_rbf_plot(out_dir, scalar_xyz, cp_targets, nbr_ids, nbr_w)

vertex_rows = [
    [str(i), f"{xy[0]:.10e}", f"{xy[1]:.10e}"] for i, xy in enumerate(mesh_xy)
]
save_csv(os.path.join(out_dir, "mesh_vertices.csv"), "vertex_id,x,y", vertex_rows)

cell_rows = []
for cell_id, cell_conn in enumerate(cells):
    centroid = mesh_xy[cell_conn].mean(axis=0)
    cell_rows.append(
        [
            str(cell_id),
            str(int(cell_conn[0])),
            str(int(cell_conn[1])),
            str(int(cell_conn[2])),
            f"{centroid[0]:.10e}",
            f"{centroid[1]:.10e}",
        ]
    )
save_csv(
    os.path.join(out_dir, "mesh_cells.csv"),
    "cell_id,vertex0,vertex1,vertex2,centroid_x,centroid_y",
    cell_rows,
)

scalar_rows = [
    [
        str(i),
        f"{xyz[0]:.10e}",
        f"{xyz[1]:.10e}",
        f"{xyz[2]:.10e}",
        str(int(i in cp_node_set)),
    ]
    for i, xyz in enumerate(scalar_xyz)
]
save_csv(
    os.path.join(out_dir, "scalar_interface_nodes.csv"),
    "scalar_node_id,x,y,z,is_coupling_node",
    scalar_rows,
)

panel_rows = []
for panel_idx, ids in enumerate(panel_node_ids):
    i_span = panel_idx // n_chord
    j_chord = panel_idx % n_chord
    for node_id in ids:
        panel_rows.append([str(panel_idx), str(i_span), str(j_chord), str(int(node_id))])
save_csv(
    os.path.join(out_dir, "panel_to_scalar_nodes.csv"),
    "panel_idx,i_span,j_chord,scalar_node_id",
    panel_rows,
)

cp_rows = []
for cp_idx, node_id in enumerate(cp_node_ids):
    xyz = scalar_xyz[node_id]
    target_xy = cp_targets[cp_idx]
    cp_rows.append(
        [
            str(cp_idx),
            str(cp_idx // n_chord),
            str(cp_idx % n_chord),
            str(int(node_id)),
            f"{target_xy[0]:.10e}",
            f"{target_xy[1]:.10e}",
            f"{xyz[0]:.10e}",
            f"{xyz[1]:.10e}",
            f"{xyz[2]:.10e}",
        ]
    )
save_csv(
    os.path.join(out_dir, "coupling_control_points.csv"),
    "cp_idx,i_span,j_chord,scalar_node_id,target_x,target_y,node_x,node_y,node_z",
    cp_rows,
)


def dof_rows_for_component(name, dof_ids):
    rows = []
    for local_id, mixed_dof in enumerate(dof_ids):
        xy = mixed_xy[mixed_dof]
        rows.append([name, str(local_id), str(int(mixed_dof)), f"{xy[0]:.10e}", f"{xy[1]:.10e}"])
    return rows


dof_rows = []
dof_rows.extend(dof_rows_for_component("u_x", dofs_u_x))
dof_rows.extend(dof_rows_for_component("u_y", dofs_u_y))
dof_rows.extend(dof_rows_for_component("w", dofs_w))
dof_rows.extend(dof_rows_for_component("theta_x", dofs_theta_x))
dof_rows.extend(dof_rows_for_component("theta_y", dofs_theta_y))
save_csv(
    os.path.join(out_dir, "mixed_dof_maps.csv"),
    "component,local_component_id,mixed_global_dof_id,x,y",
    dof_rows,
)

rbf_rows = []
for cp_idx in range(nbr_ids.shape[0]):
    for q_idx in range(nbr_ids.shape[1]):
        solid_id = int(nbr_ids[cp_idx, q_idx])
        xyz = scalar_xyz[solid_id]
        rbf_rows.append(
            [
                str(cp_idx),
                str(q_idx),
                str(solid_id),
                f"{nbr_w[cp_idx, q_idx]:.10e}",
                f"{xyz[0]:.10e}",
                f"{xyz[1]:.10e}",
                f"{xyz[2]:.10e}",
            ]
        )
save_csv(
    os.path.join(out_dir, "rbf_neighbor_map.csv"),
    "cp_idx,neighbor_rank,scalar_node_id,weight,node_x,node_y,node_z",
    rbf_rows,
)

summary_lines = [
    "Reissner-Mindlin v12 indexing summary",
    f"mesh vertices: {mesh_xy.shape[0]}",
    f"mesh cells: {cells.shape[0]}",
    f"CG1/interface nodes (used by Vt.sub(0).collapse()): {scalar_xyz.shape[0]}",
    f"control points / coupling targets: {len(cp_node_ids)}",
    f"mixed-space total dofs: {V.dim()}",
    f"u_x component dofs: {len(dofs_u_x)}",
    f"u_y component dofs: {len(dofs_u_y)}",
    f"w component dofs: {len(dofs_w)}",
    f"theta_x component dofs: {len(dofs_theta_x)}",
    f"theta_y component dofs: {len(dofs_theta_y)}",
    "",
    "Index spaces visualized here:",
    "1. mesh vertex_id and cell_id from RectangleMesh after chord/sweep mapping",
    "2. scalar_node_id from the CG1 interface space used for aerodynamic coupling",
    "3. panel_idx = i_span * n_chord + j_chord used for panel ownership and incoming force layout",
    "4. cp_idx = i_span * n_chord + j_chord used for communication control points",
    "5. mixed_global_dof_id for the Reissner-Mindlin state [u_x, u_y, w, theta_x, theta_y]",
    "6. local RBF neighbor links from each control point to interface nodes",
    "",
    "Files:",
    "mesh_vertex_indices.png",
    "cell_indices.png",
    "scalar_interface_node_indices.png",
    "panel_and_control_point_map.png",
    "dofs_u_x.png",
    "dofs_u_y.png",
    "dofs_w.png",
    "dofs_theta_x.png",
    "dofs_theta_y.png",
    "rbf_neighbor_links.png",
    "mesh_vertices.csv",
    "mesh_cells.csv",
    "scalar_interface_nodes.csv",
    "panel_to_scalar_nodes.csv",
    "coupling_control_points.csv",
    "mixed_dof_maps.csv",
    "rbf_neighbor_map.csv",
]

summary_path = os.path.join(out_dir, "summary.txt")
with open(summary_path, "w", encoding="ascii") as f:
    f.write("\n".join(summary_lines) + "\n")

print("Reissner-Mindlin v12 indexing visualization completed.")
print(f"Output directory: {out_dir}")
print(f"Mesh vertices: {mesh_xy.shape[0]}")
print(f"Mesh cells: {cells.shape[0]}")
print(f"Scalar/interface nodes: {scalar_xyz.shape[0]}")
print(f"Coupling/control points: {len(cp_node_ids)}")
print(f"Mixed-space total dofs: {V.dim()}")
print(f"Summary: {summary_path}")
