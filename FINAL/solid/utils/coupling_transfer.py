from dolfin import *
import numpy as np
from scipy.spatial import cKDTree


class CouplingTransfer:
    """Coupling-side geometry, sampling, and transfer operators."""

    def __init__(
        self,
        domain,
        model,
        eta_span_comm,
        eta_cp,
        rbf_radius=0.08,
        rbf_neighbors=24,
        max_abs_force_component=5.0e3,
        edge_eval_xi_eps=1.0e-6,
        work_conservative_mode=True,
        enforce_chord_projection=False,
        enforce_span_projection=False,
        debug_io=False,
    ):
        self.domain = domain
        self.model = model
        self.eta_span_comm = np.asarray(eta_span_comm, dtype=float).reshape(-1)
        self.eta_cp = float(eta_cp)
        self.eta_cp_comm = np.array([self.eta_cp], dtype=float)
        self.rbf_radius = float(rbf_radius)
        self.rbf_neighbors = int(rbf_neighbors)
        self.max_abs_force_component = float(max_abs_force_component)
        self.edge_eval_xi_eps = float(edge_eval_xi_eps)
        self.work_conservative_mode = bool(work_conservative_mode)
        self.enforce_chord_projection = bool(enforce_chord_projection)
        self.enforce_span_projection = bool(enforce_span_projection)
        self.debug_io = bool(debug_io)

        self.aero_node_ids, self.aero_coords = self.get_aero_surface_node_ids()
        self.interface_node_ids, self.interface_coords, self.interface_tree = (
            self._build_interface_sets()
        )
        self.cp_targets = self.domain.build_spanwise_targets(self.eta_span_comm, self.eta_cp, xi_eps=0.0)
        self.le_targets = self.domain.build_spanwise_targets(
            self.eta_span_comm, 0.0, xi_eps=self.edge_eval_xi_eps
        )
        self.te_targets = self.domain.build_spanwise_targets(
            self.eta_span_comm, 1.0, xi_eps=self.edge_eval_xi_eps
        )
        self.crm_transfer_matrix, self.crm_panel_areas, self.crm_panel_polys = (
            self.build_common_refinement_operator()
        )

        cp_targets_2d = self.cp_targets[:, :2]
        self.nbr_ids, self.nbr_w = self.build_local_rbf_map(
            cp_targets_2d, self.interface_coords, self.rbf_radius, n_neighbors=self.rbf_neighbors
        )
        self.A_diag = np.ones((self.cp_targets.shape[0],), dtype=float)
        self.S_lumped = self.compute_S_lumped(
            len(self.interface_node_ids), self.nbr_ids, self.nbr_w, self.A_diag
        )

    def get_aero_surface_node_ids(self):
        v_w = self.model.V.sub(1).collapse()
        coords_v = v_w.tabulate_dof_coordinates().reshape((-1, 2))
        ids = []
        for i_node, X in enumerate(coords_v):
            y_val = X[1]
            if 0.0 <= y_val <= self.domain.span:
                ids.append(i_node)
        return np.asarray(ids, dtype=np.int64), coords_v

    def _build_interface_sets(self):
        crm_space = FunctionSpace(self.domain.mesh, "CG", 1)
        crm_coords = crm_space.tabulate_dof_coordinates().reshape((-1, 2))
        crm_tree = cKDTree(self.aero_coords)
        crm_dist, interface_node_ids = crm_tree.query(crm_coords, k=1)
        interface_node_ids = np.asarray(interface_node_ids, dtype=np.int64)
        if np.max(crm_dist) > 1.0e-10:
            raise RuntimeError(
                f"CRM vertex-to-solid dof mapping failed: max mismatch={np.max(crm_dist):.3e}"
            )
        interface_coords = self.aero_coords[interface_node_ids]
        interface_tree = cKDTree(interface_coords)
        return interface_node_ids, interface_coords, interface_tree

    @staticmethod
    def as_eta_array(values, n):
        if values is None:
            return np.linspace(0.0, 1.0, n)
        arr = np.asarray(values, dtype=float).reshape(-1)
        if len(arr) != n:
            return np.linspace(0.0, 1.0, n)
        if n > 1:
            arr = np.clip(arr, 0.0, 1.0)
            arr = np.maximum.accumulate(arr)
            if arr[-1] > 0.0:
                arr = arr / arr[-1]
        return arr

    @staticmethod
    def interp_profile(x_src, vals_src, x_dst):
        x_src = np.asarray(x_src, dtype=float).reshape(-1)
        vals_src = np.asarray(vals_src, dtype=float)
        x_dst = np.asarray(x_dst, dtype=float).reshape(-1)
        if vals_src.ndim == 1:
            vals_src = vals_src.reshape(-1, 1)
        if len(x_src) == 0 or vals_src.shape[0] == 0:
            return np.zeros((len(x_dst), vals_src.shape[1]), dtype=float)
        if len(x_src) == 1:
            return np.repeat(vals_src[:1, :], len(x_dst), axis=0)
        out = np.zeros((len(x_dst), vals_src.shape[1]), dtype=float)
        for c_idx in range(vals_src.shape[1]):
            out[:, c_idx] = np.interp(x_dst, x_src, vals_src[:, c_idx])
        return out

    def resample_forces_to_shape(
        self,
        forces,
        n_span_out,
        n_chord_out,
        eta_span_in=None,
        eta_chord_in=None,
        eta_span_out=None,
        eta_chord_out=None,
    ):
        forces = np.asarray(forces, dtype=float).reshape(-1, 3)
        n_out = n_span_out * n_chord_out
        if len(forces) == 0:
            return np.zeros((n_out, 3), dtype=float)

        eta_span_out = self.as_eta_array(eta_span_out, n_span_out)
        eta_chord_out = self.as_eta_array(eta_chord_out, n_chord_out)

        n_in = len(forces)
        if eta_span_in is None or eta_chord_in is None:
            if n_in == n_out:
                return forces
            if n_in == 1:
                return np.repeat(forces[:1], n_out, axis=0)
            s_in = np.linspace(0.0, 1.0, n_in)
            s_out = np.linspace(0.0, 1.0, n_out)
            return self.interp_profile(s_in, forces, s_out)

        eta_span_in = np.asarray(eta_span_in, dtype=float).reshape(-1)
        eta_chord_in = np.asarray(eta_chord_in, dtype=float).reshape(-1)
        n_span_in = len(eta_span_in)
        n_chord_in = len(eta_chord_in)
        if n_span_in == 0 or n_chord_in == 0:
            return np.zeros((n_out, 3), dtype=float)
        if n_in != n_span_in * n_chord_in:
            return self.resample_forces_to_shape(forces, n_span_out, n_chord_out)

        grid_in = forces.reshape((n_span_in, n_chord_in, 3))
        eta_span_in = self.as_eta_array(eta_span_in, n_span_in)
        eta_chord_in = self.as_eta_array(eta_chord_in, n_chord_in)

        grid_span = np.zeros((n_span_out, n_chord_in, 3), dtype=float)
        for j_idx in range(n_chord_in):
            grid_span[:, j_idx, :] = self.interp_profile(
                eta_span_in, grid_in[:, j_idx, :], eta_span_out
            )

        grid_out = np.zeros((n_span_out, n_chord_out, 3), dtype=float)
        for i_idx in range(n_span_out):
            grid_out[i_idx, :, :] = self.interp_profile(
                eta_chord_in, grid_span[i_idx, :, :], eta_chord_out
            )

        return grid_out.reshape((n_out, 3))

    def parse_force_payload(self, data, n_span_out, n_chord_out, eta_span_out, eta_chord_out):
        eta_span_out = self.as_eta_array(eta_span_out, n_span_out)
        eta_chord_out = self.as_eta_array(eta_chord_out, n_chord_out)

        if "n_span" in data and "n_chord" in data and "force" in data:
            n_span_in = int(data.get("n_span", 0))
            n_chord_in = int(data.get("n_chord", 0))
            force_raw = np.asarray(data.get("force", []), dtype=float).reshape(-1, 3)
            indexing = str(data.get("indexing", "span-major"))
            if n_span_in > 0 and n_chord_in > 0 and len(force_raw) == n_span_in * n_chord_in:
                force_grid = np.zeros((n_span_in, n_chord_in, 3), dtype=float)
                k_idx = 0
                if indexing == "span-major":
                    for i_idx in range(n_span_in):
                        for j_idx in range(n_chord_in):
                            force_grid[i_idx, j_idx, :] = force_raw[k_idx, :]
                            k_idx += 1
                elif indexing == "chord-major":
                    for j_idx in range(n_chord_in):
                        for i_idx in range(n_span_in):
                            force_grid[i_idx, j_idx, :] = force_raw[k_idx, :]
                            k_idx += 1
                else:
                    raise RuntimeError(f"Unsupported force indexing '{indexing}'")
                force_raw = force_grid.reshape((-1, 3))
            eta_span_in = (
                self.as_eta_array(data.get("eta_span"), n_span_in) if n_span_in > 0 else None
            )
            eta_chord_in = (
                self.as_eta_array(data.get("eta_chord"), n_chord_in) if n_chord_in > 0 else None
            )
            if n_span_in > 0 and n_chord_in > 0:
                forces = self.resample_forces_to_shape(
                    force_raw,
                    n_span_out,
                    n_chord_out,
                    eta_span_in=eta_span_in,
                    eta_chord_in=eta_chord_in,
                    eta_span_out=eta_span_out,
                    eta_chord_out=eta_chord_out,
                )
                forces = np.clip(
                    forces, -self.max_abs_force_component, self.max_abs_force_component
                )
                return forces, True

        forces_legacy = np.asarray(data.get("force", []), dtype=float).reshape(-1, 3)
        forces = self.resample_forces_to_shape(
            forces_legacy,
            n_span_out,
            n_chord_out,
            eta_span_out=eta_span_out,
            eta_chord_out=eta_chord_out,
        )
        forces = np.clip(forces, -self.max_abs_force_component, self.max_abs_force_component)
        return forces, False

    def build_spanwise_targets(self, eta_span_vals, xi_val, xi_eps=0.0):
        return self.domain.build_spanwise_targets(eta_span_vals, xi_val, xi_eps=xi_eps)

    def sample_vector_field_at_targets(self, u_fun, targets_xyz, fallback_tree=None, fallback_vals=None):
        out = np.zeros((targets_xyz.shape[0], 3), dtype=float)
        for k_idx in range(targets_xyz.shape[0]):
            pt = Point(
                float(targets_xyz[k_idx, 0]),
                float(targets_xyz[k_idx, 1]),
                float(targets_xyz[k_idx, 2]),
            )
            try:
                val = u_fun(pt)
                out[k_idx, 0] = float(val[0])
                out[k_idx, 1] = float(val[1])
                out[k_idx, 2] = float(val[2])
            except RuntimeError:
                if fallback_tree is None or fallback_vals is None:
                    continue
                _, idx = fallback_tree.query(targets_xyz[k_idx, :2], k=1)
                out[k_idx, :] = fallback_vals[int(idx), :]
        return out

    def span_edges_from_stations(self, eta_span_vals):
        eta = np.asarray(eta_span_vals, dtype=float).reshape(-1)
        if len(eta) == 0:
            raise ValueError("eta_span_vals must contain at least one station")
        if len(eta) == 1:
            return np.array([0.0, 1.0], dtype=float)

        edges = np.zeros((len(eta) + 1,), dtype=float)
        edges[0] = 0.0
        edges[-1] = 1.0
        edges[1:-1] = 0.5 * (eta[:-1] + eta[1:])
        edges = np.clip(edges, 0.0, 1.0)
        edges = np.maximum.accumulate(edges)
        edges[-1] = 1.0
        return edges

    def panel_polygon_for_span_interval(self, y0, y1):
        x0_le = self.domain.x_leading_edge_at(y0)
        x0_te = x0_le + self.domain.chord_at(y0)
        x1_le = self.domain.x_leading_edge_at(y1)
        x1_te = x1_le + self.domain.chord_at(y1)
        return np.array([[x0_le, y0], [x0_te, y0], [x1_te, y1], [x1_le, y1]], dtype=float)

    @staticmethod
    def polygon_area(poly):
        poly = np.asarray(poly, dtype=float)
        if len(poly) < 3:
            return 0.0
        x = poly[:, 0]
        y = poly[:, 1]
        return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    @staticmethod
    def polygon_centroid(poly):
        poly = np.asarray(poly, dtype=float)
        if len(poly) == 0:
            return np.zeros((2,), dtype=float)
        return np.mean(poly, axis=0)

    @staticmethod
    def is_inside_halfplane(p, a, b, eps=1.0e-14):
        return ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) >= -eps

    @staticmethod
    def line_intersection(p1, p2, a, b):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = a
        x4, y4 = b
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1.0e-16:
            return p2.copy()
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        return np.array([px, py], dtype=float)

    def clip_polygon_to_convex(self, poly, clip_poly):
        poly = [np.asarray(p, dtype=float) for p in np.asarray(poly, dtype=float)]
        clip_poly = np.asarray(clip_poly, dtype=float)
        if len(poly) == 0:
            return np.zeros((0, 2), dtype=float)
        for i in range(len(clip_poly)):
            a = clip_poly[i]
            b = clip_poly[(i + 1) % len(clip_poly)]
            if len(poly) == 0:
                break
            out = []
            prev = poly[-1]
            prev_inside = self.is_inside_halfplane(prev, a, b)
            for cur in poly:
                cur_inside = self.is_inside_halfplane(cur, a, b)
                if cur_inside:
                    if not prev_inside:
                        out.append(self.line_intersection(prev, cur, a, b))
                    out.append(cur)
                elif prev_inside:
                    out.append(self.line_intersection(prev, cur, a, b))
                prev = cur
                prev_inside = cur_inside
            poly = out
        if len(poly) == 0:
            return np.zeros((0, 2), dtype=float)
        return np.asarray(poly, dtype=float)

    @staticmethod
    def barycentric_coords_triangle(p, tri):
        a = tri[0]
        b = tri[1]
        c = tri[2]
        v0 = b - a
        v1 = c - a
        v2 = p - a
        denom = v0[0] * v1[1] - v1[0] * v0[1]
        if abs(denom) < 1.0e-16:
            return None
        l2 = (v2[0] * v1[1] - v1[0] * v2[1]) / denom
        l3 = (v0[0] * v2[1] - v2[0] * v0[1]) / denom
        l1 = 1.0 - l2 - l3
        return np.array([l1, l2, l3], dtype=float)

    @staticmethod
    def cg1_triangle_basis(bary):
        return np.asarray(bary, dtype=float)

    @staticmethod
    def triangle_quadrature_rule():
        bary = np.array(
            [[1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0], [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0], [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0]],
            dtype=float,
        )
        weights = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=float)
        return bary, weights

    @staticmethod
    def triangulate_polygon_fan(poly):
        poly = np.asarray(poly, dtype=float)
        if len(poly) < 3:
            return []
        base = poly[0]
        tris = []
        for i in range(1, len(poly) - 1):
            tris.append(np.array([base, poly[i], poly[i + 1]], dtype=float))
        return tris

    def build_common_refinement_operator(self):
        eta_span_vals = self.eta_span_comm
        if len(eta_span_vals) == 0:
            raise ValueError("No span stations supplied to CRM operator")

        span_edges = self.span_edges_from_stations(eta_span_vals)
        panel_polys = []
        panel_areas = np.zeros((len(eta_span_vals),), dtype=float)
        for i_idx in range(len(eta_span_vals)):
            y0 = span_edges[i_idx] * self.domain.span
            y1 = span_edges[i_idx + 1] * self.domain.span
            poly = self.panel_polygon_for_span_interval(y0, y1)
            panel_polys.append(poly)
            panel_areas[i_idx] = max(self.polygon_area(poly), 1.0e-14)

        trial_space = FunctionSpace(self.domain.mesh, "CG", 1)
        dofmap = trial_space.dofmap()
        ndofs = trial_space.dim()
        n_panels = len(eta_span_vals)
        C = np.zeros((ndofs, n_panels), dtype=float)

        bary_rule, bary_weights = self.triangle_quadrature_rule()
        for cell in cells(self.domain.mesh):
            cell_index = cell.index()
            vert_ids = cell.entities(0)
            tri = self.domain.mesh.coordinates()[vert_ids, :2]
            tri_min = np.min(tri, axis=0)
            tri_max = np.max(tri, axis=0)
            local_dofs = dofmap.cell_dofs(cell_index)

            for j_idx, panel_poly in enumerate(panel_polys):
                panel_min = np.min(panel_poly, axis=0)
                panel_max = np.max(panel_poly, axis=0)
                if tri_max[0] < panel_min[0] - 1.0e-14 or tri_min[0] > panel_max[0] + 1.0e-14:
                    continue
                if tri_max[1] < panel_min[1] - 1.0e-14 or tri_min[1] > panel_max[1] + 1.0e-14:
                    continue

                clipped = self.clip_polygon_to_convex(tri, panel_poly)
                if len(clipped) < 3:
                    continue

                clipped_centroid = self.polygon_centroid(clipped)
                if not np.all(np.isfinite(clipped_centroid)):
                    continue

                for subtri in self.triangulate_polygon_fan(clipped):
                    sub_area = self.polygon_area(subtri)
                    if sub_area <= 1.0e-16:
                        continue
                    for k_idx in range(3):
                        bary_sub = bary_rule[k_idx]
                        p = (
                            bary_sub[0] * subtri[0]
                            + bary_sub[1] * subtri[1]
                            + bary_sub[2] * subtri[2]
                        )
                        bary_tri = self.barycentric_coords_triangle(p, tri)
                        if bary_tri is None:
                            continue
                        phi = self.cg1_triangle_basis(bary_tri)
                        wq = sub_area * bary_weights[k_idx]
                        C[local_dofs, j_idx] += wq * phi

        return C, panel_areas, panel_polys

    @staticmethod
    def panel_loads_to_solid_nodal_forces(panel_forces, C, panel_areas):
        panel_forces = np.asarray(panel_forces, dtype=float).reshape(-1, 3)
        densities = panel_forces / panel_areas[:, None]
        return C @ densities

    @staticmethod
    def nodal_displacements_to_panel_average(nodal_disp, C, panel_areas):
        nodal_disp = np.asarray(nodal_disp, dtype=float).reshape(-1, 3)
        return (C.T @ nodal_disp) / panel_areas[:, None]

    @staticmethod
    def project_le_te_inextensible(u_le_arr, u_te_arr, le_ref, te_ref):
        Xle = le_ref + u_le_arr
        Xte = te_ref + u_te_arr
        mid = 0.5 * (Xle + Xte)

        ref_vec = te_ref - le_ref
        ref_len = np.linalg.norm(ref_vec, axis=1)

        cur_vec = Xte - Xle
        cur_len = np.linalg.norm(cur_vec, axis=1)
        eps = 1.0e-14

        ref_dir = np.zeros_like(ref_vec)
        ok_ref = ref_len > eps
        ref_dir[ok_ref, :] = ref_vec[ok_ref, :] / ref_len[ok_ref, None]
        if np.any(~ok_ref):
            ref_dir[~ok_ref, :] = np.array([1.0, 0.0, 0.0], dtype=float)

        cur_dir = np.zeros_like(cur_vec)
        ok_cur = cur_len > eps
        cur_dir[ok_cur, :] = cur_vec[ok_cur, :] / cur_len[ok_cur, None]
        if np.any(~ok_cur):
            cur_dir[~ok_cur, :] = ref_dir[~ok_cur, :]

        Xle_p = mid - 0.5 * ref_len[:, None] * cur_dir
        Xte_p = mid + 0.5 * ref_len[:, None] * cur_dir
        u_le_p = Xle_p - le_ref
        u_te_p = Xte_p - te_ref
        return u_le_p, u_te_p, cur_len, ref_len

    @staticmethod
    def project_spanwise_inextensible_line(u_line, ref_line):
        X = ref_line + u_line
        n_pts = X.shape[0]
        if n_pts <= 1:
            return u_line, np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)

        seg_ref_vec = ref_line[1:, :] - ref_line[:-1, :]
        seg_ref_len = np.linalg.norm(seg_ref_vec, axis=1)
        seg_cur_vec = X[1:, :] - X[:-1, :]
        seg_cur_len = np.linalg.norm(seg_cur_vec, axis=1)

        Xp = np.zeros_like(X)
        Xp[0, :] = X[0, :]
        eps = 1.0e-14
        for i in range(1, n_pts):
            v = X[i, :] - X[i - 1, :]
            lv = np.linalg.norm(v)
            if lv > eps:
                d = v / lv
            else:
                rv = ref_line[i, :] - ref_line[i - 1, :]
                lrv = np.linalg.norm(rv)
                if lrv > eps:
                    d = rv / lrv
                else:
                    d = np.array([0.0, 1.0, 0.0], dtype=float)
            Xp[i, :] = Xp[i - 1, :] + seg_ref_len[i - 1] * d

        u_proj = Xp - ref_line
        return u_proj, seg_cur_len, seg_ref_len

    def build_local_rbf_map(self, fluid_points, solid_points, radius, n_neighbors=24):
        fluid_points = np.asarray(fluid_points, dtype=float)
        solid_points = np.asarray(solid_points, dtype=float)
        support_radius = float(radius)
        if support_radius <= 0.0:
            raise ValueError("COUPLING_RBF_RADIUS must be positive for compact RBF transfer")
        n_s = solid_points.shape[0]
        k = int(max(1, min(n_neighbors, n_s)))

        tree = cKDTree(solid_points)
        dists, idx = tree.query(fluid_points, k=k)
        if k == 1:
            dists = dists.reshape(-1, 1)
            idx = idx.reshape(-1, 1)

        nbr_ids = idx.astype(np.int64)
        q = dists / support_radius
        nbr_w = np.zeros_like(dists, dtype=float)
        inside = q < 1.0
        one_minus_q = 1.0 - q[inside]
        nbr_w[inside] = (one_minus_q ** 4) * (4.0 * q[inside] + 1.0)

        row_sum = np.sum(nbr_w, axis=1, keepdims=True)
        bad = np.where(row_sum[:, 0] <= 1.0e-16)[0]
        for bi in bad:
            nbr_w[bi, :] = 0.0
            nbr_w[bi, 0] = 1.0
        row_sum = np.sum(nbr_w, axis=1, keepdims=True)
        nbr_w /= np.maximum(row_sum, 1.0e-16)
        return nbr_ids, nbr_w

    @staticmethod
    def map_forces_to_solid(f_fluid, n_solid_nodes, nbr_ids, nbr_w):
        n_f, k = nbr_ids.shape
        out = np.zeros((n_solid_nodes, 3), dtype=float)
        for q_idx in range(k):
            contrib = nbr_w[:, q_idx : q_idx + 1] * f_fluid
            np.add.at(out, nbr_ids[:, q_idx], contrib)
        return out

    @staticmethod
    def compute_S_lumped(n_solid_nodes, nbr_ids, nbr_w, A_diag):
        _n_f, k = nbr_ids.shape
        S = np.zeros((n_solid_nodes,), dtype=float)
        for q_idx in range(k):
            np.add.at(S, nbr_ids[:, q_idx], nbr_w[:, q_idx] * A_diag)
        return np.maximum(S, 1.0e-14)

    @classmethod
    def apply_Tf_operator(cls, Fa, n_solid_nodes, nbr_ids, nbr_w, A_diag, S_lumped):
        FaA = Fa * A_diag[:, None]
        rhs = cls.map_forces_to_solid(FaA, n_solid_nodes, nbr_ids, nbr_w)
        Fs_coeff = rhs / S_lumped[:, None]
        return Fs_coeff, rhs

    @staticmethod
    def add_nodal_forces_to_rhs_plate(rhs_vec, nodal_forces, node_ids, dofs_u_x, dofs_u_y, dofs_w):
        arr = rhs_vec.get_local()
        arr[dofs_u_x[node_ids]] += nodal_forces[:, 0]
        arr[dofs_u_y[node_ids]] += nodal_forces[:, 1]
        arr[dofs_w[node_ids]] += nodal_forces[:, 2]
        rhs_vec.set_local(arr)
        rhs_vec.apply("insert")

    @staticmethod
    def get_nodal_displacements_plate(q_fun, node_ids, dofs_u_x, dofs_u_y, dofs_w):
        q_arr = q_fun.vector().get_local()
        out = np.zeros((len(node_ids), 3), dtype=float)
        out[:, 0] = q_arr[dofs_u_x[node_ids]]
        out[:, 1] = q_arr[dofs_u_y[node_ids]]
        out[:, 2] = q_arr[dofs_w[node_ids]]
        return out


__all__ = ["SolidCouplingTransfer"]
