from dolfin import *
import numpy as np


class SolidDomain:
    """Geometry, mesh, and coupling-station utilities for the plate domain."""

    def __init__(
        self,
        span,
        root_chord,
        tip_chord,
        thickness_ratio,
        leading_edge_sweep,
        nx,
        ny,
        n_span_comm,
        span_sampling_mode="node-stride",
        span_custom_stride=None,
    ):
        self.span = float(span)
        self.root_chord = float(root_chord)
        self.tip_chord = float(tip_chord)
        self.thickness_ratio = float(thickness_ratio)
        self.leading_edge_sweep = float(leading_edge_sweep)
        self.nx = int(nx)
        self.ny = int(ny)
        self.n_span_comm = int(n_span_comm)
        self.span_sampling_mode = str(span_sampling_mode).strip().lower()
        self.span_custom_stride = span_custom_stride

        self.eta_span_comm, self.eta_span_comm_indices = self.build_eta_span_comm(
            self.n_span_comm, self.ny, self.span_sampling_mode, self.span_custom_stride
        )

        self.mesh = self._build_mesh()
        self.facet_markers, self.ds_aero = self._build_coupling_measures()
        self.plate_thickness = self.root_chord * self.thickness_ratio
        self.h = Constant(self.plate_thickness)

    @staticmethod
    def build_eta_span_comm(n_span_vals, ny_vals, mode, span_custom_stride=None):
        if n_span_vals < 1:
            raise ValueError("COUPLING_NSPAN_COMM must be >= 1")
        if n_span_vals == 1:
            return np.array([0.0], dtype=float), np.array([0], dtype=int)

        mode = str(mode).strip().lower()
        if mode == "midpoint":
            eta = (np.arange(n_span_vals, dtype=float) + 0.5) / n_span_vals
            idx = np.round(eta * ny_vals).astype(int)
            return eta, idx

        if mode == "node-stride":
            stride = max(1, ny_vals // n_span_vals)
            idx = np.arange(n_span_vals, dtype=int) * stride
            idx = np.clip(idx, 0, max(ny_vals, 1))
            if np.unique(idx).size < n_span_vals:
                idx = np.linspace(0, max(ny_vals - 1, 0), n_span_vals).round().astype(int)
            eta = idx.astype(float) / max(float(ny_vals), 1.0)
            return eta, idx

        if mode == "custom-stride":
            if span_custom_stride is None:
                raise ValueError(
                    "COUPLING_SPAN_SAMPLING=custom-stride requires COUPLING_SPAN_STRIDE"
                )
            stride = int(span_custom_stride)
            if stride < 1:
                raise ValueError("COUPLING_SPAN_STRIDE must be >= 1")
            last_idx = (n_span_vals - 1) * stride
            if last_idx > ny_vals:
                raise ValueError(
                    "Unsafe custom span stride: "
                    f"(COUPLING_NSPAN_COMM-1)*COUPLING_SPAN_STRIDE = {last_idx} "
                    f"exceeds SOLID_NY = {ny_vals}. Reduce stride/stations or increase SOLID_NY."
                )
            idx = np.arange(n_span_vals, dtype=int) * stride
            eta = idx.astype(float) / max(float(ny_vals), 1.0)
            return eta, idx

        if mode == "linspace":
            eta = np.linspace(0.0, 1.0, n_span_vals, endpoint=False)
            idx = np.round(eta * ny_vals).astype(int)
            return eta, idx

        raise ValueError(f"Unsupported COUPLING_SPAN_SAMPLING='{mode}'")

    def chord_at(self, y_val):
        eta = min(max(float(y_val) / self.span, 0.0), 1.0)
        return self.root_chord + (self.tip_chord - self.root_chord) * eta

    def x_leading_edge_at(self, y_val):
        eta = min(max(float(y_val) / self.span, 0.0), 1.0)
        return self.leading_edge_sweep * eta

    def naca_half_thickness(self, xi):
        xi_clip = min(max(float(xi), 0.0), 1.0)
        return 5.0 * self.thickness_ratio * (
            0.2969 * np.sqrt(xi_clip)
            - 0.1260 * xi_clip
            - 0.3516 * xi_clip ** 2
            + 0.2843 * xi_clip ** 3
            - 0.1015 * xi_clip ** 4
        )

    def _build_mesh(self):
        mesh = RectangleMesh(Point(0.0, 0.0), Point(1.0, self.span), self.nx, self.ny)
        coords = mesh.coordinates()
        for i in range(coords.shape[0]):
            xi = coords[i, 0]
            y_val = coords[i, 1]
            chord = self.chord_at(y_val)
            x_le = self.x_leading_edge_at(y_val)
            coords[i, 0] = x_le + xi * chord
        return mesh

    def _build_coupling_measures(self):
        facet_markers = MeshFunction(
            "size_t", self.mesh, self.mesh.topology().dim() - 1
        )
        facet_markers.set_all(0)
        DomainBoundary().mark(facet_markers, 1)
        ds_aero = Measure("ds", domain=self.mesh, subdomain_data=facet_markers)
        return facet_markers, ds_aero

    def is_left_boundary(self, x, on_boundary):
        return near(x[1], 0.0) and on_boundary

    def build_spanwise_targets(self, eta_span_vals, xi_val, xi_eps=0.0):
        pts = np.zeros((len(eta_span_vals), 3), dtype=float)
        xi_eff = float(np.clip(xi_val, 0.0, 1.0))
        if xi_eff <= 0.0:
            xi_eff = min(1.0, xi_eff + float(xi_eps))
        elif xi_eff >= 1.0:
            xi_eff = max(0.0, xi_eff - float(xi_eps))

        for i_idx, eta_s in enumerate(eta_span_vals):
            y_val = float(eta_s) * self.span
            chord = self.chord_at(y_val)
            x_le = self.x_leading_edge_at(y_val)
            pts[i_idx, 0] = x_le + xi_eff * chord
            pts[i_idx, 1] = y_val
            pts[i_idx, 2] = 0.0
        return pts


__all__ = ["SolidDomain"]
