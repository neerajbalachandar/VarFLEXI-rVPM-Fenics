import numpy as np
import matplotlib.pyplot as plt


#just for visualization 
#naca 0012 wing
span = 1.0
root_chord = 0.12
tip_chord = 0.08
thickness_ratio = 0.12
leading_edge_sweep = 0.0


n_chord = 121
n_span = 81
n_thickness = 21
span_strips = 200

def chord_at(y_val):
    eta = np.clip(y_val / span, 0.0, 1.0)
    return root_chord + (tip_chord - root_chord) * eta


def x_leading_edge_at(y_val):
    eta = np.clip(y_val / span, 0.0, 1.0)
    return leading_edge_sweep * eta


def naca_half_thickness(xi):
    xi_clip = np.clip(xi, 0.0, 1.0)
    return 5.0 * thickness_ratio * (
        0.2969 * np.sqrt(xi_clip)
        - 0.1260 * xi_clip
        - 0.3516 * xi_clip**2
        + 0.2843 * xi_clip**3
        - 0.1015 * xi_clip**4
    )


def map_reference_to_airfoil(xi, y, z_ref):
    chord = chord_at(y)
    x_le = x_leading_edge_at(y)
    zeta = 2.0 * z_ref  # z_ref in [-0.5, 0.5] -> zeta in [-1, 1]
    half_t = chord * naca_half_thickness(xi)
    x = x_le + xi * chord
    z = zeta * half_t
    return x, y, z


def build_mapped_volume():
    xi = np.linspace(0.0, 1.0, n_chord)
    y = np.linspace(0.0, span, n_span)
    z_ref = np.linspace(-0.5, 0.5, n_thickness)
    XI, Y, ZREF = np.meshgrid(xi, y, z_ref, indexing="ij")
    X, Y, Z = map_reference_to_airfoil(XI, Y, ZREF)
    return X, Y, Z


def main():
    X, Y, Z = build_mapped_volume()
    xi = np.linspace(0.0, 1.0, n_chord)
    y = np.linspace(0.0, span, n_span)
    XI_s, Y_s = np.meshgrid(xi, y, indexing="ij")
    XU, YU, ZU = map_reference_to_airfoil(XI_s, Y_s, 0.5)
    XL, YL, ZL = map_reference_to_airfoil(XI_s, Y_s, -0.5)

    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(XU, YU, ZU, linewidth=0.0, antialiased=True, alpha=0.95)
    ax1.plot_surface(XL, YL, ZL, linewidth=0.0, antialiased=True, alpha=0.95)
    ax1.set_title("Mapped Airfoil Wing Surface")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y (span)")
    ax1.set_zlabel("z")
    ax1.set_box_aspect((np.ptp(XU), span, 0.25 * root_chord))

    ax2 = fig.add_subplot(1, 2, 2)
    y_mid = 0.5 * span
    chord_mid = chord_at(y_mid)
    x_le_mid = x_leading_edge_at(y_mid)
    x_profile = x_le_mid + xi * chord_mid
    z_half = chord_mid * naca_half_thickness(xi)
    ax2.plot(x_profile, z_half, "k-", label="upper")
    ax2.plot(x_profile, -z_half, "k-")
    ax2.axvline(x_le_mid + 0.75 * chord_mid, color="r", linestyle="--", label="3/4 chord")
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_title("Mid-Span Airfoil Section")
    ax2.set_xlabel("x")
    ax2.set_ylabel("z")
    ax2.legend(loc="best")

    plt.tight_layout()
    plt.show()

    # Top-view plot: 3/4-chord line across span and strip-center points
    y_line = np.linspace(0.0, span, 400)
    x_line = np.array([x_leading_edge_at(yv) + 0.75 * chord_at(yv) for yv in y_line])
    y_pts = (np.arange(span_strips) + 0.5) * span / span_strips
    x_pts = np.array([x_leading_edge_at(yv) + 0.75 * chord_at(yv) for yv in y_pts])
    x_le = np.array([x_leading_edge_at(yv) for yv in y_line])
    x_te = np.array([x_leading_edge_at(yv) + chord_at(yv) for yv in y_line])

    plt.figure(figsize=(7, 5))
    plt.plot(y_line, x_le, "k-", label="leading edge")
    plt.plot(y_line, x_te, "k-", label="trailing edge")
    plt.plot(y_line, x_line, "r-", linewidth=2.0, label="3/4 chord line")
    plt.scatter(y_pts, x_pts, s=10, c="b", label=f"{span_strips} strip centers")
    plt.xlabel("y (span)")
    plt.ylabel("x (chordwise)")
    plt.title("Spanwise Strip Points at 3/4 Chord")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

    surf_upper = np.column_stack([XU.ravel(), YU.ravel(), ZU.ravel()])
    surf_lower = np.column_stack([XL.ravel(), YL.ravel(), ZL.ravel()])
    np.savetxt("solid-fenics/results/airfoil_surface_upper.csv", surf_upper, delimiter=",", header="x,y,z", comments="")
    np.savetxt("solid-fenics/results/airfoil_surface_lower.csv", surf_lower, delimiter=",", header="x,y,z", comments="")


    stride = max(n_chord // 20, 1)
    vol_points = np.column_stack([X[::stride, ::stride, ::stride].ravel(),
                                  Y[::stride, ::stride, ::stride].ravel(),
                                  Z[::stride, ::stride, ::stride].ravel()])
    np.savetxt("solid-fenics/results/airfoil_volume_points.csv", vol_points, delimiter=",", header="x,y,z", comments="")


if __name__ == "__main__":
    main()
