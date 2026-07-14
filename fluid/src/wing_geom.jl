function chord_length_twist(xl, zl, xt, zt)
    dx = xt - xl
    dz = zt - zl
    c = sqrt(dx * dx + dz * dz)
    twist = atan(-dz, dx) * 180 / pi
    return c, twist
end

function make_cantilever_template(span, c_root, c_tip, x_tip, z_tip, twist_root, twist_tip, nspan)
    wing = vlm.Wing(0.0, 0.0, 0.0, c_root, twist_root)
    vlm.addchord(wing, x_tip, span, z_tip, c_tip, twist_tip, nspan; r=1.0)
    return wing
end

# wing = make_cantilever_template(
#     span, root_chord, tip_chord, leading_edge_sweep, 0.0, twist_root, twist_tip, n_span
# )
# wing_ref = deepcopy(wing)

# system = vlm.WingSystem()
# vlm.addwing(system, "Wing", wing)
# vehicle = uns.VLMVehicle(system; vlm_system=system, wake_system=system)

# Vvehicle(t) = zeros(3)
# anglevehicle(t) = zeros(3)
# maneuver = uns.KinematicManeuver((), (), Vvehicle, anglevehicle)

# simulation = uns.Simulation(
#     vehicle, maneuver, 0.0, 0.0, ttot;
#     Vinit=zeros(3), Winit=zeros(3)
# )