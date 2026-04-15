import socket
import json
import os
import errno
import subprocess
import csv
from typing import Optional, List

HOST = os.getenv("COUPLING_HOST", "127.0.0.1")
PORT = int(os.getenv("COUPLING_PORT", "9000"))
NSTEPS = int(os.getenv("COUPLING_NSTEPS", "800"))
FORCE_RELAX = float(os.getenv("COUPLING_FORCE_RELAX", "1.0"))
USE_AITKEN = os.getenv("COUPLING_AITKEN", "1").strip() not in ("0", "false", "False")

def port_owner_hint(port):
    try:
        out = subprocess.check_output(
            ["fuser", "-v", f"{port}/tcp"],
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
        if out:
            return out
    except Exception:
        pass
    return "Unable to query owner (try: `fuser -v {}/tcp`)".format(port)

def read_json_line(stream, name):
    line = stream.readline()
    if line == "":
        raise RuntimeError(f"{name} disconnected or sent empty line")
    return json.loads(line)

def accept_role(server, fluid_conn, fluid_file, solid_conn, solid_file):
    conn, addr = server.accept()
    stream = conn.makefile("r")
    hello = read_json_line(stream, f"Client@{addr}")
    role = str(hello.get("role", "")).strip().lower()

    if role == "fluid":
        if fluid_conn is not None:
            raise RuntimeError("Duplicate fluid connection")
        print(f"Fluid connected from {addr}.")
        return conn, stream, solid_conn, solid_file
    if role == "solid":
        if solid_conn is not None:
            raise RuntimeError("Duplicate solid connection")
        print(f"Solid connected from {addr}.")
        return fluid_conn, fluid_file, conn, stream

    raise RuntimeError(f"Client {addr} must send role handshake {{\"role\":\"fluid\"|\"solid\"}}")

print("Starting coupling server...")
print(f"Coupling config: NSTEPS={NSTEPS}, FORCE_RELAX={FORCE_RELAX}, AITKEN={USE_AITKEN}")

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    server.bind((HOST, PORT))
except OSError as err:
    if err.errno == errno.EADDRINUSE:
        raise RuntimeError(
            f"Port {PORT} is already in use on {HOST}.\n"
            f"{port_owner_hint(PORT)}\n"
            f"Stop the old process or run with another port:\n"
            f"`COUPLING_PORT=<port> python3 coupling.py`"
        ) from err
    raise
server.listen(2)

fluid_conn, fluid_file = None, None
solid_conn, solid_file = None, None
print("Waiting for fluid and solid role handshakes...")
while fluid_conn is None or solid_conn is None:
    fluid_conn, fluid_file, solid_conn, solid_file = accept_role(
        server, fluid_conn, fluid_file, solid_conn, solid_file
    )
print("Both participants connected.")

results_dir = os.path.join("results", "coupling")
os.makedirs(results_dir, exist_ok=True)
log_csv = os.path.join(results_dir, "coupling_history.csv")
log_fp = open(log_csv, "w", newline="")
writer = csv.writer(log_fp)
writer.writerow(["step", "n_forces", "force_relax_used", "force_residual", "sample_fx", "sample_fy", "sample_fz"])

prev_forces: Optional[List[List[float]]] = None
aitken_relax = FORCE_RELAX

for step in range(1, NSTEPS + 1):

    print(f"\n--- Step {step} ---")

    # 1) Receive geometry from solid (line-based)
    print("Waiting for geometry from solid...")
    geo_data = read_json_line(solid_file, "Solid")
    print("Geometry received.")
    geometry = geo_data.get("geometry", [])
    if not isinstance(geometry, list) or len(geometry) == 0:
        raise RuntimeError(f"Solid sent invalid geometry payload at step {step}")
    n_span = int(geo_data.get("n_span", 0)) if "n_span" in geo_data else 0
    n_chord = int(geo_data.get("n_chord", 0)) if "n_chord" in geo_data else 0
    if n_span > 0 and n_chord > 0:
        expected = n_span * n_chord
        if len(geometry) != expected:
            raise RuntimeError(
                f"Solid geometry length mismatch at step {step}: "
                f"received {len(geometry)}, expected {expected} ({n_span}x{n_chord})"
            )
        rot = geo_data.get("rotation", [])
        if isinstance(rot, list) and len(rot) not in (0, expected):
            raise RuntimeError(
                f"Solid rotation length mismatch at step {step}: "
                f"received {len(rot)}, expected {expected} ({n_span}x{n_chord})"
            )
        idxing = str(geo_data.get("indexing", "span-major"))
        if idxing not in ("span-major", "chord-major"):
            raise RuntimeError(f"Unsupported indexing '{idxing}' at step {step}")

    # 2) Send geometry to fluid
    print("Sending geometry to fluid...")
    fluid_conn.sendall((json.dumps(geo_data) + "\n").encode())

    # 3) Receive forces from fluid
    print("Waiting for forces from fluid...")
    force_data = read_json_line(fluid_file, "Fluid")
    print("Forces received.")
    forces = force_data.get("force", [])
    if not isinstance(forces, list):
        raise RuntimeError(f"Fluid sent invalid force payload at step {step}")
    if len(forces) == 0:
        raise RuntimeError(f"Fluid sent empty force payload at step {step}")
    if n_span > 0 and n_chord > 0:
        expected = n_span * n_chord
        if len(forces) != expected:
            raise RuntimeError(
                f"Fluid force length mismatch at step {step}: "
                f"received {len(forces)}, expected {expected} ({n_span}x{n_chord})"
            )

    # Optional explicit coupling relaxation for stability.
    # This does not turn the scheme into strongly coupled GS sub-iterations,
    # but helps damp loose-coupling oscillations.
    force_residual = 0.0
    relax_used = FORCE_RELAX
    if prev_forces is not None and len(prev_forces) == len(forces):
        num = 0.0
        den = 0.0
        for a, b in zip(forces, prev_forces):
            dx = float(a[0]) - float(b[0])
            dy = float(a[1]) - float(b[1])
            dz = float(a[2]) - float(b[2])
            num += dx * dx + dy * dy + dz * dz
            den += float(a[0]) ** 2 + float(a[1]) ** 2 + float(a[2]) ** 2
        force_residual = (num ** 0.5) / max(den ** 0.5, 1.0e-16)

        if USE_AITKEN:
            # Simple bounded Aitken-like update based on residual trend.
            if force_residual > 1.0e-2:
                aitken_relax = max(0.25, 0.9 * aitken_relax)
            else:
                aitken_relax = min(1.0, 1.05 * aitken_relax)
            relax_used = max(0.2, min(1.0, aitken_relax))

        if relax_used < 0.999:
            relaxed = []
            for f_new, f_old in zip(forces, prev_forces):
                relaxed.append([
                    relax_used * float(f_new[0]) + (1.0 - relax_used) * float(f_old[0]),
                    relax_used * float(f_new[1]) + (1.0 - relax_used) * float(f_old[1]),
                    relax_used * float(f_new[2]) + (1.0 - relax_used) * float(f_old[2]),
                ])
            force_data["force"] = relaxed
            forces = relaxed
    prev_forces = [list(map(float, f[:3])) for f in forces]

    # DEBUG PRINT
    print(f"Sample force[0] = {forces[0]} (relax={relax_used:.3f}, residual={force_residual:.3e})")

    writer.writerow([
        step,
        len(forces),
        f"{relax_used:.6f}",
        f"{force_residual:.6e}",
        f"{float(forces[0][0]):.6e}",
        f"{float(forces[0][1]):.6e}",
        f"{float(forces[0][2]):.6e}",
    ])

    # 4) Send forces to solid
    print("Sending forces to solid...")
    solid_conn.sendall((json.dumps(force_data) + "\n").encode())

print("Coupling finished.")
print(f"Coupling diagnostics saved at: {log_csv}")
log_fp.close()

fluid_conn.close()
solid_conn.close()
server.close()
