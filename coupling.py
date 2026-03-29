import socket
import json
import os
import errno
import subprocess

HOST = os.getenv("COUPLING_HOST", "127.0.0.1")
PORT = int(os.getenv("COUPLING_PORT", "9000"))

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

nsteps = 400

for step in range(1, nsteps+1):

    print(f"\n--- Step {step} ---")

    # 1) Receive geometry from solid (line-based)
    print("Waiting for geometry from solid...")
    geo_data = read_json_line(solid_file, "Solid")
    print("Geometry received.")
    geometry = geo_data.get("geometry", [])
    if not isinstance(geometry, list) or len(geometry) == 0:
        raise RuntimeError(f"Solid sent invalid geometry payload at step {step}")

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

    # 🔎 DEBUG PRINT
    print("Sample force[0] =", forces[0])

    # 4) Send forces to solid
    print("Sending forces to solid...")
    solid_conn.sendall((json.dumps(force_data) + "\n").encode())

print("Coupling finished.")

fluid_conn.close()
solid_conn.close()
server.close()
