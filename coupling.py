import socket
import json

HOST = "127.0.0.1"
PORT = 9000

print("Starting coupling server...")

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(2)

print("Waiting for fluid...")
fluid_conn, _ = server.accept()
fluid_file = fluid_conn.makefile("r")
print("Fluid connected.")

print("Waiting for solid...")
solid_conn, _ = server.accept()
solid_file = solid_conn.makefile("r")
print("Solid connected.")

nsteps = 200

for step in range(1, nsteps+1):

    print(f"\n--- Step {step} ---")

    # 1) Receive geometry from solid (line-based)
    print("Waiting for geometry from solid...")
    geo_line = solid_file.readline()
    print("Geometry received.")
    geo_data = json.loads(geo_line)
    geometry = geo_data["geometry"]

    # 2) Send geometry to fluid
    print("Sending geometry to fluid...")
    fluid_conn.sendall((json.dumps({"geometry": geometry}) + "\n").encode())

    # 3) Receive forces from fluid
    print("Waiting for forces from fluid...")
    force_line = fluid_file.readline()
    print("Forces received.")
    force_data = json.loads(force_line)
    forces = force_data["force"]

    # 🔎 DEBUG PRINT
    print("Sample force[0] =", forces[0])

    # 4) Send forces to solid
    print("Sending forces to solid...")
    solid_conn.sendall((json.dumps({"force": forces}) + "\n").encode())

print("Coupling finished.")

fluid_conn.close()
solid_conn.close()
server.close()