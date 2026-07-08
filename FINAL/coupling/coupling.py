def main():

    cfg = load_coupling_config()

    server = create_server(cfg)

    solid = connect_solid(server)

    fluid = connect_fluid(server)

    run_coupling(server, solid, fluid, cfg)

    shutdown(server)