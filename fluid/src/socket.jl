function read_json_line(sock::TCPSocket, tag::String)
    line = try
        readline(sock)
    catch err
        if err isa EOFError
            error("$tag: coupling socket closed")
        end
        rethrow(err)
    end
    s = String(line)
    isempty(strip(s)) && error("$tag: received empty line from coupling")
    return JSON.parse(s)
end

function connect_to_server(host::String, port::Int)
    println("Connecting to coupling server at $(host):$(port)...")
    sock = connect(host, port)
    write(sock, JSON.json(Dict("role" => "fluid")) * "\n")
    flush(sock)
    println("Fluid connected.")
    return sock
end
