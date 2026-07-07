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

println("Connecting to coupling server...")
sock = connect(get(ENV, "COUPLING_HOST", "127.0.0.1"), parse(Int, get(ENV, "COUPLING_PORT", "9000")))
write(sock, JSON.json(Dict("role" => "fluid")) * "\n")
flush(sock)
println("Fluid connected.")