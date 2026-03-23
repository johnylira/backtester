using Pkg
Pkg.activate(@__DIR__)

using Revise
using XGBoostRecencyAPI

host = get(ENV, "HOST", "0.0.0.0")
port = parse(Int, get(ENV, "PORT", "8080"))

XGBoostRecencyAPI.serve(host=host, port=port, revise=:eager)
