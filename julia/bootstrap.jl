using Pkg

Pkg.activate(@__DIR__)

deps = [
    "Oxygen",
    "HTTP",
    "JSON3",
    "DataFrames",
    "CSV",
    "SwaggerMarkdown",
    "XGBoost",
    "Revise",
]

proj = Pkg.project()
installed = Set(keys(proj.dependencies))
missing = filter(dep -> !(dep in installed), deps)

if !isempty(missing)
    Pkg.add(missing)
end

Pkg.resolve()
Pkg.instantiate()
Pkg.precompile()
