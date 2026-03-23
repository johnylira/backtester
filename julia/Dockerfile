FROM julia:1.12.5 AS base

WORKDIR /app

ENV JULIA_PROJECT=/app
ENV JULIA_NUM_THREADS=auto
ENV HOST=0.0.0.0
ENV PORT=8080

COPY Project.toml Manifest.toml* bootstrap.jl /app/
COPY src /app/src
RUN julia --project=/app /app/bootstrap.jl

FROM base AS dev
COPY dev.jl prod.jl /app/
EXPOSE 8080
CMD ["julia", "--project=/app", "--threads=auto", "/app/dev.jl"]

# FROM base AS prod
# COPY prod.jl /app/
# COPY src /app/src
# EXPOSE 8080
# CMD ["julia", "--project=/app", "--threads=auto", "/app/prod.jl"]
