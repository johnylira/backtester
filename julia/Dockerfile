FROM julia:1.12.5

WORKDIR /app

# Otimização: Instalação de dependências de sistema para o XGBoost (se necessário no Alpine)
# RUN apk add --no-cache libstdc++ gcompat

RUN julia -e 'using Pkg; Pkg.add(["Oxygen","HTTP","JSON3","DataFrames","XGBoost"]); Pkg.precompile()'

COPY app.jl /app/app.jl

ENV HOST=0.0.0.0
ENV PORT=8080
# Opção A: Definir via variável de ambiente (afeta todo o runtime)
# ENV JULIA_NUM_THREADS=auto

EXPOSE 8080

# Opção B: Passar explicitamente no CMD (mais prioritário que a ENV)
CMD ["julia", "--project=@", "--threads=auto", "/app/app.jl"]