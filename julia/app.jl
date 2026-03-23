using Oxygen
using HTTP
using JSON3
using DataFrames
using Dates
using Statistics
using XGBoost
using CSV
using SwaggerMarkdown

const DEFAULTS = Dict(
    "test_fraction" => 0.2,
    "train_window_days" => 252,
    "min_train_days" => 126,
    "recency_half_life_days" => 63.0,
    "top_k" => 3,
    "long_only_positive" => true,
    "num_round" => 150,
    "eta" => 0.05,
    "max_depth" => 4,
    "subsample" => 0.9,
    "colsample_bytree" => 0.9
)

@info "Servidor iniciado" Threads.nthreads()

function parse_ptbr_number(s::AbstractString)
    t = strip(String(s))
    t = replace(t, "." => "")
    t = replace(t, "," => ".")
    return parse(Float64, t)
end

function parse_ptbr_percent(s::AbstractString)
    t = strip(String(s))
    t = replace(t, "%" => "")
    return parse_ptbr_number(t) / 100.0
end

function parse_ptbr_volume(s::AbstractString)
    t = strip(String(s))
    mult = 1.0
    if endswith(t, "K")
        mult = 1_000.0
        t = chop(t)
    elseif endswith(t, "M")
        mult = 1_000_000.0
        t = chop(t)
    elseif endswith(t, "B")
        mult = 1_000_000_000.0
        t = chop(t)
    end
    return parse_ptbr_number(t) * mult
end

function asset_from_filename(filename::AbstractString)
    base = split(strip(filename), " - ")[1]
    return strip(base)
end

function parse_investing_csv_text(filename::AbstractString, content::AbstractString)
    asset = asset_from_filename(filename)

    io = IOBuffer(content)
    raw = CSV.read(
        io,
        DataFrame;
        delim = ',',
        quotechar = '"',
        ignorerepeated = false,
        normalizenames = false,
        stringtype = String
    )

    rename!(raw, Dict(
        "Data" => :date_str,
        "Último" => :last_str,
        "Abertura" => :open_str,
        "Máxima" => :high_str,
        "Mínima" => :low_str,
        "Vol." => :vol_str,
        "Var%" => :var_str
    ))

    raw[!, :datetime] = DateTime.(Date.(raw.date_str, dateformat"d.m.y"))
    raw[!, :asset] = fill(asset, nrow(raw))
    raw[!, :asset_open] = parse_ptbr_number.(raw.open_str)
    raw[!, :asset_high] = parse_ptbr_number.(raw.high_str)
    raw[!, :asset_low] = parse_ptbr_number.(raw.low_str)
    raw[!, :asset_close] = parse_ptbr_number.(raw.last_str)
    raw[!, :asset_volume] = parse_ptbr_volume.(raw.vol_str)
    raw[!, :asset_var_pct] = parse_ptbr_percent.(raw.var_str)

    sort!(raw, :datetime)
    return select(raw, :datetime, :asset, :asset_open, :asset_high, :asset_low, :asset_close, :asset_volume, :asset_var_pct)
end

function build_long_from_investing_payload(payload)
    files = payload["files"]
    benchmark_asset = String(payload["benchmark_asset"])

    parts = DataFrame[]
    for f in files
        filename = String(f["filename"])
        content = String(f["content"])
        push!(parts, parse_investing_csv_text(filename, content))
    end

    all_df = vcat(parts...)
    sort!(all_df, [:asset, :datetime])

    bench = all_df[all_df.asset .== benchmark_asset, [:datetime, :asset_open, :asset_high, :asset_low, :asset_close]]
    rename!(bench, Dict(
        :asset_open => :bench_open,
        :asset_high => :bench_high,
        :asset_low => :bench_low,
        :asset_close => :bench_close
    ))

    panel = leftjoin(all_df, bench, on=:datetime)

    if any(ismissing.(panel.bench_open)) || any(ismissing.(panel.bench_high)) || any(ismissing.(panel.bench_low)) || any(ismissing.(panel.bench_close))
        error("Benchmark asset $(benchmark_asset) não cobre todas as datas necessárias.")
    end

    return select(panel,
        :datetime,
        :asset,
        :asset_open,
        :asset_high,
        :asset_low,
        :asset_close,
        :bench_open,
        :bench_high,
        :bench_low,
        :bench_close
    )
end

function run_pipeline_from_investing_csv(payload)
    params = haskey(payload, "params") ? payload["params"] : Dict{String, Any}()
    df = build_long_from_investing_payload(payload)
    panel = make_features(df)

    feature_cols = [
        :asset_id,
        :ret1,
        :mom3,
        :mom5,
        :mom10,
        :vol5,
        :vol10,
        :range1,
        :body1,
        :close_to_sma5,
        :close_to_sma10,
        :bench_ret1,
        :bench_mom3,
        :bench_mom5,
        :bench_range1,
        :bench_body1,
        :rel_ret1,
        :rel_mom5,
        :mom5_rank,
        :vol10_rank,
        :relret1_rank
    ]

    bt = forward_backtest(panel, feature_cols, params)
    rank = next_day_ranking(panel, feature_cols, params)

    return Dict(
        "status" => "ok",
        "input_rows" => nrow(df),
        "assets" => sort(unique(df.asset)),
        "benchmark_asset" => String(payload["benchmark_asset"]),
        "feature_columns" => string.(feature_cols),
        "backtest" => bt,
        "next_day_ranking" => rank
    )
end

safe_float(x) = x === nothing ? NaN : Float64(x)

function getparam(params, key, default)
    try
        return haskey(params, key) ? params[key] : default
    catch
        return default
    end
end

function lagvec(v::Vector{Float64}, k::Int)
    n = length(v)
    out = fill(NaN, n)
    if n > k
        out[(k+1):end] = v[1:(end-k)]
    end
    out
end

function rollmean(v::Vector{Float64}, w::Int)
    n = length(v)
    out = fill(NaN, n)
    if w <= 0
        return out
    end
    s = 0.0
    for i in 1:n
        s += v[i]
        if i > w
            s -= v[i-w]
        end
        if i >= w
            out[i] = s / w
        end
    end
    out
end

function rollstd(v::Vector{Float64}, w::Int)
    n = length(v)
    out = fill(NaN, n)
    if w <= 1
        return out
    end
    for i in w:n
        out[i] = std(@view v[(i-w+1):i]; corrected=false)
    end
    out
end

function normalized_rank(v::Vector{Float64})
    n = length(v)
    out = similar(v)
    idx = sortperm(v)
    if n == 1
        out[1] = 0.5
        return out
    end
    for (r, i) in enumerate(idx)
        out[i] = (r - 1) / (n - 1)
    end
    out
end

function canonical_df(payload)
    schema = payload["schema"]
    rows = payload["rows"]

    dt_col = String(schema["datetime"])
    asset_col = String(schema["asset"])
    ao_col = String(schema["asset_open"])
    ah_col = String(schema["asset_high"])
    al_col = String(schema["asset_low"])
    ac_col = String(schema["asset_close"])
    bo_col = String(schema["bench_open"])
    bh_col = String(schema["bench_high"])
    bl_col = String(schema["bench_low"])
    bc_col = String(schema["bench_close"])

    dt = DateTime[]
    asset = String[]
    ao = Float64[]
    ah = Float64[]
    al = Float64[]
    ac = Float64[]
    bo = Float64[]
    bh = Float64[]
    bl = Float64[]
    bc = Float64[]

    for row in rows
        push!(dt, DateTime(String(row[dt_col])))
        push!(asset, String(row[asset_col]))
        push!(ao, safe_float(row[ao_col]))
        push!(ah, safe_float(row[ah_col]))
        push!(al, safe_float(row[al_col]))
        push!(ac, safe_float(row[ac_col]))
        push!(bo, safe_float(row[bo_col]))
        push!(bh, safe_float(row[bh_col]))
        push!(bl, safe_float(row[bl_col]))
        push!(bc, safe_float(row[bc_col]))
    end

    df = DataFrame(
        datetime = dt,
        asset = asset,
        asset_open = ao,
        asset_high = ah,
        asset_low = al,
        asset_close = ac,
        bench_open = bo,
        bench_high = bh,
        bench_low = bl,
        bench_close = bc
    )

    sort!(df, [:asset, :datetime])
    return df
end

function make_features(df::DataFrame)
    parts = DataFrame[]
    asset_ids = Dict{String, Int}()

    for sdf in groupby(df, :asset)
        s = DataFrame(sdf)
        asset_name = String(s.asset[1])
        asset_ids[asset_name] = get(asset_ids, asset_name, length(asset_ids) + 1)

        c = Vector{Float64}(s.asset_close)
        o = Vector{Float64}(s.asset_open)
        h = Vector{Float64}(s.asset_high)
        l = Vector{Float64}(s.asset_low)
        bc = Vector{Float64}(s.bench_close)
        bo = Vector{Float64}(s.bench_open)
        bh = Vector{Float64}(s.bench_high)
        bl = Vector{Float64}(s.bench_low)

        logc = log.(c)
        blogc = log.(bc)

        ret1 = fill(NaN, length(c))
        bench_ret1 = fill(NaN, length(c))
        if length(c) >= 2
            ret1[2:end] = log.(c[2:end] ./ c[1:end-1])
            bench_ret1[2:end] = log.(bc[2:end] ./ bc[1:end-1])
        end

        target = fill(NaN, length(c))
        bench_target = fill(NaN, length(c))
        if length(c) >= 2
            target[1:end-1] = log.(c[2:end] ./ c[1:end-1])
            bench_target[1:end-1] = log.(bc[2:end] ./ bc[1:end-1])
        end

        sma5 = rollmean(c, 5)
        sma10 = rollmean(c, 10)
        vol5 = rollstd(ret1, 5)
        vol10 = rollstd(ret1, 10)

        mom3 = logc .- lagvec(logc, 3)
        mom5 = logc .- lagvec(logc, 5)
        mom10 = logc .- lagvec(logc, 10)

        bench_mom3 = blogc .- lagvec(blogc, 3)
        bench_mom5 = blogc .- lagvec(blogc, 5)

        range1 = log.(h ./ l)
        body1 = log.(c ./ o)
        bench_range1 = log.(bh ./ bl)
        bench_body1 = log.(bc ./ bo)

        feat = DataFrame(
            datetime = s.datetime,
            asset = s.asset,
            asset_id = fill(asset_ids[asset_name], nrow(s)),
            asset_close = c,
            bench_close = bc,
            ret1 = ret1,
            mom3 = mom3,
            mom5 = mom5,
            mom10 = mom10,
            vol5 = vol5,
            vol10 = vol10,
            range1 = range1,
            body1 = body1,
            close_to_sma5 = (c ./ sma5) .- 1.0,
            close_to_sma10 = (c ./ sma10) .- 1.0,
            bench_ret1 = bench_ret1,
            bench_mom3 = bench_mom3,
            bench_mom5 = bench_mom5,
            bench_range1 = bench_range1,
            bench_body1 = bench_body1,
            rel_ret1 = ret1 .- bench_ret1,
            rel_mom5 = mom5 .- bench_mom5,
            target = target,
            bench_target = bench_target
        )

        push!(parts, feat)
    end

    panel = vcat(parts...)
    sort!(panel, [:datetime, :asset])

    panel[!, :mom5_rank] = similar(panel.mom5)
    panel[!, :vol10_rank] = similar(panel.vol10)
    panel[!, :relret1_rank] = similar(panel.rel_ret1)

    for sdf in groupby(panel, :datetime)
        idx = parentindices(sdf)[1]
        panel[idx, :mom5_rank] = normalized_rank(Vector{Float64}(sdf.mom5))
        panel[idx, :vol10_rank] = normalized_rank(Vector{Float64}(sdf.vol10))
        panel[idx, :relret1_rank] = normalized_rank(Vector{Float64}(sdf.rel_ret1))
    end

    return panel
end

function finite_mask(df::DataFrame, cols::Vector{Symbol}; need_target::Bool=true)
    mask = trues(nrow(df))
    usecols = need_target ? vcat(cols, [:target, :bench_target]) : cols
    for c in usecols
        mask .&= map(x -> x isa Real && isfinite(Float64(x)), df[!, c])
    end
    return mask
end

function fit_xgb(train_df::DataFrame, feature_cols::Vector{Symbol}, params, reference_date::DateTime)
    X = Matrix{Float32}(select(train_df, feature_cols))
    y = Float32.(train_df.target)

    half_life = Float64(getparam(params, "recency_half_life_days", DEFAULTS["recency_half_life_days"]))
    ages = Float32.(Dates.value.(Date(reference_date) .- Date.(train_df.datetime)))
    weights = Float32.(exp.(-log(2.0) .* ages ./ half_life))

    dtrain = DMatrix(X, label=y, weight=weights)

    booster = xgboost(
        dtrain;
        num_round = Int(getparam(params, "num_round", DEFAULTS["num_round"])),
        objective = "reg:squarederror",
        eta = Float64(getparam(params, "eta", DEFAULTS["eta"])),
        max_depth = Int(getparam(params, "max_depth", DEFAULTS["max_depth"])),
        subsample = Float64(getparam(params, "subsample", DEFAULTS["subsample"])),
        colsample_bytree = Float64(getparam(params, "colsample_bytree", DEFAULTS["colsample_bytree"])),
        tree_method = "hist"
    )

    return booster
end

function predict_xgb(model, df::DataFrame, feature_cols::Vector{Symbol})
    X = Matrix{Float32}(select(df, feature_cols))
    d = DMatrix(X)
    return Vector{Float64}(XGBoost.predict(model, d))
end

function max_drawdown_from_logrets(logrets::Vector{Float64})
    if isempty(logrets)
        return 0.0
    end
    equity = exp.(cumsum(logrets))
    peak = similar(equity)
    peak[1] = equity[1]
    for i in 2:length(equity)
        peak[i] = max(peak[i-1], equity[i])
    end
    dd = equity ./ peak .- 1.0
    return minimum(dd)
end

function forward_backtest(panel::DataFrame, feature_cols::Vector{Symbol}, params)
    all_dates = sort(unique(panel.datetime))
    labeled = panel[finite_mask(panel, feature_cols; need_target=true), :]
    sample_dates = sort(unique(labeled.datetime))

    test_fraction = Float64(getparam(params, "test_fraction", DEFAULTS["test_fraction"]))
    train_window_days = Int(getparam(params, "train_window_days", DEFAULTS["train_window_days"]))
    min_train_days = Int(getparam(params, "min_train_days", DEFAULTS["min_train_days"]))
    top_k = Int(getparam(params, "top_k", DEFAULTS["top_k"]))
    long_only_positive = Bool(getparam(params, "long_only_positive", DEFAULTS["long_only_positive"]))

    n_test = max(1, floor(Int, length(sample_dates) * test_fraction))
    test_dates = sample_dates[(end - n_test + 1):end]

    series = Vector{Dict{String, Any}}()
    strategy_logrets = Float64[]
    benchmark_logrets = Float64[]

    for d in test_dates
        train_cut = d - Day(1)
        min_dt = train_cut - Day(train_window_days)

        train_df = labeled[(labeled.datetime .< d) .& (labeled.datetime .>= min_dt), :]
        if length(unique(Date.(train_df.datetime))) < min_train_days
            continue
        end

        score_df = panel[(panel.datetime .== d) .& finite_mask(panel[(panel.datetime .== d), :], feature_cols; need_target=false), :]
        if nrow(score_df) == 0
            continue
        end

        model = fit_xgb(train_df, feature_cols, params, maximum(train_df.datetime))
        preds = predict_xgb(model, score_df, feature_cols)

        score = DataFrame(score_df)
        score[!, :prediction] = preds
        sort!(score, :prediction, rev=true)

        picks = long_only_positive ? score[score.prediction .> 0.0, :] : score
        picks = first(picks, min(top_k, nrow(picks)))

        strat_lr = nrow(picks) > 0 ? mean(Vector{Float64}(picks.target)) : 0.0
        bench_lr = Float64(score.bench_target[1])

        push!(strategy_logrets, strat_lr)
        push!(benchmark_logrets, bench_lr)

        strat_cum = exp(sum(strategy_logrets)) - 1.0
        bench_cum = exp(sum(benchmark_logrets)) - 1.0

        push!(series, Dict(
            "feature_date" => string(d),
            "prediction_date" => string(Date(d) + Day(1)),
            "strategy_log_return" => strat_lr,
            "benchmark_log_return" => bench_lr,
            "strategy_cumulative_return" => strat_cum,
            "benchmark_cumulative_return" => bench_cum,
            "selected_assets" => Vector{String}(picks.asset),
            "avg_prediction" => nrow(picks) > 0 ? mean(Vector{Float64}(picks.prediction)) : 0.0
        ))
    end

    mdd = max_drawdown_from_logrets(strategy_logrets)
    last_cum = isempty(strategy_logrets) ? 0.0 : exp(sum(strategy_logrets)) - 1.0
    ret_dd = mdd == 0.0 ? Inf : last_cum / abs(mdd)

    return Dict(
        "series" => series,
        "metrics" => Dict(
            "test_points" => length(series),
            "max_drawdown" => mdd,
            "last_cumulative_return_test" => last_cum,
            "return_over_drawdown" => ret_dd
        )
    )
end

function next_day_ranking(panel::DataFrame, feature_cols::Vector{Symbol}, params)
    labeled = panel[finite_mask(panel, feature_cols; need_target=true), :]
    latest_feature_df = panel[finite_mask(panel, feature_cols; need_target=false), :]
    latest_dt = maximum(latest_feature_df.datetime)

    score_df = latest_feature_df[latest_feature_df.datetime .== latest_dt, :]
    model = fit_xgb(labeled, feature_cols, params, maximum(labeled.datetime))
    preds = predict_xgb(model, score_df, feature_cols)

    out = DataFrame(
        asset = score_df.asset,
        prediction = preds
    )
    sort!(out, :prediction, rev=true)

    return Dict(
        "feature_date" => string(latest_dt),
        "prediction_date" => string(Date(latest_dt) + Day(1)),
        "ranking" => [
            Dict(
                "rank" => i,
                "asset" => String(out.asset[i]),
                "predicted_next_day_log_return" => Float64(out.prediction[i])
            ) for i in 1:nrow(out)
        ]
    )
end

function run_pipeline(payload)
    params = haskey(payload, "params") ? payload["params"] : Dict{String, Any}()

    df = canonical_df(payload)
    panel = make_features(df)

    feature_cols = [
        :asset_id,
        :ret1,
        :mom3,
        :mom5,
        :mom10,
        :vol5,
        :vol10,
        :range1,
        :body1,
        :close_to_sma5,
        :close_to_sma10,
        :bench_ret1,
        :bench_mom3,
        :bench_mom5,
        :bench_range1,
        :bench_body1,
        :rel_ret1,
        :rel_mom5,
        :mom5_rank,
        :vol10_rank,
        :relret1_rank
    ]

    bt = forward_backtest(panel, feature_cols, params)
    rank = next_day_ranking(panel, feature_cols, params)

    return Dict(
        "status" => "ok",
        "input_rows" => nrow(df),
        "assets" => sort(unique(df.asset)),
        "feature_columns" => string.(feature_cols),
        "backtest" => bt,
        "next_day_ranking" => rank
    )
end

@get "/health" function()
    return Dict("status" => "ok", "service" => "xgboost-recency-api")
end

@post "/v1/backtest" function(req::HTTP.Request)
    try
        payload = JSON3.read(String(req.body))
        result = run_pipeline(payload)
        return result
    catch e
        return HTTP.Response(
            400,
            ["Content-Type" => "application/json"],
            body = JSON3.write(Dict(
                "status" => "error",
                "message" => sprint(showerror, e)
            ))
        )
    end
end

@swagger """
/v1/backtest/investing-csv:
  post:
    summary: Backtest multiativo a partir de CSVs no formato Investing
    description: |
      Recebe uma lista de arquivos CSV em texto bruto. O ativo é inferido do nome do arquivo,
      por exemplo: "PETR4 - Historico.csv". Um dos ativos deve ser indicado como benchmark.
    requestBody:
      required: true
      content:
        application/json:
          example:
            benchmark_asset: "IBOV"
            params:
              test_fraction: 0.2
              train_window_days: 252
              min_train_days: 126
              recency_half_life_days: 63.0
              top_k: 3
              long_only_positive: true
              num_round: 150
              eta: 0.05
              max_depth: 4
              subsample: 0.9
              colsample_bytree: 0.9
            files:
              - filename: "PETR4 - Historico.csv"
                content: |
                  "Data","Último","Abertura","Máxima","Mínima","Vol.","Var%"
                  "03.10.2025","106,28","106,47","106,48","105,85","23,62K","0,10%"
                  "02.10.2025","106,17","107,10","107,56","105,72","55,59K","-0,61%"
                  "01.10.2025","106,82","107,99","108,48","106,62","31,85K","-0,62%"
                  "30.09.2025","107,49","107,51","108,17","107,00","80,15K","0,43%"
                  "29.09.2025","107,03","108,00","108,00","107,03","28,57K","0,25%"
              - filename: "VALE3 - Historico.csv"
                content: |
                  "Data","Último","Abertura","Máxima","Mínima","Vol.","Var%"
                  "03.10.2025","62,11","62,00","62,35","61,80","18,20M","0,40%"
                  "02.10.2025","61,86","62,40","62,70","61,55","21,05M","-0,77%"
                  "01.10.2025","62,34","63,10","63,20","62,10","17,44M","-0,52%"
                  "30.09.2025","62,67","62,80","63,01","62,22","19,80M","0,18%"
                  "29.09.2025","62,56","62,20","62,70","61,95","15,33M","0,61%"
              - filename: "IBOV - Historico.csv"
                content: |
                  "Data","Último","Abertura","Máxima","Mínima","Vol.","Var%"
                  "03.10.2025","135.120,32","134.900,00","135.300,00","134.500,00","1,25B","0,30%"
                  "02.10.2025","134.716,10","135.400,00","135.800,00","134.200,00","1,18B","-0,41%"
                  "01.10.2025","135.270,80","136.000,00","136.220,00","135.010,00","1,05B","-0,25%"
                  "30.09.2025","135.609,90","135.100,00","135.900,00","134.980,00","1,11B","0,44%"
                  "29.09.2025","135.016,00","134.700,00","135.200,00","134.410,00","980,00M","0,20%"
    responses:
      '200':
        description: Resultado do backtest e ranking do próximo dia
"""
@post "/v1/backtest/investing-csv" function(req::HTTP.Request)
    try
        payload = JSON3.read(String(req.body))
        result = run_pipeline_from_investing_csv(payload)
        return result
    catch e
        return HTTP.Response(
            400,
            ["Content-Type" => "application/json"],
            body = JSON3.write(Dict(
                "status" => "error",
                "message" => sprint(showerror, e)
            ))
        )
    end
end


info = Dict(
    "title" => "XGBoost Recency API",
    "version" => "1.1.0"
)

openApi = OpenAPI("3.0.0", info)
swagger_document = build(openApi)
mergeschema(swagger_document)

host = get(ENV, "HOST", "0.0.0.0")
port = parse(Int, get(ENV, "PORT", "8080"))

serve(host=host, port=port)

