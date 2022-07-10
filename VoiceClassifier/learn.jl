# ---------------------------------------------------------------------------- #
#                                    Imports                                   #
# ---------------------------------------------------------------------------- #

using Pkg 
Pkg.activate(".")

using Flux
using Feather, DelimitedFiles, DataFrames
using Statistics

# ---------------------------------------------------------------------------- #
#                                   Load data                                  #
# ---------------------------------------------------------------------------- #

function load_data(prefix::String, datadir::String)::Tuple
    datadir *= prefix 

    splits, _ = readdlm(
        datadir * "_splits.csv", 
        ',', Int, header=true
    )

    features:: DataFrame = Feather.read(
        "$datadir-features_shuffled.feather"
    )
    
    labels, _ = readdlm(
        "$datadir-labels_shuffled.csv", 
        ',', Int, header=true
    )

    X = Vector{Matrix{Float64}}(undef, 3)
    Y = Vector{Vector{Int}}(undef, 3)

    i0 = 1
    for (i, split) in enumerate(splits)
        X[i] = Matrix{Float64}(features[i0:i0+split-1, :])
        Y[i] = labels[i0:i0+split-1]
        i0 += split 
    end 
    return X, Y 
end 

prefix = "ichinose_tamaki_taidan"
X, Y = load_data(prefix, "./data/")

# X_train, X_valid, X_test = X 
# Y_train, Y_valid, Y_test = Y 

# ---------------------------------------------------------------------------- #
#                                  Scale data                                  #
# ---------------------------------------------------------------------------- #


size(μ_X), size(σ_X)

function __get_mu_std__(arr::Array{Float64,2})::Tuple
    mu, sig = vec.([
        mean(arr, dims=1), 
        std(arr, dims=1)]
    )
    return mu', sig'
end 

"""
Scale training data to have zero mean and unit variance, 
and apply same transform to the validation and testing sets.
"""
function StandardScale!(
    features::Vector{Matrix{Float64}})::Vector{Matrix{Float64}}
    
    # mean and standard deviation for each feature in the training set 
    μ_X, σ_X = __get_mu_std__(features[1])
    @info size(μ_X), size(σ_X)

    n_feats = length(μ_X)   # number of features

    for i = 1:length(features)
        features[i] .= (features[i] .- μ_X) ./ σ_X
    end 

    # verify scaling worked 
    μ_X, σ_X = __get_mu_std__(features[1])

    @assert isapprox(mean(μ_X), 0., atol=1e-12, rtol=0)
    @assert isapprox(mean(σ_X), 1., atol=1e-12, rtol=0)

    return features 
end 

X = StandardScale!(X)

# ---------------------------------------------------------------------------- #
#                                  Build model                                 #
# ---------------------------------------------------------------------------- #

"""An example: model that predicts 0.5 for each vector

_d, _l = rand(10, 100), fill(0.5, 2, 100)
_m = Chain(Dense(10, 5, relu), Dense(5, 2), softmax)
_loss(x,y) = Flux.Losses.crossentropy(_m(x), y)

Flux.@epochs 250 Flux.train!(
    _loss,
    Flux.params(_m),
    [(_d, _l)],
    Flux.Optimise.Descent(0.01)
)

# accuracy 
mean(isapprox.(_m(_d), 0.5, atol=0.05, rtol=1e-2))
"""

n_feats = size(X[1], 2)
n_classes = length(unique(Y[1])) - 1

X_T = transpose.(X)
Y_T = transpose.(Y)

model = Flux.Chain(
    Flux.Dense(n_feats, 128), 
    Flux.Dropout(0.1),
    # Flux.Dense(128, 20),
    # Flux.Dropout(0.4),
    # Flux.Dense(20, 1),
    Flux.softmax
)

model(X_T[1])
loss(x, y) = Flux.Losses.crossentropy(model(x), y)
loss(X_T[1], Y_T[1])

function accuracy(x, y; threshold=0.5)
    preds = model(x)
    preds[preds .>= threshold] .= 1 
    preds[preds .< threshold] .= 0
    return mean(preds)
end 

evalcb() = @show(loss(X_T[1], Y_T[1]), accuracy(X_T[2], Y_T[2]))

epochs = 50
opt = Flux.Optimise.Adam()
for epoch = 1:epochs
    gs = Flux.gradient(Flux.params(model)) do 
        l = accuracy(X_T[1], Y_T[1])
    end 
    Flux.update!(opt, Flux.params(model), gs)
    @show loss(X_T[1], Y_T[1]), accuracy(X_T[2], Y_T[2])
end

