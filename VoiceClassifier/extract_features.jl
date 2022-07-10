# ENV["PYTHON"] = "C:\\Users\\delbe\\AppData\\Local\\Programs\\Python\\Python310\\python.EXE"
# Pkg.build("PyCall")

# ---------------------------------------------------------------------------- #
#                                    Imports                                   #
# ---------------------------------------------------------------------------- #

using Pkg 
Pkg.activate(".")

using PyCall
using Parquet, Feather, DataFrames, CSV
using Statistics
using Flux: onehot

librosa = pyimport("librosa")

librosa_feat = pyimport("librosa.feature")
keys(librosa_feat)

librosa_effects = pyimport("librosa.effects")
keys(librosa_effects)

audiofile = pyimport("audiofile")
keys(audiofile)

# ---------------------------------------------------------------------------- #
#                                   Load data                                  #
# ---------------------------------------------------------------------------- #

PREFIX = "ichinose_tamaki_taidan"

fname = "./data/" * PREFIX * "_merge-subs_processed.parquet"
if isfile(fname)
    df_subs = Parquet.read_parquet(fname) |> DataFrame 
    df_subs = select(df_subs, 1:5)
end 

fname = "./data/$PREFIX.m4a"
if isfile(fname)
    audio, rate = audiofile.read(fname)
end 
size(audio)

function nz_frac(arr::Array{Float32, 2}):: Nothing 
    N = size(arr, 2)
    @inbounds for i = 1:size(arr, 1)
        z = count(arr[1,:] .== 0.)
        zfrac = round(z/(N-z), digits=5)
        zperc = round(100*z/N, digits=3)
        @info "Channel $i has $z zeroes\n z/nz = $zfrac; z/n = $zperc %"
        if zperc < 50 || zfrac < 0.8
            @info "Don't use sparse arrays"
        end 
    end 
    nothing
end 

nz_frac(audio)

# take mean of channels 
audio = (audio[1,:] + audio[2,:])/2

# truncate audio data with the earliest and last samples
minmax_table = describe(df_subs, :min, :max)[3:4, :]
ti, tf = minmax_table[1,2], minmax_table[2,3]

# we could potentially extract 'silent' portions, but these may also just be sections that were not translated, so let's refrain from that 

signal = audio[ti:tf]

inds = Array{Int32, 2}(sort(df_subs[:, 3:4], :Start_samples)) .- (ti - 1)
n_subs = size(inds, 1)

# ---------------------------------------------------------------------------- #
#                              Feature calculation                             #
# ---------------------------------------------------------------------------- #

"""
Compute librosa features. Each are multidimensional, so we take the mean along each bin, collapsing `(N x T)` arrays into length-`N` vectors. 
"""
function _get_feats3_(y::Vector{Float32}, sr::Int)::Matrix{Float64}
    #   526.301 ms (452 allocations: 7.40 MiB)
    feats = [
        librosa_feat.mfcc(y=y, sr=sr, n_mfcc=40),
        librosa_feat.chroma_stft(
            S=abs.(librosa.stft(y)),
            sr=sr
        ),
        librosa_feat.melspectrogram(y=y, sr=sr),
        librosa_feat.spectral_contrast(y=y, sr=sr),
        librosa_feat.tonnetz(
            y=librosa_effects.harmonic(y), 
            sr=sr
        )
    ] 
    n_cols = size(feats[1], 2)
    feats[2:end] .= [x[:,1:n_cols] for x in feats[2:end]]
    return mean(vcat(feats...), dims=2)
end 

"""
Compute librosa features for each row (subtitle) of the dataframe 
```
    632×193 Matrix{Float64}:
    -336.255  112.007   -24.19    30.1746  …   0.00896775   -0.0102192
```
"""
function featcalc(
    inds::Matrix{Int64}, sgnl::Vector{Float32}, n_subs::Int; sr::Int=44100,
    filename::String=""
)::Array{Float64, 2}
    # 193 features per subtitle
    feats = Array{Float64, 2}(undef, (193, n_subs))
    
    for i = 1:n_subs
        a, b = inds[i,:]
        y = sgnl[a:b]
        @inbounds feats[:, i] .= _get_feats3_(y, sr)
    end 

    if length(filename) > 1
        if isfile(filename)
            @warn "$filename\n is an existing file. Overwrite? [y/n]"
            ow = readline()
            if ow != "y" 
                return feats 
            end 
        end 
        Feather.write(filename, Tables.table(feats))
        @info "Saved features to $filename"
    end 

    return feats 
end 

# 436.379395 seconds (340.98 k allocations: 5.767 GiB, 0.09% gc time, 0.00% compilation time)
feats = featcalc(
    inds, signal, n_subs;
    filename="./data/$PREFIX-features.feather"
)

# ---------------------------------------------------------------------------- #
#                             One hot encode labels                            #
# ---------------------------------------------------------------------------- #

using Random
Random.seed!(1234)

labels = Vector{Int32}(df_subs[:, :Speaker])

function get_splits(n_subs::Int)::Vector{Int32}
    n_train = round(0.7*n_subs) |> Int
    n_valid = round(0.2*n_subs) |> Int 
    n_test = n_subs - n_train - n_valid 
    return [n_train, n_valid, n_test] 
end 

# ensure proportions of speakers are roughly equal in each split 
function check_speaker_fracs(labels::Vector{Int32}, splits::Vector{Int32})
    i = 1
    for j = 1:3 
        labs = labels[i:i+splits[j]]
        @info sum.([labs .== 0, labs .== 1])
        i = splits[j] + 1  
    end 
end 

splits = get_splits(n_subs)
shuffled = collect(shuffle(1:n_subs))

# [ Info: [223, 220]
# [ Info: [72, 55]
# [ Info: [31, 34]
check_speaker_fracs(labels[shuffled], splits)

feats[shuffled, :]
labels[shuffled, :]

# ---------------------------------------------------------------------------- #
#                       Save shuffled features and labels                      #
# ---------------------------------------------------------------------------- #

CSV.write("./data/$PREFIX" * "_splits.txt", Tables.table(splits))

CSV.write(
    "./data/$PREFIX-labels_shuffled.csv",
    Tables.table(labels[shuffled, :])
)

Feather.write(
    "./data/$PREFIX-features_shuffled.feather", 
    Tables.table(feats[shuffled, :])
)