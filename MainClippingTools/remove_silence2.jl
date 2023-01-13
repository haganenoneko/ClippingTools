# ---------------------------------------------------------------------------- #
#                               Setup and imports                              #
# ---------------------------------------------------------------------------- #

using Pkg 
Pkg.activate(".")
Pkg.precompile()

using PyCall
using LIBSVM
using StatsBase
using PyPlot
PyPlot.pygui(true)

VIDEO_PATH = "../../../../Videos/subtitles/full_raw/"

# Import FFMPEG functions
include("../MainClippingTools/splice_silence.jl")

# use audiofile for reading `.wav` audio 
if !@isdefined(audiofile)
    const audiofile = PyCall.pyimport("audiofile")
end 

# import functions for audio processing 
function __init__()
    py"""
    import sys 
    from pathlib import Path 
    sys.path.insert(
        0, 
        str(Path.cwd() / "MainClippingTools")
    )
    print(sys.path[0])

    from pyAudioAnalysis_ShortTermFeatures import feature_extraction
    from numpy import convolve
    from scipy.signal import savgol_filter 
    """
end 

# verify that `__init__` works 
try 
    py"feature_extraction"
catch e 
    if isa(e, PyCall.PyError)
        __init__()
    else 
        throw(e)
    end 
end 

# define imported Python functions as `const`ants 
if !@isdefined(feature_extraction)
    const feature_extraction = py"feature_extraction"
    const savgol_filter = py"savgol_filter"
    const convolve = py"convolve"
end 


# ---------------------------------------------------------------------------- #
#                              Audio input/output                              #
# ---------------------------------------------------------------------------- #

"""
Since we can't read the audio from video, we extract the audio from a video, and then detect silence in the audio
"""
function ffmpeg_extract_audio(filename:: String):: Bool 
    outname = "$filename.m4a"
    
    if !isfile(outname)
        cmd = "ffmpeg -hide_banner -i $filename.mp4 -vn -acodec copy $outname"
        run(`powershell $cmd`)
    end 

    if isfile(outname)
        return true 
    else 
        error("Failed to extract audio from $filename.mp4")
    end 
end 

"""
Open the `.m4a` file corresponding to a given video. If unavailable, it is created with `ffmpeg_extract_audio`
"""
function open_video(filepath:: String):: Tuple{Matrix{Float32}, Int64}
    if !isfile(filepath * ".m4a")
        ffmpeg_extract_audio(filepath)
    end 
    signal, sampling_rate = audiofile.read(filepath * ".m4a")
    if size(signal, 2) > size(signal, 1)
        return Matrix{Float32}(signal'), sampling_rate
    else 
        return Matrix{Float32}(signal), sampling_rate 
    end 
end 

"""
Convert stereo signal to mono
"""
function stereo_to_mono(signal:: Array{Float32})::Vector{Float32}
    if ndims(signal) == 2
        if size(signal, 2) == 1
            return vec(signal[:,1])
        elseif size(signal, 2) == 2 
            return vec(mean(signal ./ 2, dims=2))
        end 
    end 
    return vec(signal)
end 

# ---------------------------------------------------------------------------- #
#                    Utility functions for `detect_silence`                    #
# ---------------------------------------------------------------------------- #

function dB_to_AR(dB::Number):: Float64
    return sqrt(10^(dB/10))
end 

function AR_to_dB(AR:: Float64):: Float64
    return log10(AR^2) * 10
end 

get_downsample(signal:: Vector) = round(Int, 3*log10(length(signal)))

"""
Smooth a 1-D signal

# Arguments 

- `signal:: Vector`: 1-D signal  
- `window=11`: window size for smoothing, in seconds. If not an `Int`eger, will try to convert it by calling `round(Int, window)`, but must be greater than one to do so. 

# Returns 

`Vector{Float64}`: smoothed signal 
"""
function smooth_moving_avg(signal:: Vector; window=11)::Vector{Float64}
    
    if window >= 1 
        window = round(Int, window)
    else 
        throw(ArgumentError("window must be >= 1. Gave $window"))
    end 

    if length(signal) < window 
        throw(KeyError("Input vector needs to be bigger than window size, $window"))
    elseif window < 3
        return signal 
    else 
        s = vcat(
            [
                2*signal[1] .- signal[window:-1:1],
                signal,
                2*signal[end] .- signal[end-1:-1:end-window+1]
            ]...
        )

        y = convolve(
            ones(window) ./ window, 
            s,
            mode="same"
        )
        return y[window:end-window]
    end 
end 


"""Return concatenated feature matrix and class labels"""
function features_to_matrix(features:: Vector{Array{Float64, 2}}):: Tuple{Matrix{Float64}, Vector{Int64}}
    # labels = np.array([])
    # feature_matrix = np.array([])
    # for i, f in enumerate(features):
    #     if i == 0:
    #         feature_matrix = f
    #         labels = i * np.ones((len(f), 1))
    #     else:
    #         feature_matrix = np.vstack((feature_matrix, f))
    #         labels = np.append(labels, i * np.ones((len(f), 1)))
    # return feature_matrix, labels
    feature_matrix:: Matrix{Float64} = vcat(features...)

    labels:: Vector{Int64} = (
        collect(0:length(features)-1) .* 
        ones.(Int64, size.(features, 1))
    ) |> Base.Iterators.flatten |> collect 

    return feature_matrix, labels 
end 

"""Plot detected segments of silence over raw audio and SVM probabilities"""
function plot_silent_segments(
    seg_limits:: Vector{Vector{Float64}}, 
    prob_on_set:: Vector,
    signal:: Vector, 
    sampling_rate:: Number,
    st_step:: Float64; 
    downsample:: Int=2
):: Nothing 
    
    try 
        @assert isa(downsample, Int)
        @assert 0 < downsample < length(signal)
    catch 
        throw(ArgumentError("`downsample` must be an Int and between 1-n"))
    end 
    
    t_stop = length(signal) / sampling_rate 
    time_x = collect(0 : (1/sampling_rate) : t_stop)[1:end-1]
    
    if t_stop > 600 
        time_x ./= 60
    end 

    _, axs = PyPlot.subplots(
        nrows=2, ncols=1, 
        figsize=(6, 3), 
        constrained_layout=true
    )

    axs[1].set_ylabel("Raw Signal")
    axs[2].set_ylabel("Probability")
    
    axs[1].set_title("Detected segments of silence")
    axs[2].set_title("SVM Probability")
    
    axs[1].plot(
        time_x[1:downsample:end], signal[1:downsample:end], 
        alpha=0.65, lw=0.5, color="blue"
    )

    x = collect(0 : st_step : (st_step * size(prob_on_set, 1)))
    axs[2].plot(x[1:downsample:end-1], prob_on_set[1:downsample:end], lw=1, alpha=0.75)

    @inbounds for lim in seg_limits
        axs[1].axvline.(lim, color="red", lw=0.7, alpha=0.55)
        axs[2].axvline.(lim, color="red", lw=0.7, alpha=0.55)
    end     
    
    for i = 1:2 
        axs[i].locator_params(axis="both", nbins=5)
    end 
    
    if t_stop > 600 
        axs[2].set_xlabel("Time (min)")
    else 
        axs[2].set_xlabel("Time (s)")
    end 

    PyPlot.show()
    return nothing 
end 

"""Plot intervals without probability"""
function plot_silent_segments(
    intervals:: Vector{Tuple{Float64, Float64}}, 
    signal:: Vector, 
    sampling_rate:: Number;
    downsample:: Int=2
):: Nothing 
    
    try 
        @assert isa(downsample, Int)
        @assert 0 < downsample < length(signal)
    catch 
        throw(ArgumentError("`downsample` must be an Int and between 1-n"))
    end 
    
    t_stop = length(signal) / sampling_rate 
    time_x = collect(0 : (1/sampling_rate) : t_stop)[1:end-1]
    
    # scaling factor from seconds to minutes if > 10 min
    𝓈 = (t_stop > 600) ? 1/60 : 1 
    time_x .*= 𝓈

    _, ax = PyPlot.subplots(
        figsize=(6, 3), 
        constrained_layout=true
    )

    ax.set_ylabel("Raw Signal")
    ax.set_title("Detected segments of silence")
    
    ax.plot(
        time_x[1:downsample:end], 
        signal[1:downsample:end], 
        alpha=0.65, lw=0.5, color="blue"
    )

    @inbounds for lim in intervals
        ax.axvline.(lim .* 𝓈, color="red", lw=0.7, alpha=0.55)
    end     
    
    ax.locator_params(axis="both", nbins=5)
    
    (t_stop > 600) ? 
        ax.set_xlabel("Time (min)") : 
        ax.set_xlabel("Time (s)")

    PyPlot.show()
    return nothing 
end 

# ---------------------------------------------------------------------------- #
#                               Silence detection                              #
# ---------------------------------------------------------------------------- #

@doc """
Detect intervals of silence in an audio sample and return intervals of non-silence.

# Positional Arguments  

- `signal:: Matrix`: audio signal 
- `sampling_rate:: Number`: sampling frequency in Hz 

# Keyword Arguments 

- `st_win=0.02`: window size (in seconds) for computing short-term features 
- `st_step=0.02`: step size (in seconds) for computing short-term features
- `weight:: Float64=0.5`: factor between 0 and 1 that specifies how "strict" the thresholding is.
- `smooth_window:: Float64=0.5`: window size (in seconds) to smooth the SVM probabilistic sequence. Smaller values tend to do better for recordings with more vocal content.
- `energy_threshold:: Float64=0.1`: fraction of lowest- and highest-energy features to use for training the model. 
- `min_segment_duration:: Float64=0.25`: minimum duration required for non-silent segments to be kept.
- `plot:: Bool=false`: whether to plot results.

# Returns 

`Vector{Vector{Float64}}`: `Vector` of `[start, end]` times corresponding to non-silent intervals in the audio.

# Tips 

1. For audio with high vocal content, **decrease** `smooth_window` and **increase** `weight`. 
2. Play around with `smooth_window`, `st_win`, `st_step`, and `weight`.

# References.

This is a Julia implementation of `silence_removal` from `pyAudioAnalysis.audioSegmentation`:

1. [Original code from `pyAudioAnalysis.audioSegementation`](https://github.com/tyiannak/pyAudioAnalysis/blob/master/py@AudioAnalysis/audioSegmentation.py)
2. [Documentation of `pyAudioAnalysis`](https://github.com/tyiannak/pyAudioAnalysis/wiki/5.-Segmentation)
"""
function detect_silence(
    signal:: Vector,
    sampling_rate:: Number;
    st_win:: Float64=0.02,
    st_step:: Float64=0.02,
    weight:: Float64=0.5,
    energy_threshold:: Float64=0.1,
    smooth_window:: Float64=0.5,
    min_segment_duration:: Float64=0.25,
    plot:: Bool=false
):: Vector{Tuple{Float64, Float64}}

    @assert 0 < energy_threshold < 0.5
    @assert 0 <= weight <= 1

    # 1. Feature extraction 
    # st_feats = F x N, where N = no. samples, F = no. features 
    st_feats, _ = feature_extraction(
        signal, 
        sampling_rate,
        st_win * sampling_rate, 
        st_step * sampling_rate
    )

    # 2. Train binary SVM classifier on Low vs High energy frames 
    # Energy short-term sequence (2nd feature)
    st_energy:: Vector{Float64} = vec(st_feats[2, :])
    en = sort(st_energy)
    # number of 10% of the total short-term windows 
    st_windows_fraction = round(
        Int, 
        length(en) * energy_threshold
    )

    # "lower" 10% energy threshold 
    low_threshold = mean(en[1:st_windows_fraction]) + 1e-15
    # "higher" 10% energy threshold
    high_threshold = mean(en[end-st_windows_fraction:end]) + 1e-15

    # get low energy features 
    low_energy = st_feats[:, st_energy .<= low_threshold]
    # get high energy features 
    high_energy = st_feats[:, st_energy .>= high_threshold]

    # assemble low/energy features for SVM training 
    features = Matrix.([low_energy', high_energy'])
    features, labels = features_to_matrix(features)

    scaler = fit(ZScoreTransform, features; dims=1)
    μ = mean(features, dims=1)
    σ = std(features, dims=1)
    features_norm = StatsBase.transform(scaler, features)
    
    # train model with the reduced, normalized features 
    model = LIBSVM.fit!(
        LIBSVM.SVC(
            kernel=LIBSVM.Kernel.Linear,
            probability=true,
            gamma=:auto,
            cost=1.0,
        ), 
        features_norm, 
        labels 
    )

    # 3. Compute onset probability based on the trained SVM 
    # standardize each frame (row) in each column of `st_feats`
    st_feats .= (st_feats .- μ') ./ σ'
    # predict probability that each frame is onset using full features 
    prob_on_set = LIBSVM.predict(model, st_feats')
    # get SVM probability that each frame belongs to the ONSET class 
    prob_on_set = smooth_moving_avg(
        prob_on_set;
        window=smooth_window / st_step
    )

    # 4a. Detect onset frame indices.
    # find probability threshold as a wavg of top + bottom 10%
    prob_on_set_sort = sort(prob_on_set)
    nt = round(Int, size(prob_on_set_sort, 1)/10)
    threshold = mean((1-weight) .* prob_on_set_sort[1:nt]) + 
        mean(weight .* prob_on_set_sort[end-nt:end])
    
    # get indices of frames that satisfy thresholding 
    max_indices = findall(>(threshold), prob_on_set)
    
    index = 1 
    time_clusters:: Vector{Vector{Int64}} = []
    seg_limits:: Vector{Vector{Float64}} = [] 
    
    # 4b. Group frame indices to onset segments 
    while index < length(max_indices)
        # for all detected onset indices 
        cluster = [max_indices[index]]
        if index == length(max_indices)-1
            break 
        end 

        while max_indices[index+1] - cluster[end] <= 2
            push!(cluster, max_indices[index+1])
            index += 1 
            if index == length(max_indices)-1
                break 
            end 
        end 

        index += 1 
        push!(time_clusters, cluster)
        push!(seg_limits, st_step .* cluster[[1, end]])
    end 

    # 5. Post process by removing very small segments 
    seg_limits_post:: Vector{Vector{Float64}} = seg_limits[
        findall(
            >(min_segment_duration),
            map(x -> x[2] - x[1], seg_limits)
        )
    ]

    if plot 
        plot_silent_segments(
            seg_limits_post, 
            prob_on_set, 
            signal,
            sampling_rate,
            st_step;
            downsample=get_downsample(signal)
        )
    end 

    # summary 
    tot = (length(signal)/sampling_rate)
    total = tot - sum(
        map(
            x -> x[2] - x[1], 
            seg_limits_post
        )
    )
    
    @info "Total time removed: $(round(total/60, digits=2)) min ($(round(total, digits=2)) sec) ($(tot/total)%)"
    @info "Number of streams: $(length(seg_limits_post))"

    return Tuple.(seg_limits_post)
end 

"""
Detection algorithm purely based on audio volume. 
"""
function detect_silence(
    signal:: Vector,
    sampling_rate:: Number;
    silence_threshold:: Float64=0.01,
    min_segment_duration:: Float64=0.25,
    min_silence_duration:: Float64=0.8,
    savgol_window_size:: Int64=121,
    savgol_poly_order:: Int64=2,
    plot:: Bool=false
):: Vector{Tuple{Float64, Float64}}
    
    # lens[i] = length of i-th run with i-th value (_[i])
    # we ignore the values here because they are all boolean (0 or 1)
    _, lens = (
        savgol_filter(
            abs.(signal), 
            savgol_window_size,
            savgol_poly_order
        ) .<= silence_threshold
    ) |> StatsBase.rle 
    
    # flatten lens 
    flat = ((0,), cumsum(lens)) |> Iterators.flatten 

    # get tuples of (start, end) times for intervals 
    chunks:: Vector{Int64} = [
        (a+1, b) for (a,b) in partition(flat, 2, 1)
        if (b-a) >= min_silence_duration*sampling_rate
    ] |> Iters2Vec

    # ensure that `chunks`` contains first and last indices 
    if chunks[1] == 1
        chunks = chunks[2:end]
    else 
        insert!(chunks, 1, 1)
    end 

    if chunks[end] == length(signal)
        chunks = chunks[1:end-1]
    else 
        push!(chunks, length(signal))
    end 

    # group into 2-tuples 
    intervals:: Vector{Tuple{Float64, Float64}} = 
        partition(
            chunks ./ sampling_rate, 2
        ) |> collect 

    # select intervals with suprathreshold duration
    mask = map(
        x -> (x[2] - x[1]) >= min_segment_duration, 
        intervals
    )

    if plot
        plot_silent_segments(
            intervals,
            signal,
            sampling_rate;
            downsample=get_downsample(signal)
        )
    end 

    return intervals[mask]
end

# ---------------------------------------------------------------------------- #
#                        Silence removal (main function)                       #
# ---------------------------------------------------------------------------- #

@doc """
Detect and remove intervals of silence from an audio sample using `ffmpeg`. 

# Arguments 

- `filename:: String`: name of the video file 
- `video_dir:: String=VIDEO_PATH`: directory containing the video file, and where output will be saved 
- `splice:: Bool=true`: whether to run `ffmpeg`
- `min_silence_duration:: Float64=0.6`: minimum duration (in seconds) of silence intervals. Adjacent non-silent intervals are concatenated if they are separated by a smaller interval. Ignored if 0.
- `interval_padding:: Float64=0.1` = padding (in seconds) to add to start and end of each interval. Ignored if 0.
- `detection_kwargs...`: other keyword arguments are passed to `detect_silence`

# Returns 

`Vector{Vector{Float64}}`: intervals of non-silence, in seconds. 
"""
function remove_silence(
    filename:: String; 
    min_silence_duration:: Float64=0.6,
    interval_padding:: Float64=0.1,
    video_dir=VIDEO_PATH, 
    splice=true,
    plot=false, 
    detection_kwargs...
):: Vector{Tuple{Float64, Float64}}
    
    filepath = video_dir * filename 
    if !isfile(filepath)
        SystemError("No file exists at $(filepath)!")        
    end 

    signal, sampling_rate = open_video(filepath)
    signal = stereo_to_mono(signal)
    
    intervals:: Vector{Tuple{Float64, Float64}} = 
        detect_silence(
            signal, 
            sampling_rate; 
            plot=plot,
            detection_kwargs...
        )
    
    # post process intervals by adding padding and/or 
    # concatenate intervals with subthreshold time difference 
    if min_silence_duration > 0 || interval_padding > 0
        intervals = post_process_intervals(
            intervals,
            length(signal) / sampling_rate;
            min_silence_duration=min_silence_duration,
            pad=interval_padding
        )

        if plot 
            plot_silent_segments(
                intervals, signal, sampling_rate;
                downsample=get_downsample(signal)
            )
        end 
    else 
        # round to 3 digits 
        intervals = map(x -> round.(x, digits=3), intervals)
    end 
    
    if splice 
        splice_video_together(filepath, intervals)
    end 
    return intervals 
end 

# ------------------------ Algorithm keyword settings ------------------------ #

svm_kwargs = Dict(
    :smooth_window => 0.1,
    :energy_threshold => 0.2,
    :weight => 0.3, 
    :st_win => 2e-2,
    :st_step => 2e-2
)

vol_kwargs = Dict(
    :silence_threshold => dB_to_AR(-41),
    :min_segment_duration => 0.25,
    :min_silence_duration => 0.45,
    :savgol_window_size => 181,
    :savgol_poly_order => 2
)

# ----------------------------- Run the function! ---------------------------- #
"""
# HOW TO USE 

1. Enter filename (below).
2. Select which algorithm to use. 
    1. Use `vol_kwargs...` for simple, amplitude-based silence detection. 
    2. Use `svm_kwargs...` if you want to use the SVM algorithm.
3. Update the respective kwargs (above).
4. Update kwargs in the `remove_silence` call below. 

# Notes 
- If using `vol_kwargs`, `vol_kwargs["min_silence_duration"]` takes precedence over the value set below!! 
- For `vol_kwargs`, you can choose a value for `:silence_threshold` using `AR_to_dB` and `dB_to_AR` to convert between decibels and absolute amplitudes
- As noted in the docstring for the SVM version of `detect_silence`, vocal-heavy tracks work better with **low** values of `smooth_window` and **high** values of `weight`. However, you need to play with the values to get good results.
"""
filename = "rabbit/otaku_voice"

remove_silence(
    filename;
    splice=true,
    min_silence_duration=0.35,
    interval_padding=0.3,
    plot=true,
    vol_kwargs...
)
