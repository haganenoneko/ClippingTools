using PyCall
using Statistics
using StatsBase
using IterTools
using Printf
using DelimitedFiles
using Plots 

# import audiofile 
audiofile = PyCall.pyimport("audiofile")

# constants 
VIDEO_PATH = "../../../../Videos/subtitles/full_raw"
SECS_PATH = VIDEO_PATH * "removesilence_timecodes/"

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
Open the `.m4a` file corresponding to a given video. If the `.m4a` file is not found, it is created with `ffmpeg_extract_audio`
"""
function open_video(filepath:: String):: Tuple{Matrix{Float32}, Int64}
    if !isfile(filepath * ".m4a")
        ffmpeg_extract_audio(filepath)
    end 
    return audiofile.read(filepath * ".m4a")
end 

"""Get start and end times of non-silent intervals"""
function get_ranges(
    signal:: Vector{Float32}, 
    rate:: Float64, 
    threshold:: Float64, 
    duration:: Float64,
    min_concat_interval:: Float64;
    interval_padding:: Float64=0.):: Vector{Tuple{Int64, Int64}}

    # lens[i] = length of i-th run with i-th value (_[i])
    # we ignore the values here because they are all boolean (0 or 1)
    _, lens = StatsBase.rle(signal .<= threshold)
    
    # flatten lens 
    flat = Iterators.flatten(
        ((0,), cumsum(lens))
    )

    # vector of Int64s  
    chunks::Vector{Int64} = [
        ((from + 1), to) 
        for (from, to) in partition(flat, 2, 1) 
        if (to - from) >= duration*rate 
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

    
    intervals = partition(chunks, 2) |> collect 
    mask = map(x -> (x[2] - x[1]) >= min_concat_interval, intervals)

    if interval_padding > 0
        # N x 2 array with start indices in the first column 
        # and end indices in the second column 
        mat = reinterpret(reshape, Int64, intervals[mask])'
        pad = round(Int, interval_padding*rate/2)
        mat[2:end, 1] .-= pad 
        mat[1:end-1, 2] .+= pad 
        return tuple.(eachcol(mat)...)
    else 
        return intervals[mask]
    end 
end 

"""
Return the command for including one interval (from `start_end[1]` to `start_end[2]` of the video) in the `concat` filter of `ffmpeg`
"""
function create_splice_pair_str(ind:: Int, start_end:: Tuple{Float64, Float64}):: String
    t0, t1 = start_end 
        return "[0:v]trim=start=$t0:end=$t1,setpts=PTS-STARTPTS[$(ind)v];
        [0:a]atrim=start=$t0:end=$t1,asetpts=PTS-STARTPTS[$(ind)a];"
end 

Iters2Vec(x) = collect(Iterators.flatten(x))
joinNewline(x) = join(x, "\n")

"""Return the suffix to the `concat` filter for `ffmpeg`"""
function get_concat(num:: Int64, isVideo:: Bool):: String 
    head = "concat=n=$(num)"
    if isVideo
        return head * ":v=1:a=0 [outv]"
    else 
        return head * ":v=0:a=1 [outa]"
    end 
end 

function splice_video(filepath:: String, intervals:: Vector{Tuple{Float64, Float64}}) 
    
    outname = "$(filepath)_silenceremove"
    
    num = length(intervals)
    inds = 0:num-1

    pairs = create_splice_pair.(inds, intervals) |> Iters2Vec
    v_pairs, a_pairs = joinNewline(pairs[1:2:end]), joinNewline(pairs[2:2:end])
    streams = [("[$(i)v]", "[$(i)a]") for i = inds] |> Iters2Vec
    v_streams, a_streams = *(streams[1:2:end]...), *(streams[2:2:end]...)

    concats = [
        "concat=n=$(num)" * s for s in 
        (":v=1:a=0 [v]\" -map [v]", ":v=0:a=1 [a]\" -map [a]")
    ]
    @info concats 

    cmd_header = "ffmpeg -hide_banner -i \"$filepath.mp4\" -filter_complex "

    for i = 1:2 
        pairs_i = join(pairs[i:2:end], "\n")
        streams_i = *(streams[i:2:end]...)
        filter_i = "\"$pairs_i $streams_i $(concats[i])"
        
        cmd_i = "$cmd_header $filter_i $(outname)_$i.mp4"
        run(`powershell.exe $cmd_i`)
    end 
    
    merge = "ffmpeg -hide_banner -i $(outname)_1.mp4 -i $(outname)_2.mp4 -c:v copy -c:a copy $(outname).mp4"
    run(`powershell.exe $merge`)
end 

"""
Similar to `splice_video`, but tries to trim and concatenate video and audio streams simultaneously, instead of one at a time.
"""
function splice_video_together(filepath:: String, intervals:: Vector{Tuple{Float64, Float64}}) 
    output_name = "$(filepath)_silenceremove.mp4"
    num = length(intervals)
    inds = 0:num-1

    pairs = join(create_splice_pair_str.(inds, intervals), "\n")
    concat_fn = *(["[$(i)v][$(i)a]" for i = inds]...) * 
        "concat=n=$(num):v=1:a=1[outv][outa]"

    tmp = "$(filepath)__filter.txt"
    open("$(filepath)__filter.txt", "w") do io 
        write(io, "$(pairs * concat_fn)")
    end 

    # tmp = replace(tmp, "/" => "//")
    map_fn = "-map [outv] -map [outa]"
    cmd = `powershell.exe ffmpeg -hide_banner -i "$filepath.mp4" -filter_complex_script $tmp $map_fn $output_name`
    
    run(cmd)

    if !isfile(output_name)
        error("Output file not found:\n $output_name")
    end 
end 

struct TooManyIntervalsException <: Exception
    num_intervals:: Int64 
    threshold:: Float64 
    duration:: Float64 
end 

Base.showerror(io:: IO, e::TooManyIntervalsException) = print(io,
    """
    too many intervals were found: $(e.num_intervals)
    \t Threshold: $(e.threshold)
    \t Duration: $(e.duration)

    Consider lowering the threshold or raising the duration.
    """
)

"""Remove silence from a given .mp4 file 
"""
function remove_silence(
    filename:: String; 
    splice:: Bool=true, 
    video_dir:: String=VIDEO_PATH, 
    silence_duration:: Float64=1.0,
    min_concat_interval:: Float64=0.25, 
    silence_threshold:: Float64=1e-2, 
    interval_padding:: Float64=5e-2,
    return_intervals:: Bool=false,
    plot_signal=false)
    
    filepath = video_dir * filename 
    audio_data, audio_sr = open_video(filepath)
    signal = abs.(audio_data[1,:])
    
    if plot_signal == true 
        p = Plots.plot(
            (1:length(signal)) ./ audio_sr,
            log.(signal .^ 2) .* 10,
            serestype=:scatter
        )
        display(p)
    end 
    
    intervals = get_ranges(
        signal, 
        Float64(audio_sr), 
        silence_threshold, 
        silence_duration, 
        min_concat_interval*audio_sr;
        interval_padding=interval_padding
    )

    if length(intervals) < 1 
        error("Empty intervals")
    end 
    
    seconds = map(x -> round.(x ./ audio_sr, digits=3), intervals)
    
    total = (length(signal)/audio_sr) - 
        sum(map(x -> x[2] - x[1], seconds))
    
    @info "Total time removed: $(total/60) min ($total sec)"
    @info "Number of streams: $(length(seconds))"

    if splice 
        # splice_video(filepath, seconds)
        splice_video_together(filepath, seconds)
    end 

    if return_intervals
        return intervals, seconds 
    end
end 

"""NOT IMPLEMENTED! 
Another version that accepts pre-computed intervals. 
Thus, it skips computing intervals and goes directly to splicing."""
function remove_silence(
    filename:: String, seconds:: Vector{Tuple{Float64, Float64}};
    video_dir:: String=VIDEO_PATH)
    filepath = video_dir * filename 
    splice_video_together(filepath, seconds)
end 

"""Iteratively runs `remove_silence` to check for a working `silence_threshold`, and then runs `remove_silence` up to some maximum number of times. 
"""
function iterative_removal(
    filename:: String, 
    threshold:: Float64, 
    init_dur:: Float64; 
    min_dur:: Float64=0.8,
    dur_step_up:: Float64=1.0,
    dur_step_down:: Float64=0.5,
    max_steps:: Int64=10,
    min_num_intervals:: Int64=10,
    kwargs...)
    
    error("This isn't well-tested. Error until otherwise.")

    try 
        @assert min_dur < init_dur 
    catch 
        throw(
            DomainError(
                init_dur,
                "Initial duration $(init_dur)-s >= minimum duration $(min_dur)-s"
            )
        )
    end 
    
    fname = filename 
    dur = init_dur 
    num_steps = 0 

    while dur >= min_dur 
        if num_steps >= max_steps
            @info "Number of steps $(num_steps) >= max steps $(max_steps)"
            break 
        end 

        @info "Step $(num_steps) of $(max_steps).....
        Current duration: $(dur). Target: $(min_dur)"
        
        try 
            _, secs = remove_silence(
                fname;
                silence_duration=dur, 
                silence_threshold=threshold,
                splice=false,
                return_intervals=true,
                kwargs...
                )
                if length(secs) < min_num_intervals
                    @info "Not enough intervals. Stepping down."
                    dur -= dur_step_down
                else 
                    @info "\tIntervals: $(length(secs))"
                    remove_silence(fname, secs)
                    fname *= "_$(@sprintf("%.0f", dur))-s"
                end 
        catch e 
            if isa(e, TooManyIntervalsException)
                dur += dur_step_up
            else 
                throw(e)
            end 
        end 
    end 
    @info "Final removal iteration completed with 
        Duration $(dur)
        Threshold $(threshold)

        The output file is:
        \t$(fname)"
end 

dB_to_AR(dB::Number) = sqrt(10^(dB/10))
AR_to_dB(AR::Float32) = log10(AR ^ 2) .* 10 

function save_secs(
    seconds:: Vector, filename:: String; 
    outdir=SECS_PATH) 
    
    outname = outdir * "$(filename).csv"
    open(outname, "w") do io 
        writedlm(io, seconds, ',')
    end 

    @info "Intervals saved to $outname"
end 

# ---------------------------------------------------------------------------- #
#                                  Test usage                                  #
# ---------------------------------------------------------------------------- #

"""
Helps determine values for `remove_silence`. Example:
```
get_dbs(
    filename, 
    [
        (50.053, 5.118), # silent 
        (50.497, 2.559), # silent
        (6.75, 0.33),   # soft 
        (40.465, 0.409) # normal 
    ];
    rle_estimate_dB=-44.
)
```
"""
function get_dbs(
    filename:: String, 
    time_lengths:: Vector{Tuple{Float64, Float64}}; 
    rle_estimate_dB:: Float64=nothing,
    video_dir=VIDEO_PATH,
    quantiles:: Vector{Float64}=[0.05, 0.5, 0.95]
)::Vector{Vector{Float64}}

    filepath = video_dir * filename  
    audio_data, audio_sr = open_video(filepath)
    signal = abs.(audio_data[1,:])

    dBs = Vector{Vector{Float64}}(
        undef,
        length(time_lengths) + 1
    )

    for i = 1:length(time_lengths)
        s, ns = round.(Int, time_lengths[i] .* audio_sr)
        x = AR_to_dB.(signal[s:s+ns])
        
        dBs[i+1] = quantile(x, quantiles)
    end 

    dBs[1] = quantile(signal, quantiles)

    if rle_estimate_dB !== nothing 
        _, lens = StatsBase.rle(signal .<= dB_to_AR(rle_estimate_dB))
        flat = Iterators.flatten(((0,), cumsum(lens)))
        chunks:: Vector{Int64} = partition(flat, 2, 1) |> Iters2Vec
        intervals = partition(chunks, 2) |> collect 
        mat = reinterpret(reshape, Int, intervals)
        delta_t = (mat[:,2] .- mat[:,1]) ./ audio_sr 
        @info quantile(delta_t, quantiles)
    end 
    @info "Total, Intervals"
    return dBs 
end 

# ---------------------------------------------------------------------------- #

filename = "/rabbit/mimi_bday_2023_1"

get_dbs(
    filename, 
    [
        (15.05, 1.32), # silent 
    ];
    rle_estimate_dB=-25.
)

remove_silence(
    filename; 
    splice=false, 
    silence_threshold=dB_to_AR(-30), 
    silence_duration=0.6,
    min_concat_interval=0.25, 
    interval_padding=0.2,
    return_intervals=true,
)[2]

# save_secs(secs, filename)

"""TODO 
Visualize amplitude
"""