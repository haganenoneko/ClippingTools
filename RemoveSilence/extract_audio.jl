using PyCall
using Statistics
using StatsBase
using IterTools

# import audiofile 
audiofile = PyCall.pyimport("audiofile")

# constants 
VIDEO_PATH = "C:/Users/delbe/Videos/subtitles/full_raw/"

function ffmpeg_extract_audio(filename:: String):: Bool 
    outname = "$filename.m4a"
    
    if !isfile(outname)
        cmd = "ffmpeg -i $filename.mp4 -vn -acodec copy $outname"
        run(`powershell $cmd`)
    end 

    if isfile(outname)
        return true 
    else 
        error("Failed to extract audio from $filename.mp4")
    end 
end 

function open_video(filepath:: String):: Tuple{Matrix{Float32}, Int64}
    if !isfile(filepath * ".m4a")
        ffmpeg_extract_audio(filepath)
    end 
    return audiofile.read(filepath * ".m4a")
end 

function get_ranges(signal:: Vector{Float32}, rate:: Float64, threshold:: Float64, duration:: Float64):: Vector{Tuple{Int64, Int64}}
    _, lens = StatsBase.rle(signal .<= threshold)
    
    chunks = [
        ((from + 1), to) for (from, to) in 
        partition(
            Iterators.flatten(
                ((0,),cumsum(lens))
            ), 2, 1
        ) if (to - from) >= duration*rate 
    ] |> Iters2Vec
    
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

    return partition(chunks, 2) |> collect 
end 

function create_splice_pair(ind:: Int, start_end:: Tuple{Float64, Float64}):: Tuple{String, String}
    t0, t1 = start_end 
    return ("[0:v]trim=start=$t0:end=$t1,setpts=PTS-STARTPTS[$(ind)v];", 
        "[0:a]atrim=start=$t0:end=$t1,asetpts=PTS-STARTPTS[$(ind)a];")
end 


function create_splice_pair_str(ind:: Int, start_end:: Tuple{Float64, Float64}):: String
    t0, t1 = start_end 
    return "[0:v]trim=start=$t0:end=$t1,setpts=PTS-STARTPTS[$(ind)v];
    [0:a]atrim=start=$t0:end=$t1,asetpts=PTS-STARTPTS[$(ind)a];"
end 

Iters2Vec(x) = collect(Iterators.flatten(x))
joinNewline(x) = join(x, "\n")

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

    cmd_header = "ffmpeg -i \"$filepath.mp4\" -filter_complex "

    for i = 1:2 
        pairs_i = join(pairs[i:2:end], "\n")
        streams_i = *(streams[i:2:end]...)
        filter_i = "\"$pairs_i $streams_i $(concats[i])"
        
        cmd_i = "$cmd_header $filter_i $(outname)_$i.mp4"
        run(`powershell.exe $cmd_i`)
    end 
    
    merge = "ffmpeg -i $(outname)_1.mp4 -i $(outname)_2.mp4 -c:v copy -c:a copy $(outname).mp4"
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
    filter_ = "\"$(pairs * concat_fn)\""
    map_fn = "-map [outv] -map [outa]"
    cmd = `powershell.exe ffmpeg -i "$filepath.mp4" -filter_complex $filter_ $map_fn $output_name`
    
    run(cmd)

    if !isfile(output_name)
        error("Output file not found:\n $output_name")
    end 
end 

function remove_silence(filename:: String; splice=true, video_dir=VIDEO_PATH, silence_duration=1.0, silence_threshold=1e-2)
    filepath = video_dir * filename 
    audio_data, audio_sr = open_video(filepath)
    signal = abs.(audio_data[1,:])

    intervals = get_ranges(signal, Float64(audio_sr), silence_threshold, silence_duration)
    if length(intervals) < 1 
        error("Empty intervals")
    end 

    seconds = map(x -> round.(x ./ audio_sr, digits=3), intervals)
    total = (length(signal)/audio_sr) - sum(map(x -> x[2] - x[1], seconds))
    @info "Total time removed: $(total/60) minutes ($total seconds)"
    @info "Number of streams: $(length(seconds))"

    if length(seconds) > 300 
        error("Too many streams!")
    end 

    if splice 
        # splice_video(filepath, seconds)
        splice_video_together(filepath, seconds)
    end 

    return intervals, seconds 
end 

filename = "uruha_unei-gifting_voice-turtle"
remove_silence(filename; silence_duration=1.0, silence_threshold=2e-2, splice=true)