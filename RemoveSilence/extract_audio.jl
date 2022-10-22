using PyCall
using Statistics
using StatsBase
using IterTools

# import audiofile 
audiofile = PyCall.pyimport("audiofile")

# constants 
VIDEO_PATH = "C:/Users/delbe/Videos/subtitles/full_raw/"

function ffmpeg_extract_audio(filename:: String):: Bool 
    output_name = "$filename.m4a"
    
    if !isfile(output_name)
        cmd = "ffmpeg -i $filename.mp4 -vn -acodec copy $output_name"
        run(`powershell $cmd`)
    end 

    if isfile(output_name)
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
    
    return [
        ((from + 1), to) for (from, to) in 
        partition(
            Iterators.flatten(
                ((0,),cumsum(lens))
            ), 2, 1
        ) if (to - from) >= duration*rate 
    ]
end 

function create_splice_pair(ind:: Int, start_end:: Tuple{Float64, Float64}):: String
    t0, t1 = start_end 
    return "[0:v]trim=start=$t0:end=$t1,setpts=PTS-STARTPTS[$(ind)v];
    [0:a]atrim=start=$t0:end=$t1,asetpts=PTS-STARTPTS[$(ind)a];"
end 

function splice_video(filepath:: String, intervals:: Vector{Tuple{Float64, Float64}}) 
    
    output_name = "$(filepath)_silenceremove.mp4"
    # if isfile(output_name)
    #     println("$output_name alreay exists. Overwrite? [y/n]")
    #     overwrite = lowercase(readline(stdin))
    #     if overwrite != "y"
    #         return 
    #     end 
    # end 
    
    num = length(intervals)
    inds = 0:num-1

    pairs = join(create_splice_pair.(inds, intervals), "\n")
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
    total = sum(map(x -> x[2] - x[1], seconds))
    @info "Total time removed: $(total/60) minutes ($total seconds)"

    if splice 
        splice_video(filepath, seconds)
    end 

    return intervals, seconds 
end 

filename = "nazuna_4y_2"
remove_silence(filename; silence_duration=1.0, silence_threshold=5e-3, splice=false)