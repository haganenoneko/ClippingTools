using IterTools

# ---------------------------------------------------------------------------- #
#                            Post process intervals                            #
# ---------------------------------------------------------------------------- #

Iters2Vec(x) = x |> Iterators.flatten |> collect 

function post_process_intervals(
    intervals:: Union{Vector{Vector{T}}, Vector{Tuple{T,T}}},
    total_seconds:: Number;
    min_silence_duration:: Float64=0.6,
    pad:: Float64=0.1,
) where {T<:Number} 

    @assert all([total_seconds, min_silence_duration, pad] .> 0)

    if pad == 0 && min_silence_duration == 0
        return intervals 
    end 

    # flatten intervals 
    flat = Iters2Vec(intervals)

    # add padding to start and end times 
    if pad > 0 
        flat[2:2:end-2] .+= pad
        flat[3:2:end-1] .-= pad
        if flat[1] > pad
            flat[1] -= pad
        elseif flat[end] + pad <= total_seconds 
            flat[end] += pad
        end 
    end 

    if min_silence_duration <= 0 
        return partition(flat, 2) |> collect 
    end 

    # compute elapsed time between [end(i), start(i+1)] of adjacent intervals 
    hasOverlap = findall(
        <=(min_silence_duration),
        flat[3:2:end] .- flat[2:2:end-1]       
    ) .* 2 

    # get indices of original [start, end] times that need 
    # to be dropped 
    inds = vcat(
        [hasOverlap, hasOverlap .+ 1]...
        ) |> sort |> unique
        
    # set start/end times with sub-threshold differences to -1 
    flat[inds] .= -1 

    # output new intervals 
    pruned:: Vector{Tuple{Float64, Float64}} = 
        partition(
            round.(flat[flat .> 0], digits=3), 2
        ) |> collect 
    
    @info "No. pruned intervals: $(length(pruned)) ($(length(intervals)-length(pruned)) removed)"

    tot = total_seconds - sum(map(x -> x[2]-x[1], pruned))
    @info "Total time removed: $(round(tot/60, digits=2)) min ($(round(tot, digits=2)) sec)"

    return pruned 
end 


# ---------------------------------------------------------------------------- #
#                         Build and run ffmpeg command                         #
# ---------------------------------------------------------------------------- #

"""
Return the command for including one interval (from `start_end[1]` to `start_end[2]` of the video) in the `concat` filter of `ffmpeg`

# Arguments 

- `ind:: Int`: index of current splice segment  
- `start_end:: Union{Vector, Tuple}`

# Returns 

- `String`: filter for `ffmpeg` to splice a section of video (and the associated audio)

"""
function create_splice_pair_str(
    ind:: Int, start_end:: Union{Vector, Tuple}
):: String
    t0, t1 = start_end 
    return "[0:v]trim=start=$t0:end=$t1,setpts=PTS-STARTPTS[$(ind)v];
    [0:a]atrim=start=$t0:end=$t1,asetpts=PTS-STARTPTS[$(ind)a];"
end 


"""
Trim and concatenate video and audio streams simultaneously.

# Arguments 

- `filepath::String`: path to the video file 
- `intervals::Vector{Union{Tuple, Vector}}`: a `Vector` of `[start, end]` times corresponding to each segment of video to be spliced in (i.e. all other parts of the video will be removed). 

# Returns 

`::Nothing`

"""
function splice_video_together(
    filepath:: String, 
    intervals:: Vector
):: Nothing 

    output_name = "$(filepath)_silenceremove.mp4"
    num = length(intervals)
    inds = 0:num-1

    # join all intervals into a single filter 
    pairs = join(create_splice_pair_str.(inds, intervals), "\n")
    concat_fn = *(["[$(i)v][$(i)a]" for i = inds]...) * 
        "concat=n=$(num):v=1:a=1[outv][outa]"

    # write all filters to a text file to prevent overloading stdout
    filterpath = "$(filepath)__filter.txt"
    open(filterpath, "w") do io 
        write(io, "$(pairs * concat_fn)")
    end 
    
    # map output video and audio streams 
    map_fn = "-map [outv] -map [outa]"

    # assemble final command
    cmd = `powershell.exe ffmpeg -hide_banner -i "$filepath.mp4" -filter_complex_script $filterpath $map_fn $output_name`
    
    Base.run(cmd)

    if !isfile(output_name)
        error("Output file not found:\n $output_name")
    else
        @info "Output file saved at:\n$output_name"
    end 

    return nothing 
end 