
type minibatch_iter
		filename::AbstractString
		fp::IOStream
		num_passes::Int64
		mb_size::Int64 #default 1000
end

minibatch_iter(fn::AbstractString) = minibatch_iter(fn, open(fn, "r"), 0, 1000) 

minibatch_iter(fn::AbstractString, mbsize::Int64) = minibatch_iter(fn, open(fn, "r"), 0, mbsize)

function read_mb(mb_iter::minibatch_iter)
	mbsize = mb_iter.mb_size
	if (!isopen(mb_iter.fp))
		mb_iter.fp = open(filename, "r")
		println(STDERR, "SHOULD NOT REACH HERE IN READ_MB")
	end
	idxs = Int[]
	vals = Float64[]
	offset = Int64[]
	labels = Int64[]
	i = 1
	num_rows = 1
	has_value = true
	while (!eof(mb_iter.fp) && num_rows <= mbsize)
		num_rows += 1
		line = readline(mb_iter.fp)	
		push!(offset, i)
		ix = findfirst(line, ' ')
		y = parse(Int64, strip(line[1:ix-1]))
		if (y != 1)
			y = -1
		end
		push!(labels, y)
		tokens = split(strip(line[ix+1:end]), ' ')
		for token in tokens
			i += 1
			colon_ix = findfirst(token, ':')
			if (colon_ix != 0)
				ix = parse(Int, token[1:colon_ix-1])
				e = parse(Float64, token[colon_ix+1:end])
				push!(idxs, ix)
				push!(vals, e)
			else
				ix = parse(Int, strip(token))
				push!(idxs, ix)
				has_value = false
			end
		end	
	end
	push!(offset, i)
	if (eof(mb_iter.fp))
		mb_iter.num_passes += 1
		mb_iter.fp = open(mb_iter.filename, "r")
	end
	return RowBlock(offset, idxs, has_value, vals, labels)
end

