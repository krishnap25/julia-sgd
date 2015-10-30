module sgd

type minibatch_iter
		filename::AbstractString
		fp::IOStream
		num_passes::Int64
end

minibatch_iter(fn::AbstractString) = minibatch_iter(fn, open(fn, "r"), 0)

function read_mb(mb_iter::minibatch_iter, mbsize::Int64)
	if (!isopen(fp))
		fp = open(filename, "r")
		println(STDERR, "SHOULD NOT REACH HERE IN READ_MB")
	end
	idxs = Int[]
	vals = Float64[]
	offset = Int64[]
	labels = Int64[]
	i = 1
	while (!eof(fout) && i <= mbsize)
		line = readline(fout)	
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
			ix = parse(Int, token[1:colon_ix-1])
			e = parse(Float64, token[colon_ix+1:end])
			push!(idxs, ix)
			push!(vals, e)
		end	
	end
	if (eof(fout))
		mb_iter.num_passes += 1
		fp = open(filename, "r")
	end
	return RowBlock(offset, idxs, true, vals, label)
end

end
