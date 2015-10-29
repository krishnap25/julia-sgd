module sgd

import ArrayViews, Base.show

type Row
	idx::ArrayViews.ContiguousView{Int64,1,Array{Int64,1}}
	has_value::Bool #if true, value is non-empty
	value::ArrayViews.ContiguousView{Float64,1,Array{Float64,1}}
	label::Int64
end

function get_value(r::Row, i::Int64)
	if (r.has_value) return r.value[i]
	else return 1.0
	end
end

function show(io::IO, r::Row)
	print(io, r.label)
	for i in 1:length(r.idx)
		print(io, " $(r.idx[i]):$(get_value(r, i))")
	end
	println(io, "")
end

type RowBlock
	offset::Vector{Int64} #offset[i] gives starting element of i^th row
	idx::Vector{Int64}
	has_value::Bool
	value::Vector{Float64}
	label::Vector{Int64}
end

function getindex(rb::RowBlock, i::Int64)
	#if (i > length(offset))
	j = rb.offset[i]
	if (i >= length(rb.offset)) j1 = length(rb.offset)
	else j1 = rb.offset[i+1] - 1
	end
	if (rb.has_value)
		val_view = ArrayViews.view(rb.value, j:j1)
	else
		val_view = Float64[]
	end
	return Row(ArrayViews.view(rb.idx, j:j1), rb.has_value, val_view, rb.label[i])
end


function read_svfile(name::AbstractString)
	fout = open(name, "r")
	idxs = Int[]
	vals = Float64[]
	offset = Int64[]
	labels = Int64[]
	i = 1
	for line in eachline(fout)
		push!(offset, i)
		ix = findfirst(line, ' ')
		y = parse(Int64, strip(line[1:ix-1]))
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
	return RowBlock(offset, idxs, true, vals, labels)
end




end
