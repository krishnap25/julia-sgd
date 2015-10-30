import ArrayViews, Base.show

typealias SgdModel Dict{Int64, Float64}

type Row
	idxs::ArrayViews.ContiguousView{Int64,1,Array{Int64,1}}
	has_values::Bool #if true, values is non-empty
	values::ArrayViews.ContiguousView{Float64,1,Array{Float64,1}}
	label::Int64
end

function get_value(r::Row, i::Int64)
	if (r.has_values) return r.values[i]
	else return 1.0
	end
end

function show(io::IO, r::Row)
	print(io, r.label)
	for i in 1:length(r.idxs)
		print(io, " $(r.idxs[i]):$(get_value(r, i))")
	end
	println(io, "")
end

function dot(r::Row, w::SgdModel)
	res = 0
	for i in 1:length(r.idxs)
		idx = r.idxs[i]
		if (haskey(w, idx))
			res += w[idx] * get_value(r, i)
		end	
	end
	return res
end

type RowBlock
	offset::Vector{Int64} #offset[i] gives starting element of i^th row
	idxs::Vector{Int64}
	has_values::Bool
	values::Vector{Float64}
	label::Vector{Int64}
end

function size(rb::RowBlock)
	return length(rb.label)
end

function getindex(rb::RowBlock, i::Int64)
	#if (i > length(offset))
	j = rb.offset[i]
	#if (i >= length(rb.offset)) j1 = rb.offset[end]
	#else j1 = rb.offset[i+1] - 1	
	#end
	j1 = rb.offset[i+1]-1
	println("$(j) $(j1)")
	if (rb.has_values)
		val_view = ArrayViews.view(rb.values, j:j1)
	else
		val_view = ArrayViews.view(Float64[], :)
	end
	return Row(ArrayViews.view(rb.idxs, j:j1), rb.has_values, val_view, rb.label[i])
end


function read_svfile(name::AbstractString)
	fout = open(name, "r")
	idxs = Int[]
	vals = Float64[]
	offset = Int64[]
	labels = Int64[]
	i = 1
	has_value = true
	for line in eachline(fout)
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
	return RowBlock(offset, idxs, has_value, vals, labels)
end




