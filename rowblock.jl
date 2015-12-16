module rowblock

import ArrayViews, Base.show

export Row, dot, RowBlock, size, getindex, read_svfile, get_value, localize

typealias SgdModel Dict{UInt64, Float64}
import Base.getindex
type Row
	idxs::ArrayViews.ContiguousView{UInt64,1,Array{UInt64,1}}
	has_values::Bool #if true, values is non-empty
	values::ArrayViews.ContiguousView{Float64,1,Array{Float64,1}}
	label::Int64
end

size(r::Row) = Base.length(r.idxs)

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
	res = 0.0
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
	idxs::Vector{UInt64}
	has_values::Bool
	values::Vector{Float64}
	label::Vector{Int64}
end

size(rb::RowBlock) = Base.length(rb.label)

#length(rb::RowBlock) = Base.length(rb.label)

function getindex(rb::RowBlock, i::Int64)
	#if (i > length(offset))
	j = rb.offset[i]
	#if (i >= length(rb.offset)) j1 = rb.offset[end]
	#else j1 = rb.offset[i+1] - 1	
	#end
	j1 = rb.offset[i+1]-1
	if (rb.has_values)
		val_view = ArrayViews.view(rb.values, j:j1)
	else
		val_view = ArrayViews.view(Float64[], :)
	end
	return Row(ArrayViews.view(rb.idxs, j:j1), rb.has_values, val_view, rb.label[i])
end	


function localize(rb::RowBlock)
	
	idxs = rb.idxs
	pa = Vector{Tuple{UInt64, Int64}}()
	#populate pair
	for i in 1:length(idxs)
		push!(pa, (idxs[i], i))
	end

	#sort pair
	sort!(pa)
	println(pa)
	#copy rb
	rb1 = deepcopy(rb)
	uniq_id = Uint64[]
	#renumber
	prev_id = 0
	id_new = 0
	for (id, posn) in pa
		if (id == prev_id)
			rb1.idxs[posn] = id_new
		else
			push!(uniq_id, id)
			prev_id = id
			id_new += 1
			rb1.idxs[posn] = id_new
		end
	end
	return rb1, uniq_id
end


function read_svfile(name::AbstractString)
	fout = open(name, "r")
	idxs = UInt64[]
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
				ix = parse(UInt64, token[1:colon_ix-1])
				e = parse(Float64, token[colon_ix+1:end])
				push!(idxs, ix)
				push!(vals, e)
			else
				ix = parse(UInt64, strip(token))
				push!(idxs, ix)
				has_value = false
			end
		end
	end
	push!(offset, i)
	return RowBlock(offset, idxs, has_value, vals, labels)
end



end #module
