typealias SgdModel Dict{Int64, Float64}

#abstract loss

function loss(losstype::AbstractString, x::Float64)
	
	if (losstype == "logistic" || losstype == "logloss")
		return log(1 + exp(-x))

	elseif (losstype == "hinge" || losstype == "l1svm" || losstype == "svm")
		return max(0, 1-x)

	elseif (losstype == "sqhinge" || losstype == "l2svm")
		return max(0, 1-x)^2

	else
		println(STDERR, "unknown loss")
	end
end

function loss(losstype::AbstractString, w::SgdModel, mb::RowBlock)
	res = 0
	for i in 1:size(m)
		r = mb[i]
		res += loss(losstype, r.label * dot(r, w))	
	end
	return res
end

function lossGradient(losstype::AbstractString, x::Float64)
	if (losstype == "logistic")
		return exp(log(1-exp(-x)) - log(1+exp(-x)))

	elseif (losstype == "hinge")
		if (x < 1)
			return 0
		else
			return -1
		end

	elseif (losstype == "sqhinge")
		if (x < 1)
			return 0
		else
			return 2*(x-1)
		end

	else
		println(STDERR, "unknown loss")
	end
end

function lossGradient(losstype::AbstractString, w::SgdModel, mb::RowBlock)
	grad = Dict{Int64, Float64}()
	for ii in 1:size(m)
		r = mb[ii]
		lossD = lossGradient(losstype, r.label * dot(r, w))
		for j in 1:size(r)
			idx = r.idxs[i]
			val = get_value(r, j) * r.label * lossD
			grad[idx] = get(grad, idx, 0) + val
		end
	end
	return grad
end
function lossGradientNormalized(losstype::AbstractString, w::SgdModel, mb::RowBlock)
	grad = Dict{Int64, Float64}()
	for ii in 1:size(m)
		r = mb[ii]
		lossD = lossGradient(losstype, r.label * dot(r, w))
		for j in 1:size(r)
			idx = r.idxs[i]
			val = get_value(r, j) * r.label * lossD
			grad[idx] = get(grad, idx, 0) + val
		end
	end
	for (idx, g) in grad
		grad[idx] = g/size(m)
	end
	return grad
end



