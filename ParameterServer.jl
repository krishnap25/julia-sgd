module ParameterServer
using l1l2Penalty
export PS, init_PS, pull, push


type PS
	w::Dict{UInt64, Float64}
	penalty::L1L2Penalty
end


function init_PS(lambda_l1::Float64, lambda_l2::Float64)
	penalty = L1L2Penalty(lambda_l1, lambda_l2)
	return PS(Dict{UInt64, Float64}(), penalty)
end

function pull(ps::PS, keys::Vector{UInt64})
	ks = UInt64[]
	vs = Float64[]
	for k in keys
		if (haskey(ps.w, k))
			push!(ks, k)
			push!(vs, ps.w[k])
		end
	end
	return ks, vs
end

function push(ps::PS, grad_keys::Vector{UInt64}, grad_vals::Vector{Float64}, eta)
	for i in 1:length(grad_keys)
		idx = grad_keys[i]
		grad_val = grad_vals[i]
		old_w = get(ps.w, idx, 0.0)
		new_w = update_model(ps.penalty, old_w, grad_val, eta)
		if (new_w == 0.0)
			delete!(ps.w, idx)
		else
			ps.w[idx] = new_w
		end
	end
end



end #module
