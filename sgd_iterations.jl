

function init_sgd(lambda_l1::Float64, lambda_l2::Float64, filename::AbstractString, mb_size::Int64)
	w = SgdModel()
	mb_iter = minibatch_iter(filename, mb_size)
	penalty = L1L2Penalty(lambda_l1, lambda_l2)
	return w, mb_iter, penalty
end


function sgd_one_iter(losstype::AbstractString, w::SgdModel, mb_iter::minibatch_iter, penalty::L1L2Penalty, timestep::Float64)
	beta = 1 
	alpha = 0.1 #defaults
	eta = (beta + sqrt(timestep))/ alpha #step size
	
	grad = lossGradient(losstype, w, read_mb(mb_iter))
	for (idx, grad_val) in grad
		#update
		old_w = get(w, idx, 0)
		new_w = update_model(penalty, old_w, grad_val, eta)
		if (new_w == 0)
			delete!(w, idx)
		else
			w[idx] = new_w
		end
	end
end


function run_sgd(losstype::AbstractString, lambda_l1::Float64, lambda_l2::Float64, trainingfile::AbstractString, mb_size::Int64, max_data_pass::Int64)

	w, mb_iter, penalty = init_sgd(lambda_l1, lambda_l2, trainingfile, mb_size)
	t = 1
	while (mb_iter.num_passes < max_data_pass)
		sgd_one_iter(losstype, w, mb_iter, penalty, t)
		t += 1
	end
	return w
end
