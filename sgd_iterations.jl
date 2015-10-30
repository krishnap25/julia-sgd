

function init_sgd(lambda_l1::Float64, lambda_l2::Float64, filename::AbstractString)
	w = SgdModel()
	mb_iter = minibatch_iter(filename)
	penalty = L1L2Penalty(lambda_l1, lambda_l2)
	return w, mb_iter, penalty
end


function sgd_iter(w::SgdModel, mb_iter::minibatch_iter, penalty::L1L2Penalty, t::Float64)
	beta = 1 
	alpha = 0.1 #defaults
	eta = (beta + sqrt(t))/ alpha #step size


end
