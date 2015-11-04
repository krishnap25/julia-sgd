function norm(w::SgdModel)
	n = 0.0
	for x in values(w)
		n += x^2
	end
	return sqrt(n)
end

function init_sgd(lambda_l1::Float64, lambda_l2::Float64, filename::AbstractString, mb_size::Int64)
	w = SgdModel()
	mb_iter = minibatch_iter(filename, mb_size)
	penalty = L1L2Penalty(lambda_l1, lambda_l2)
	return w, mb_iter, penalty
end

function run_sgd(losstype::Int, lambda_l1::Float64, lambda_l2::Float64, trainingfile::AbstractString, testfile::AbstractString, mb_size::Int64, max_data_pass::Int64)
	beta = 1 
	alpha = 0.1 #defaults
	w, mb_iter, penalty = init_sgd(lambda_l1, lambda_l2, trainingfile, mb_size)
	t = 1.0
	while (mb_iter.num_passes < max_data_pass)
		eta = (beta + sqrt(t))/ alpha #step size
		old_iter = mb_iter.num_passes
		grad = lossGradientNormalized(losstype, w, read_mb(mb_iter))
		#println(norm(grad))
		new_iter = mb_iter.num_passes
		old_w::Float64 = 0.0
		new_w::Float64 = 0.0
		for (idx, grad_val) in grad
			#update
			old_w = get(w, idx, 0.0)
			new_w = update_model(penalty, old_w, grad_val, eta)
			if (new_w == 0.0)
				delete!(w, idx)
			else
				w[idx] = new_w
			end
		end
		#println("norm : $(norm(w))")
		if (new_iter != old_iter)
			acc = predict(testfile, w)
			println("Iteration $(new_iter): Accuracy $(acc), Sparsity $(length(collect(keys(w))))")
		end	
		t += 1
	end
	return w
end

#function sgd_one_iter(losstype::AbstractString, w::SgdModel, mb_iter::minibatch_iter, penalty::L1L2Penalty, timestep::Float64)
#	beta = 1 
#	alpha = 0.1 #defaults
#	eta = (beta + sqrt(timestep))/ alpha #step size
#	old_iter = mb_iter.num_passes
#	grad = lossGradientNormalized(losstype, w, read_mb(mb_iter))
#	new_iter = mb_iter.num_passes
#	old_w::Float64 = 0.0
#	new_w::Float64 = 0.0
#	if (new_iter != old_iter)
#		println("Iteration $(new_iter) complete")
#	end	
#	for (idx, grad_val) in grad
#		#update
#		old_w = get(w, idx, 0.0)
#		new_w = update_model(penalty, old_w, grad_val, eta)
#		if (new_w == 0)
#			delete!(w, idx)
#		else
#			w[idx] = new_w
#		end
#	end
#	println("norm : $(norm(w))")
#end

#function run_sgd(losstype::AbstractString, lambda_l1::Float64, lambda_l2::Float64, trainingfile::AbstractString, mb_size::Int64, max_data_pass::Int64)
#
#	w, mb_iter, penalty = init_sgd(lambda_l1, lambda_l2, trainingfile, mb_size)
#	t = 1.0
#	while (mb_iter.num_passes < max_data_pass)
#		sgd_one_iter(losstype, w, mb_iter, penalty, t)
#		t += 1
#	end
#	return w
#end

function predict(testfile::AbstractString, w::SgdModel)
	correct::Int64 = 0
	total::Int64 = 0
	fout = open(testfile, "r")
	has_value = true
	ix = 0; e = 0.0; 
	for line in eachline(fout)
		dotp = 0.0
		ix = findfirst(line, ' ')
		y = parse(Int64, strip(line[1:ix-1]))
		if (y != 1)
			y = -1
		end
		tokens = split(strip(line[ix+1:end]), ' ')
		for token in tokens
			colon_ix = findfirst(token, ':')
			if (colon_ix != 0)
				ix = parse(Int, token[1:colon_ix-1])
				e = parse(Float64, token[colon_ix+1:end])
			else
				ix = parse(Int, strip(token))
				e = 1
			end
			dotp += (get(w, ix, 0) * e)
		end
		if (sign(dotp) == y)
			correct += 1
		end
		total += 1
	end
	println("$(correct)/$(total)")

	return correct/total	
end
