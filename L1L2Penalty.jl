package sgd



type L1L2Penalty
	lambda_l1::Float64
	lamdba_l2::Float64
end

function update(reg::L1L2Penalty, old_w::Float64, grad::Float64, eta::Float64) 
	#eta: step-size
	temp = eta*old_w - grad
	if (abs(temp) < reg.lambda_l1)
		return 0
	else
		ret = sign(temp) * (abs(temp) - lambda_l1)
		return ret / (eta + reg.lambda_l2)
	end
end







end
