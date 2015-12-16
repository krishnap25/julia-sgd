#!/usr/bin/env julia

using sgd

function main()
	#args: dataset trainfilename testfilename, num_passes, lambda_l1, lambda_l2, 	
	num_args::Int8 = 1
  dataset = ARGS[num_args] ; num_args += 1
	const trainfile = ARGS[num_args] ; num_args += 1
	const testfile = ARGS[num_args] ; num_args += 1
	const num_passes = parse(Int, ARGS[num_args]) ; num_args += 1

  #params: lambda_l1 lambda_l2 alpha beta

  if (dataset == "ctra")
    params = [1e-2 1e-3  0.2 1]
  elseif (dataset == "ctrb")
    params = [5e-3 1e-4  0.09 1]
  elseif (dataset == "criteo_s")
    params = [5e-4 1e-4  3 1]
  else
    println("Unrecognized dataset. Supply params!")
    return 0
  end


	lambda_l1 = params[1]
	lambda_l2 = params[2]
	mb_size = 10000
	#println("Starting")
	println("Starting: lambda_l1 = $(lambda_l1) ; lambda_l2 = $(lambda_l2)")

	#function run_sgd(losstype::AbstractString, lambda_l1::Float64, lambda_l2::Float64, trainingfile::AbstractString, mb_size::Int64, max_data_pass::Int64)
	loss_dict = Dict("logistic" => 1, "logloss" => 1, "1" => 1, "hinge" => 2, "2" => 2, "l1svm" => 2, "svm" => 2, "sqhinge" => 3, "l2svm" => 3, "3" => 3)
	loss = "sqhinge"
	w = run_sgd(loss_dict[loss], lambda_l1, lambda_l2, trainfile, testfile,  mb_size, num_passes, params[3], params[4])
	println("Done learning")
	acc = predict(testfile, w)
	println("Accuracy = $(acc)")
end

main()
