#!/usr/bin/env julia

using sgd

function main()
	#args: trainfilename testfilename, num_passes, lambda_l1, lambda_l2, 	
	num_args::Int8 = 1
	const trainfile = ARGS[num_args] ; num_args += 1
	const testfile = ARGS[num_args] ; num_args += 1
	const num_passes = parse(Int, ARGS[num_args]) ; num_args += 1
	const lambda_l1 = 1e-2
	const lambda_l2 = 1e-3
	const mb_size = 10000
	println("Starting")

	#function run_sgd(losstype::AbstractString, lambda_l1::Float64, lambda_l2::Float64, trainingfile::AbstractString, mb_size::Int64, max_data_pass::Int64)
	loss_dict = Dict("logistic" => 1, "logloss" => 1, "1" => 1, "hinge" => 2, "2" => 2, "l1svm" => 2, "svm" => 2, "sqhinge" => 3, "l2svm" => 3, "3" => 3)
	loss = "sqhinge"
	w = run_sgd(loss_dict[loss], lambda_l1, lambda_l2, trainfile, testfile,  mb_size, num_passes)
	println("Done learning")
	acc = predict(testfile, w)
	println("Accuracy = $(acc)")
end

main()
