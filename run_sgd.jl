#!/usr/bin/env julia

using sgd

function main()
	#args: trainfilename testfilename, num_passes, lambda_l1, lambda_l2, 	
	num_args::Int8 = 1
	const trainfile = ARG[num_args] ; num_args += 1
	const testfile = ARGS[num_args] ; num_args += 1
	const num_passes = parse(Int, ARGS[num_args]) ; num_args += 1
	const lambda_l1 = 0
	const lambda_l2 = 1e-6
	#function run_sgd(losstype::AbstractString, lambda_l1::Float64, lambda_l2::Float64, trainingfile::AbstractString, mb_size::Int64, max_data_pass::Int64)


	w = run_sgd("sqhinge", lambda_l1, lambda_l2, trainfile, testfile, 1000, 3)
	acc = predict(testfile, w)
	println("Accuracy = $(acc)")
end
