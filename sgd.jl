module sgd

importall rowblock 
using minibatch, Loss, l1l2Penalty, ParameterServer

export run_sgd, predict

#include("RowBlock.jl")
#include("Minibatch.jl")
#include("Loss.jl")
#include("L1L2Penalty.jl")
include("sgd_iterations.jl")

end
