using KernelAbstractions, AMDGPU, LinearAlgebra

SqExp(x1,x2, θ) = θ[1]^2 * exp(-1/2 * LinearAlgebra.norm(x2 - x1, 2)/θ[2]) 
export SqExp

ZeroMean(x) = 0.0
export ZeroMean
