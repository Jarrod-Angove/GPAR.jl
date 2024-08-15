# GPAR.jl: A simple package for autoregressive gaussian process functionals in Julia

A Julia package for modeling hysteretic data with Gaussian process regression!

# What is an autoregressive Guassian Process functional? 

This package is built using `KernelAbstractions.jl` to enable all models to run on parallel hardware (GPU or CPUs). My intention
is not to replace existing GP packages such as `AbstractGPs.jl` or `GaussianProcesses.jl`. Rather, I have a very specific usecase that 
requires a special kind of gaussian process and I need it to run on GPU. These packages are not compatible with GPU hardware, so 
this is my best crack at an implementation. 

