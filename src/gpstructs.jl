
abstract type ModInput end

struct GPARinput{T<:Real} <: ModInput
    response::AbstractVector{T} # the 'y' value
    input::AbstractMatrix{T}    # the 'X' value. data in an nxN matrix, where n is the length of the time vector
    t::AbstractVector{T}        # time must be on a regular grid (effectively acts as an index)
end
export GPARinput

struct GPinput{T<:Real} <: ModInput
    response::AbstractVector{T}
    input::AbstractMatrix{T}
end
export GPinput

struct GPA{T}
    k::Function     # Kernel function
    θk::Vector{T}   # Kernel function hyperparameters
    μ::Function     # Mean function
    θμ::Vector{T}   # Mean function hyperparameters
    σₙ::T           # signal noise standard deviation
end
export GPA

struct GPE{T<:Real}
    K::AbstractMatrix{T} # Covariance matrix
    μ::AbstractVector{T} # Mean function evaluated on inputs
end
export GPE


function build_GPE(gpa::GPA, input::GPinput)
    X = [i for i in eachrow(input.input)] 
    g = gpa.k
    k(x1,x2) = g(x1,x2, gpa.θk)
    K = [k(x1, x2) for x1 in X, x2 in X] .+ gpa.σₙ^2
    μ = gpa.μ.(X)
    return GPE(K, μ)
end
export build_GPE
