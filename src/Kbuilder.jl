using KernelAbstractions, AMDGPU, LinearAlgebra

function build_K(X::AbstractMatrix{<:Real}, times::AbstractVector{<:Real}, N::Int, n::Int)
    T = KernelAbstractions.zeros(get_backend(X), N, N, n)
    @kernel function build_tensor!(T, @Const(X), @Const(times))
        i,j,k = @index(Global, NTuple)
        T[i,j,k] = i + j + k
    end
    return T
end

function build_tensor!(X , Δt)
    n, N = size(X)
    T = AMDGPU.zeros(N,N,n)
    @kernel function build_tensor_kernel!(T, @Const(X))
        i,j,k = @index(Global, NTuple)
        
        # The lower and upper triangular portions
        # will be equal so we don't need to calc both
        #if j ≥ i
        @inbounds T[i,j,k] = (abs(X[k, i] - X[k, j])) * Δt
        #end
        
        for m in 2:size(X)[1]
            @inbounds T[i,j,m] = T[i,j,m] + T[i, j, m - 1]
            @synchronize
        end
    end

    backend=KernelAbstractions.get_backend(T)
    kernel! = build_tensor_kernel!(backend)
    kernel!(T, X; ndrange = size(T))
    
    # This is the slowest possible way to do this
    # Look up "parallel prefix sum kernel" if this
    # is a bottleneck
    #for k in 2:size(T)[3]
    #        T[:,:,k] = T[:,:,k] .+ T[:,:,k-1]
    #end

    #for k in 1:size(T)[3]
    #    cholesky!(T[:,:,k])
    #end
    return T
end
export build_tensor!

