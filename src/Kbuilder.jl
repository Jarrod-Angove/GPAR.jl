using AMDGPU, LinearAlgebra, KernelAbstractions

function get_test_x(n, N)
    times = collect(1:n)./10.0
    X = Matrix{Float32}(undef, n, N)
    for i in 1:N
    X[:, i] = sqrt(i) .* times .+ randn()/20
    end
    return X
end
export get_test_x

function build_tensor(X , Δt)
    # T likely needs to be a N,N,n,d matrix if 
    # you want to use input with more dimensions
    n, N = size(X)
    T = AMDGPU.zeros(N,N,n)
    @kernel function build_tensor_kernel!(T, @Const(X))
        i,j,k = @index(Global, NTuple)
        
        @inbounds T[i,j,k] = (X[k, i] - X[k, j])^2 * Δt
    end

    backend=KernelAbstractions.get_backend(T)
    kernel! = build_tensor_kernel!(backend)
    kernel!(T, X; ndrange = size(T))
    
    @kernel function scan_sum_kernel!(T, T_out)
        i,j = @index(Global, NTuple)
        #k = @index(Local, Linear)
        @inbounds for k in 1:n
            if k == 1
                T_out[i,j,k] = T[i,j,k]
            else
                T_out[i,j,k] = T[i,j,k] + T_out[i,j,k-1]
            end
            @synchronize
        end
    end
    scan_sum_kernel! = scan_sum_kernel!(backend)

    T_out = KernelAbstractions.zeros(backend, Float32, N,N,n)
    scan_sum_kernel!(T,T_out; ndrange=(N,N))
    KernelAbstractions.unsafe_free!(T)

    return T_out
end
export build_tensor

function build_tensor_cpu(X::AbstractArray{Float32}, Δt)::AbstractArray{Float32, 3}
    n,N = size(X)
    T = [(X[k,i] - X[k,j])^2 * Δt for i in 1:N, j in 1:N, k in 1:n]
    T_out = zeros(size(T)...)
    for k in 1:n
        if k == 1
            T_out[:,:,k] .= T[:,:,k]
        else
            T_out[:,:,k] .= T[:,:,k] .+ T_out[:,:, k-1]
        end
    end
    T_out = exp.(-0.5 .* T_out/100.0)

    return T_out
end
export build_tensor_cpu

# Untested!!!
function build_K(X, t, θ)
    dt = diff(t); 
    for i in eachindex(dt)
        if dt[i] !== dt[1]
            throw("Times are not evenly spaced")
        end
    end
    Δt = dt[1]
    T = build_tensor(X, Δt)
    K = θ[1]^2 .* exp.(-0.5 .* T ./θ[2]^2)
    return K
end

# UNOPTIMIZED and untested!!!!!
function lml_K(X, Y, t, θ)
    n, N = size(X)
    K = build_K(X, t, θ)
    # Might want to move everything to shared memory here
    logp = Vector{Float32}(undef, n)
    for k in 1:n
        y = Y[k, :]
        L = cholesky(K[:,:,k])
        α = L.U \ (L.L \ y)
        logp[k] = -0.5 .* y' * α .- tr(L.L) .- n/2 * log(2π)
    end
    return logp
end




