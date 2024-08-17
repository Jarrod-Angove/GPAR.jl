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
    #TODO: Add signal noise
    backend=KernelAbstractions.get_backend(X)
    n, N = size(X)
    nzeros = KernelAbstractions.zeros(backend, Float32, N,N)
    noise_tensor = repeat(nzeros + I.*θ[3], 1, 1, n)
    σ = θ[1]; l = θ[2];
    dt = diff(t); 
    dtc = Array(dt)
    for i in eachindex(dtc)
        if dtc[i] !== dtc[1]
            throw("Times are not evenly spaced")
        end
    end
    Δt = dtc[1]
    T = build_tensor(X, Δt)
    K = σ^2 .* exp.(-0.5 .* T ./l^2) .+ noise_tensor
    return K
end
export build_K

# UNOPTIMIZED and untested!!!!!
# Look into Krylov.jl for generating α
function lml_K(X, Y, t, θ)
    n, N = size(X)
    K = build_K(X, t, θ)

    # Might want to move everything to shared memory here
    logp_v = Vector{Float32}(undef, n)
    for k in 1:n
        y = Y[k, :]
        nK = K[:, :, k]
        try
           lp = logp(nK, y, N)
           logp_v[k] = lp
        catch err
            logp_v[k] = -1000000.0
            println("timestep $k failed -- probably due to posdef error")
            println(err)
        end
    end
    return -sum(logp_v)
end
export lml_K

function logp(k, y, N)
    L = cholesky(k)
    α = L.U \ (L.L \ y)
    p = -0.5 * y' * α - tr(L.L) - N/2 * log(2π)
    return p
end
export logp

function build_loss_function(X, Y, t)
    return loss(θ, p) = lml_K(X, Y, t, θ)
end
export build_loss_function

