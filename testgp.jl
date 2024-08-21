using LinearAlgebra

X = collect(0:1.0:100.0)
function gen_test_data(;X=X)
    times = X
    noise = rand(length(times))
    f = sqrt.((sin.(sqrt.(times))).^4) .+ noise/2   
end
export gen_test_data

y = gen_test_data

function get_dmat(X)
    D = [(i - j) for i in X, j in X]
end

function build_K(D, θ)
    n = size(D, 1)
    K = θ[1] .* exp.(-0.5 .* D.^2 ./θ[2]^2)
    σ_noise = diagm(ones(n)) .* θ[3]
    K = K .+ σ_noise
    return K
end

function nlogp(D, y, θ)
    n = size(D, 1)
    K = build_K(D, θ)
    L = cholesky(K)
    α = L.U \ (L.L \ y)
    logp = 1/2 * (y' * α) + tr(log.(L.L)) + n/2 * log(2π)
    return logp
end

function build_loss(D, y)
    loss(θ, p) = nlogp(D, y, θ)
    return loss
end

function predict_y(X, y, X_s, θ)
    D_ss = [(i - j) for i in X, j in X_s]
    D = get_dmat(X)
    K_ss = θ[1] .* exp.(-0.5 .* D_ss.^2 ./θ[2]^2)
    K = build_K(D, θ)
    L = cholesky(K)
    α = L.U \ (L.L \ y)
    μ = K_ss' * α
    return μ
end
