# File to define the mathematical kernel operator

struct KernelOperator{F,Θ}
    f::F # f is a function that takes in first an array of parameters (θ), and then two vectors, x and y. It computes some kernel function between the vectors
    θ::Θ
end

function KernelOperator(f, θ)
    Ker = KernelOperator{typeof(f),typeof(θ)}(f, θ)
    if !isbitstype(typeof(Ker))
        throw(ArgumentError("KernelOperator fields must be of bits type to be GPU compatable"))
    end
    return Ker
end

function (Ker::KernelOperator)(X, Y, i::Int, j::Int)
    # X and Y are vectors of vectors
    return Ker.f(Ker.θ, X, Y, i, j)
end
