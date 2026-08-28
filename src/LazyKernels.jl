module LazyKernels
include("KernelOperator.jl")
include("ExecutionPlan.jl")
using KernelAbstractions

struct LazyKernelMatrix{T,Ke} <: AbstractMatrix{T}
    X::AbstractArray{T} # X points
    Y::AbstractArray{T} # Y points
    Ker::Ke
    Plan::ExecutionPlan
end

Base.size(K::LazyKernelMatrix) = (size(K.X, 1), size(K.Y, 1))

function apply!(y::AbstractVector, K::LazyKernelMatrix, v::AbstractVector)
    # check dimensions
    if (size(K, 2) != length(v)) || (size(K, 1) != length(y))
        throw(ArgumentError("Matrix-vector size missmatch. LazyKernelMatrix is $(size(K,1)) x $(size(K,2)), input vector is length $(length(v)), output vector is length $(length(y))"))
    end

    @kernel function mulker!(y, Ker, X, Y, v)
        i = @index(Global)
        tmp_sum = zero(eltype(y))
        for j in 1:size(Y, 1)
            tmp_sum += Ker(X, Y, i, j) * v[j]
        end
        y[i] = tmp_sum
    end

    # check for backend mismatch
    if (get_backend(K.X) != K.Plan.backend) || (get_backend(K.Y) != K.Plan.backend)
        throw(ArgumentError("Backends do not match. X is on $(get_backend(K.X)), Y is on $(get_backend(K.Y)), and the LazyKernelMatrix object has backend specified as $(K.Plan.backend)"))
    end
    if (get_backend(v) != K.Plan.backend)
        throw(ArgumentError("input vector is not on the same device as is specified by LazyKernelMatrix"))
    end
    if (get_backend(y) != K.Plan.backend)
        throw(ArgumentError("output vector is not on the same device as is specified by LazyKernelMatrix"))
    end

    kernel! = mulker!(K.Plan.backend, K.Plan.workgroup_n)
    kernel!(y, K.Ker, K.X, K.Y, v; ndrange=length(y))
    synchronize(K.Plan.backend)
end

export LazyKernelMatrix
export apply!
export KernelOperator, ExecutionPlan
end
