module LazyKernels
using KernelAbstractions
using LinearAlgebra

include("KernelOperator.jl")
include("ExecutionPlan.jl")
using KernelAbstractions

struct LazyKernelMatrix{T,Ke,P<:AbstractExecutionPlan} <: AbstractMatrix{T}
    X::AbstractArray{T} # X points
    Y::AbstractArray{T} # Y points
    Ker::Ke
    Plan::P
end

Base.size(K::LazyKernelMatrix) = (size(K.X, 1), size(K.Y, 1))

function LinearAlgebra.mul!(y::AbstractVector, K::LazyKernelMatrix, v::AbstractVector)
    # check dimensions
    if (size(K, 2) != length(v)) || (size(K, 1) != length(y))
        throw(ArgumentError("Matrix-vector size missmatch. LazyKernelMatrix is $(size(K,1)) x $(size(K,2)), input vector is length $(length(v)), output vector is length $(length(y))"))
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
    apply!(K.Plan, K.Ker, K.X, K.Y, y, v)
    return y
end

export LazyKernelMatrix
export apply!
export KernelOperator, DirectExecutionPlan, BlockedExecutionPlan
end
