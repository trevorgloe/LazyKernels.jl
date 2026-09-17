# File containing the definitions for LazyKernels execution plans

abstract type AbstractExecutionPlan end

struct DirectExecutionPlan <: AbstractExecutionPlan
    backend::KernelAbstractions.Backend
    workgroup_n::Int
end

struct BlockedExecutionPlan <: AbstractExecutionPlan
    backend::KernelAbstractions.Backend
    workgroup_n::Int
    block_size::Int
end

# ExecutionPlan implementation

"""
Direct multiplication - multiply Av = y, distributing the calculation of y_i = sum_j A_{i,j}v_j over threads, so each i gets a different thread. I.e. distribute over rows of A
"""
function apply!(Plan::DirectExecutionPlan, Ker::KernelOperator, X::AbstractArray, Y::AbstractArray, y::AbstractVector, v::AbstractVector)
    # direct multiplication kernel    
    @kernel function mulker!(y, Ker, X, Y, v)
        i = @index(Global)
        tmp_sum = zero(eltype(y))
        for j in 1:size(Y, 1)
            tmp_sum += Ker(X, Y, i, j) * v[j]
        end
        y[i] = tmp_sum
    end

    kernel! = mulker!(Plan.backend, Plan.workgroup_n)
    kernel!(y, Ker, X, Y, v; ndrange=length(y))
    synchronize(Plan.backend)
end


