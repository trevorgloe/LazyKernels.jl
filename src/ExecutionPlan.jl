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


"""
Blocked multiplication - split the matrix up into blocks of size block_size x block_size. Compute each block A_{i,j}x_J with a different thread. Then add up the blocked resultant vector. Adds additional step for summation, but involves less sharing of the x (each thread row only needs a subset of x
"""
function apply!(Plan::BlockedExecutionPlan, Ker::KernelOperator, X::AbstractArray, Y::AbstractArray, y::AbstractVector, v::AbstractVector)
    # Stage 1:
    # block multiply kernel
    @kernel function stage1ker!(res, Ker, X, Y, v, d, r)
	# d is the number of blocks, r is the size of each block
	# res is an nxd array which will contain the outputs
	i,j = @index(Global, NTuple)
	for row in r*(i-1)+1:r*i
	    tmp_sum = zero(eltype(res))
	    for col in r*(j-1)+1:r*j
		tmp_sum += Ker(X, Y, row, col) * v[j]
	    end
	    res[row, j] = tmp_sum
	end
    end

    stage1kernel! = stage1ker!(Plan.backend, Plan.workgroup_n)
    r = Plan.block_size # block size
    d = ceil(Int, length(y)/r) # number of blocks, note that the dimensions should have been checked by now
    println(Plan.backend)
    println(eltype(y))
    println(length(y))
    println(d)
    res = KernelAbstractions.zeros(Plan.backend, eltype(y), length(y), d)
    stage1kernel!(res, Ker, X, Y, v, d, r; ndrange=d^2)


end
