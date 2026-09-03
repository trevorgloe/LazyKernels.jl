# LazyKernels.jl

[![Build Status](https://github.com/trevorgloe/LazyKernels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/trevorgloe/LazyKernels.jl/actions/workflows/CI.yml?query=branch%3Amain)

A julia package to easily implement lazily evaluated, non-allocating kernel matrices on the GPU/CPU through a platform-agnostic interface.

## TO DO
- [ ] Overload regular linear algebra `mul!` and test with Krylov.jl
- [ ] Add symmetric variant so only one set of points is stored
- [ ] Test with KernelFunctions.jl to see if we can get compatability
- [ ] Add batched matrix multiply and move low level kernels to ExecutionPlan.jl
- [ ] Add tests

## TO DO (further down the road)
- [ ] Sparse approximation methods
    - [ ] Hierarchical matrices (possibly integrate with Hmatrices.jl
    - [ ] Nystrom approximation (possibly integrate with RPCholseky.jl or AcceleratedRPCholseky.jl)
- [ ] Log determinant calculation
- [ ] $tr(A^{-1}B)$ calculation
