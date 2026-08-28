using LinearAlgebra
using LazyKernels
using KernelAbstractions
using CUDA

n = 5000
println("Starting example")
X = randn(Float32, n, 2)
Y = randn(Float32, n, 2)
function RBF(θ, X, Y, I, J)
        normsq = zero(eltype(X))
        for i in 1:size(X, 2)
                normsq += (X[I, i] - Y[J, i])^2
        end
        return θ[1] * exp(-normsq / θ[2])
end
θ = (1.0f0, 1.0f0)
Ker = KernelOperator(RBF, θ)
back = CPU()
plan = ExecutionPlan(back, 64)
K = LazyKernelMatrix(X, Y, Ker, plan)
v = randn(Float32, n)
y = zeros(Float32, n)
println("Executing CPU kernel")
apply!(y, K, v)

testK = zeros(Float32, n, n)
for i in 1:n
        for j in 1:n
                testK[i, j] = RBF(θ, X, Y, i, j)
        end
end
testy = zeros(Float32, n)
mul!(testy, testK, v)
# println(y)
# println(testy)
println(maximum(abs.(y - testy)))

println("Executing GPU kernel")
Xcu = CuArray(X)
Ycu = CuArray(Y)
back2 = get_backend(Xcu)
plan2 = ExecutionPlan(back2, 64)
K2 = LazyKernelMatrix(Xcu, Ycu, Ker, plan2)
y2 = CUDA.zeros(n)
apply!(y2, K2, CuArray(v))
y2cpu = Array(y2)
println(maximum(abs.(y - y2cpu)))
