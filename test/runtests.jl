using LazyKernels
using Test
using KernelAbstractions
using LinearAlgebra

@testset "RBF kernel tests" begin
        n = 20
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
        plan = DirectExecutionPlan(back, 64)
        K = LazyKernelMatrix(X, Y, Ker, plan)
        v = randn(Float32, n)
        y = zeros(Float32, n)
        testK = zeros(Float32, n, n)
        for i in 1:n
                for j in 1:n
                        testK[i, j] = RBF(θ, X, Y, i, j)
                end
        end
        testy = zeros(Float32, n)
        mul!(y, K, v) # lazy kernel
        mul!(testy, testK, v) # dense kernel
        @test maximum(abs.(y - testy)) < 1e-3

        vbad = randn(Float32, n + 1)
        ybad = zeros(Float32, n + 1)
        @test_throws ArgumentError mul!(y, K, vbad)
        @test_throws ArgumentError mul!(ybad, K, v)
end
