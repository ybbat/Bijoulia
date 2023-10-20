using Test
using Bijoulia.Variables

@testset "Variable constructors" begin
    @testset "construct from float array" begin
        a = Variable([1. 2. 3.; 4. 5. 6.])
        @test a.data == [1. 2. 3.; 4. 5. 6.]
        @test a.grad == [0. 0. 0.; 0. 0. 0.]
        @test a.children == Set()
        @test a.backward == Returns(nothing)
    end

    @testset "construct from other <:Real array" begin
        a = Variable([1 2 3; 4 5 6; 7 8 9])
        @test a.data == [1. 2. 3.; 4. 5. 6.; 7. 8. 9.]
        @test a.grad == [0. 0. 0.; 0. 0. 0.; 0. 0. 0.]
        @test a.children == Set()
        @test a.backward == Returns(nothing)
    end

    @testset "construct from data, children, function" begin
        s = Set([Variable([1.]), Variable([2.])])
        test_func() = 1.
        a = Variable([1. 2. 3.; 4. 5. 6.; 7. 8. 9.], s, test_func)
        @test a.data == [1. 2. 3.; 4. 5. 6.; 7. 8. 9.]
        @test a.grad == [0. 0. 0.; 0. 0. 0.; 0. 0. 0.]
        @test a.children == s
        @test a.backward == test_func
    end

    @testset "construct from scalar" begin
        a = Variable(1.0)
        @test a.data == 1.0
        @test a.grad == 0.0
        @test a.children == Set()
        @test a.backward == Returns(nothing)
    end

    @testset "construct from other <:Real" begin
        a = Variable(1)
        @test a.data == 1.0
        @test a.grad == 0.0
        @test a.children == Set()
        @test a.backward == Returns(nothing)
    end
end

@testset "adding variables" begin
    @testset "basic add behaviour" begin
        @test (Variable(2) + Variable(5)).data == 7.0
        @test (Variable([1 2 3]) + Variable(5)).data == [6. 7. 8.]
        @test (Variable([1 2 3]) + Variable([4 5 6])).data == [5. 7. 9.]
        @test_throws DimensionMismatch (Variable([1 2 3]) + Variable([4 5]))
    end

    @testset "basic backward behaviour" begin
        a = Variable(2)
        b = Variable(5)
        c = a + b
        c.backward(1.0)
        @test a.grad == 1.0
        @test b.grad == 1.0

        a = Variable([1 2 3])
        b = Variable([4 5 6])
        c = a + b
        c.backward([1. 1. 1.])
        @test a.grad == [1. 1. 1.]
        @test b.grad == [1. 1. 1.]

    end

    @testset "gradiant correct when broadcasting" begin
        a = Variable([1 2 3])
        b = Variable(1)
        c = a + b
        c.backward([1.0 1.0 1.0])
        @test a.grad == [1. 1. 1.]
        @test b.grad == 3.

        a = Variable([1 2 3; 4 5 6])
        b = Variable([1 2 3])
        c = a + b
        c.backward([1. 1. 1.; 1. 1. 1.])
        @test a.grad == [1. 1. 1.; 1. 1. 1.]
        @test b.grad == [2. 2. 2.]
    end
end