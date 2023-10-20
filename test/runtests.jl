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