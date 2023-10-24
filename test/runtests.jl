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

@testset "operators" begin
    @testset "addition" begin
        @testset "basic add behaviour" begin
            @test (Variable(2) + Variable(5)).data == 7.0
            @test (Variable([1 2 3]) .+ Variable(5)).data == [6. 7. 8.]
            @test (Variable([1 2 3]) .+ Variable([4 5 6])).data == [5. 7. 9.]
            @test_throws DimensionMismatch (Variable([1 2 3]) + Variable([4 5]))
            @test_throws DimensionMismatch (Variable([1 2 3]) .+ Variable([4 5]))
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
            c = a .+ b
            c.backward([1.0 1.0 1.0])
            @test a.grad == [1. 1. 1.]
            @test b.grad == 3.

            a = Variable([1 2 3; 4 5 6])
            b = Variable([1 2 3])
            c = a .+ b
            c.backward([1. 1. 1.; 1. 1. 1.])
            @test a.grad == [1. 1. 1.; 1. 1. 1.]
            @test b.grad == [2. 2. 2.]
        end
    end

    @testset "multiplication" begin
        @testset "basic multiplication behaviour" begin
            @test (Variable(2) * Variable(5)).data == 10.
            @test (Variable([1 2 3]) * Variable(5)).data == [5. 10. 15.]
            @test (Variable([1 2 3]) * Variable([1;2;3])).data == 14.
            @test (Variable([1;2;3]) * Variable([1 2 3])).data == [1. 2. 3.; 2. 4. 6.; 3. 6. 9.]
            @test (Variable([1 2 3; 4 5 6]) * Variable([4 5 6; 1 2 3; 8 9 10])).data == [30. 36. 42.; 69. 84. 99.]
            @test size((Variable(rand(10, 5)) * Variable(rand(5, 15))).data) === (10, 15)
        end

        @testset "multiplication backwards" begin
            a = Variable(2)
            b = Variable(5)
            c = a * b
            c.backward(1.0)
            @test a.grad == 5.0
            @test b.grad == 2.0

            a = Variable([1 2 3])
            b = Variable(5)
            c = a * b
            c.backward([1. 1. 1.])
            @test a.grad == [5. 5. 5.]
            @test b.grad == 6.

            a = Variable([7 2 2; 9 6 5])
            b = Variable(8)
            c = a * b
            c.backward([1. 1. 1.; 1 1 1])
            @test a.grad == [8. 8. 8.; 8. 8. 8.]
            @test b.grad == 31.

            a = Variable([7 5 8; 8 3 9])
            b = Variable([2 4 8 8; 5 6 8 3; 9 9 5 8])
            c = a * b
            c.backward([1. 1. 1. 1.; 1. 1. 1. 1.])
            @test a.grad == [22. 22. 31.; 22. 22. 31.]
            @test b.grad == [15. 15. 15. 15.; 8. 8. 8. 8.; 17. 17. 17. 17]
        end
    end

    @testset "transpose" begin
        a = Variable([1 2 3; 4 5 6])
        @test a'.data == [1. 4.; 2. 5.; 3. 6.]
        @test a'.backward([1. 1.; 1. 1.; 1. 1.;]) == [1. 1. 1.; 1. 1. 1.]
    end

    @testset "pow" begin
        @test (Variable(2)^2).data == 4.
        a = Variable([1 2 3; 4 5 6])
        @test (a^2).data == [1. 4. 9.; 16. 25. 36.]
        @test (a^3).data == [1. 8. 27.; 64. 125. 216.]

        c = a^2
        c.backward([1. 1. 1.; 1. 1. 1.])
        @test a.grad == [2. 4. 6.; 8. 10. 12.]

        a = Variable([6 8 2; 3 1 8; 3 8 3])
        c = a ^ 3
        c.backward([1. 1. 1.; 1. 1. 1.; 1. 1. 1.])
        @test a.grad == [108. 192 12.; 27. 3. 192.; 27. 192. 27.]
    end

    @testset "negation" begin
        @test (-Variable(2)).data == -2.
        @test (-Variable([1 2 3])).data == [-1. -2. -3.]
        @test (-Variable([1 2 3; 4 5 6])).data == [-1. -2. -3.; -4. -5. -6.]
    end
end

@testset "backpropagate" begin
    @testset "scalar variable reuse" begin
        a = Variable(-2)
        b = Variable(3)
        c = a * b
        d = a + b
        e = c + d
        backpropagate(e)
        @test a.grad == 4.
        @test b.grad == -1.
        @test c.grad == 1.
        @test d.grad == 1.
        @test e.grad == 1.
    end

    @testset "simple matrix backprop" begin
        x = Variable([1; 2])
        W = Variable([-3. 0.5])
        b = Variable([1.])
        z = W * x + b
        a = tanh(z)
        backpropagate(a)
        @test isapprox.(x.grad, [-1.26; 0.21], atol=0.01) |> all
        @test isapprox.(W.grad, [0.42 0.84], atol=0.01) |> all
    end
end