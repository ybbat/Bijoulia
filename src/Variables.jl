module Variables

export Variable

mutable struct Variable
    data::Union{Array{Float64}, Float64}
    grad::Union{Array{Float64}, Float64}
    children::Set{Variable}
    backward::Function

    function Variable(data::Array{Float64}, children::Set{Variable}, backward::Function)
        if typeof(data) <: Vector
            data = reshape(data, (length(data), 1))
        end
        new(data, zeros(size(data)), children, backward)
    end

    function Variable(data::Float64, children::Set{Variable}, backward::Function)
        new(data, zero(Float64), children, backward)
    end

    function Variable(data::Array{Float64})
        if typeof(data) <: Vector
            data = reshape(data, (length(data), 1))
        end
        new(data, zeros(size(data)), Set(), Returns(nothing))
    end

    function Variable(data::Float64)
        new(data, zero(Float64), Set(), Returns(nothing))
    end

    function Variable(data::Array{<:Real}, children::Set{Variable}, backward::Function)
        if typeof(data) <: Vector
            data = reshape(data, (length(data), 1))
        end
        new(convert.(Float64, data), zeros(size(data)), children, backward)
    end

    function Variable(data::Real, children::Set{Variable}, backward::Function)
        new(convert(Float64, data), zero(Float64), children, backward)
    end

    function Variable(data::Array{<:Real})
        if typeof(data) <: Vector

            data = reshape(data, (length(data), 1))
        end
        new(convert.(Float64, data), zeros(size(data)), Set(), Returns(nothing))
    end

    function Variable(data::Real)
        new(convert(Float64, data), zero(Float64), Set(), Returns(nothing))
    end
end

# TODO: allow broadcasting over Variable so all this logic isnt needed for each operation
function broadcast_dims(a, b)::Union{Array{Float64}, Float64}
    size_a = size(a)
    size_b = size(b)
    if size_a == size_b
        return b
    end

    size_a = [i <= length(size_a) ? size_a[i] : 1 for i ∈ 1:length(size_b)]

    indices = []
    for (i, (x, y)) ∈ enumerate(collect(zip(size_a, size_b)))
        if x == 1 && y > 1
            push!(indices, i)
        end
    end

    summed = sum(b, dims=indices)
    if length(summed) == 1
        return summed[1]
    else
        return summed
    end
end

function Base.:+(a::Variable, b::Variable)
    if size(a.data) == size(b.data)
        back_func(p_grad) = (a.grad += p_grad; b.grad += p_grad)
        Variable(a.data + b.data, Set([a, b]), back_func)
    else
        try
            broadcast_add = a.data .+ b.data
            bback_func(p_grad) = begin
                a.grad = a.grad + broadcast_dims(a.grad, p_grad)
                b.grad = b.grad + broadcast_dims(b.grad, p_grad)
            end
            Variable(broadcast_add, Set([a, b]), bback_func)
        catch
            throw(DimensionMismatch("Could not add variables of size $(size(a.data)) and $(size(b.data))"))
        end
    end
end

function Base.:*(a::Variable, b::Variable)
    # scalar * scalar
    if typeof(a.data) == Float64 && typeof(b.data) == Float64
        backward_ss(p_grad) = (a.grad += p_grad * b.data; b.grad += p_grad * a.data)
        Variable(a.data * b.data, Set([a, b]), backward_ss)
    # scalar * array
    elseif typeof(a.data) == Float64 || typeof(b.data) == Float64
        backwards_sa(p_grad) = begin
            if typeof(a.grad) == Float64
                a.grad += sum(p_grad .* b.data)
                b.grad += p_grad .* a.data
            else
                a.grad += p_grad .* b.data
                b.grad += sum(p_grad .* a.data)
            end
        end
        Variable(a.data * b.data, Set([a, b]), backwards_sa)
    # array * array
    elseif last(size(a.data)) == first(size(b.data))
        mul = a.data * b.data
        if length(mul) == 1
            mul = mul[1]
        end
        backwards_aa(p_grad) = begin
            a.grad += p_grad * b.data'
            b.grad += a.data' * p_grad
        end
        Variable(mul, Set([a, b]), backwards_aa)
    end
end

function Base.adjoint(a::Variable)
    backward(p_grad) = a.grad += p_grad'
    Variable(copy(a.data'), Set([a]), backward)
end

function Base.:^(a::Variable, b::Number)
    backward(p_grad) = a.grad += b .* p_grad .* a.data.^(b - 1)
    Variable(a.data.^b, Set([a]), backward)
end

function Base.:-(a::Variable)
    backward(p_grad) = a.grad -= p_grad
    Variable(-a.data, Set([a]), backward)
end

end # module Variables