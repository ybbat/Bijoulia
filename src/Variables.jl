module Variables

export Variable

mutable struct Variable
    data::Union{Array{Float64}, Float64}
    grad::Union{Array{Float64}, Float64}
    children::Set{Variable}
    backward::Function

    function Variable(data::Array{Float64}, children::Set{Variable}, backward::Function)
        new(data, zeros(size(data)), children, backward)
    end

    function Variable(data::Float64, children::Set{Variable}, backward::Function)
        new(data, zero(Float64), children, backward)
    end

    function Variable(data::Array{Float64})
        new(data, zeros(size(data)), Set(), Returns(nothing))
    end

    function Variable(data::Float64)
        new(data, zero(Float64), Set(), Returns(nothing))
    end

    function Variable(data::Array{<:Real}, children::Set{Variable}, backward::Function)
        new(convert.(Float64, data), zeros(size(data)), children, backward)
    end

    function Variable(data::Real, children::Set{Variable}, backward::Function)
        new(convert(Float64, data), zero(Float64), children, backward)
    end

    function Variable(data::Array{<:Real})
        new(convert.(Float64, data), zeros(size(data)), Set(), Returns(nothing))
    end

    function Variable(data::Real)
        new(convert(Float64, data), zero(Float64), Set(), Returns(nothing))
    end
end

function broadcast_dims(a, b)::Union{Array{Float64}, Float64}
    size_a = size(a)
    size_b = size(b)

    size_a = [i <= length(size_a) ? size_a[i] : 1 for i ∈ 1:length(size_b)]

    indices = []
    for (i, (x, y)) ∈ enumerate(collect(zip(size_a, size_b)))
        if x == 1 && y > 1
            push!(indices, i)
        end
    end

    if indices != []
        summed = sum(b, dims=indices)
        if length(summed) == 1
            return summed[1]
        else
            return summed
        end
    else
        return b
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

end # module Variables