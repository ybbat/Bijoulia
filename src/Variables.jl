module Variables

export Variable, backpropagate!, zero_grads!

#
# Variable definitions and constructors
#

abstract type Variable end

mutable struct VariableArray <: Variable
    data::Array{Float64}
    grad::Array{Float64}
    children::Set{Variable}
    backward::Function

    VariableArray(data) = new(data, zeros(size(data)), Set(), Returns(nothing))
    VariableArray(data, children, backward) = new(data, zeros(size(data)), children, backward)
end

mutable struct VariableScalar <: Variable
    data::Float64
    grad::Float64
    children::Set{Variable}
    backward::Function

    VariableScalar(data) = new(data, zero(Float64), Set(), Returns(nothing))
    VariableScalar(data, children, backward) = new(data, zero(Float64), children, backward)
end

function Variable(data::Array{<:Real})
    data = convert.(Float64, data)
    length(data) == 1 && return VariableScalar(data[1])
    VariableArray(data)
end

function Variable(data::Array{<:Real}, children::Set{<:Variable}, backward::Function)
    data = convert.(Float64, data)
    length(data) == 1 && return VariableScalar(data[1], children, backward)
    VariableArray(data, children, backward)
end

function Variable(data::Real)
    data = convert(Float64, data)
    VariableScalar(data)
end

function Variable(data::Real, children::Set{<:Variable}, backward::Function)
    data = convert(Float64, data)
    VariableScalar(data, children, backward)
end

#
# Base methods
#

Base.size(v::Variable) = size(v.data)
Base.size(v::Variable, d::Integer) = size(v.data, d)
Base.length(v::Variable) = length(v.data)
Base.getindex(v::Variable, inds...) = v.data[inds...]
Base.setindex!(v::Variable, val, inds...)= v.data[inds...] = val
Base.firstindex(v::Variable) = firstindex(v.data)
Base.lastindex(v::Variable) = lastindex(v.data)
Base.iterate(v::Variable) = iterate(v.data)
Base.iterate(v::Variable, state=firstindex(v.data)) = iterate(v.data, state)
Base.eltype(v::Variable) = eltype(v.data)

#
# Printing and string representations
#

function Base.show(io::IO, v::Variable)
    io_buff = IOBuffer()
    buffer_context = IOContext(io_buff, :limit => true)
    print(buffer_context, v.data)
    limited_data = String(take!(io_buff))
    print(buffer_context, v.grad)
    limited_grad = String(take!(io_buff))

    print(io, "Variable(data=$(limited_data), grad=$(limited_grad))")

end

# This method is used when printing arrays to the REPL implicitly (not using print)
function Base.show(io::IO, m::MIME"text/plain", v::VariableArray)
    dims = join(string.(size(v)), "×")
    println(io, "$dims VariableArray{$(eltype(v))}:")

    # I trust the std lib to display matrices within the REPL better than I can
    remove_firstline = s -> join(split(s, "\n")[2:end], "\n")
    io_buff = IOBuffer()
    buffer_context = IOContext(io_buff, :limit => true, :compact => true)
    show(buffer_context, m, v.data)
    limited_data = take!(io_buff) |> String |> remove_firstline
    show(buffer_context, m, v.grad)
    limited_grad = take!(io_buff) |> String |> remove_firstline

    println(io, "data =\n", limited_data)
    println(io, "grad =\n", limited_grad)
end

#
# Graph traversal
#

function dfs(root::Variable,
             visited::Set{Variable}=Set{Variable}(),
             sorted::Array{Variable}=Array{Variable}([]))
    if root ∈ visited
        return
    end
    push!(visited, root)
    for child ∈ root.children
        dfs(child, visited, sorted)
    end
    push!(sorted, root)

    return sorted
end

function backpropagate!(root::VariableArray)
    fill!(root.grad, one(Float64))

    for v ∈ reverse(dfs(root))
        v.backward(v.grad)
    end
end

function backpropagate!(root::VariableScalar)
    root.grad = one(Float64)

    for v ∈ reverse(dfs(root))
        v.backward(v.grad)
    end
end

function zero_grads!(root::Variable)
    for v ∈ dfs(root)
        zero_grad!(v)
    end
end

zero_grad!(v::VariableArray) = v.grad = zeros(size(v.grad))
zero_grad!(v::VariableScalar) = v.grad = zero(Float64)

#
# Basic custom broadcast methods
#

struct VariableStyle <: Broadcast.BroadcastStyle end
Base.BroadcastStyle(::Type{<:Variable}) = VariableStyle()

Base.BroadcastStyle(::VariableStyle, ::Broadcast.BroadcastStyle) = VariableStyle()

function Base.similar(bc::Broadcast.Broadcasted{VariableStyle}, ::Type{ElType}) where {ElType}
    if all(typeof.(bc.args) .== VariableScalar)
        Variable(zero(Float64))
    else
        Variable(similar(Array{ElType}, axes(bc)))
    end
end

function unbroadcast(a::VariableArray, b::Union{AbstractArray,Number})
    dims = findall(size(b) .> size(a))
    sum(b, dims=dims)
end

function unbroadcast(::VariableScalar, b::Union{AbstractArray,Number})
    return sum(b)
end

Base.broadcastable(v::Variable) = v

# Swap position of "a" and "b" arguments in function
# TODO: could probably work for more than just a, b by creating all permutations of specified arguments
macro commutative(ex)
    func_call = ex.args[1]
    body = ex.args[2]
    func_name, func_args... = func_call.args

    indices = findall(e -> e.args[1] ∈ (:a, :b), func_args)
    @assert length(indices) == 2
    swapped = copy(func_args)
    swapped[indices[1]], swapped[indices[2]] = func_args[indices[2]], func_args[indices[1]]

    return quote
        $ex
        function $(func_name)($(swapped...))
            $body
        end
    end
end

#
# Adding
#

function backward(::typeof(+), a::Variable, b::Variable)
    function ret(pgrad::Union{AbstractArray,Number})
        a.grad += unbroadcast(a, pgrad)
        b.grad += unbroadcast(b, pgrad)
    end
end

function Base.broadcasted(::VariableStyle, ::typeof(+), a::Variable, b::Variable)
    back = backward(+, a, b)
    Variable(a.data .+ b.data, Set([a, b]), back)
end

function Base.:+(a::Variable, b::Variable)
    back = backward(+, a, b)
    Variable(a.data + b.data, Set([a, b]), back)
end

function backward(::typeof(+), a::Variable)
    function ret(pgrad::Union{AbstractArray,Number})
        a.grad += unbroadcast(a, pgrad)
    end
end

@commutative function Base.broadcasted(::VariableStyle, ::typeof(+), a::Variable, b::Union{AbstractArray,Number})
    back = backward(+, a)
    Variable(a.data .+ b, Set([a]), back)
end


@commutative function Base.:+(a::Variable, b::Union{AbstractArray,Number})
    back = backward(+, a)
    Variable(a.data .+ b, Set([a]), back)
end

#
# Multiplication
#

function backward(::typeof(*), a::VariableArray, b::VariableArray)
    function ret(pgrad::Union{AbstractArray,Number})
        a.grad += pgrad * b.data'
        b.grad += a.data' * pgrad
    end
end

function backward(::typeof(*), a::Variable, b::Variable)
    function ret(pgrad::Union{AbstractArray,Number})
        a.grad += unbroadcast(a, pgrad .* b.data)
        b.grad += unbroadcast(b, pgrad .* a.data)
    end
end

function backward(::typeof(.*), a::Variable, b::Variable)
    function ret(pgrad::Union{AbstractArray,Number})
        a.grad += unbroadcast(a, pgrad .* b.data)
        b.grad += unbroadcast(b, pgrad .* a.data)
    end
end

function Base.broadcasted(::VariableStyle, ::typeof(*), a::Variable, b::Variable)
    Variable(a.data .* b.data, Set([a, b]), backward(.*, a, b))
end

function Base.:*(a::Variable, b::Variable)
    Variable(a.data * b.data, Set([a, b]), backward(*, a, b))
end

@commutative function Base.broadcasted(::VariableStyle, ::typeof(*), a::Variable, b::Union{AbstractArray,Number})
    println("test")
    @show a
    @show b
    back = backward(*, a)
    Variable(a.data .* b, Set([a]), back)
end


@commutative function Base.:*(a::Variable, b::Union{AbstractArray,Number})
    println("test2")
    @show a
    @show b
    back = backward(*, a)
    Variable(a.data * b, Set([a]), back)
end

#
# Transpose
#
function Base.adjoint(a::Variable)
    Variable(collect(a.data'), Set([a]), p_grad -> a.grad += p_grad')
end

#
# Negate
#
function Base.:-(a::Variable)
    Variable(-a.data, Set([a]), p_grad -> a.grad -= p_grad)
end

#
# Subtract
#
function Base.:-(a::Variable, b::Variable)
    a + (-b)
end

function Base.broadcasted(::VariableStyle, ::typeof(-), a::Variable, b::Variable)
    a .+ (-b)
end

@commutative function Base.:-(a::Variable, b::Number)
    a + (-b)
end

@commutative function Base.broadcasted(::VariableStyle, op::typeof(-), a::Variable, b::Number)
    a .+ (-b)
end

#
# Power
#
function backward(::typeof(^), a::Variable, b::Number)
    function ret(pgrad::Union{AbstractArray,Number})
        a.grad += b .* pgrad .* a.data.^(b .- 1)
    end
end

function Base.:^(a::VariableScalar, b::Number)
    back = backward(^, a, b)
    Variable(a.data^b, Set([a]), back)
end

function Base.broadcasted(::VariableStyle, ::typeof(^), a::Variable, b::Number)
    back = backward(^, a, b)
    Variable(a.data.^b, Set([a]), back)
end

# Needed when exponent is an int literal due to funky stuff (official term) for compile time specialization
deval(::Base.RefValue{Base.Val{n}}) where n = n
function Base.broadcasted(::VariableStyle, ::typeof(Base.literal_pow), op, a::Variable, b::Base.RefValue)
    result = broadcast(Base.literal_pow, ^, a.data, b)
    back = backward(^, a, deval(b))
    Variable(result, Set([a]), back)
end

#
# Tanh
#
function backward(::typeof(tanh), a::Variable)
    function ret(pgrad::Union{AbstractArray,Number})
        a.grad += (1. .- tanh.(a.data).^2 .* pgrad)
    end
end

function Base.tanh(a::Variable)
    back = backward(tanh, a)
    Variable(tanh(a.data), Set([a]), back)
end

function Base.broadcasted(::VariableStyle, ::typeof(tanh), a::Variable)
    back = backward(tanh, a)
    Variable(tanh.(a.data), Set([a]), back)
end

#
# Sum
#
function backward(::typeof(sum), a::Variable)
    function ret(pgrad::Union{AbstractArray,Number})
        a.grad .+= pgrad
    end
end

function Base.sum(a::Variable)
    back = backward(sum, a)
    Variable(sum(a.data), Set([a]), back)
end

end # module Variables