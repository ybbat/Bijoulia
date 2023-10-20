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

end # module Variables