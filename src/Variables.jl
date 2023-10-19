module Variables

export Variable

mutable struct Variable
    data::Array{Float64}
    grad::Array{Float64}
    children::Set{Variable}
    backward::Function

    function Variable(data::Array{Float64}, children::Set{Variable}, backward::Function)
        new(data, zeros(size(data)), children, backward)
    end

    function Variable(data::Array{<:Real}, children::Set{Variable}, backward::Function)
        new(convert.(Float64, data), zeros(size(data)), children, backward)
    end

    function Variable(data::Array{Float64})
        new(data, zeros(size(data)), Set(), Returns(nothing))
    end

    function Variable(data::Array{<:Real})
        new(convert.(Float64, data), zeros(size(data)), Set(), Returns(nothing))
    end
end

end # module Variables