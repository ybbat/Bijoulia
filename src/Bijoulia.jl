module Bijoulia

include("Variables.jl")
using .Variables: Variable, backpropagate


abstract type Module end

mutable struct Linear <: Module
    params::NamedTuple

    function Linear(input_size::Integer, neurons::Integer)
        new((weights=Variable(rand(input_size, neurons)),
             bias=Variable(zeros(neurons))))
    end
end

function forward(m::Linear, inp::Variable)::Variable
    inp * m.params.weights + m.params.bias'
end


abstract type Activation <: Module end

mutable struct Tanh <: Activation
end
function forward(m::Tanh, inp::Variable)::Variable
    tanh(inp)
end


abstract type Network end

end # module Bijoulia