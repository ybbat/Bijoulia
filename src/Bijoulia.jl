module Bijoulia

include("Variables.jl")

using .Variables


abstract type Module end

mutable struct Linear <: Module
    params::NamedTuple

    function Linear(input_size::Integer, neurons::Integer)
        new((weights=Variable(rand(input_size, neurons)),
            bias=Variable(zeros(neurons))))
    end
end

function forward(m::Linear, inp::Variable)::Variable
    inp * m.params.weights .+ m.params.bias'
end

mutable struct Apply <: Module
    func::Function

    function Apply(func::Function)
        new(func)
    end
end

function forward(m::Apply, inp::Variable)::Variable
    m.func.(inp)
end

Tanh = Apply(tanh)
ReLU = Apply(relu)


abstract type Activation <: Module end

abstract type Network end

end # module Bijoulia