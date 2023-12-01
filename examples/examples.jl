Revise.includet("../src/Bijoulia.jl")
# include("../src/Bijoulia.jl")
using .Bijoulia: Variable, Linear, forward, Network, Activation, Tanh, forward, backpropagate!, zero_grads!, Apply, ReLU

# f(x) = x^3 - 6x^2 + 11x - 6
# f(x) = sin(x)
# f(x) = 2x
# f(x) = x + 2
# X = collect(range(start=0, stop=250, length=500))
# y = f.(X)
X = [1.0 0.0; 0.0 1.0; 1.0 1.0; 0.0 0.0] # 2 bit inputs
y = [1; 1; 0; 0] # XOR

mutable struct MyNet <: Network
    fc1::Linear
    fc2::Linear
    act::Apply

    MyNet(input::Int, output::Int) = new(Linear(input, 3),
        Linear(3, output),
        ReLU)
end

function fw(n::MyNet, x::Variable)::Variable
    x = forward(n.fc1, x)
    x = forward(n.act, x)
    x = forward(n.fc2, x)
end

net = MyNet(2, 1)

for i in 1:10000
    local preds = fw(net, Variable(X))
    local loss = sum(0.5 .* (preds .- y) .^ 2.0)
    backpropagate!(loss)
    for p in (net.fc1.params.weights, net.fc1.params.bias,
        net.fc2.params.weights, net.fc2.params.bias)
        p.data -= 0.05 .* p.grad
    end
    println("epoch: $i -> loss: $(loss.data)")
    zero_grads!(loss)
end

preds = fw(net, Variable(X))
for i ∈ eachindex(y)
    println("X: $(X[i, :]) -> y: $(y[i]) -> pred: $(preds.data[i])")
end