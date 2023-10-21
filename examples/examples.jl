using Bijoulia: Variable, Linear, forward, Network, Activation, Tanh, forward, backpropagate

# f(x) = x^3 - 6x^2 + 11x - 6
f(x) = x^2
X = collect(range(start=-100, stop=100, length=500))
y = f.(X)

mutable struct MyNet <: Network
    fc1::Linear
    fc2::Linear
    fc3::Linear
    act::Activation

    MyNet(input::Int, output::Int) = new(Linear(input, 20),
        Linear(20, 20),
        Linear(20, output),
        Tanh())
end

# temp
function zero_grads(n::MyNet)
    n.fc1.params.weights.grad = zeros(size(n.fc1.params.weights.grad))
    n.fc1.params.bias.grad = zeros(size(n.fc1.params.bias.grad))
    n.fc2.params.weights.grad = zeros(size(n.fc2.params.weights.grad))
    n.fc2.params.bias.grad = zeros(size(n.fc2.params.bias.grad))
    n.fc3.params.weights.grad = zeros(size(n.fc3.params.weights.grad))
    n.fc3.params.bias.grad = zeros(size(n.fc3.params.bias.grad))
    return nothing
end

function fw(n::MyNet, x::Variable)::Variable
    x = forward(n.fc1, x)
    x = forward(n.act, x)
    x = forward(n.fc2, x)
    x = forward(n.act, x)
    x = forward(n.fc3, x)
end

net = MyNet(1, 1)

for i in 1:1000
    zero_grads(net)
    preds = fw(net, Variable(X))
    loss = sum((preds - Variable(y))^2)
    backpropagate(loss)
    for p in (net.fc1.params.weights, net.fc1.params.bias,
              net.fc2.params.weights, net.fc2.params.bias,
              net.fc3.params.weights, net.fc3.params.bias)
        p.data .-= 0.0001 .* p.grad
    end
    println("epoch: $i -> loss: $(loss.data)")
end