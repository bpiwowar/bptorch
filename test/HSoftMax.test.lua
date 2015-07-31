NOOPTIM = false
require 'bptorch'


-- a,b is the span of the subtree
-- terminates when a == b
function construct_tree(parents, a, b, state)
    state = state or { leaveIx=0, innerIx=(parents:size(1) + 1) / 2}
    -- print("Split ", a, b, state, "\n")
    if a == b then
        state.leaveIx = state.leaveIx + 1
        return state.leaveIx
    end

    -- Construct subtrees
    local split = torch.random(0, b - a - 1)
    local leftIx = construct_tree(parents, a, a + split, state)
    local rightIx = construct_tree(parents, a + split + 1, b, state)

    state.innerIx = state.innerIx + 1
    -- print(string.format("Node %d (%d-%d), children %d and %d", state.innerIx, a, b, leftIx, rightIx))
    parents[leftIx] = -state.innerIx
    parents[rightIx] = state.innerIx
    return state.innerIx
end

nleaves = 100000
parents = torch.LongTensor(2 * nleaves - 1)
construct_tree(parents, 1, nleaves)

-- print(parents)
-- exit()

input_size = 100
hs = nn.HSoftMax(input_size, parents)

words = torch.range(torch.LongTensor(nleaves), 1, nleaves)

x = torch.randn(1, input_size)
input = torch.expand(x, words:size(1), input_size)

-- print("words:"); print(words)
-- print(input)

output = hs:forward({input, words})
-- print(output)

s = torch.sum(output:exp())
delta = torch.abs(s - 1)
print("[batch] sum(p) = " .. s)
assert(delta < 1e-10, "Sum of probabilities should be 1, but |p - 1| is " .. delta)

s = 0
for i = 1, words:size(1) do
    s = s + torch.exp(hs:forward({ x[1], words[{{i}}] }))[1]
end
delta = torch.abs(s - 1)
print("sum(p) = " .. s)
assert(delta < 1e-10, "Sum of probabilities should be 1, but |p - 1| is " .. delta)



ninput = 10000
input = torch.randn(ninput, input_size)
targets = torch.LongTensor(ninput)
targets:random(nleaves)

function test_time()
    local ntimes = 10
    timer = torch.Timer() -- the Timer starts to count now
    for i=1,ntimes do
      output = hs:updateOutput({input, targets})
    end
    print('Time elapsed for ' .. ninput ..' inputs: ' .. timer:time().real / ntimes .. ' seconds, input size = ' .. (input:size(1)) .. "x" .. (input:size(2)) .. " , # leaves = " .. nleaves)
end
test_time()

exit()
local SumCriterion, parent = torch.class('nn.SumCriterion', 'nn.Criterion')

function SumCriterion:__init()
   parent.__init(self)
   self.sizeAverage = true
end

function SumCriterion:updateOutput(input, target)
    self.output = -torch.sum(input)
    return self.output
end

function SumCriterion:updateGradInput(input, target)
    self.gradInput:resizeAs(input)
    -- Fill with -1 since we need to increase the value
    self.gradInput:fill(-1)
    return self.gradInput
end


function test_gradient()

    -- Test gradient
    local ninput = 100000
    input = torch.randn(ninput, input_size)
    targets = torch.LongTensor(ninput)
    targets:random(nleaves)

    criterion = nn.SumCriterion()

    hs:zeroGradParameters()
    output = hs:updateOutput({input, targets})
    err = criterion:forward(output, nil)
    df_do = criterion:backward(output, nil)
    hs:backward({input, targets}, df_do)
    grad = hs.gradInput[1]:clone()

    for i = 1, 2 do
      delta = torch.randn(ninput, input_size)

      for i, eps in pairs({1, 1e-1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-14}) do
        input_delta = input:clone()
        input_delta:add(eps, delta)

        local output = hs:updateOutput({input_delta, targets})
        local err_eps = criterion:forward(output, nil)

        delta_err = delta:clone()
        delta_err:cmul(grad)

        -- print(delta:sum())
        print(eps, (err - err_eps - delta_err:sum()) / ninput )
      end
      print("")
    end
end

function test_learn()
    require 'optim'
    require 'nn'
    -- Batch size
    local nsample = 100000
    local modelinputsize = 100

    input = torch.randn(nsample, modelinputsize)
    criterion = nn.SumCriterion()


    models = nn.ParallelTable()
    -- models:add(nn.Identity())
    models:add(nn.Linear(modelinputsize, input_size))
    models:add(nn.Identity())

    model = nn.Sequential()
    model:add(models)
    model:add(hs)
    -- print(model)

    optimState = {
      maxIter = 10,
      learningRate = 1e-2,
      nCorrection = 10
    }
    optimMethod = optim.lbfgs

    print("Getting model parameters...")
    parameters, gradParameters = model:getParameters()


    -- create closure to evaluate f(X) and df/dX
    local feval = function(x)
       -- get new parameters
       if x ~= parameters then
          parameters:copy(x)
       end

       -- reset gradients
       gradParameters:zero()

       -- f is the average of all criterions
       local f = 0

      -- estimate f
      local output = model:forward({input, targets})
      local err = criterion:forward(output, nil)
      f = f + err

      -- estimate df/dW
      local df_do = criterion:backward(output, nil)
      model:backward({input, targets}, df_do)

       -- normalize gradients and f(X)
       gradParameters:div(input:size(1))
       f = f / input:size(1)

       -- return f and df/dX
       return f, gradParameters
    end

    print("Starting to learn...")
    timer = torch.Timer() -- the Timer starts to count now
    a, b = optimMethod(feval, parameters, optimState)
    print(b)

    print('Time elapsed for ' .. nsample ..' inputs: ' .. timer:time().real .. ' seconds')

end

-- test_gradient()
-- test_gradient()
test_learn()