require 'bptorch'

ninputx = 10
ninputy = 12

noutput = 7
noperators = 5

nsample = 6

mop = nn.MOperator(ninputx, ninputy, noutput, noperators)

x = torch.rand(nsample, ninputx)
y = torch.rand(nsample, ninputy)
print(mop:forward({x, y}))

x1 = torch.rand(ninputx)
print(mop:forward({x1, y}))

y1 = torch.rand(1, ninputy):expand(nsample, ninputy):contiguous()
print(mop:forward({x1, y1}))
