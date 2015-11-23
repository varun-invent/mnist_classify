require 'torch'
require 'nn'
require 'image'

noutputs = 10

ndim = 1
row = 32
col = 32

ninputs = ndim*row*col

nhidden = ninputs/2 -- for MLP only

-- for covnets
--nstates =  {ninputs,64,64,128}
--filtsize = 5
--poolsize = 2

mod = 'linear'

if mod == 'linear' then
	model = nn.Sequential()
	model:add(nn.Reshape(ninputs))
	model:add(nn.Linear(ninputs,noutputs))
end

print('Here is the linear model:')
print(model)

print ('Visualization of weights')
image.display(model:get(2).weight)  -- get(2) coz layer 1 is just reshaping of input and layer 2 actually transforms 1024->10

