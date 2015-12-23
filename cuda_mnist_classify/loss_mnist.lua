require 'torch'
require 'nn'

loss = 'nll'
if model == nil then
	print('Model not created! Exiting..')
	os.exit()

elseif loss == 'nll' then
	model:add(nn.LogSoftMax())
	criterion =  nn.ClassNLLCriterion()
end

print('Here is the loss function')
print(criterion)
