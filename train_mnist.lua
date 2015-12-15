require 'torch'
require 'xlua'
require 'optim'

if model then
	parameters,gradParameters =  model.getParameters()
end 

trainLogger  = 	optim.Logger('/home/varunk/Projects/mnist_classify/train.log')
testLogger  = 	optim.Logger('/home/varunk/Projects/mnist_classify/test.log')

optimization = 'sgd'

--Defining params for sgd
if optimization == 'sgd' then
	optimState = {
	learningRate = 0.001
	learningRateDecay = 1e-7
	weightDecay = 0.001
	momentum =  0.001
	}
	optimMethod = optim.sgd
end

batchSize = 50
-- Training procedure
function train()

	-- epoch tracker
	epoch = epoch or 1

	local  time = sys.clock()

	-- set model to training mode (for modules that differ in training and testing, like Dropout)
    model:training()

   -- shuffle at each epoch
   shuffle = torch.randperm(trsize)

   -- do one epoch
   print('Doing epoch on training data')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')
   for t=1,trainData.data:size(),batchSize do
   	--display progess
   	xlua.progess(1,trainData.data:size())
   end

   --create a minibatch
   local inputs = {}
   local targets = {}

   for i = 1, math.min(t+batchSize-1,trainData.data:size()) do

	   --load new samples
	   local input = trainData.data[shuffle[i]]
	   local target = trainData.labels[shuffle[i]]

	   table.insert(inputs,input)
	   table.insert(targets,target)
	   
	end

	-- create a closure to evaluate f(x) and df(x)/dW i.e. dZ/dW
	
	local feval =  function(x)
		--get new parameters
		if x ~= parameters then
			parameters:copy(x)
		end

		--reset gradients
		gradParameters:zero()

		--f is the average error of criterion
		f = 0

		--evaluate f and grad for all inputs in a minibatch

		for i=1,#inputs do
			
			--estimate f
			local output = model:forward(inputs[i])
			local err =  criterion:forward(output,targets[i])
			f = f+err

			--estimate df/dW
			local df_do = criterion:backward(output,targets[i])
			model:backward(inputs,df_do)
			---------------------------------------------------------------------Try to understand this !! backward!!!
		end

	end






end