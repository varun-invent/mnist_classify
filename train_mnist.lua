require 'torch'
require 'xlua'
require 'optim'

-- Continue from here ---> implement the fewal with sgd 

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

   







end