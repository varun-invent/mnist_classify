require 'torch'
require 'xlua'
require 'optim'

if model then
	print('Model whose params are used is:')
	print(model)
	parameters,gradParameters =  model:getParameters()
	print('Params Extracted')
end 




proj_path = '/home/varunk/Projects/mnist_classify/'
trainLogger  = 	optim.Logger(paths.concat(proj_path,'train.log'))
testLogger  = 	optim.Logger(paths.concat(proj_path,'test.log'))

optimization = 'sgd'

--Defining params for sgd
if optimization == 'sgd' then
	optimState = {
	learningRate = 0.001,
	learningRateDecay = 1e-7,
	weightDecay = 0.001,
	momentum =  0.001
	}
	optimMethod = optim.sgd
end

batchSize = 50
-- Training procedure
function train()

	-- epoch tracker
	epoch = epoch or 1 -- global variable (local keyword is not present)

	local  time = sys.clock()

	-- set model to training mode (for modules that differ in training and testing, like Dropout)
    model:training()

   -- shuffle at each epoch
   shuffle = torch.randperm(trsize)

   -- do one epoch
   print('Doing epoch on training data')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')
   for t=1,trsize,batchSize do

	   	--display progess
	   	xlua.progress(t,trsize)
	   

	   --create a minibatch
	   local inputs = {}
	   local targets = {}
	   -- print('data',trainData)
	   for i = t, math.min(t+batchSize-1,trsize) do

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
				model:backward(inputs[i],df_do)
				
			end

			f = f/#inputs
			gradParameters:div(#inputs)

			return f,gradParameters


		end

		optimMethod(feval,parameters,optimState)
	end

	--total time taken
	time = sys.clock() - time

	-- time taken by one sample
	time = time/trsize
	print('Time taken to learn 1 sample ' .. (time*1000) .. 'ms')

	local filename =  paths.concat(proj_path,'model.net')
	os.execute('mkdir -p ' .. sys.dirname(filename))    -- sys.dirname gives only the directory, not the filename i.e it skips the last name
	print('=> Saving model to ' .. filename)
	torch.save(filename,model)

	epoch =  epoch + 1

end