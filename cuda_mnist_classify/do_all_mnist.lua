require 'torch'

-- Keeping all the parameters here
opt = {
	max_epoch = 5,
	gpu = true,
	trsize_custom = 20000 , -- If you want to train with less data, change this parameter
	normalize = true,
	model = 'linear',
	loss = 'nll',
	optimization = 'sgd',
	batchSize = 50
}


print('Executing all the files')

dofile 'data_mnist.lua'
dofile 'model_mnist.lua'
dofile 'loss_mnist.lua'
dofile 'train_mnist.lua'
dofile 'test_mnist.lua'

for i=1, opt.max_epoch do
    train()
    test()
end

-- Normalize the data   -- done
-- Check the function to find max index -- done
-- Move all the params in 'opt' table  --done
-- Cuda implementation of the code
-- Change the model to NLP  