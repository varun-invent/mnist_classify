require 'torch'
require 'image'

train  = '/home/varunk/Projects/datasets/mnist_torch/mnist.t7/train_32x32.t7'
test =  '/home/varunk/Projects/datasets/mnist_torch/mnist.t7/test_32x32.t7'

trsize_custom = 20000  -- If you want to train with less data, change this parameter
normalize = false
loaded_train = torch.load(train,'ascii')
loaded_test = torch.load(test,'ascii')
trsize = math.min(loaded_train.labels:size()[1],trsize_custom)
tesize = loaded_test.labels:size()[1]

--image.display(loaded_train.data[{{1,25},{},{},{}}])
-- print('this is ')
-- print(loaded_train.labels[{{1,25}}])

trainData = { 
    --Conversion to double was necessary coz the model:forward() doesnot work on ByteTensor
	data = loaded_train.data[{{1,trsize},{},{},{}}]:double(),
	labels = loaded_train.labels[{{1,trsize}}],
	size = function() return trsize end
	}

testData = {
	data = loaded_test.data:double(),
	labels = loaded_test.labels,
	size = function () return tesize end
	}

--print(trainData:size())

if normalize == true then
	mean = trainData.data:mean()
	std =  trainData.data:std()
	trainData.data:add(-mean)
	trainData.data:div(std)
	testData.data:add(mean)
	testData.data:div(std)
end