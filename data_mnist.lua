require 'torch'
require 'image'

train  = '/home/varunk/Projects/datasets/mnist_torch/mnist.t7/train_32x32.t7'
test =  '/home/varunk/Projects/datasets/mnist_torch/mnist.t7/test_32x32.t7'

loaded_train = torch.load(train,'ascii')
loaded_test = torch.load(test,'ascii')
trsize = loaded_train.labels:size()[1]

--image.display(loaded_train.data[{{1,25},{},{},{}}])
-- print('this is ')
-- print(loaded_train.labels[{{1,25}}])

trainData = { 
	data = loaded_train.data,
	labels = loaded_train.labels
	}

--print(trainData:size())