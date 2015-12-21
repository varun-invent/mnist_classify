require 'torch'

print('Executing all the files')

dofile 'data_mnist.lua'
dofile 'model_mnist.lua'
dofile 'loss_mnist.lua'
dofile 'train_mnist.lua'

for i=1,20 do
    train()
end