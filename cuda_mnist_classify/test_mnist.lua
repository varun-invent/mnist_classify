require 'torch'
require 'xlua'

print('Defining test procedure =>')

function test()
	local time = sys.clock()

	model:evaluate() -- necessary while testing to supress modules like dropout


	--Test on test set

	print('Testing on test set=> ')
	print("Test data size ",testData:size())
	local correct_pred_count = 0

	for t=1,testData:size() do
		xlua.progress(t,testData:size())
		local target = testData.labels[t]
		local pred = model:forward(testData.data[t])
		pred_max,pred_index = torch.max(pred,2) -- 2 coz the size of pred is 1x10 and not just 10
		--print("target ",target)
		--print('pred ',pred_index)
		if target == pred_index[1][1] then
			correct_pred_count = correct_pred_count + 1
		end
	end

	acc = correct_pred_count/testData:size()
	print('Testing Accuracy is: ',acc*100)

end


