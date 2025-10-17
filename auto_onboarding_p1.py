import torch
from torch import nn

# Creating 5 sets of two random nubmers
samplesize = 5

x = torch.randn(samplesize, 2)
y = (x[:,0] + x[:,1]).unsqueeze(1)

# split 80%-20%
trainsplit = int(0.8 * len(x))
xtrain, ytrain = x[:trainsplit], y[:trainsplit]
xtest, ytest = x[trainsplit:], y[trainsplit:]

# Create class model
class add(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2,1) 

    def forward(self, x):
        pred = self.linear(x) # take the two number sets as input and return one output
        return pred


# Create a manual seed and an instance
torch.manual_seed(1)
model = add()

print(model.state_dict())
# the weigths coff to the input and the bias is the constant and we want them to be 1,1 and zero respectivly

#Creating the loss and optimizer fucntions to begin traning
lossfn = nn.L1Loss()
optimfn = torch.optim.SGD(model.parameters(),0.01)

# Create testing and traning loops
iter = 500

for i in range(iter):
    model.train()
    ypred = model(xtrain) # not x because we only want the part we are training
    loss = lossfn(ypred,ytrain)
    optimfn.zero_grad()
    loss.backward()
    optimfn.step()

    model.eval()
    with torch.inference_mode():
        testpred = model(xtest)
        testloss = lossfn(testpred, ytest)
    
    #print the numbers
    if i % 10 == 0:
        print(f'Iter #: {i}')
        print(model.state_dict())



