import matplotlib.pyplot as plt
import numpy as np 
import torch
from torchvision import datasets, transforms

state_space_model_type = "complex" # or "real"
optimizer = "adam" # or "psgd"

print(f"Domain of state vectors: {state_space_model_type}")
print(f"Optimizer: {optimizer}\n")

if state_space_model_type == "complex":
    from state_space_models import ComplexStateSpaceModel as SSM
    increase_state_size = 1
else:
    from state_space_models import RealStateSpaceModel as SSM
    increase_state_size = 2
   
if optimizer == "psgd":
    print("Need to download the psgd optimizer (https://github.com/lixilinx/psgd_torch or from other places)")
    import preconditioned_stochastic_gradient_descent as psgd

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    ),
    batch_size=60,
    shuffle=True,
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data", train=False, transform=transforms.Compose([transforms.ToTensor()])
    ),
    batch_size=60,
    shuffle=False,
)


class SSMNet(torch.nn.Module):
    def __init__(self):
        super(SSMNet, self).__init__()
        self.ssm1 = SSM(1, increase_state_size * 16, 16, resample_down=4)
        self.ssm2 = SSM(16, increase_state_size * 128, 128, resample_down=4)
        self.linear = torch.nn.Linear(128, 10)

    def forward(self, u):
        x, _ = self.ssm1(u)
        x = x * torch.rsqrt(1 + x*x)
        
        x, _ = self.ssm2(x)
        x = x[:, -1]
        x = x * torch.rsqrt(1 + x*x)
        
        x = self.linear(x)
        return x

device = torch.device("cuda:0")  
ssmnet = SSMNet().to(device)
lr0 = 1e-3
if optimizer == "psgd":
    opt = psgd.Kron(ssmnet.parameters(), lr_params=lr0, lr_preconditioner=0.1, 
                    momentum=0.9, preconditioner_type="whitening", grad_clip_max_norm=100)
else:
    opt = torch.optim.Adam(ssmnet.parameters(), lr=lr0)

num_epochs = 20 
TrainLosses, TestErrs = [], []
for epoch in range(num_epochs):
    for batch, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        def closure():
            y = ssmnet(torch.reshape(data, [-1, 28*28, 1]))
            y = torch.nn.functional.log_softmax(y, dim=-1)
            xentropy = torch.nn.functional.nll_loss(y, target)
            return xentropy
        
        if optimizer == "psgd":
            loss = opt.step(closure)
        else: # Adam 
            opt.zero_grad()
            loss = closure() 
            loss.backward()
            opt.step() 

        TrainLosses.append(loss.item())
        if (batch+1) % 100 == 0:
            print(f"Epoch: {epoch + 1}; train loss: {np.mean(TrainLosses[-1000:])}")
           
    # test loss
    with torch.no_grad():
        num_errs = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            y = ssmnet(torch.reshape(data, [-1, 28*28, 1]))
            _, pred = torch.max(y, dim=1)
            num_errs += torch.sum(pred != target)
        test_err_rate = num_errs.item() / len(test_loader.dataset)
        TestErrs.append(test_err_rate)
    print(f"Epoch: {epoch + 1}; test classification error rate: {TestErrs[-1]}")
    
    # linear lr schedule 
    if optimizer == "psgd":
        opt.preconditioner_update_probability = 0.1 
        opt.lr_params -= lr0 / num_epochs
    else:
        opt.param_groups[0]["lr"] -= lr0 / num_epochs
    
plt.plot(TrainLosses)
plt.show()
plt.plot(TestErrs)
plt.show() 
        

