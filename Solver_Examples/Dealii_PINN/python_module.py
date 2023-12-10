print("Entered Python module")
import torch 
from torch import nn
import math as ma
import numpy as np
from pathlib import Path
import torch
from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.data import *
import os
from torchvision import transforms
import matplotlib.pyplot as plt

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

def my_loss(output,coords,ground_truth,bvals):
    coords.flatten()
    output.flatten()

    # Taking the first gradient
    grad1 = torch.autograd.grad(outputs=output,inputs=coords,grad_outputs=torch.ones_like(ground_truth),create_graph=True,retain_graph=True)[0]
    
    # Separating components
    dphidx1 = grad1[:,0]
    dphidx2 = grad1[:,1]
    
    grad2 = torch.autograd.grad(outputs=dphidx1,inputs=coords,grad_outputs=torch.ones_like(dphidx1),create_graph=True,retain_graph=True)[0]
    
    grad3 = torch.autograd.grad(outputs=dphidx2,inputs=coords,grad_outputs=torch.ones_like(dphidx2),create_graph=True,retain_graph=True)[0]
    
    dphi2dx12 = grad2[:,0]
    dphi2dx22 = grad3[:,1]

    laplacian = dphi2dx12+dphi2dx22

    x_int = torch.pow(coords,2) 
    x = ((x_int[:,1]+x_int[:,0]))
   
    xloss = torch.nn.MSELoss()
    loss_physics = xloss(laplacian-x,torch.zeros_like(laplacian))
    loss_MSE = xloss(output,ground_truth)
    #Soft constraint for the homogenous dirichlet BCs applied to the domain
    loss_boundary = xloss(bvals,torch.zeros_like(bvals)) 
    loss = loss_physics +loss_MSE +loss_boundary
    
    return loss

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2,320),
            nn.Tanh(),
            nn.Linear(320,160),
            nn.Tanh(),
            nn.Linear(160,80),
            nn.Tanh(),
            nn.Linear(80,40),
            nn.Tanh(),
            nn.Linear(40,20),
            nn.Tanh(),
            nn.Linear(20,1)
            
                       
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
def train(X,y,z):
    
    model = NeuralNetwork().to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.5e-4)
    loss_fn = my_loss 
    X,y,z = X.to(device), y.to(device), z.to(device)
    X = X.float()
    y = y.float()
    z = z.float()
    
    # Compute prediction error        
    pred = model(y)
    loss = loss_fn(pred,y,X,z)
    
    # Backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    loss = loss.item()
    print(f"loss: {loss:>7f}")
    return loss, pred

              
def run_training(output,coords,bdofs):        
 print('Training function starts here')
 
 data = np.zeros((4225,1))
 data2 = np.zeros((4225,2))
 data3 = np.zeros((np.shape(bdofs)[0]),dtype=int)
 data[:,0] = output
 data2[:,:] = coords
 data3[:] = bdofs
 data4 = data[data3]
 #Converting all Deal.ii data to Torch tensors
 X = torch.tensor(data,requires_grad=True)
 y = torch.tensor(data2,requires_grad=True)
 z = torch.tensor(data4,requires_grad=True)
 #z = torch.tensor(data3)
 
 epochs = 100
 for t in range(epochs):
      print(f"Epoch {t+1}\n-------------------------------")
      train_losses, pred = train(X,y,z)     
 
 #model2 = NeuralNetwork().to(device)
 #model2.eval()
 #with torch.no_grad():
 #     pred = model2(y)
 pred = pred.cpu().detach().numpy()
 xv = np.reshape(data2[:,0], (65,65))
 yv = np.reshape(data2[:,1], (65,65))

 #xv,yv = np.meshgrid(xcoords,ycoords)
 zvals_PINN = np.reshape(pred,(np.shape(xv)[0],np.shape(xv)[1]))
 zvals_FEM = np.reshape(data,(65,65))
 
 print(zvals_FEM)
 print(data2)
 fig, (ax1, ax2) = plt.subplots(1, 2)
 fig.suptitle('PINN solution (left) vs FEM solution (right)')
 ax1.contourf(xv,yv,zvals_PINN, 200, cmap=plt.cm.viridis)
 ax2.contourf(xv,yv,zvals_FEM, 200, cmap=plt.cm.viridis)
 fig.savefig('PINN_Output.png')
 
 


 print("Training Completed")
 torch.save(model, "model4.pth")
 print("Saved PyTorch Model State to model4.pth")

print('python_module imported successfully')