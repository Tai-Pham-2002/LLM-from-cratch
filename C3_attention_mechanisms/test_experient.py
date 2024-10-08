import torch

a= torch.tensor([0.55, 0.87, 0.66])
b = torch.tensor([0.43, 0.15, 0.89])
print(b @ a.T)
print(torch.dot(b,a))
print(torch.softmax(a,dim=0))
print(torch.softmax(a,dim=-1))
