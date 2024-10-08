
import torch
torch.manual_seed(123)
input_ids = torch.tensor([2,3,5,1])
# print(input_ids)

vocab_size = 6
output_dim = 3
embedding_layer = torch.nn.Embedding(vocab_size,output_dim)

print(embedding_layer.weight)
print(embedding_layer(torch.tensor([3])))
print(embedding_layer(input_ids))
