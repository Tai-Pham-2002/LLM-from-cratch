import re
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)
        for i in range(0,len(token_ids)-max_length,stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256,
    stride=128, shuffle=True, drop_last=True,
    num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")  # initialize tokenizer
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride) # create dataset
    dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    drop_last=drop_last,
                    num_workers=num_workers
                    )
    return dataloader

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=8, stride =2, shuffle=False)
data_iter = iter(dataloader) 
first_batch = next(data_iter)
second_batch = next(data_iter)
print(first_batch)
print(second_batch)
# tokenizer = tiktoken.get_encoding("gpt2")
# enc_text = tokenizer.encode(raw_text)

# # Create dataset
# enc_sample = enc_text[:50]
# context_size = 4
# x = enc_sample[:context_size]
# y = enc_sample[1:context_size+1]
# # print(f"x: {x}")
# # print(f"y:     {y}")

# for i in range(1, context_size+1):
#     context = enc_sample[:i]
#     desired = enc_sample[i]
#     print(context, "---->", desired)

# for i in range(1, context_size+1):
#     context = enc_sample[:i]
#     desired = enc_sample[i]
#     print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))