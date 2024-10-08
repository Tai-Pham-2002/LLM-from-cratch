import torch
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your
    [0.55, 0.87, 0.66], # journey
    [0.57, 0.85, 0.64], # starts
    [0.22, 0.58, 0.33], # with
    [0.77, 0.25, 0.10], # one
    [0.05, 0.80, 0.55]] # step
)
# ===============================================================================================================================    

# Calculate the s attention scores of second input token

# query = inputs[1]   # The second input token is the query.
# attn_scores_2 = torch.empty(inputs.shape[0])
# for i, x_i in enumerate(inputs):
#     attn_scores_2[i] = torch.dot(x_i, query)
# print(attn_scores_2)

# attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
# print("Attention weights:", attn_weights_2_tmp)
# print("Sum:", attn_weights_2_tmp.sum())
# def softmax_naive(x):
#     return torch.exp(x) / torch.exp(x).sum(dim=0)
# attn_weights_2_naive = softmax_naive(attn_scores_2)
# print("Attention weights:", attn_weights_2_naive)
# print("Sum:", attn_weights_2_naive.sum())

# attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
# print("Attention weights:", attn_weights_2)
# print("Sum:", attn_weights_2.sum())

# ===============================================================================================================================    
# calculate the s attention scores of all input token

# ==> Way 1
# attn_scores_all = torch.empty(inputs.shape[0], inputs.shape[0])
# print(attn_scores_all.shape)
# for i, x_i in enumerate(inputs):
#     for j, x_j in enumerate(inputs):
#         attn_scores_all[i, j] = torch.dot(x_i, x_j)
# print(attn_scores_all)

# ==> Way 2
# Step 1: Compute attention scores
attn_scores = inputs @ inputs.T
print(attn_scores)
# Step 2: Normalize attention scores
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)
# Step 3: Sum attention scores for each row
row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Row 2 sum:", row_2_sum)
print("All row sums:", attn_weights.sum(dim=-1))