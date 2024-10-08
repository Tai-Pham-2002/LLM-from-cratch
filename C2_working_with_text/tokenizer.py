import re

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

with open('the-verdict.txt', 'r',encoding='utf-8') as f:
    raw_text = f.read()
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
# print(len(preprocessed))
# all_words = sorted(set(preprocessed))
# vocab = {token:integer for integer,token in enumerate(all_words)} 

# # inference example 
# tokenizer = SimpleTokenizerV1(vocab)
# text = """"It's the last he painted, you know,"
# Mrs. Gisburn said with pardonable pride."""
# ids = tokenizer.encode(text)
# print(ids)
# print(tokenizer.decode(ids))

# ===============================================================================================================================================================
# Add the special token: <|unk|> token to represent new and unknown words and <|endoftext|> token that we can use to separate two unrelated text sources.

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}
print(len(vocab.items())) 
# for i in list(vocab.items())[-5:]:
#     print(i)

class SimpleTokenizerV2:
    def __init__(self,vocab) -> None:
        self.str_to_int = vocab
        self.int_to_str = {integer:token for token, integer in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int 
                        else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

# text1 = "Hello, do you like tea?"
# text2 = "In the sunlit terraces of the palace."
# text = " <|endoftext|> ".join((text1, text2))
# print(text)

# tokenizer = SimpleTokenizerV2(vocab)
# ids = tokenizer.encode(text)
# print(ids)
# print(tokenizer.decode(ids))


# ===============================================================================================================================================================
#  tiktoken
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
text = (
"Hello, do you like tea? <|endoftext|> In the sunlit terraces"
"of someunknownPlace."
) 
# exercise = "Akwirw ier"
# integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"}) # encode
# # print(integers)

# strings = tokenizer.decode(integers)  # decode
# print(strings)

# Exercise 1: try with "Akwirw ier"
exercise = "Akwirw ier"
integers = tokenizer.encode(exercise, allowed_special={"<|endoftext|>"}) # encode
print(integers)

strings = tokenizer.decode(integers)  # decode
print(strings)