import torch 
import torch.nn as nn
from torch.nn import functional as F

#setting hyperparameters
batch_size = 32 
block_size = 8
max_iters = 3000
eval_interval = 300
learing_rate = 1e-3
eval_iters = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embed = 32 # number of embeding dimensions 

#-------

torch.manual_seed(1337)

#opening the file
with open('tiny_shakespeare.txt', 'r') as file:
    text = file.read()

#processing of the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

#creating custom basic embedings 
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

#spliting data into test and train
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

#data loading
def get_batch(split):
    #generating a small batch of data of inputs for x and y
    data = train_data if split == 'train' else val_data # we ste our data set
    ix = torch.randint(len(data) - block_size, (batch_size,)) # here we randomize 4 (since this is the batch size) x's to have as a starting point 
    x = torch.stack([data[i:i+block_size] for i in ix]) # we fill in the full x vector from the one we randomized
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # we fill in the y vector to be able to follow the context
    return x,y

@torch.no_grad
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

#self attention
class Head(nn.Module):
    """one head self attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=F)
        self.query = nn.Linear(n_embed, head_size, bias=F)
        self.value = nn.Linear(n_embed, head_size, bias=F)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape()
        k = self.key(x)
        q = self.query(x)
        #compute scores for attention
        wei = q @ k.transpose(-2, -1) * C**-0.5 
        wei = wei.masked_fill(self.tril[:T, T:] == 0, float('-inf')) 
        wei = F.softmax(wei, dim=-1)

        v = self.value(x)
        out = wei @ v 

        return out 

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out 


class FeedFoward(nn.Module):

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computations"""

    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x



#defining the bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly looks up next token from lookup array of all token relation values
        self.token_embeding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embeding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            Block(n_embed, n_head=4),
            Block(n_embed, n_head=4),
            Block(n_embed, n_head=4),
        )
        self.sa_head = MultiHeadAttention(4, n_embed//4)
        self.ffwd = FeedFoward(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):

        B,T = idx.shape

        #idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embeding_table(idx) # (B,T,C)
        pos_emb = self.position_embeding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x)
        logits = self.lm_head(x) # (B,T,vocab_size)
        #B - batch size, how many sequences we look over
        #T - time, which means how long of a context we look into
        #C - channels or how many different characters we have

        if targets is None:
            loss = None
        else:
        
            B, T, C = logits.shape # this is needed after the investigation of loss function required inputs 
            logits = logits.view(B*T, C) 
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


    def generate(self, idx, max_new_tokens):
        #idx is (B,T) array of indices in the current context

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            #get the predictions 
            logits, loss = self(idx_cond)
            #focus only on the last time step
            #we reduce dimensionality to 2 dimension basically since we need only the last output to make the new token prediction
            logits = logits[:, -1, :] # becomes B,C
            #apply softmax to probabilities
            probs = F.softmax(logits, dim=-1)
            #sample from distribution 
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            #append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        return idx

model = BigramLanguageModel()
m = model.to(device)

#create pytorch optimizer 
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for iter in range(max_iters):

    #every once in a while evaluate the loss on the train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss: {losses['val']:.4f}")

    #sample a batch of data 
    xb, yb = get_batch('train')

    #evaluate loss 
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#generate the text from the given model 
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))


