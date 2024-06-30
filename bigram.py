import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batchSize = 32  # num of independent sequences to process in parallel
blockSize = 8  # max context length for prediction
maxIters = 5000
evalInterval = 300
learningRate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # runs on GPU if available
evalIters = 200
nEmbed = 32

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

with open('input.txt', 'r', encoding='utf-8') as file:
    text = file.read()

chars = sorted(set(text))
vocabSize = len(chars)

# Tokenization
stoi = {ch: i for i, ch in enumerate(chars)}  # dict mapping all chars to their ASCII values
itos = {i: ch for i, ch in enumerate(chars)}  # dict mapping all ASCII values (ints) to their chars
encode = lambda s: [stoi[c] for c in s]  # encoder --> string to list of ints
decode = lambda l: ''.join([itos[i] for i in l])  # decoder --> list of ints to string

data = torch.tensor(encode(text), dtype=torch.long)  # makes the basic datatype in PyTorch, like a NumPy array

n = int(0.9 * len(data))
trainData = data[:n]
valData = data[n:]

def getBatch(split):
    """
    Function that generates a small batch of 
    data of inputs x and targets y
    """
    if split == 'train':
        data = trainData
    else:
        data = valData

    ix = torch.randint(len(data) - blockSize, (batchSize,))  # randomly generated offsets into the dataset
    x = torch.stack([data[i:i + blockSize] for i in ix])
    y = torch.stack([data[i + 1:i + blockSize + 1] for i in ix])
    x, y = x.to(device), y.to(device)  # load data into relevant device
    return x, y



@torch.no_grad()  # telling PyTorch to not store intermediate vars, makes it more efficient
def estimateLoss():
    out = {}
    model.eval()  # setting to eval mode
    for split in ['train', 'val']:
        losses = torch.zeros(evalIters)
        for k in range(evalIters):
            X, Y = getBatch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # setting to train mode
    return out



class Head(nn.Module):
    """
    One head of self attention
    """
    def __init__(self, headSize):
        super().__init__()
        self.key = nn.Linear(nEmbed, headSize, bias=False)
        self.query = nn.Linear(nEmbed, headSize, bias=False)
        self.value = nn.Linear(nEmbed, headSize, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(blockSize, blockSize)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, headSize)
        q = self.query(x)  # (B, T, headSize)

        # No communication has happened yet, now time for communication
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B, T, headSize) @ (B, headSize, T) --> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # ensuring tokens from the past can't communicate
        wei = F.softmax(wei, dim=-1)  # (B, T, T)

        v = self.value(x)  # (B, T, headSize)
        out = wei @ v  # (B, T, T) @ (B, T, headSize) --> (B, T, headSize)
        return out


class MultiHeadAttention(nn.Module):
    """
    multiple heads of attention in parallel
    """

    def __init__(self, numHeads, headSize):
        super().__init__()
        self.heads = nn.ModuleList((Head(headSize) for _ in range(numHeads)))
    
    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)
    
    
class FeedForward(nn.Module):
    """
    A simple linear later followed by a non-linearity
    """

    def __init__(self, nEmbed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nEmbed, nEmbed),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block"""

class BigramLanguageModel(nn.Module):  # simplest language model
    def __init__(self):
        super().__init__()
        # Token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocabSize, nEmbed)
        self.position_embedding_table = nn.Embedding(blockSize, nEmbed)
        self.sa_head = MultiHeadAttention(4, nEmbed//4)
        self.lm_head = nn.Linear(nEmbed, vocabSize)
        self.ffwd = FeedForward(nEmbed)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tokEmb = self.token_embedding_table(idx)  # (Batch, Time, nEmbed)
        posEmb = self.position_embedding_table(torch.arange(T, device=device))  # (T, nEmbed)
        x = tokEmb + posEmb  # (B, T, nEmbed)
        x = self.sa_head(x)  # Apply one layer of self attention
        x = self.ffwd(x)
        logits = self.lm_head(x)  # (B, T, vocabSize)
        # logits = scores for next char in the series

        # Transform logits, targets shape to better conform to what PyTorch expects
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            # - log likelihood loss (cross-entropy)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, maxNewTokens):
        # idx is (B, T) array of indices in current context
        for _ in range(maxNewTokens):
            idxCond = idx[:, -blockSize:]  # cropping context
            logits, loss = self(idxCond)  # get the new preds (logits)
            logits = logits[:, -1, :]  # transforms from (B, T, C) to (B, C)
            probs = F.softmax(logits, dim=-1)  # apply softmax to get probs
            idxNext = torch.multinomial(probs, num_samples=1)  # sample from the distribution (B, 1)
            idx = torch.cat((idx, idxNext), dim=1)  # append sampled index to running seq (B, T + 1)
        return idx

model = BigramLanguageModel()
m = model.to(device)  # make sure the embedding table, etc, is moved to the correct device

optimizer = torch.optim.AdamW(m.parameters(), lr=learningRate)  # AdamW optimizer

# Training loop
for iter in range(maxIters):
    # Every once in a while, evaluate the loss on train and val data
    if iter % evalInterval == 0:
        losses = estimateLoss()
        print(f"Step {iter}: train loss = {losses['train']:.4f}, val loss = {losses['val']:.4f}")

    # Sample a batch of data
    xb, yb = getBatch('train')

    # Evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate from model
context = torch.zeros((1, 1), dtype=torch.long)  # creating a tensor (batch = 1, time = 1), initialized with 0
print(decode(m.generate(context, maxNewTokens=400)[0].tolist()))  # asking for 400 tokens