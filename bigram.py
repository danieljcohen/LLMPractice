import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparams
batchSize = 32 #num of independent sequences to process in parallel
blockSize = 8 #max context length for prediction
maxIters = 3000
evalInterval = 300
learningRate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
evalIters = 200

torch.manual_seed(1337)


#wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

with open('input.txt', 'r', encoding = 'utf-8') as file:
    text = file.read()

chars = sorted(set(text))
vocabSize = len(chars)

#tokenization, redo with google SentencePiece or openAI tiktoken
stoi = {ch:i for i, ch in enumerate(chars)}  #dict mapping all chars to their sascii vals
itos = {i:ch for i, ch in enumerate(chars)}  #dict mapping all ascii vals(ints) to their chars
encode = lambda s: [stoi[c] for c in s] #lambda func encoder --> string to list of ints
decode = lambda l: ''.join([itos[i] for i in l]) #decoder --> list of ints to str

data = torch.tensor(encode(text), dtype = torch.long) #makes the basic datatype in spytorch, like a numpy array

n = int(.9 * len(data))
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
        valData

    ix = torch.randint(len(data) - blockSize, (batchSize,)) #4 randomly generated numbers that are offsets into the training setss
    x = torch.stack([data[i:i + blockSize] for i in ix])
    y = torch.stack([data[i + 1 : i + blockSize + 1] for i in ix])
    return x, y

@torch.no_grad()
def estimateLoss():
    out = {}
    model.eval()
    for split in ['train', 'eval']:
        losses = torch.zeros(evalIters)
        for k in range(evalIters):
            X, Y = getBatch(split)
            logit, loss = model(X, Y)
            losses[k] = losses.item()
        out[split] = losses.mean()
    model.train()
    return out

class BigramLanguageModel(nn.Module): #simplest language model

    def __init__(self, vocabSize):
        super().__init__()
        #token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocabSize, vocabSize)

    def forward(self, idx, targets=None):
        #idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) #(Batch, Time, Channel (vocabSize))
        #logits = scores for next char in the series

        #transform logits, targets shape to better conform to what pytorch expects 
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # - log likelihood loss (cross-entropy)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    #history is not used right now, but will be used later
    def generate(self, idx, maxNewTokens):
        #idx is (B, T) array of indices in curr context
        for _ in range(maxNewTokens):
            logits, loss = self(idx) # get the new preds (logits)
            logits = logits[:, -1, :] #transforms from B, C to B, T, C
            probs = F.softmax(logits, dim=-1) #apply softmax to get probs
            idxNext = torch.multinomial(probs, num_samples = 1) # sample from the distr (B, 1)
            idx = torch.cat((idx, idxNext), dim=1) #append samled index to running seq(B, T + 1)
        return idx

model = BigramLanguageModel(vocabSize)
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3) #AdamW as opposed to SGD, more advanced and popular

for iter in range(maxIters):

    #every once in a while eval the loss on train and val data
    if iter % evalInterval == 0:
        losses = estimateLoss()
        print(f"Step {iter}: train loss = {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    #sample a batch of datas
    xb, yb = getBatch('train')

    #eval the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#generate from model
context = torch.zeros((1, 1), dtype = torch.long) #creating  a tensore (batch = 1x time = 1), initialized with 0
print(decode(m.generate(context, maxNewTokens=400)[0].tolist())) # asking for 100 tokens