import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparams (lecture)
# batch_size = 32
# block_size = 8  # Maximum context length for prediction
# emb_size = 32
# num_heads = 4
# num_layers = 4  # Number of consecutive transformer blocks
# max_iters = 5000
# eval_interval = 500
# eval_iters = 200
# learning_rate = 1e-3
# dropout = 0.2
# ------

# hyperparams (compromise for CPU, ~6hr training without parallelization)
batch_size = 64
block_size = 128  # Maximum context length for prediction
emb_size = 128
num_heads = 4
num_layers = 6  # Number of consecutive transformer blocks
max_iters = 10000
eval_interval = 250
eval_iters = 100
learning_rate = 5e-4
dropout = 0.2
# ------


device = "cpu"
if hasattr(torch.backends.mps, "is_available") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = "mps"
if hasattr(torch.backends.cuda, "is_available") and torch.backends.cuda.is_available():
    device = "cuda"
# TODO fix temporary override due to mpu-related bug with torch.multinomial:
device = "cpu"  # Hmm, CPU is twice as fast as MPS for some reason (even though MPS uses the GPU according to activity monitor) :/

torch.manual_seed(1337)

# Get dataset
# !curl "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt" >input.txt

with open('input.txt', 'r', encoding="utf-8") as f:
    text = f.read()
print(f"Loaded data: {len(text)=}")

# Tokenization/Encoding

# There are other encoding tools (byte-pair encoding based) like Google's SentencePiece
# (which uses sub-word units). Also: OpenAI's tiktoken). They have ca. 50000 possible tokens.
# Cf. my syllable-tokenization approach, which is maybe a good compromise without dependencies
vocab = sorted(list(set(text)))
print(f"{len(vocab)=}\n{''.join(vocab)=}")
tok2idx = { c:i for i, c in enumerate(vocab) }

def encode(s):  # string to list of integers
    return [tok2idx[c] for c in s]

def decode(l):  # list of integers to string
    return ''.join([vocab[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

# Train-validation split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"{len(train_data)=}, {len(val_data)=}")

def get_batch(split):
    data = train_data if split == 'train' else val_data
    num_blocks = len(data) - block_size
    shuffled_batch_indices = torch.randint(num_blocks, (batch_size,))  # 1D
    x = torch.stack([data[i:i+block_size] for i in shuffled_batch_indices])  # 2D: N x T
    y = torch.stack([data[i+1:i+block_size+1] for i in shuffled_batch_indices])  # one y for every time-shifted part of block-size
    # But the time-shifted stucture is not created here, but later.
    # e.g. if we return x:[3, 4, 7], y:[4, 7, 2] then this will later be used as:
    # [3] -> [4]
    # [3, 4] -> [7]
    # [3, 4, 7] -> [2]
    return x.to(device), y.to(device)  # move the data to the device

# xb, yb = get_batch('train')
# print('inputs')
# print(xb.shape)
# print(xb)
# print('targets')
# print(yb.shape)
# print(yb)
#
# print('----')

# Example for building the training samples as outlined in the comments in `get_batch`:
# for b in range(batch_size):  # batch dimension
#     for t in range(block_size):  # time dimension
#         context = xb[b, :t+1]
#         target = yb[b, t]
#         print(f"when input is {context.tolist()}, the target is {target}")


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()  # enter evaluation mode (no tracking of gradients etc.)
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, y = get_batch(split)
            logits, loss = model(X, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # return model to training mode
    return out


class Head(nn.Module):
    """ Self-attention head """

    def __init__(self, head_size):
        super().__init__()
        #print(f"{emb_size=}")
        #print(f"{head_size=}")
        # See bottom of nanogpt notebook for explanation
        self.key = nn.Linear(emb_size, head_size, bias=False)  # Typically no biases are used
        self.query = nn.Linear(emb_size, head_size, bias=False)  # Typically no biases are used
        self.value = nn.Linear(emb_size, head_size, bias=False)  # Typically no biases are used
        # Triangular matrix, need it as parameter (have to register it as pytorch buffer)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))  # block_size = T

        self.dropout = nn.Dropout(dropout)  # regularization (as we scale up the model, we might overfit drastically)

    def forward(self, x):
        N, T, C_head = x.shape  # 'C' here is the head embedding size, not necessarily equal to the token embedding size
        key = self.key(x)      # N x T x C_head  -- which kinds of tokens to share info with
        query = self.query(x)  # N x T x C_head  -- which kinds of tokens to desire info from

        # Match up keys & queries (determines flow of communication between tokens)
        # Keyword: scaled dot product attention
        wei = query @ key.transpose(-2, -1) * C_head**-0.5  # N x T x T
        # The masking step is optional (not wanted for encoder blocks, e.g. for sentiment analysis).
        # Our generative thing is a decoder block
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # [:T,:T] to shrink tril with T of x if needed
        wei = wei.softmax(dim=-1)  # N x T x T
        wei = self.dropout(wei)  # randomly keep some of these nodes from communicating

        value = self.value(x)  # Generate info to communicate to matched-up tokens
        out = wei @ value  # Actually share information. BxTxT @ BxTxC --> BxTxC
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(emb_size, emb_size)  # For the residual pathway/connection
        self.dropout = nn.Dropout(dropout)  # regularization (as we scale up the model, we might overfit drastically)

    def forward(self, x):
        # Multi-head attention: simply concatenate head results along the channel dimension (last dimension, index -1).
        # This concatenation makes the channel longer, of course, and, since we don't want extremely long
        # channels, is the reason for choosing head_size = emb_size/num_heads
        # According to Andrej, this is kind of similar to group convolutions
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.projection(x)
        out = self.dropout(x)
        return out


class FeedForward(nn.Module):
    """ A simple MLP to process information that has been shared previously
    """
    def __init__(self, emb_size):
        super().__init__()
        # remember: we're acting on last dimension = channel = single-token level
        self.net = nn.Sequential(
            nn.Linear(emb_size, 4 * emb_size),  # inner layer should be 4x as wide as input/output according to paper
            nn.ReLU(),
            nn.Linear(4 * emb_size, emb_size),  # For the residual pathway/connection
            nn.Dropout(dropout),  # regularization (as we scale up the model, we might overfit drastically)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """ Grouping the parts of a repeatable transformer block according to Attention is all you Need paper.
    "Communication", followed by "computation"
    """
    def __init__(self, emb_size, num_heads):
        super().__init__()
        head_size = emb_size // num_heads  # keep total channel size constant at emb_size
        self.self_attention_heads = MultiHeadAttention(num_heads, head_size)
        self.feed_forward = FeedForward(emb_size)
        # Note that we'll apply the "pre-norm": While in the original paper, normalization was
        # applied *after* the self-att & ff steps, we now apply it *before*:
        self.layer_norm1 = nn.LayerNorm(emb_size)
        self.layer_norm2 = nn.LayerNorm(emb_size)

    def forward(self, x):
        # The addition of x is part of the residual pathway (gradients can be passed straight through)
        # Part of this are also the projections by additional linear layers in the heads and ffwd modules.
        x = x + self.self_attention_heads(self.layer_norm1(x))  # Apply one head of self-attention, NxTxC
        out = x + self.feed_forward(self.layer_norm1(x))
        return out


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Dimension shortcuts used:
        # N = (mini)batch dimension, batch_size (also called B by Andrej)
        # T = time dimension, block_size
        # C = channel dimension (number of "features"), emb_size (can also be head_size, depending on context)
        self.token_embedding_table = nn.Embedding(len(vocab), emb_size)  # encode the token identities
        self.position_embedding_table = nn.Embedding(block_size, emb_size)  # encode the positions as well
        self.attn_blocks = nn.Sequential(
            *[TransformerBlock(emb_size, num_heads=num_heads) for _ in range(num_layers)],
        )
        self.layer_norm_f = nn.LayerNorm(emb_size)  # Typically, there's another normalization at the end, before creating logits
        self.lm_head = nn.Linear(emb_size, len(vocab))

    def forward(self, x, y=None):
        B, T = x.shape
        # Embedding expects N x T as well, and outputs N x T x C
        token_embeddings = self.token_embedding_table(x)  # N x T x C (C=embed_size)
        position_embeddings = self.position_embedding_table(torch.arange(T, device=device))  # T x C
        combined_embeddings = token_embeddings + position_embeddings  # N x T x C (pos_emb gets broadcast along N)
        out = self.attn_blocks(combined_embeddings)  # NxTxC
        out = self.layer_norm_f(out)
        logits = self.lm_head(out)  # N x T x vocab_size
        # print(f"{logits.shape=}")
        # print(f"{logits=}")
        loss = None

        if y is not None:
            # cross_entropy expects C as the second dimension, so we need to reshape our logits tensor
            N, T, C = logits.shape
            logits_view = logits.view(N * T,
                                      C)  # We keep the C dimension intact, just "stack them above each other" instead of in a NxT matrix
            target_view = y.view(N * T)  # align targets to reshaped logits
            # print(f"{logits_view.shape=}, {target_view.shape=}")
            loss = F.cross_entropy(logits_view, target_view)

        return logits, loss

    def generate(self, context, max_new_tokens):  # TODO BUG: only produces newlines with device=mps https://github.com/pytorch/pytorch/issues/92752
        # context: N x T matrix of token indices
        for _ in range(max_new_tokens):
            # crop context length to `block_size` num of tokens (so we don't exceed the dimensions of our positional embeddings)
            idx_cond = context[:, -block_size:]
            #print(f"{idx_cond.shape=}")
            logits, _ = self(idx_cond)  # N x T x C
            logits = logits[:, -1, :]  # Drop all but most recent tokens --> N x C
            # print(f"{logits.shape=}")
            probs = F.softmax(logits, dim=-1)  # N x C
            #print(f"{probs=}")
            #print(f"{probs.shape=}")
            new_token = torch.multinomial(probs, num_samples=1)  # N x 1
            #print(f"{new_token=}")
            context = torch.cat((context, new_token), dim=1)  # N x T+1
        return context


model = BigramLanguageModel()
model = model.to(device)  # move the model parameters to the device
# logits, loss = model(xb, yb)
# print("Untrained model loss:")
# print(f"{loss.item()=}")
# print("Untrained model output:")
# print(decode(model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

# Used SGB before, but AdamW is really good so we'll use it now
# Suitable LR for AdamW is usually 1e-4, but for small models you can get away with 1e-3 or even higher
# But self-attention can't tolerate very high learning rates, so 1e-3 or smaller it is.
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training run
from time import time
print(f"Training model with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters for {max_iters} steps...")
start_time = time()
for iter in range(max_iters):
    # Evaluate every once in a while
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        end_time = time()
        print(f"Step {iter} training loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}, time: {(end_time-start_time)/(iter+1):.4f} s/iter")

    # Sample a batch of data
    xb, yb = get_batch('train')

    # evaluate loss and optimize
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
end_time = time()
print(f"Training {max_iters} iterations took {end_time-start_time:.4f} seconds"
      f" ({(end_time-start_time)/max_iters:.4f} s/iter).")

print("Trained model output:")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

prompt = "BOZO THE CLOWN:"
print(decode(model.generate(torch.tensor([encode(prompt)], dtype=torch.long, device=device), max_new_tokens=100)[0].tolist()))
