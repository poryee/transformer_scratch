import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers

gpus = tf.config.list_physical_devices('GPU')

print(gpus)
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel

batch_size_train = 800000
batch_size_val = 100000

block_size = 32 # what is the maximum context length for predictions
max_iters = 1000
eval_iters = 200
eval_interval = 100
learning_rate = 1e-3


n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
# ------------


# read it in to inspect it
with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("length of dataset in characters: ", len(text))

# let's look at the first 1000 characters
print(text[:1000])

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# unique 64 char
print(''.join(chars))
# count
print(vocab_size)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

# encode lambda function to map string to integer iteratively using dict vice versa
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

encoded=encode(text)
data = tf.constant(encoded, dtype=tf.int64)
# data shape
print(data.shape, data.dtype)
# the 1000 characters we looked at earier will to the GPT look like this
print(data[:1000])

# Let's now split up the data into train and validation sets
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


def estimate_loss():
    out = {}
    for split in ['train', 'val']:
        losses = []
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, _ = model(X, training=False)  # Ignore the loss output
            loss = tf.keras.losses.sparse_categorical_crossentropy(Y, logits, from_logits=True)
            losses.append(loss)
        out[split] = tf.reduce_mean(losses)
    return out


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = tf.random.uniform((batch_size,), maxval=len(data) - block_size, dtype=tf.int64)
    x = tf.stack([data[i:i+block_size] for i in ix])
    y = tf.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

def get_batch_split(split):

    if split == 'train':
        data = train_data
        batch=batch_size_train
    else:
        data= val_data
        batch=batch_size_val
    ix = tf.random.uniform((batch,), maxval=len(data) - block_size, dtype=tf.int64)
    x = tf.stack([data[i:i+block_size] for i in ix])
    y = tf.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


# self attention head
class Head(layers.Layer):

    def __init__(self, head_size):
        super().__init__()
        self.key = layers.Dense(head_size, use_bias=False)
        self.query = layers.Dense(head_size, use_bias=False)
        self.value = layers.Dense(head_size, use_bias=False)
        self.tril = tf.linalg.band_part(tf.ones((block_size, block_size)), -1, 0)

    def call(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        wei = tf.matmul(q, k, transpose_b=True) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = tf.where(self.tril[:T, :T] == 0, float('-inf'), wei) # (B, T, T)
        wei = tf.nn.softmax(wei, axis=-1) # (B, T, T)
        v = self.value(x) # (B,T,C)
        out = tf.matmul(wei, v) # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class BigramLanguageModel(keras.Model):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_layer = layers.Embedding(vocab_size, n_embd)
        self.position_embedding_layer = layers.Embedding(block_size, n_embd)

        # add sa head
        self.sa_head = Head(n_embd)

        # final layer that maps back to vocab size for all possible output to be decoded
        self.lm_head = layers.Dense(vocab_size)



    def call(self, idx):
        B, T = idx.shape
        # so here we just passing char position to predict what comes next using just embedding lookup
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_layer(idx) # (B,T,C)
        pos_emb = self.position_embedding_layer(tf.range(T, dtype=tf.int64)) # (T,C)

        # combine embedding with positional information
        x = tok_emb + pos_emb
        # single head
        x= self.sa_head(x)

        # multi head implementation and layer norm

        #x = self.blocks(x) # (B,T,C)
        #x = self.ln_f(x) # (B,T,C)
        return self.lm_head(x)


        # if targets is None:
        #     loss = None
        # else:
        #     # stretch out b*t cause pytorch expects (b, c, t)
        #     logits = tf.reshape(logits, [B*T, -1])
        #     targets = tf.reshape(targets, [B*T])
        #     loss = tf.keras.losses.sparse_categorical_crossentropy(targets, logits)
        #
        # return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            # sliding window cause positional emb is based on block side if not will index out of bound
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits = self.call(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = tf.nn.softmax(logits, axis=-1)  # (B, C)
            # sample from the distribution
            idx_next = tf.random.categorical(probs, num_samples=1, dtype=tf.int64)  # (B, 1)
            # append sampled index to the running sequence (very important for nlp we adding to x+=x to find next y)
            idx = tf.concat([idx, idx_next], axis=1)  # (B, T+1)
        return idx

model = BigramLanguageModel()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#model.compile(optimizer=optimizer)

train_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy",metrics=["accuracy"])


xb, yb = get_batch('train')
xv, yv = get_batch_split('val')

model.fit(x=xb, y=yb, epochs=max_iters, validation_data=(xv,yv))


#for iter in range(max_iters): # increase number of steps for good results...

    # every once in a while evaluate the loss on train and val sets
    # if iter % eval_interval == 0 or iter == max_iters - 1:
    #     losses = estimate_loss()
    #     print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    #xb, yb = get_batch('train')


    # Train for one step
    #history = model.train_on_batch(xb, yb)


    # # evaluate the loss
    # with tf.GradientTape() as tape:
    #     logits, loss = model(xb, yb)
    # gradients = tape.gradient(loss, model.trainable_variables)
    # optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# 2.5 loss with just tok and pos emb
# 2.27 loss with single self attention head
print(model.loss)

# generate from the model
context = tf.zeros((1, 1), dtype=tf.int64)
res= model.generate(context, max_new_tokens=2000)
print(decode(res.numpy().tolist()[0]))

