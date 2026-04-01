import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#Using GPU for faster processing

words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
chars = ['.'] + chars

vocab_size = len(chars)

stoi = {s:i for i, s in enumerate(chars)}
itos = {i:s for i, s in enumerate(chars)}

block_size = 3
xs = []
ys = []

for word in words:
  context = [0, 0, 0]
  for ch in word + '.':
    xs.append(context)
    ys.append(stoi[ch])
    context = context[1:] + [stoi[ch]]

xs = torch.tensor(xs).to(device)
ys = torch.tensor(ys).to(device)

class MLP(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, 5)
    self.layer1 = nn.Linear(15, 1000)
    self.layer2 = nn.Linear(1000, vocab_size)

  def forward(self, x):
    emb = self.embedding(x)
    emb = emb.view(-1, 15)
    out = torch.tanh(self.layer1(emb))
    out = self.layer2(out)
    return out

model = MLP(vocab_size).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(1000):
  logits = model(xs)
  loss = loss_fn(logits, ys)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  if epoch % 100 == 0:
    print(loss)

def generate_name(model, block_size=3, max_length=15):
  model.eval()
  name = []
  context = [0] * block_size

  with torch.no_grad():
    while True:
      x = torch.tensor([context]).to(device)
      logits = model(x)
      probs = torch.softmax(logits, dim=-1)
      next_char = torch.multinomial(probs, num_samples=1).item()
      if next_char == 0 or len(name) >= max_length:
        break
      name.append(itos[next_char])
      context = context[1:] + [next_char]
  return ''.join(name)

for i in range(50):
  print(generate_name(model))