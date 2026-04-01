# MLP Name Generator

A name generator built with PyTorch that uses a Multi Layer Perceptron 
to learn patterns from 32,000 real names and invent new human-like names.

This is built on top of my 
previous bigram (bigram-name-generator) model — but smarter!

---

## Sample generated names

amaritish, freigaely, norianiela, meily, joskailo, avio, kira, dane,
jakenley, faimooy, zimus, kalen, adeesaia, rufaraan, ziv, elina, fina,
alishi, hantabianna, philey, brena, khaira, keilishka, lani, cashlynn,
yeshur, tarek, atill, viah, diayshi, sannady, christeriks, conna, nair,
fuell, autum, zed, aya, adela, harlynn, nea, xani, damournie

---

## Advantages over bigram?

| | Bigram  | MLP  |
|---|---|---|
| Looks at | 1 character | 3 characters |
| Layers | 1 | 3 |
| Names quality | okay | much better! |

---

## How it works

1. Reads 32,000 real names from a dataset
2. Builds a context window of 3 characters at a time
3. Each character is turned into 5 numbers (embedding)
4. Those numbers pass through 2 neural network layers
5. Model outputs a probability for every possible next character
6. We sample from those probabilities to generate new names!

---

## Files

| File | Description |
|---|---|
| `mlp.py` | The MLP model code |
| `names.txt` | Dataset of 32,000 names |

---

## How to run

1. Install PyTorch
   ```pip install torch```

2. Run the model
   ```python mlp.py```

---

## What I learned

- What a context window is and why it matters
- How embedding layers represent characters as vectors
- How Linear layers transform data
- What tanh activation does and why we need it
- The difference between logits, softmax and probabilities
- How multinomial sampling picks the next character
- How to use a GPU instead of CPU

---

⭐ If you like this project, give it a star!
Have fun ❤️
