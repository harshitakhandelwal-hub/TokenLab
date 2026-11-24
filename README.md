# TokenLab

A browser-based LLM tokenizer and vocabulary visualizer. It trains a real
tokenizer on text you type, builds a vocabulary with token IDs, encodes and
decodes new sentences, charts how often each token is used, and trains real
token embeddings (word2vec) that you can explore in 3D.

## Run it

No install, no server, no internet needed (fonts fall back to system fonts offline).

1. Unzip the folder.
2. Double-click `index.html` (or drag it into Chrome, Edge, Firefox or Safari).

## What's inside

```
index.html          page structure, one section per pipeline step
css/styles.css      all styling
js/tokenizer.js     the tokenizer itself: no DOM code, works in Node too
js/presets.js       sample training corpora and test sentences
js/chart.js         frequency charts drawn as plain SVG
js/embedding.js     word2vec skip-gram training + PCA down to 3D
js/chart3d.js       small hand-written 3D renderer (canvas) for the embedding view
js/app.js           connects the controls to the tokenizer and draws the UI
```

## The two algorithms (js/tokenizer.js)

**Words + pieces (WordPiece-style, like BERT and the class demo)**
1. Lowercase, then split text into words and punctuation.
2. Keep the N most frequent words as whole tokens.
3. Add every single character as a piece (`c` at the start of a word, `##c` inside one).
4. Encode a new word by greedily taking the longest piece the vocabulary knows.
   Anything with no matching piece becomes `[UNK]`.

**BPE merges (Byte-Pair Encoding, like GPT)**
1. Start with every word split into characters.
2. Count all neighbouring pairs, weighted by word frequency.
3. Glue the most frequent pair into a new token. Repeat for N merges
   (stops early when no pair appears twice).
4. Encode a new word by replaying the merges in the order they were learned.

## Embeddings (js/embedding.js)

Each token gets 16 random numbers. word2vec (skip-gram with negative sampling)
then trains them on the training corpus: for every token, its neighbours within
2 positions should score high and random tokens should score low. Tokens used
in similar places end up pointing the same way. PCA squashes the 16 numbers to
3 so they can be drawn. With a tiny corpus, expect rough groupings only:
real LLMs learn embeddings with thousands of dimensions from billions of tokens.

Special tokens: `[PAD]` = 0, `[UNK]` = 1, `[BOS]` = 2, `[EOS]` = 3.

## Use the tokenizer from Node

```js
const T = require('./js/tokenizer.js');
const tk = T.train('The cat sat on the mat.', {
  algorithm: 'wordpiece', lowercase: true, maxWords: 20,
  subwordFallback: true, baseAlphabet: true,
});
const enc = T.encode(tk, 'The zebra sat.', { addBos: true, addEos: true });
console.log(enc.items.map(i => i.token), enc.ids);
console.log(T.decode(tk, enc.ids, { skipSpecial: true }).text);
```
