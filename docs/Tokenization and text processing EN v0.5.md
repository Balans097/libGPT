# API Documentation: Tokenization Library

**Version:** 0.5  
**Date:** 2026-01-31  
**Author:** github.com/Balans097

---

## Table of Contents

1. [Introduction](#introduction)
2. [Data Types](#data-types)
3. [Training Tokenizers](#training-tokenizers)
4. [Tokenization and Decoding](#tokenization-and-decoding)
5. [Batch Processing](#batch-processing)
6. [Text Processing](#text-processing)
7. [Metrics and Analysis](#metrics-and-analysis)
8. [Saving and Loading](#saving-and-loading)
9. [Utilities](#utilities)
10. [Advanced Features](#advanced-features)

---

## Introduction

This library provides a comprehensive set of tools for text tokenization with support for four main algorithms:
- **BPE** (Byte Pair Encoding) — classic algorithm
- **WordPiece** — used in BERT
- **SentencePiece** — universal algorithm
- **ByteLevel BPE** — used in GPT-2/3

### Key Features:
- ✅ Full Unicode and Cyrillic support
- ✅ Caching for faster repeated tokenizations
- ✅ Streaming processing for large files
- ✅ Token position tracking (for NER/QA)
- ✅ BPE-dropout for regularization
- ✅ Export vocabularies to JSON
- ✅ Batch processing with padding/truncation

---

## Data Types

### TokenizerKind
```nim
type TokenizerKind* = enum
  tkBPE = 0              # Byte Pair Encoding
  tkWordPiece = 1        # WordPiece (BERT-style)
  tkSentencePiece = 2    # SentencePiece (universal)
  tkByteLevelBPE = 3     # ByteLevel BPE (GPT-2/3)
```

### SpecialTokens
```nim
type SpecialTokens* = object
  padToken*: string      # Padding token
  unkToken*: string      # Unknown token
  bosToken*: string      # Beginning of sequence token
  eosToken*: string      # End of sequence token
  sepToken*: string      # Separator token
  clsToken*: string      # Classification token
  maskToken*: string     # Mask token
```

### Tokenizer
```nim
type Tokenizer* = ref object
  kind*: TokenizerKind                    # Tokenizer type
  vocab*: Table[string, int]              # Vocabulary: token -> ID
  inverseVocab*: seq[string]              # Inverse vocab: ID -> token
  merges*: seq[BPEMerge]                  # BPE merges (BPE only)
  specialTokens*: SpecialTokens           # Special tokens
  specialTokenIds*: Table[string, int]    # Special token IDs
  maxInputCharsPerWord*: int              # Max word length (WordPiece)
  continuingSubwordPrefix*: string        # Subword prefix (usually "##")
  scores*: Table[string, float]           # Scores (SentencePiece)
  preserveCase*: bool                     # Preserve case
  cache*: Table[string, seq[int]]         # Tokenization cache
  cacheMaxSize*: int                      # Cache size
  cacheHits*: int                         # Cache hits
  cacheMisses*: int                       # Cache misses
  byteEncoder*: Table[int, string]        # Byte encoder (ByteLevel BPE)
  byteDecoder*: Table[string, int]        # Byte decoder (ByteLevel BPE)
```

### TokenOffset
```nim
type TokenOffset* = object
  token*: string         # Token text
  tokenId*: int          # Token ID
  startChar*: int        # Start position (in characters)
  endChar*: int          # End position (in characters)
  startByte*: int        # Start position (in bytes)
  endByte*: int          # End position (in bytes)
```

### BatchEncoding
```nim
type BatchEncoding* = object
  inputIds*: seq[seq[int]]        # Token IDs for each text
  attentionMask*: seq[seq[int]]   # Attention masks
  tokenTypeIds*: seq[seq[int]]    # Token type IDs
  lengths*: seq[int]              # Sequence lengths
```

### TokenizerMetrics
```nim
type TokenizerMetrics* = object
  vocabSize*: int              # Vocabulary size
  compressionRatio*: float     # Compression ratio
  avgTokensPerWord*: float     # Average tokens per word
  vocabUtilization*: float     # Vocabulary utilization (0.0-1.0)
  unkTokenRate*: float         # Unknown token rate
  tokensPerSecond*: float      # Tokenization speed
```

### VocabAnalysis
```nim
type VocabAnalysis* = object
  vocabSize*: int                                        # Vocabulary size
  avgTokenLength*: float                                 # Average token length
  typeTokenRatio*: float                                 # Type/Token ratio
  coverageRate*: float                                   # Corpus coverage
  oovRate*: float                                        # Out-of-vocabulary rate
  mostFrequent*: seq[tuple[token: string, freq: int]]    # Top tokens
  leastFrequent*: seq[tuple[token: string, freq: int]]   # Rare tokens
  lengthDistribution*: CountTable[int]                   # Length distribution
```

---

## Training Tokenizers

### trainBPE
Train a BPE (Byte Pair Encoding) tokenizer.

```nim
proc trainBPE*(
  corpus: seq[string],
  vocabSize: int = 8000,
  minFrequency: int = 2,
  preserveCase: bool = false
): Tokenizer
```

**Parameters:**
- `corpus` - training corpus
- `vocabSize` - desired vocabulary size (default: 8000)
- `minFrequency` - minimum token frequency (default: 2)
- `preserveCase` - preserve letter case (default: false)

**Returns:** trained tokenizer

**Example:**
```nim
let corpus = @["Hello world", "Text tokenization"]
var tokenizer = trainBPE(corpus, vocabSize = 1000)
```

---

### trainWordPiece
Train a WordPiece tokenizer (used in BERT).

```nim
proc trainWordPiece*(
  corpus: seq[string],
  vocabSize: int = 8000,
  minFrequency: int = 2,
  continuingSubwordPrefix: string = "##",
  preserveCase: bool = false
): Tokenizer
```

**Parameters:**
- `corpus` - training corpus
- `vocabSize` - desired vocabulary size (default: 8000)
- `minFrequency` - minimum frequency (default: 2)
- `continuingSubwordPrefix` - subword continuation prefix (default: "##")
- `preserveCase` - preserve letter case (default: false)

**Returns:** trained tokenizer

**Features:**
- Supports lowercase fallback for uppercase letters
- Preserves spaces as separate tokens
- Optimized for long Cyrillic words (n-grams up to 15 characters)

**Example:**
```nim
var tokenizer = trainWordPiece(corpus, vocabSize = 1500, preserveCase = true)
```

---

### trainSentencePiece
Train a SentencePiece tokenizer.

```nim
proc trainSentencePiece*(
  corpus: seq[string],
  vocabSize: int = 8000,
  characterCoverage: float = 0.9995,
  preserveCase: bool = false
): Tokenizer
```

**Parameters:**
- `corpus` - training corpus
- `vocabSize` - desired vocabulary size (default: 8000)
- `characterCoverage` - character coverage (default: 0.9995)
- `preserveCase` - preserve letter case (default: false)

**Returns:** trained tokenizer

**Features:**
- Uses ▁ symbol to denote spaces
- Assigns scores to each token
- Works well with languages without clear word boundaries

**Example:**
```nim
var tokenizer = trainSentencePiece(corpus, vocabSize = 2000)
```

---

### trainByteLevelBPE
Train a ByteLevel BPE tokenizer (GPT-2/3 style).

```nim
proc trainByteLevelBPE*(
  corpus: seq[string],
  vocabSize: int = 8000,
  minFrequency: int = 2
): Tokenizer
```

**Parameters:**
- `corpus` - training corpus
- `vocabSize` - desired vocabulary size (default: 8000)
- `minFrequency` - minimum frequency (default: 2)

**Returns:** trained tokenizer

**Features:**
- Each byte is mapped to a unique Unicode character
- Guarantees processing of any UTF-8 text without UNK tokens
- Preserves spaces correctly
- Compatible with GPT-2/3

**Example:**
```nim
var tokenizer = trainByteLevelBPE(corpus, vocabSize = 2000)
```

---

## Tokenization and Decoding

### tokenize
Tokenize text into a sequence of token IDs.

```nim
proc tokenize*(
  text: string,
  tokenizer: var Tokenizer,
  flag: int = 0,
  addSpecialTokens: bool = false
): seq[int]
```

**Parameters:**
- `text` - text to tokenize
- `tokenizer` - trained tokenizer
- `flag` - additional flag (unused)
- `addSpecialTokens` - add BOS/EOS tokens (default: false)

**Returns:** sequence of token IDs

**Features:**
- Uses caching for faster repeated tokenizations
- For WordPiece, supports lowercase fallback
- Correctly handles spaces

**Example:**
```nim
let text = "Hello world"
let tokens = tokenize(text, tokenizer, addSpecialTokens = true)
# Result: @[1, 145, 289, 2]  # [BOS, "Hello", "world", EOS]
```

---

### decode
Decode a sequence of token IDs back into text.

```nim
proc decode*(
  tokenizer: Tokenizer,
  tokens: seq[int],
  skipSpecialTokens: bool = false
): string
```

**Parameters:**
- `tokenizer` - tokenizer
- `tokens` - sequence of token IDs
- `skipSpecialTokens` - skip special tokens (default: false)

**Returns:** decoded text

**Features:**
- For WordPiece, removes "##" prefix
- For SentencePiece, replaces ▁ with spaces
- For ByteLevel BPE, correctly decodes UTF-8

**Example:**
```nim
let decoded = tokenizer.decode(tokens, skipSpecialTokens = true)
# Result: "Hello world"
```

---

### tokenizeWithOffsets
Tokenize with token position tracking.

```nim
proc tokenizeWithOffsets*(
  text: string,
  tokenizer: var Tokenizer,
  addSpecialTokens: bool = true
): seq[TokenOffset]
```

**Parameters:**
- `text` - text to tokenize
- `tokenizer` - tokenizer
- `addSpecialTokens` - add special tokens (default: true)

**Returns:** sequence of `TokenOffset` objects with positions

**Use Cases:**
- Named Entity Recognition (NER)
- Question Answering (QA)
- Information extraction
- Token-to-text alignment

**Example:**
```nim
let offsets = tokenizeWithOffsets("Moscow is capital", tokenizer)
for offset in offsets:
  echo offset.token, " -> chars: ", offset.startChar, "..", offset.endChar
# Result:
# "Moscow" -> chars: 0..6
# " " -> chars: 6..7
# "is" -> chars: 7..9
# " " -> chars: 9..10
# "capital" -> chars: 10..17
```

---

### streamTokenize
Streaming tokenization for large files.

```nim
iterator streamTokenize*(
  filePath: string,
  tokenizer: var Tokenizer,
  chunkSize: int = 8192,
  addSpecialTokens: bool = true
): seq[int]
```

**Parameters:**
- `filePath` - file path
- `tokenizer` - tokenizer
- `chunkSize` - chunk size for reading (default: 8192 bytes)
- `addSpecialTokens` - add special tokens (default: true)

**Returns:** iterator over token chunks

**Use Cases:**
- Processing files > 1GB
- Memory efficiency
- Data processing pipelines

**Example:**
```nim
for chunk in streamTokenize("large_file.txt", tokenizer):
  # Process each token chunk
  processTokens(chunk)
```

---

## Batch Processing

### encodeBatch
Batch tokenization with padding and truncation.

```nim
proc encodeBatch*(
  tokenizer: var Tokenizer,
  texts: seq[string],
  maxLength: int = 512,
  padding: bool = true,
  truncation: bool = true,
  addSpecialTokens: bool = true,
  returnAttentionMask: bool = true,
  returnTokenTypeIds: bool = false
): BatchEncoding
```

**Parameters:**
- `tokenizer` - tokenizer
- `texts` - list of texts
- `maxLength` - maximum sequence length (default: 512)
- `padding` - pad to maxLength (default: true)
- `truncation` - truncate long sequences (default: true)
- `addSpecialTokens` - add BOS/EOS (default: true)
- `returnAttentionMask` - return attention mask (default: true)
- `returnTokenTypeIds` - return token type IDs (default: false)

**Returns:** `BatchEncoding` object

**Example:**
```nim
let texts = @["First text", "Second longer text"]
let batch = encodeBatch(tokenizer, texts, maxLength = 10)
echo batch.inputIds        # Tokens with padding
echo batch.attentionMask   # Attention masks
echo batch.lengths         # Actual lengths
```

---

### decodeBatch
Batch decoding.

```nim
proc decodeBatch*(
  tokenizer: Tokenizer,
  tokensBatch: seq[seq[int]],
  skipSpecialTokens: bool = true
): seq[string]
```

**Parameters:**
- `tokenizer` - tokenizer
- `tokensBatch` - list of token sequences
- `skipSpecialTokens` - skip special tokens (default: true)

**Returns:** list of decoded texts

**Example:**
```nim
let decoded = decodeBatch(tokenizer, batch.inputIds)
```

---

## Text Processing

### cleanText
Universal text cleaning.

```nim
proc cleanText*(
  text: string,
  removeHtml: bool = true,
  removeUrls: bool = true,
  removeEmails: bool = true,
  removeExtraWhitespace: bool = true,
  removeEmoji: bool = false,
  removeNumbers: bool = false,
  removePunctuation: bool = false,
  normalizeQuotes: bool = true,
  normalizeDashes: bool = true,
  removeControlChars: bool = true
): string
```

**Parameters:** multiple flags for different types of cleaning

**Returns:** cleaned text

**Example:**
```nim
let cleaned = cleanText(
  "<p>Email: test@example.com</p>",
  removeHtml = true,
  removeEmails = true
)
# Result: ""
```

---

### toLowerUnicode / toUpperUnicode
Convert to lowercase/uppercase with Unicode support.

```nim
proc toLowerUnicode*(s: string): string
proc toUpperUnicode*(s: string): string
```

**Example:**
```nim
echo toLowerUnicode("HELLO")  # "hello"
echo toUpperUnicode("world")  # "WORLD"
```

---

### normalizeText
Basic text normalization.

```nim
proc normalizeText*(text: string): string
```

Performs:
- Removes extra whitespace
- Unicode normalization
- Lowercase conversion

---

### splitIntoWords
Split text into words.

```nim
proc splitIntoWords*(text: string): seq[string]
```

**Features:**
- Correctly handles Unicode
- Accounts for punctuation
- Optimized for Cyrillic

---

### splitIntoSentences
Split text into sentences.

```nim
proc splitIntoSentences*(text: string): seq[string]
```

**Example:**
```nim
let sentences = splitIntoSentences("First. Second! Third?")
# Result: @["First.", "Second!", "Third?"]
```

---

### normalizeWhitespace
Normalize whitespace.

```nim
proc normalizeWhitespace*(
  text: string,
  preserveNewlines: bool = false
): string
```

**Parameters:**
- `text` - text
- `preserveNewlines` - preserve line breaks (default: false)

---

### removeAccents
Remove diacritical marks.

```nim
proc removeAccents*(text: string): string
```

**Example:**
```nim
echo removeAccents("café")  # "cafe"
```

---

### truncateText
Truncate text to maximum length.

```nim
proc truncateText*(
  text: string,
  maxLength: int,
  addEllipsis: bool = true
): string
```

**Example:**
```nim
echo truncateText("Long text", 7)  # "Long..."
```

---

### countWords / countCharacters / countSentences
Count words, characters, sentences.

```nim
proc countWords*(text: string): int
proc countCharacters*(text: string, excludeWhitespace: bool = false): int
proc countSentences*(text: string): int
```

---

## Metrics and Analysis

### getMetrics
Calculate tokenizer metrics.

```nim
proc getMetrics*(
  tokenizer: var Tokenizer,
  corpus: seq[string]
): TokenizerMetrics
```

**Returns:** object with metrics:
- `vocabSize` - vocabulary size
- `compressionRatio` - compression ratio
- `avgTokensPerWord` - average tokens per word
- `vocabUtilization` - vocabulary utilization
- `unkTokenRate` - unknown token rate
- `tokensPerSecond` - tokenization speed

**Example:**
```nim
let metrics = getMetrics(tokenizer, corpus)
echo "Compression ratio: ", metrics.compressionRatio
echo "Vocab size: ", metrics.vocabSize
```

---

### analyzeVocabulary
Analyze tokenizer vocabulary.

```nim
proc analyzeVocabulary*(
  tokenizer: var Tokenizer,
  corpus: seq[string],
  topN: int = 10
): VocabAnalysis
```

**Parameters:**
- `tokenizer` - tokenizer
- `corpus` - corpus for analysis
- `topN` - number of top tokens to return (default: 10)

**Returns:** `VocabAnalysis` object with detailed statistics

**Example:**
```nim
let analysis = analyzeVocabulary(tokenizer, corpus, topN = 20)
echo "Average token length: ", analysis.avgTokenLength
echo "Coverage rate: ", analysis.coverageRate
echo "Most frequent tokens:"
for (token, freq) in analysis.mostFrequent:
  echo "  ", token, ": ", freq
```

---

### printMetrics
Print metrics in readable format.

```nim
proc printMetrics*(metrics: TokenizerMetrics)
```

**Example:**
```nim
printMetrics(metrics)
# Outputs:
# Vocabulary size: 8000
# Compression ratio: 2.45
# ...
```

---

### compareTokenizers
Compare multiple tokenizers on the same text.

```nim
proc compareTokenizers*(
  text: string,
  tokenizers: seq[Tokenizer]
): seq[tuple[kind: TokenizerKind, tokens: seq[int], decoded: string]]
```

**Example:**
```nim
let results = compareTokenizers("Test text", @[bpe, wordpiece, sp])
for result in results:
  echo result.kind, ": ", result.tokens.len, " tokens"
```

---

### benchmark
Measure tokenization speed.

```nim
proc benchmark*(
  tokenizer: var Tokenizer,
  texts: seq[string]
): float
```

**Returns:** tokens per second

**Example:**
```nim
let speed = benchmark(tokenizer, corpus)
echo "Speed: ", speed, " tokens/sec"
```

---

## Saving and Loading

### saveTokenizer
Save tokenizer to disk.

```nim
proc saveTokenizer*(tokenizer: Tokenizer, path: string)
```

**Format:** JSON

**Saves:**
- Tokenizer type
- Full vocabulary
- Special tokens
- Merges (for BPE)
- Scores (for SentencePiece)
- All parameters

**Example:**
```nim
saveTokenizer(tokenizer, "my_tokenizer.json")
```

---

### loadTokenizer
Load tokenizer from disk.

```nim
proc loadTokenizer*(path: string): Tokenizer
```

**Example:**
```nim
let tokenizer = loadTokenizer("my_tokenizer.json")
```

---

### exportTokenizerToJson
Export partial tokenizer data to JSON (for viewing).

```nim
proc exportTokenizerToJson*(tokenizer: Tokenizer, filepath: string)
```

**Saves:**
- First 100 vocabulary entries
- First 50 merges/scores
- All parameters

**Use Case:** for viewing and debugging vocabulary

**Example:**
```nim
exportTokenizerToJson(tokenizer, "vocab_preview.json")
```

---

## Utilities

### Getting Special Tokens

```nim
proc getUnkTokenId*(tokenizer: Tokenizer): int
proc getPadTokenId*(tokenizer: Tokenizer): int
proc getBosTokenId*(tokenizer: Tokenizer): int
proc getEosTokenId*(tokenizer: Tokenizer): int
proc getSepTokenId*(tokenizer: Tokenizer): int
proc getClsTokenId*(tokenizer: Tokenizer): int
proc getMaskTokenId*(tokenizer: Tokenizer): int
```

**Example:**
```nim
let unkId = tokenizer.getUnkTokenId()
let padId = tokenizer.getPadTokenId()
```

---

### Working with Vocabulary

```nim
proc getTokenById*(tokenizer: Tokenizer, id: int): string
proc getIdByToken*(tokenizer: Tokenizer, token: string): int
proc hasToken*(tokenizer: Tokenizer, token: string): bool
proc getVocabSize*(tokenizer: Tokenizer): int
proc getVocabTokens*(tokenizer: Tokenizer): seq[string]
```

**Example:**
```nim
if tokenizer.hasToken("hello"):
  let id = tokenizer.getIdByToken("hello")
  echo "Token 'hello' has ID: ", id
```

---

### Caching

```nim
proc initCache*(maxSize: int): Table[string, seq[int]]
proc clearCache*(tokenizer: var Tokenizer)
```

**Example:**
```nim
tokenizer.clearCache()
echo "Cache hits: ", tokenizer.cacheHits
echo "Cache misses: ", tokenizer.cacheMisses
```

---

## Advanced Features

### tokenizeWithDropout
BPE-dropout for model regularization.

```nim
proc tokenizeWithDropout*(
  text: string,
  tokenizer: var Tokenizer,
  dropoutProb: float = 0.1,
  minDropped: int = 0,
  seed: int = -1
): seq[int]
```

**Parameters:**
- `text` - text to tokenize
- `tokenizer` - BPE tokenizer
- `dropoutProb` - merge skip probability (0.0-1.0)
- `minDropped` - minimum dropped merges
- `seed` - random seed (-1 = random)

**Use Cases:**
- Regularization during model training
- Data augmentation
- Improved generalization

**Example:**
```nim
# Normal tokenization
let normal = tokenize("text", tokenizer)

# With dropout - different tokenization each time
let dropout1 = tokenizeWithDropout("text", tokenizer, 0.3)
let dropout2 = tokenizeWithDropout("text", tokenizer, 0.3)
# dropout1 != dropout2 (most likely)
```

---

### incrementalTrain
Continue training tokenizer on new data.

```nim
proc incrementalTrain*(
  tokenizer: var Tokenizer,
  newCorpus: seq[string],
  maxNewTokens: int = 1000,
  minFrequency: int = 2
)
```

**Parameters:**
- `tokenizer` - existing tokenizer
- `newCorpus` - new texts
- `maxNewTokens` - maximum new tokens to add
- `minFrequency` - minimum frequency

**Use Cases:**
- Domain adaptation
- Adding terminology
- Online learning

**Example:**
```nim
incrementalTrain(tokenizer, newDomainTexts, maxNewTokens = 500)
```

---

### pruneVocabulary
Remove rare tokens from vocabulary.

```nim
proc pruneVocabulary*(
  tokenizer: var Tokenizer,
  corpus: seq[string],
  minFrequency: int = 2,
  keepTopN: int = -1
)
```

**Parameters:**
- `tokenizer` - tokenizer
- `corpus` - corpus for frequency counting
- `minFrequency` - minimum frequency to keep
- `keepTopN` - keep top-N tokens (-1 = all)

**Use Cases:**
- Reduce vocabulary size
- Remove noise
- Production optimization

**Example:**
```nim
# Before: 10000 tokens
pruneVocabulary(tokenizer, corpus, minFrequency = 5)
# After: ~7000 tokens (rare ones removed)
```

---

### maskTokens
Mask tokens for MLM (Masked Language Modeling).

```nim
proc maskTokens*(
  tokens: seq[int],
  tokenizer: Tokenizer,
  maskProb: float = 0.15,
  replaceMaskProb: float = 0.8,
  replaceRandomProb: float = 0.1
): seq[int]
```

**Parameters:**
- `tokens` - original tokens
- `tokenizer` - tokenizer
- `maskProb` - masking probability (default: 0.15)
- `replaceMaskProb` - probability to replace with [MASK] (default: 0.8)
- `replaceRandomProb` - probability to replace with random token (default: 0.1)

**Use Cases:**
- Training BERT-like models
- Masked Language Modeling
- Self-supervised learning

**Example:**
```nim
let original = tokenize("This is test sentence", tokenizer)
let masked = maskTokens(original, tokenizer, maskProb = 0.15)
# masked might be: @[1, [MASK], 3, 4, [MASK]]
```

---

### encodeWithPadding
Tokenization with automatic padding/truncation.

```nim
proc encodeWithPadding*(
  tokenizer: var Tokenizer,
  text: string,
  maxLength: int,
  padding: bool = true,
  truncation: bool = true
): seq[int]
```

**Example:**
```nim
let padded = encodeWithPadding(tokenizer, "text", maxLength = 10)
# Result always has length 10
```

---

### getSubwordBreakdown
Get text breakdown into subwords.

```nim
proc getSubwordBreakdown*(
  text: string,
  tokenizer: var Tokenizer
): seq[string]
```

**Returns:** list of subwords as strings

**Use Cases:**
- Tokenization visualization
- Debugging
- Vocabulary quality analysis

**Example:**
```nim
let breakdown = getSubwordBreakdown("unknown", tokenizer)
# Result: @["un", "##known"]
```

---

### estimateTokenCount
Quick token count estimation without tokenization.

```nim
proc estimateTokenCount*(
  text: string,
  avgCharsPerToken: float = 4.0
): int
```

**Use Cases:**
- Preliminary size estimation
- Input length validation
- Batching

**Example:**
```nim
let estimate = estimateTokenCount("Long text...")
if estimate > 512:
  echo "Text too long"
```

---

### validateTokenizer
Check tokenizer correctness.

```nim
proc validateTokenizer*(tokenizer: Tokenizer): seq[string]
```

**Returns:** list of errors (empty list = valid)

**Checks:**
- Vocab and inverseVocab consistency
- Special tokens presence
- Merges correctness (for BPE)
- Scores correctness (for SentencePiece)

**Example:**
```nim
let errors = validateTokenizer(tokenizer)
if errors.len == 0:
  echo "Tokenizer is valid"
else:
  echo "Errors found:"
  for error in errors:
    echo "  - ", error
```

---

### Byte-level encoding (for ByteLevel BPE)

```nim
proc initBytePairEncoder*(): Table[int, string]
proc initByteDecoder*(encoder: Table[int, string]): Table[string, int]
proc byteLevelEncode*(text: string): seq[int]
proc byteLevelDecode*(bytes: seq[int]): string
```

**Use Case:** low-level work with byte-level encoding

---

## Usage Examples

### Basic Example

```nim
import tokenization

# Training
let corpus = @[
  "This is the first training text",
  "And this is the second text",
  "Tokenization works great"
]

var tokenizer = trainBPE(corpus, vocabSize = 500)

# Tokenization
let text = "New test text"
let tokens = tokenize(text, tokenizer)
echo "Tokens: ", tokens

# Decoding
let decoded = tokenizer.decode(tokens)
echo "Decoded: ", decoded

# Metrics
let metrics = getMetrics(tokenizer, corpus)
echo "Compression ratio: ", metrics.compressionRatio
```

---

### Batch Processing

```nim
let texts = @[
  "First sentence",
  "Second long sentence with many words",
  "Third"
]

let batch = encodeBatch(
  tokenizer,
  texts,
  maxLength = 20,
  padding = true,
  truncation = true
)

echo "Input IDs: ", batch.inputIds
echo "Attention masks: ", batch.attentionMask
echo "Lengths: ", batch.lengths
```

---

### Working with Positions (NER)

```nim
let text = "John lives in London"
let offsets = tokenizeWithOffsets(text, tokenizer)

for offset in offsets:
  echo offset.token, " [", offset.startChar, ":", offset.endChar, "]"
# Result:
# "John" [0:4]
# " " [4:5]
# "lives" [5:10]
# ...
```

---

### Streaming Large File Processing

```nim
var totalTokens = 0
for chunk in streamTokenize("huge_file.txt", tokenizer):
  totalTokens += chunk.len
  # Process chunk...

echo "Total tokens processed: ", totalTokens
```

---

### Text Cleaning and Preprocessing

```nim
let dirty = """
  <html>
    <p>Email: spam@example.com</p>
    <p>Visit www.example.com</p>
  </html>
"""

let clean = cleanText(
  dirty,
  removeHtml = true,
  removeUrls = true,
  removeEmails = true
)

echo clean  # "Visit"
```

---

### Comparing Tokenizers

```nim
var bpe = trainBPE(corpus, vocabSize = 1000)
var wordpiece = trainWordPiece(corpus, vocabSize = 1000)
var sentencepiece = trainSentencePiece(corpus, vocabSize = 1000)

let results = compareTokenizers(
  "Test sentence",
  @[bpe, wordpiece, sentencepiece]
)

for result in results:
  echo result.kind, ": ", result.tokens.len, " tokens"
  echo "  Decoded: ", result.decoded
```

---

### BPE Dropout for Regularization

```nim
let text = "model training"

# Create 5 different tokenizations for augmentation
var augmented: seq[seq[int]]
for i in 0..4:
  augmented.add(tokenizeWithDropout(text, tokenizer, dropoutProb = 0.2))

# All 5 will be different!
for i, tokens in augmented:
  echo "Variant ", i, ": ", tokens
```

---

### Adapting Tokenizer to New Domain

```nim
# Original tokenizer trained on general texts
var tokenizer = trainBPE(generalCorpus, vocabSize = 5000)

# Add medical terminology
let medicalTexts = @[
  "disease diagnosis",
  "pharmacological drugs",
  "clinical practice"
]

incrementalTrain(tokenizer, medicalTexts, maxNewTokens = 500)

# Now tokenizer knows medical terms
```

---

## Performance Optimization

### Optimization Tips:

1. **Use caching:**
   - Cache is automatically active
   - To clear: `clearCache(tokenizer)`
   - Check efficiency: `tokenizer.cacheHits / (tokenizer.cacheHits + tokenizer.cacheMisses)`

2. **Batch processing:**
   - Use `encodeBatch` instead of looping `tokenize`
   - Significantly faster for multiple texts

3. **Streaming:**
   - For files > 100MB use `streamTokenize`
   - Saves memory

4. **Tokenizer choice:**
   - **BPE**: balance of speed and quality
   - **WordPiece**: slower, better for BERT
   - **SentencePiece**: universal, good for any language
   - **ByteLevel BPE**: fastest, no UNK tokens

5. **Vocabulary size:**
   - Smaller vocab (1000-2000) → faster
   - Larger vocab (30000-50000) → better quality
   - Optimal balance: 8000-16000

---

## Frequently Asked Questions (FAQ)

### Q: How to choose vocabulary size?
**A:** 
- Small corpora (<1M words): 1000-5000
- Medium corpora (1M-10M words): 8000-16000
- Large corpora (>10M words): 30000-50000

### Q: When to use preserveCase=true?
**A:** When case matters (Named Entity Recognition, code generation). For most NLP tasks, false is better.

### Q: Why do [UNK] tokens appear?
**A:**
- Vocabulary too small
- minFrequency too high
- Text very different from training corpus

**Solution:**
- Increase vocabSize
- Decrease minFrequency
- Use ByteLevel BPE (no UNK)
- Retrain: `incrementalTrain`

### Q: How to speed up tokenization?
**A:**
1. Use cache (enabled by default)
2. Use `encodeBatch` for multiple texts
3. Compile with `-d:danger --opt:speed`
4. Choose ByteLevel BPE for speed

### Q: Is ByteLevel BPE compatible with GPT-2/3?
**A:** Yes, fully compatible. Uses the same byte-pair encoding principles.

---

## Changelog

### Version 0.5 (2026-01-31)
- ✅ Fixed space preservation in WordPiece
- ✅ Fixed uppercase letter handling (lowercase fallback)
- ✅ Added vocabulary export to JSON
- ✅ Improved time measurement in tests
- ✅ Fixed vocabulary size checks

### Version 0.4
- ✅ ByteLevel BPE (GPT-2/3 compatible)
- ✅ Token position tracking
- ✅ Streaming tokenization
- ✅ Vocabulary pruning
- ✅ Performance optimization (20-50x)
- ✅ Caching
- ✅ BPE-dropout

### Version 0.3
- ✅ Extended text cleaning
- ✅ Token masking
- ✅ Tokenizer comparison

### Version 0.2
- ✅ Batch processing
- ✅ Special tokens
- ✅ Metrics

### Version 0.1
- ✅ Basic tokenizers: BPE, WordPiece, SentencePiece

---

## License

MIT License

---

## Contact

**Author:** github.com/Balans097  
**E-mail:** vasil.minsk@yahoo.com  
**Version:** 0.5  
**Date:** 2026-01-31

---

## Resources

Implementation based on the following works:
- BPE: Sennrich et al. (2016)
- WordPiece: Wu et al. (2016) 
- SentencePiece: Kudo & Richardson (2018)
- ByteLevel BPE: Radford et al. (2019)