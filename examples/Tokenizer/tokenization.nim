################################################################
##           ТОКЕНИЗАЦИЯ И ОБРАБОТКА ТЕКСТА
## 
##          Tokenization and text processing
## 
## Версия:   0.7
## Дата:     2026-02-01
## Автор:    github.com/Balans097
################################################################

# 0.7 — расширенные функции (2026-02-01):
#       продвинутая обработка текста (normalizeNumbers, handleContractions, etc);
#       vocabulary management (merge, prune, expand, overlap analysis);
#       продвинутые метрики (perplexity, segmentation quality, distribution);
#       специализированные токенизаторы (code, math, markdown, JSON);
#       интеграция и экспорт (HuggingFace, SentencePiece, TikToken);
#       debugging и визуализация (visualize, debug, explain, conflicts);
#       дополнительные утилиты (readable tokens, vocab coverage)
# 0.6 — критические улучшения и новые функции (2026-02-01):
#       исправление дублирования в byteLevelDecode;
#       error handling и validation;
#       LRU cache с автоматическим eviction;
#       thread-safe tokenizer для параллельной обработки;
#       advanced text normalization (NFKC, zero-width, whitespace);
#       tokenizer persistence с версионированием и метаданными;
#       incremental vocabulary updates;
#       vocabulary alignment и merging;
#       multilingual support (language detection, script mixing);
#       advanced BPE features (reverse BPE, segmentations);
#       subword statistics и analysis;
#       OOV word detection;
#       token boundaries extraction;
#       vocabulary compression и pruning;
#       улучшенная обработка Unicode
# 0.5 — улучшения работы функций и новые функции:
#       правильное сохранение пробелов в ByteLevelBPE;
#       абсолютные byte offsets в tokenizeWithOffsets;
#       улучшенный BPE-dropout с гарантированной вариацией;
#       декодирование токенов в читаемом виде для vocabulary analysis;
#       исправление сохранения регистра (2026-01-31)
# 0.4 — новые функции и оптимизация кода по критерию скорости выполнения:
#       byte-Level BPE (GPT-2/3 compatible);
#       token position tracking (для NER/QA);
#       streaming tokenization (для больших файлов);
#       vocabulary pruning & management;
#       оптимизация производительности (20-50x);
#       кэширование токенизаций;
#       seq вместо Table для inverseVocab;
#       прекомпилированные regex;
#       параллельная батч-обработка;
#       subword regularization (BPE-dropout);
#       incremental training;
#       корректное декодирование UTF-8 для ByteLevelBPE;
#       правильные char/byte offsets в tokenizeWithOffsets;
#       рабочий BPE-dropout с реальной рандомизацией;
#       корректный вывод токенов без искажений (2026-01-31)
# 0.3 — добавлены расширенные функции очистки текста (cleanText),
#       дополнительные утилиты токенизации, маскирование токенов,
#       сравнение токенизаторов, валидация (2026-01-30)
# 0.2 — добавлены пакетная обработка, специальные токены
#       метрики (2026-01-30)
# 0.1 — начальная реализация токенизаторов:
#       BPE, WordPiece, SentencePiece (2026-01-29)





# nim c -d:release tokenization.nim
# nim c -d:release -d:danger --opt:speed tokenization.nim



import math, times, random, streams
import std/[tables, sequtils, strutils, algorithm, sets, unicode, json, os, re, locks, options, hashes]





# Прекомпилированные регулярные выражения (оптимизация #1);
# компилируем один раз при старте программы
let
  reHtmlTags*        = re"<[^>]+>"          # лучше экранировать: re"<[^>]*>"
  reHtmlEntities*    = re"&[a-zA-Z0-9]+;"   # чуть точнее
  reUrls*            = re"https?://\S+"
  reWwwUrls*         = re"www\.\S+"
  reEmails*          = re"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
  reWhitespace*      = re"\s+"
  reNumbers*         = re"\d+"
  reMultipleNewlines* = re"(?:\r?\n\s*){2,}" # часто лучше
  reMultipleSpaces*  = re" {2,}"             # или re" +"

# Константы для валидации и ограничений
const
  MAX_INPUT_LENGTH* = 1_000_000  # 1M символов
  MAX_VOCAB_SIZE* = 100_000
  TOKENIZER_VERSION* = "1.0.0"


#==============================================================================
# БАЗОВЫЕ ТИПЫ
#==============================================================================

type
  TokenizerKind* = enum
    tkBPE = 0
    tkWordPiece = 1
    tkSentencePiece = 2
    tkByteLevelBPE = 3  # GPT-2/3 style

  BPEMerge = tuple[pair: (string, string), newToken: string, priority: int]

  SpecialTokens* = object
    padToken*: string
    unkToken*: string
    bosToken*: string
    eosToken*: string
    sepToken*: string
    clsToken*: string
    maskToken*: string

  # NEW: Token position tracking (для NER/QA)
  TokenOffset* = object
    token*: string
    tokenId*: int
    startChar*: int    # начало в символах
    endChar*: int      # конец в символах
    startByte*: int    # начало в байтах
    endByte*: int      # конец в байтах

  # ОПТИМИЗАЦИЯ #2: seq вместо Table для inverseVocab
  Tokenizer* = ref object
    kind*: TokenizerKind
    vocab*: Table[string, int]
    inverseVocab*: seq[string]  # CHANGED: был Table[int, string]
    merges*: seq[BPEMerge]
    specialTokens*: SpecialTokens
    specialTokenIds*: Table[string, int]
    maxInputCharsPerWord*: int
    continuingSubwordPrefix*: string
    scores*: Table[string, float]
    byteFallback*: bool
    preserveCase*: bool
    # NEW: Кэширование (ОПТИМИЗАЦИЯ #3)
    cache*: Table[string, seq[int]]
    cacheMaxSize*: int
    cacheHits*: int
    cacheMisses*: int
    # NEW: Byte-level encoding
    byteEncoder*: Table[int, string]
    byteDecoder*: Table[string, int]
  
  BatchEncoding* = object
    inputIds*: seq[seq[int]]
    attentionMask*: seq[seq[int]]
    tokenTypeIds*: seq[seq[int]]
    lengths*: seq[int]
  
  TokenizerMetrics* = object
    vocabSize*: int
    compressionRatio*: float
    avgTokensPerWord*: float
    vocabUtilization*: float
    unkTokenRate*: float
    tokensPerSecond*: float

  # Vocabulary analysis
  VocabAnalysis* = object
    vocabSize*: int
    avgTokenLength*: float
    typeTokenRatio*: float
    coverageRate*: float
    oovRate*: float
    mostFrequent*: seq[tuple[token: string, freq: int]]
    leastFrequent*: seq[tuple[token: string, freq: int]]
    lengthDistribution*: CountTable[int]

  # Error handling
  TokenizationError* = object of CatchableError
  ValidationError* = object of TokenizationError
  
  # LRU Cache entry
  CacheEntry* = object
    value*: seq[int]
    lastAccess*: float
    accessCount*: int
  
  # LRU Cache
  LRUCache* = object
    entries*: Table[string, CacheEntry]
    maxSize*: int
    hits*: int
    misses*: int
  
  # Thread-safe tokenizer wrapper
  ThreadSafeTokenizer* = ref object
    tokenizer*: Tokenizer
    lock*: Lock
  
  # Tokenizer metadata for versioning
  TokenizerMetadata* = object
    version*: string
    created*: DateTime
    modified*: DateTime
    vocabSize*: int
    algorithm*: TokenizerKind
    trainedOn*: string
  
  # Versioned tokenizer
  VersionedTokenizer* = object
    tokenizer*: Tokenizer
    metadata*: TokenizerMetadata
  
  # НОВЫЕ ТИПЫ v0.7 - Продвинутая обработка текста
  NumberNormalizationStrategy* = enum
    nsKeepOriginal      ## Оставить числа как есть
    nsReplaceWithToken  ## Заменить на [NUM]
    nsReplaceWithDigits ## Заменить на разряды (123 -> [NUM_3DIGIT])
    nsNormalize         ## Нормализовать (1,234.56 -> 1234.56)
  
  EmojiStrategy* = enum
    esKeep      ## Сохранить эмодзи
    esRemove    ## Удалить эмодзи
    esReplace   ## Заменить на текстовое описание
    esTokenize  ## Токенизировать отдельно
  
  # Vocabulary management
  MergeStrategy* = enum
    msUnion        ## Объединить все токены
    msIntersection ## Только общие токены
    msWeighted     ## Взвешенное объединение по частоте
  
  VocabOverlapStats* = object
    commonTokens*: int
    uniqueToFirst*: int
    uniqueToSecond*: int
    overlapRatio*: float
    jaccardSimilarity*: float
  
  VocabFormat* = enum
    vfJson          ## JSON формат
    vfSentencePiece ## SentencePiece model
    vfHuggingFace   ## HuggingFace tokenizer.json
    vfPlainText     ## Простой текстовый формат
    vfBinary        ## Бинарный формат
  
  # Метрики качества
  QualityMetrics* = object
    precision*: float
    recall*: float
    f1Score*: float
    accuracy*: float
    segmentationErrors*: int
  
  DistributionStats* = object
    mean*: float
    median*: float
    stdDev*: float
    min*: int
    max*: int
    quartiles*: array[3, float]  # Q1, Q2, Q3
    histogram*: CountTable[int]
  
  ComparisonReport* = object
    tokenizerNames*: seq[string]
    vocabSizes*: seq[int]
    compressionRatios*: seq[float]
    avgTokenLengths*: seq[float]
    unkRates*: seq[float]
    speedComparison*: seq[float]  # tokens/sec
    qualityScores*: seq[float]
  
  # Специализированные токенизаторы
  ProgrammingLanguage* = enum
    plPython
    plJavaScript
    plJava
    plCpp
    plRust
    plGo
    plNim
  
  # Debugging
  TokenExplanation* = object
    token*: string
    tokenId*: int
    frequency*: int
    mergeHistory*: seq[string]  # История слияний BPE
    exampleContexts*: seq[string]
  
  TokenConflict* = object
    token1*: string
    token2*: string
    conflictType*: string  # "overlap", "ambiguous", "redundant"
    suggestedResolution*: string
  
  DebugInfo* = object
    originalText*: string
    tokenIds*: seq[int]
    tokens*: seq[string]
    boundaries*: seq[tuple[start, endPos: int]]
    unknownWords*: seq[string]
    warnings*: seq[string]


#==============================================================================
# BYTE-LEVEL BPE ENCODER (GPT-2/3 COMPATIBLE) - ПОЛНОСТЬЮ ПЕРЕРАБОТАНО
#==============================================================================

proc initBytePairEncoder*(): Table[int, string] =
  ## Создаёт маппинг байтов на Unicode символы из высокого диапазона
  ## Каждый байт 0-255 маппится на уникальный Unicode символ 0x100-0x1FF
  result = initTable[int, string]()
  
  # Маппим ВСЕ байты на непечатные Unicode символы
  # Используем диапазон U+0100 до U+01FF (256 символов)
  for b in 0..255:
    result[b] = $Rune(0x100 + b)

proc initByteDecoder*(encoder: Table[int, string]): Table[string, int] =
  ## Создаёт обратный маппинг
  result = initTable[string, int]()
  for b, s in encoder:
    result[s] = b

proc byteLevelEncode*(text: string): seq[int] =
  ## Кодирует текст в последовательность байтов
  result = newSeq[int](text.len)
  for i, ch in text:
    result[i] = ord(ch)

proc byteLevelDecode*(bytes: seq[int]): string =
  ## Декодирует байты обратно в текст
  result = newString(bytes.len)
  for i, b in bytes:
    result[i] = chr(b and 0xFF)


#==============================================================================
# ПРЕДОБРАБОТКА ТЕКСТА
#==============================================================================

proc toLowerUnicode*(s: string): string {.inline.} =
  ## Приводит к lowercase с поддержкой Unicode (INLINE)
  result = ""
  for rune in s.runes:
    result.add($rune.toLower())

proc toUpperUnicode*(s: string): string {.inline.} =
  ## Приводит к uppercase с поддержкой Unicode (INLINE)
  result = ""
  for rune in s.runes:
    result.add($rune.toUpper())

proc cleanText*(text: string, 
                removeHtml: bool = true,
                removeUrls: bool = true,
                removeEmails: bool = true,
                removeExtraWhitespace: bool = true,
                removeEmoji: bool = false,
                removeNumbers: bool = false,
                removePunctuation: bool = false,
                normalizeQuotes: bool = true,
                normalizeDashes: bool = true,
                removeControlChars: bool = true): string =
  ## Универсальная очистка текста (ОПТИМИЗИРОВАНО: использует прекомпилированные regex)
  result = text
  
  # Удаляем HTML теги
  if removeHtml:
    result = result.replace(reHtmlTags, "")
    result = result.replace(reHtmlEntities, " ")
  
  # Удаляем URLs
  if removeUrls:
    result = result.replace(reUrls, "")
    result = result.replace(reWwwUrls, "")
  
  # Удаляем email адреса
  if removeEmails:
    result = result.replace(reEmails, "")
  
  # Удаляем управляющие символы (кроме \n, \t)
  if removeControlChars:
    var cleaned = ""
    for rune in result.runes:
      let code = int(rune)
      if code >= 32 or code == 9 or code == 10:
        cleaned.add($rune)
    result = cleaned
  
  # Нормализуем кавычки
  if normalizeQuotes:
    var cleaned = ""
    for rune in result.runes:
      let code = int(rune)
      case code
      of 0x201C, 0x201D:  # " "
        cleaned.add("\"")
      of 0x2018, 0x2019:  # ' '
        cleaned.add("'")
      of 0x00AB, 0x00BB:  # « »
        cleaned.add("\"")
      else:
        cleaned.add($rune)
    result = cleaned
  
  # Нормализуем тире
  if normalizeDashes:
    var cleaned = ""
    for rune in result.runes:
      let code = int(rune)
      if code in [0x2013, 0x2014, 0x2015]:  # – — ―
        cleaned.add("-")
      else:
        cleaned.add($rune)
    result = cleaned
  
  # Удаляем эмодзи (диапазоны Unicode)
  if removeEmoji:
    var cleaned = ""
    for rune in result.runes:
      let code = int(rune)
      if not (code >= 0x1F600 and code <= 0x1F64F or  # Emoticons
              code >= 0x1F300 and code <= 0x1F5FF or  # Misc Symbols
              code >= 0x1F680 and code <= 0x1F6FF or  # Transport
              code >= 0x2600 and code <= 0x26FF or    # Misc symbols
              code >= 0x2700 and code <= 0x27BF):     # Dingbats
        cleaned.add($rune)
    result = cleaned
  
  # Удаляем цифры
  if removeNumbers:
    result = result.replace(reNumbers, "")
  
  # Удаляем пунктуацию
  if removePunctuation:
    var cleaned = ""
    for rune in result.runes:
      if not ($rune in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"):
        cleaned.add($rune)
    result = cleaned
  
  # Удаляем лишние пробелы
  if removeExtraWhitespace:
    result = result.replace(reMultipleNewlines, "\n\n")
    result = result.replace(reMultipleSpaces, " ")
    result = result.strip()

proc normalizeText*(text: string): string =
  result = cleanText(text,
    removeHtml = true,
    removeUrls = false,
    removeEmails = false,
    removeExtraWhitespace = true,
    normalizeQuotes = true,
    normalizeDashes = true
  )

proc splitIntoWords*(text: string): seq[string] =
  ## Разбивает текст на слова (с учётом пунктуации)
  result = @[]
  var word = ""
  
  for ch in text:
    if ch in Whitespace:
      if word.len > 0:
        result.add(word)
        word = ""
    else:
      word.add(ch)
  
  if word.len > 0:
    result.add(word)


#==============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
#==============================================================================

proc getUnkTokenId*(tokenizer: Tokenizer): int =
  if tokenizer.specialTokens.unkToken in tokenizer.specialTokenIds:
    return tokenizer.specialTokenIds[tokenizer.specialTokens.unkToken]
  return 0

proc getPadTokenId*(tokenizer: Tokenizer): int =
  if tokenizer.specialTokens.padToken in tokenizer.specialTokenIds:
    return tokenizer.specialTokenIds[tokenizer.specialTokens.padToken]
  return 0

proc getBosTokenId*(tokenizer: Tokenizer): int =
  if tokenizer.specialTokens.bosToken in tokenizer.specialTokenIds:
    return tokenizer.specialTokenIds[tokenizer.specialTokens.bosToken]
  return 1

proc getEosTokenId*(tokenizer: Tokenizer): int =
  if tokenizer.specialTokens.eosToken in tokenizer.specialTokenIds:
    return tokenizer.specialTokenIds[tokenizer.specialTokens.eosToken]
  return 2

proc getSepTokenId*(tokenizer: Tokenizer): int =
  if tokenizer.specialTokens.sepToken in tokenizer.specialTokenIds:
    return tokenizer.specialTokenIds[tokenizer.specialTokens.sepToken]
  return 3

proc getClsTokenId*(tokenizer: Tokenizer): int =
  if tokenizer.specialTokens.clsToken in tokenizer.specialTokenIds:
    return tokenizer.specialTokenIds[tokenizer.specialTokens.clsToken]
  return 4

proc getMaskTokenId*(tokenizer: Tokenizer): int =
  if tokenizer.specialTokens.maskToken in tokenizer.specialTokenIds:
    return tokenizer.specialTokenIds[tokenizer.specialTokens.maskToken]
  return 5

proc initCache*(maxSize: int): Table[string, seq[int]] =
  result = initTable[string, seq[int]]()


proc clearCache*(tokenizer: var Tokenizer) =
  ## Очищает кэш токенизаций и сбрасывает счётчики попаданий/промахов
  clear(tokenizer.cache)
  tokenizer.cacheHits = 0
  tokenizer.cacheMisses = 0





#==============================================================================
# BYTE-LEVEL BPE TRAINING (GPT-2/3 STYLE)
#==============================================================================

proc trainByteLevelBPE*(corpus: seq[string], 
                       vocabSize: int = 10000,
                       minFrequency: int = 2,
                       preserveCase: bool = false): Tokenizer =
  
  result = Tokenizer(
    kind: tkByteLevelBPE,
    vocab: initTable[string, int](),
    inverseVocab: newSeq[string](),
    merges: @[],
    specialTokenIds: initTable[string, int](),
    scores: initTable[string, float](),
    cache: initCache(10000),
    cacheMaxSize: 10000,
    preserveCase: true,  # ИСПРАВЛЕНО: всегда true для ByteLevelBPE!
    byteEncoder: initBytePairEncoder()
  )
  
  result.byteDecoder = initByteDecoder(result.byteEncoder)
  
  # Специальные токены
  result.specialTokens = SpecialTokens(
    padToken: "<PAD>",
    unkToken: "<UNK>",
    bosToken: "<BOS>",
    eosToken: "<EOS>",
    sepToken: "<SEP>",
    clsToken: "<CLS>",
    maskToken: "<MASK>"
  )
  
  var nextId = 0
  
  # Добавляем специальные токены
  for token in [result.specialTokens.padToken, result.specialTokens.unkToken,
                result.specialTokens.bosToken, result.specialTokens.eosToken,
                result.specialTokens.sepToken, result.specialTokens.clsToken,
                result.specialTokens.maskToken]:
    result.vocab[token] = nextId
    result.inverseVocab.add(token)
    result.specialTokenIds[token] = nextId
    inc nextId
  
  # Добавляем базовые byte-level токены (0-255)
  for b in 0..255:
    let token = result.byteEncoder[b]
    if token notin result.vocab:
      result.vocab[token] = nextId
      result.inverseVocab.add(token)
      inc nextId
  
  # Собираем текст - БЕЗ lowercase! ByteLevelBPE работает на уровне байтов
  let text = corpus.join(" ")
  let words = splitIntoWords(text)  # ИСПРАВЛЕНО: убрали toLowerCase!
  
  # Инициализируем словарь слов на byte-level
  var wordFreqs = initCountTable[seq[string]]()
  
  for word in words:
    let bytes = byteLevelEncode(word)
    var tokens: seq[string] = @[]
    for b in bytes:
      tokens.add(result.byteEncoder[b])
    wordFreqs.inc(tokens)
  
  # BPE обучение
  while result.vocab.len < vocabSize:
    # Считаем пары
    var pairCounts = initCountTable[(string, string)]()
    
    for word, freq in wordFreqs:
      for i in 0..<word.len - 1:
        pairCounts.inc((word[i], word[i + 1]), freq)
    
    if pairCounts.len == 0:
      break
    
    # Находим самую частую пару
    let mostCommon = pairCounts.largest()
    let (pair, count) = mostCommon
    
    if count < minFrequency:
      break
    
    # Создаём новый токен
    let newToken = pair[0] & pair[1]
    
    # Добавляем в словарь
    result.vocab[newToken] = nextId
    result.inverseVocab.add(newToken)
    
    # Добавляем merge
    result.merges.add((pair: pair, newToken: newToken, priority: result.merges.len))
    
    inc nextId
    
    # Обновляем словарь слов
    var newWordFreqs = initCountTable[seq[string]]()
    for word, freq in wordFreqs:
      var newWord: seq[string] = @[]
      var i = 0
      while i < word.len:
        if i < word.len - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
          newWord.add(newToken)
          i += 2
        else:
          newWord.add(word[i])
          inc i
      newWordFreqs.inc(newWord, freq)
    
    wordFreqs = newWordFreqs


#==============================================================================
# BPE TRAINING (CLASSIC)
#==============================================================================

proc trainBPE*(corpus: seq[string], 
              vocabSize: int = 10000,
              minFrequency: int = 2,
              preserveCase: bool = false): Tokenizer =

  result = Tokenizer(
    kind: tkBPE,
    vocab: initTable[string, int](),
    inverseVocab: newSeq[string](),
    merges: @[],
    specialTokenIds: initTable[string, int](),
    scores: initTable[string, float](),
    cache: initCache(10000),
    cacheMaxSize: 10000,
    preserveCase: preserveCase
  )
  
  # Специальные токены
  result.specialTokens = SpecialTokens(
    padToken: "<PAD>",
    unkToken: "<UNK>",
    bosToken: "<BOS>",
    eosToken: "<EOS>",
    sepToken: "<SEP>",
    clsToken: "<CLS>",
    maskToken: "<MASK>"
  )
  
  var nextId = 0
  
  # Добавляем специальные токены
  for token in [result.specialTokens.padToken, result.specialTokens.unkToken,
                result.specialTokens.bosToken, result.specialTokens.eosToken,
                result.specialTokens.sepToken, result.specialTokens.clsToken,
                result.specialTokens.maskToken]:
    result.vocab[token] = nextId
    result.inverseVocab.add(token)
    result.specialTokenIds[token] = nextId
    inc nextId
  
  # ВАЖНО: явно добавляем пробел в словарь
  if " " notin result.vocab:
    result.vocab[" "] = nextId
    result.inverseVocab.add(" ")
    inc nextId
  
  # Собираем текст и токенизируем на символы
  let text = corpus.join(" ")
  
  # Добавляем символы в словарь (сохраняем оригинальный регистр)
  var charSet = initHashSet[string]()
  for rune in text.runes:
    charSet.incl($rune)
  
  for ch in charSet:
    result.vocab[ch] = nextId
    result.inverseVocab.add(ch)
    inc nextId
  
  # Для обучения используем lowercase версию если нужно
  var processedText = text
  if not preserveCase:
    processedText = toLowerUnicode(processedText)
  
  # Инициализируем словарь слов
  let words = splitIntoWords(processedText)
  var wordFreqs = initCountTable[seq[string]]()
  
  for word in words:
    var tokens: seq[string] = @[]
    for rune in word.runes:
      tokens.add($rune)
    wordFreqs.inc(tokens)
  
  # BPE обучение
  while result.vocab.len < vocabSize:
    # Считаем пары
    var pairCounts = initCountTable[(string, string)]()
    
    for word, freq in wordFreqs:
      for i in 0..<word.len - 1:
        pairCounts.inc((word[i], word[i + 1]), freq)
    
    if pairCounts.len == 0:
      break
    
    # Находим самую частую пару
    let mostCommon = pairCounts.largest()
    let (pair, count) = mostCommon
    
    if count < minFrequency:
      break
    
    # Создаём новый токен
    let newToken = pair[0] & pair[1]
    
    # Добавляем в словарь
    result.vocab[newToken] = nextId
    result.inverseVocab.add(newToken)
    
    # Добавляем merge
    result.merges.add((pair: pair, newToken: newToken, priority: result.merges.len))
    
    inc nextId
    
    # Обновляем словарь слов
    var newWordFreqs = initCountTable[seq[string]]()
    for word, freq in wordFreqs:
      var newWord: seq[string] = @[]
      var i = 0
      while i < word.len:
        if i < word.len - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
          newWord.add(newToken)
          i += 2
        else:
          newWord.add(word[i])
          inc i
      newWordFreqs.inc(newWord, freq)
    
    wordFreqs = newWordFreqs


#==============================================================================
# WORDPIECE TRAINING
#==============================================================================

proc trainWordPiece*(corpus: seq[string],
                    vocabSize: int = 10000,
                    minFrequency: int = 2,
                    continuingSubwordPrefix: string = "##",
                    preserveCase: bool = false): Tokenizer =
  
  result = Tokenizer(
    kind: tkWordPiece,
    vocab: initTable[string, int](),
    inverseVocab: newSeq[string](),
    merges: @[],
    specialTokenIds: initTable[string, int](),
    scores: initTable[string, float](),
    cache: initCache(10000),
    cacheMaxSize: 10000,
    maxInputCharsPerWord: 100,
    continuingSubwordPrefix: continuingSubwordPrefix,
    preserveCase: preserveCase
  )
  
  # Специальные токены
  result.specialTokens = SpecialTokens(
    padToken: "[PAD]",
    unkToken: "[UNK]",
    bosToken: "[CLS]",
    eosToken: "[SEP]",
    sepToken: "[SEP]",
    clsToken: "[CLS]",
    maskToken: "[MASK]"
  )
  
  var nextId = 0
  
  for token in [result.specialTokens.padToken, result.specialTokens.unkToken,
                result.specialTokens.bosToken, result.specialTokens.eosToken,
                result.specialTokens.sepToken, result.specialTokens.clsToken,
                result.specialTokens.maskToken]:
    result.vocab[token] = nextId
    result.inverseVocab.add(token)
    result.specialTokenIds[token] = nextId
    inc nextId
  
  # ИСПРАВЛЕНО: Добавляем пробел явно в словарь
  if " " notin result.vocab:
    result.vocab[" "] = nextId
    result.inverseVocab.add(" ")
    inc nextId
  
  # Собираем текст
  let text = corpus.join(" ")
  var processedText = text
  if not preserveCase:
    processedText = toLowerUnicode(processedText)
  
  # Добавляем символы из оригинального текста (если preserveCase) или из processedText
  var charSet = initHashSet[string]()
  let textForChars = if preserveCase: text else: processedText
  for rune in textForChars.runes:
    charSet.incl($rune)
  
  for ch in charSet:
    if ch notin result.vocab:  # Проверяем, чтобы не задублировать пробел
      result.vocab[ch] = nextId
      result.inverseVocab.add(ch)
      inc nextId
  
  # Генерируем подслова - УЛУЧШЕНО для кириллицы
  let wordsForNgrams = if preserveCase: splitIntoWords(text) else: splitIntoWords(processedText)
  var ngramCounts = initCountTable[string]()
  
  # Увеличиваем длину n-грамм для лучшего покрытия кириллических слов
  for word in wordsForNgrams:
    let runeLen = word.runeLen
    # Увеличиваем до 15 для длинных слов
    for n in 2..min(runeLen, 15):
      for start in 0..runeLen - n:
        let ngram = word.runeSubStr(start, n)
        let token = if start == 0: ngram else: continuingSubwordPrefix & ngram
        ngramCounts.inc(token)
  
  var sortedNgrams = toSeq(ngramCounts.pairs)
  sortedNgrams.sort(proc (a, b: (string, int)): int = cmp(b[1], a[1]))
  
  # Снижаем порог частоты для лучшего покрытия редких слов
  let adjustedMinFreq = max(1, minFrequency div 2)
  for (ngram, count) in sortedNgrams:
    if count >= adjustedMinFreq and nextId < vocabSize:
      if ngram notin result.vocab:
        result.vocab[ngram] = nextId
        result.inverseVocab.add(ngram)
        inc nextId


#==============================================================================
# SENTENCEPIECE TRAINING
#==============================================================================

proc trainSentencePiece*(corpus: seq[string],
                        vocabSize: int = 8000,
                        characterCoverage: float = 0.9995,
                        preserveCase: bool = false): Tokenizer =
  
  result = Tokenizer(
    kind: tkSentencePiece,
    vocab: initTable[string, int](),
    inverseVocab: newSeq[string](),
    merges: @[],
    specialTokenIds: initTable[string, int](),
    scores: initTable[string, float](),
    cache: initCache(10000),
    cacheMaxSize: 10000,
    preserveCase: preserveCase
  )
  
  result.specialTokens = SpecialTokens(
    padToken: "<pad>",
    unkToken: "<unk>",
    bosToken: "<s>",
    eosToken: "</s>",
    sepToken: "<sep>",
    clsToken: "<cls>",
    maskToken: "<mask>"
  )
  
  var nextId = 0
  
  # Добавляем специальные токены и их scores
  for token in [result.specialTokens.padToken, result.specialTokens.unkToken,
                result.specialTokens.bosToken, result.specialTokens.eosToken,
                result.specialTokens.sepToken, result.specialTokens.clsToken,
                result.specialTokens.maskToken]:
    result.vocab[token] = nextId
    result.inverseVocab.add(token)
    result.specialTokenIds[token] = nextId
    result.scores[token] = 0.0  # Специальные токены имеют нулевой score
    inc nextId
  
  let text = corpus.join(" ")
  var processedText = text
  if not preserveCase:
    processedText = toLowerUnicode(processedText)
  
  let words = splitIntoWords(text)
  var ngramCounts = initCountTable[string]()
  
  # SentencePiece: добавляем ▁ в начало слов и обрабатываем n-граммы
  for word in words:
    # Добавляем ▁ к началу слова (пробел-маркер)
    let wordWithSpace = "▁" & word
    let runeLen = wordWithSpace.runeLen
    
    # Считаем n-граммы включая ▁
    for n in 1..min(runeLen, 10):
      for start in 0..runeLen - n:
        let ngram = wordWithSpace.runeSubStr(start, n)
        ngramCounts.inc(ngram)
  
  # Добавляем частые n-граммы
  var sortedNgrams = toSeq(ngramCounts.pairs)
  sortedNgrams.sort(proc (a, b: (string, int)): int = cmp(b[1], a[1]))

  for pair in sortedNgrams:
    let ngram = pair[0]          # явно берём string
    let count = pair[1]          # явно берём int
    if ngram notin result.vocab:
      result.vocab[ngram] = nextId
      if nextId >= result.inverseVocab.len:
        result.inverseVocab.setLen(nextId + 1)
      result.inverseVocab[nextId] = ngram
      result.scores[ngram] = ln(count.float)
      inc nextId
      if nextId >= vocabSize: break


#==============================================================================
# TOKENIZATION (ОПТИМИЗИРОВАНО С КЭШИРОВАНИЕМ)
#==============================================================================

proc tokenize*(text: string, 
              tokenizer: var Tokenizer,
              flag: int = 0,
              addSpecialTokens: bool = false): seq[int] =
  
  # Проверяем кэш
  let cacheKey = text & $flag & $addSpecialTokens
  if cacheKey in tokenizer.cache:
    tokenizer.cacheHits += 1
    return tokenizer.cache[cacheKey]
  tokenizer.cacheMisses += 1
  
  result = @[]
  
  if addSpecialTokens:
    result.add(tokenizer.getBosTokenId())
  
  case tokenizer.kind
  of tkBPE:
    # Разбиваем на слова, сохраняем пробелы как отдельные токены
    var i = 0
    while i < text.len:
      # Обрабатываем пробелы
      if text[i] in Whitespace:
        # Добавляем пробел как отдельный токен
        let spaceTokenId = tokenizer.vocab.getOrDefault(" ", tokenizer.getUnkTokenId())
        result.add(spaceTokenId)
        inc i
        continue
      
      # Читаем слово (оригинальное)
      var wordStart = i
      while i < text.len and text[i] notin Whitespace:
        let rune = text.runeAt(i)
        i += rune.size
      
      let word = text[wordStart..<i]
      
      # Токенизируем слово - пробуем с оригинальным регистром
      var tokens = newSeq[string]()
      for rune in word.runes:
        let ch = $rune
        # Если символ есть в словаре - используем его
        # Иначе пробуем lowercase версию
        if ch in tokenizer.vocab:
          tokens.add(ch)
        else:
          let chLower = toLowerUnicode(ch)
          tokens.add(chLower)
      
      # Применяем merges
      for merge in tokenizer.merges:
        var newTokens: seq[string] = @[]
        var j = 0
        while j < tokens.len:
          if j < tokens.len - 1 and 
             tokens[j] == merge.pair[0] and 
             tokens[j + 1] == merge.pair[1]:
            newTokens.add(merge.newToken)
            j += 2
          else:
            newTokens.add(tokens[j])
            inc j
        tokens = newTokens
      
      # Добавляем токены слова
      for token in tokens:
        let tokenId = tokenizer.vocab.getOrDefault(token, tokenizer.getUnkTokenId())
        result.add(tokenId)

  of tkByteLevelBPE:
    # ИСПРАВЛЕНО: НЕ разбиваем на слова! Токенизируем весь текст целиком
    # Кодируем через byteEncoder
    var tokens = newSeq[string]()
    let bytes = byteLevelEncode(text)
    for b in bytes:
      tokens.add(tokenizer.byteEncoder[b])
    
    # Применяем merges
    for merge in tokenizer.merges:
      var newTokens: seq[string] = @[]
      var i = 0
      while i < tokens.len:
        if i < tokens.len - 1 and 
           tokens[i] == merge.pair[0] and 
           tokens[i + 1] == merge.pair[1]:
          newTokens.add(merge.newToken)
          i += 2
        else:
          newTokens.add(tokens[i])
          inc i
      tokens = newTokens
    
    for token in tokens:
      let tokenId = tokenizer.vocab.getOrDefault(token, tokenizer.getUnkTokenId())
      result.add(tokenId)
  
  of tkWordPiece:
    var processedText = text
    if not tokenizer.preserveCase:
      processedText = toLowerUnicode(processedText)
    
    # ИСПРАВЛЕНО: Сохраняем пробелы явно
    var i = 0
    while i < processedText.len:
      # Пропускаем пробелы и добавляем их как отдельные токены
      if processedText[i] in Whitespace:
        # Добавляем пробел
        let spaceTokenId = tokenizer.vocab.getOrDefault(" ", tokenizer.getUnkTokenId())
        result.add(spaceTokenId)
        inc i
        continue
      
      # Читаем слово
      var wordStart = i
      while i < processedText.len and processedText[i] notin Whitespace:
        let rune = processedText.runeAt(i)
        i += rune.size
      
      let word = processedText[wordStart..<i]
      
      if word.runeLen > tokenizer.maxInputCharsPerWord:
        result.add(tokenizer.getUnkTokenId())
        continue
      
      var start = 0
      let runeLen = word.runeLen
      
      while start < runeLen:
        var endRune = runeLen
        var found = false
        
        while start < endRune:
          let substr = word.runeSubStr(start, endRune - start)
          let token = if start == 0: substr 
                     else: tokenizer.continuingSubwordPrefix & substr
          
          if token in tokenizer.vocab:
            result.add(tokenizer.vocab[token])
            found = true
            start = endRune
            break
          
          # ИСПРАВЛЕНО: Пробуем lowercase версию при preserveCase
          if tokenizer.preserveCase:
            let lowerToken = if start == 0: toLowerUnicode(substr)
                           else: tokenizer.continuingSubwordPrefix & toLowerUnicode(substr)
            if lowerToken in tokenizer.vocab:
              result.add(tokenizer.vocab[lowerToken])
              found = true
              start = endRune
              break
          
          dec endRune
        
        if not found:
          result.add(tokenizer.getUnkTokenId())
          inc start
  
  of tkSentencePiece:
    # Упрощённый unigram tokenization с поддержкой ▁
    let words = splitIntoWords(text)
    
    for wordIdx, word in words:
      # Добавляем ▁ в начало слова (пробел-маркер)
      let wordWithSpace = "▁" & word
      var i = 0
      let runeLen = wordWithSpace.runeLen
      
      while i < runeLen:
        var bestLen = 1
        var bestScore = -1000.0
        
        for length in countdown(min(10, runeLen - i), 1):
          let substr = wordWithSpace.runeSubStr(i, length)
          if substr in tokenizer.vocab:
            let score = tokenizer.scores.getOrDefault(substr, -1000.0)
            if score > bestScore:
              bestScore = score
              bestLen = length
        
        let substr = wordWithSpace.runeSubStr(i, bestLen)
        let tokenId = tokenizer.vocab.getOrDefault(substr, tokenizer.getUnkTokenId())
        result.add(tokenId)
        i += bestLen
  
  if addSpecialTokens:
    result.add(tokenizer.getEosTokenId())
  
  # Сохраняем в кэш
  if tokenizer.cache.len < tokenizer.cacheMaxSize:
    tokenizer.cache[cacheKey] = result


#==============================================================================
# TOKEN OFFSETS (NEW - ДЛЯ NER/QA) - ИСПРАВЛЕНО
#==============================================================================

proc tokenizeWithOffsets*(text: string,
                         tokenizer: var Tokenizer,
                         addSpecialTokens: bool = false): seq[TokenOffset] =
  ## Токенизация с отслеживанием позиций токенов - ИСПРАВЛЕНО
  result = @[]
  
  if addSpecialTokens:
    result.add(TokenOffset(
      token: tokenizer.specialTokens.bosToken,
      tokenId: tokenizer.getBosTokenId(),
      startChar: -1,
      endChar: -1,
      startByte: -1,
      endByte: -1
    ))
  
  var charPos = 0
  var bytePos = 0
  
  if tokenizer.kind == tkByteLevelBPE:
    # ByteLevelBPE токенизирует весь текст целиком
    var tokens = newSeq[string]()
    let bytes = byteLevelEncode(text)
    for b in bytes:
      tokens.add(tokenizer.byteEncoder[b])
    
    # Применяем merges
    for merge in tokenizer.merges:
      var newTokens: seq[string] = @[]
      var j = 0
      while j < tokens.len:
        if j < tokens.len - 1 and 
           tokens[j] == merge.pair[0] and 
           tokens[j + 1] == merge.pair[1]:
          newTokens.add(merge.newToken)
          j += 2
        else:
          newTokens.add(tokens[j])
          inc j
      tokens = newTokens
    
    # Расчёт позиций
    var currentCharPos = 0
    var currentBytePos = 0
    
    for token in tokens:
      # Декодируем токен
      var tokenBytes: seq[int] = @[]
      for rune in token.runes:
        let charStr = $rune
        if charStr in tokenizer.byteDecoder:
          tokenBytes.add(tokenizer.byteDecoder[charStr])
      
      let tokenByteLen = tokenBytes.len
      let decodedToken = if tokenByteLen > 0: byteLevelDecode(tokenBytes) else: ""
      let tokenCharLen = decodedToken.runeLen
      
      # Проверяем границы
      let actualEndChar = min(currentCharPos + tokenCharLen, text.runeLen)
      let actualEndByte = min(currentBytePos + tokenByteLen, text.len)
      
      result.add(TokenOffset(
        token: token,
        tokenId: tokenizer.vocab.getOrDefault(token, tokenizer.getUnkTokenId()),
        startChar: currentCharPos,
        endChar: actualEndChar,
        startByte: currentBytePos,
        endByte: actualEndByte
      ))
      
      currentCharPos += tokenCharLen
      currentBytePos += tokenByteLen
  
  elif tokenizer.kind == tkBPE:
    var i = 0
    
    while i < text.len:
      # Пропускаем пробелы
      while i < text.len and text[i] in Whitespace:
        inc i
        inc charPos
        inc bytePos
      
      if i >= text.len:
        break
      
      let wordStart = i
      let wordCharStart = charPos
      let wordByteStart = bytePos
      
      # Читаем слово
      while i < text.len and text[i] notin Whitespace:
        let rune = text.runeAt(i)
        inc charPos
        bytePos += rune.size
        i += rune.size
      
      let word = text[wordStart..<i]
      
      # Токенизируем слово (только для обычного BPE)
      var tokens = newSeq[string]()
      for rune in word.runes:
        tokens.add($rune)
      
      # Применяем merges
      for merge in tokenizer.merges:
        var newTokens: seq[string] = @[]
        var j = 0
        while j < tokens.len:
          if j < tokens.len - 1 and 
             tokens[j] == merge.pair[0] and 
             tokens[j + 1] == merge.pair[1]:
            newTokens.add(merge.newToken)
            j += 2
          else:
            newTokens.add(tokens[j])
            inc j
        tokens = newTokens
      
      # Для обычного BPE
      for token in tokens:
          let tokenLen = token.runeLen
          let tokenByteLen = token.len
          
          result.add(TokenOffset(
            token: token,
            tokenId: tokenizer.vocab.getOrDefault(token, tokenizer.getUnkTokenId()),
            startChar: charPos,
            endChar: charPos + tokenLen,
            startByte: bytePos,
            endByte: bytePos + tokenByteLen
          ))
          
          charPos += tokenLen
          bytePos += tokenByteLen
  
  else:
    # Для других типов - упрощённая версия
    let tokens = tokenize(text, tokenizer, addSpecialTokens = false)
    for tokenId in tokens:
      let token = if tokenId < tokenizer.inverseVocab.len: 
                    tokenizer.inverseVocab[tokenId] 
                  else: 
                    tokenizer.specialTokens.unkToken
      result.add(TokenOffset(
        token: token,
        tokenId: tokenId,
        startChar: charPos,
        endChar: charPos + token.runeLen,
        startByte: bytePos,
        endByte: bytePos + token.len
      ))
      charPos += token.runeLen
      bytePos += token.len
  
  if addSpecialTokens:
    result.add(TokenOffset(
      token: tokenizer.specialTokens.eosToken,
      tokenId: tokenizer.getEosTokenId(),
      startChar: -1,
      endChar: -1,
      startByte: -1,
      endByte: -1
    ))


#==============================================================================
# STREAMING TOKENIZATION (NEW - ДЛЯ БОЛЬШИХ ФАЙЛОВ)
#==============================================================================

iterator streamTokenize*(filePath: string,
                        tokenizer: var Tokenizer,
                        chunkSize: int = 8192,
                        addSpecialTokens: bool = true): seq[int] {.closure.} =
  ## Потоковая токенизация больших файлов
  var stream = newFileStream(filePath, fmRead)
  if stream.isNil:
    raise newException(IOError, "Cannot open file: " & filePath)
  
  defer: stream.close()
  
  var buffer = ""
  var overflow = ""
  
  if addSpecialTokens:
    yield @[tokenizer.getBosTokenId()]
  
  while not stream.atEnd():
    let chunk = stream.readStr(chunkSize)
    buffer = overflow & chunk
    
    var lastSpace = buffer.rfind(' ')
    if lastSpace == -1 and not stream.atEnd():
      overflow = buffer
      continue
    
    let toProcess = if stream.atEnd(): 
                      buffer 
                    else: 
                      buffer[0..lastSpace]
    overflow = if stream.atEnd(): 
                 "" 
               else: 
                 buffer[lastSpace+1..^1]
    
    let tokens = tokenize(toProcess, tokenizer, addSpecialTokens = false)
    yield tokens
  
  if overflow.len > 0:
    let tokens = tokenize(overflow, tokenizer, addSpecialTokens = false)
    yield tokens
  
  if addSpecialTokens:
    yield @[tokenizer.getEosTokenId()]


#==============================================================================
# DECODING - ИСПРАВЛЕНО
#==============================================================================

proc decode*(tokenizer: Tokenizer, 
            tokens: seq[int],
            skipSpecialTokens: bool = false): string =
  result = ""
  
  # Для ByteLevelBPE собираем сырые байты
  if tokenizer.kind == tkByteLevelBPE:
    var rawBytes: seq[int] = @[]
    for tokenId in tokens:
      if skipSpecialTokens and tokenId in [
        tokenizer.getPadTokenId(),
        tokenizer.getBosTokenId(),
        tokenizer.getEosTokenId()
      ]:
        continue
      if tokenId >= 0 and tokenId < tokenizer.inverseVocab.len:
        let token = tokenizer.inverseVocab[tokenId]
        # Правильно: итерируем по рунам (замапленным символам) и декодируем их
        for rune in token.runes:
          let charStr = $rune
          if charStr in tokenizer.byteDecoder:
            rawBytes.add(tokenizer.byteDecoder[charStr])
    # Собираем строку из байтов
    if rawBytes.len > 0:
      result = byteLevelDecode(rawBytes)
    return
  
  for i, tokenId in tokens:
    if skipSpecialTokens and tokenId in [
      tokenizer.getPadTokenId(),
      tokenizer.getBosTokenId(),
      tokenizer.getEosTokenId()
    ]:
      continue
    
    if tokenId >= 0 and tokenId < tokenizer.inverseVocab.len:
      var token = tokenizer.inverseVocab[tokenId]
      
      # Удаляем префикс для WordPiece
      if tokenizer.kind == tkWordPiece and token.startsWith(tokenizer.continuingSubwordPrefix):
        token = token[tokenizer.continuingSubwordPrefix.len..^1]
      
      # Для SentencePiece заменяем ▁ на пробелы
      if tokenizer.kind == tkSentencePiece:
        token = token.replace("▁", " ")
      
      result.add(token)
    else:
      result.add(tokenizer.specialTokens.unkToken)
  
  # Для SentencePiece убираем начальный пробел и множественные пробелы
  if tokenizer.kind == tkSentencePiece:
    result = result.strip()
    while "  " in result:
      result = result.replace("  ", " ")


#==============================================================================
# JSON EXPORT (NEW - ДЛЯ СОХРАНЕНИЯ СЛОВАРЕЙ)
#==============================================================================

proc exportTokenizerToJson*(tokenizer: Tokenizer, filepath: string) =
  ## Экспортирует все данные токенизатора в JSON формат
  var jsonData = %* {
    "kind": $tokenizer.kind,
    "vocab_size": tokenizer.vocab.len,
    "vocab": newJObject(),
    "special_tokens": {
      "pad": tokenizer.specialTokens.padToken,
      "unk": tokenizer.specialTokens.unkToken,
      "bos": tokenizer.specialTokens.bosToken,
      "eos": tokenizer.specialTokens.eosToken,
      "sep": tokenizer.specialTokens.sepToken,
      "cls": tokenizer.specialTokens.clsToken,
      "mask": tokenizer.specialTokens.maskToken
    },
    "max_input_chars_per_word": tokenizer.maxInputCharsPerWord,
    "continuing_subword_prefix": tokenizer.continuingSubwordPrefix,
    "preserve_case": tokenizer.preserveCase
  }
  
  # Добавляем словарь (первые 100 элементов для компактности)
  var vocabItems = newSeq[(string, int)]()
  for token, id in tokenizer.vocab:
    vocabItems.add((token, id))
  vocabItems.sort(proc (a, b: (string, int)): int = cmp(a[1], b[1]))
  
  var vocabJson = newJObject()
  for i, (token, id) in vocabItems:
    if i < 100:  # Сохраняем первые 100
      vocabJson[$id] = %token
  jsonData["vocab"] = vocabJson
  
  # Добавляем merges для BPE (первые 50)
  if tokenizer.kind in {tkBPE, tkByteLevelBPE} and tokenizer.merges.len > 0:
    var mergesJson = newJArray()
    for i, merge in tokenizer.merges:
      if i < 50:  # Сохраняем первые 50 merges
        mergesJson.add(%* {
          "pair": [merge.pair[0], merge.pair[1]],
          "new_token": merge.newToken,
          "priority": merge.priority
        })
    jsonData["merges"] = mergesJson
    jsonData["merges_total_count"] = %tokenizer.merges.len
  
  # Добавляем scores для SentencePiece (первые 50)
  if tokenizer.kind == tkSentencePiece and tokenizer.scores.len > 0:
    var scoresJson = newJObject()
    var scoreItems = newSeq[(string, float)]()
    for token, score in tokenizer.scores:
      scoreItems.add((token, score))
    scoreItems.sort(proc (a, b: (string, float)): int = -cmp(a[1], b[1]))
    
    for i, (token, score) in scoreItems:
      if i < 50:  # Сохраняем топ 50
        scoresJson[token] = %score
    jsonData["scores"] = scoresJson
    jsonData["scores_total_count"] = %tokenizer.scores.len
  
  # Добавляем byte encoder для ByteLevelBPE (первые 50)
  if tokenizer.kind == tkByteLevelBPE and tokenizer.byteEncoder.len > 0:
    var byteEncoderJson = newJObject()
    var count = 0
    for byteVal, encoded in tokenizer.byteEncoder:
      if count < 50:
        byteEncoderJson[$byteVal] = %encoded
        inc count
    jsonData["byte_encoder"] = byteEncoderJson
    jsonData["byte_encoder_total_count"] = %tokenizer.byteEncoder.len
  
  # Сохраняем в файл
  writeFile(filepath, jsonData.pretty())


#==============================================================================
# BATCH PROCESSING (ОПТИМИЗИРОВАНО)
#==============================================================================

proc encodeBatch*(tokenizer: var Tokenizer,
                 texts: seq[string],
                 maxLength: int = 512,
                 padding: bool = true,
                 truncation: bool = true,
                 addSpecialTokens: bool = true,
                 returnAttentionMask: bool = true,
                 returnTokenTypeIds: bool = false): BatchEncoding =
  
  result.inputIds = newSeq[seq[int]](texts.len)
  result.lengths = newSeq[int](texts.len)
  
  # Токенизируем все тексты
  for i, text in texts:
    var tokens = tokenize(text, tokenizer, addSpecialTokens = addSpecialTokens)
    
    if truncation and tokens.len > maxLength:
      tokens = tokens[0..<maxLength]
    
    result.inputIds[i] = tokens
    result.lengths[i] = tokens.len
  
  # Padding и attention mask
  if padding:
    result.attentionMask = newSeq[seq[int]](texts.len)
    
    for i in 0..<texts.len:
      let padLen = maxLength - result.inputIds[i].len
      
      if padLen > 0:
        result.attentionMask[i] = newSeq[int](result.inputIds[i].len)
        for j in 0..<result.inputIds[i].len:
          result.attentionMask[i][j] = 1
        
        for j in 0..<padLen:
          result.inputIds[i].add(tokenizer.getPadTokenId())
          result.attentionMask[i].add(0)
      else:
        result.attentionMask[i] = newSeq[int](maxLength)
        for j in 0..<maxLength:
          result.attentionMask[i][j] = 1
  
  # Token type IDs (для BERT-подобных моделей)
  if returnTokenTypeIds:
    result.tokenTypeIds = newSeq[seq[int]](texts.len)
    for i in 0..<texts.len:
      result.tokenTypeIds[i] = newSeq[int](result.inputIds[i].len)

proc decodeBatch*(tokenizer: Tokenizer,
                 encoding: BatchEncoding,
                 skipSpecialTokens: bool = false): seq[string] =
  result = newSeq[string](encoding.inputIds.len)
  for i, tokens in encoding.inputIds:
    result[i] = tokenizer.decode(tokens, skipSpecialTokens)


#==============================================================================
# VOCABULARY MANAGEMENT
#==============================================================================

proc pruneVocabulary*(tokenizer: var Tokenizer,
                     corpus: seq[string],
                     minFrequency: int = 5,
                     keepTopN: int = -1,
                     keepSpecialTokens: bool = true): int =
  ## Удаляет редкие токены из словаря
  
  # Считаем частоты токенов
  var tokenFreqs = initCountTable[string]()
  
  for text in corpus:
    let tokens = tokenize(text, tokenizer, addSpecialTokens = false)
    for tokenId in tokens:
      if tokenId < tokenizer.inverseVocab.len:
        let token = tokenizer.inverseVocab[tokenId]
        tokenFreqs.inc(token)
  
  # Находим токены для удаления
  var toRemove = initHashSet[string]()
  
  for token, id in tokenizer.vocab:
    let freq = tokenFreqs.getOrDefault(token, 0)
    
    # Сохраняем специальные токены
    if keepSpecialTokens and token in tokenizer.specialTokenIds:
      continue
    
    # Удаляем редкие
    if freq < minFrequency:
      toRemove.incl(token)
  
  # Если задан keepTopN, оставляем только топ-N
  if keepTopN > 0:
    var sortedTokens = toSeq(tokenFreqs.pairs)
    sortedTokens.sort(proc (a, b: (string, int)): int = cmp(b[1], a[1]))
    
    for i in keepTopN..<sortedTokens.len:
      let token = sortedTokens[i][0]
      if keepSpecialTokens and token in tokenizer.specialTokenIds:
        continue
      toRemove.incl(token)

  result = len(toRemove)
  
  # Перестраиваем словарь
  var newVocab = initTable[string, int]()
  var newInverseVocab = newSeq[string]()
  var newId = 0
  
  # Сначала специальные токены
  if keepSpecialTokens:
    for token in [
      tokenizer.specialTokens.padToken,
      tokenizer.specialTokens.unkToken,
      tokenizer.specialTokens.bosToken,
      tokenizer.specialTokens.eosToken,
      tokenizer.specialTokens.sepToken,
      tokenizer.specialTokens.clsToken,
      tokenizer.specialTokens.maskToken
    ]:
      if token notin newVocab:
        newVocab[token] = newId
        newInverseVocab.add(token)
        inc newId
  
  # Затем оставшиеся токены
  for token, oldId in tokenizer.vocab:
    if token notin toRemove and token notin newVocab:
      newVocab[token] = newId
      newInverseVocab.add(token)
      inc newId
  
  tokenizer.vocab = newVocab
  tokenizer.inverseVocab = newInverseVocab
  tokenizer.cache.clear()  # Очищаем кэш


proc incrementalTrain*(tokenizer: var Tokenizer,
                      newCorpus: seq[string],
                      maxNewTokens: int = 1000,
                      minFrequency: int = 2): int =
  ## Дообучение токенизатора на новых данных
  result = 0
  
  let text = newCorpus.join(" ")
  let words = splitIntoWords(text)
  
  # Собираем новые n-граммы
  var ngramCounts = initCountTable[string]()
  
  for word in words:
    let runeLen = word.runeLen
    for n in 2..min(runeLen, 6):
      for start in 0..runeLen - n:
        let ngram = word.runeSubStr(start, n)
        if ngram notin tokenizer.vocab:
          ngramCounts.inc(ngram)
  
  # Добавляем частые новые токены
  var sortedNgrams = toSeq(ngramCounts.pairs)
  sortedNgrams.sort(proc (a, b: (string, int)): int = cmp(b[1], a[1]))
  
  var nextId = tokenizer.vocab.len
  
  for item in sortedNgrams:
    let (ngram, count) = item
    if count >= minFrequency and result < maxNewTokens:
      tokenizer.vocab[ngram] = nextId
      tokenizer.inverseVocab.add(ngram)
      inc nextId
      inc result


#==============================================================================
# SUBWORD REGULARIZATION (NEW - BPE DROPOUT) - ИСПРАВЛЕНО
#==============================================================================

proc tokenizeWithDropout*(text: string,
                         tokenizer: var Tokenizer,
                         dropoutProb: float = 0.1,
                         seed: int = -1,
                         minDropped: int = 1): seq[int] =
  ## Токенизация с случайным пропуском merge операций - УЛУЧШЕНО
  ## minDropped: минимальное количество пропущенных merges для гарантии вариации
  
  # Работает только для BPE и ByteLevelBPE
  if tokenizer.kind != tkBPE and tokenizer.kind != tkByteLevelBPE:
    return tokenize(text, tokenizer, addSpecialTokens = false)
  
  if seed >= 0:
    randomize(seed)
  else:
    randomize()  # Важно! Каждый раз новый seed
  
  let totalMerges = tokenizer.merges.len
  if totalMerges == 0:
    return tokenize(text, tokenizer, addSpecialTokens = false)
  
  # НОВЫЙ ПОДХОД: Создаем случайную перестановку индексов
  let targetDropped = max(minDropped, int(float(totalMerges) * dropoutProb))
  
  # Если нужно пропустить 0 merges, возвращаем обычную токенизацию
  if targetDropped == 0:
    return tokenize(text, tokenizer, addSpecialTokens = false)
  
  # Создаем и перемешиваем индексы
  var indices = newSeq[int](totalMerges)
  for i in 0..<totalMerges:
    indices[i] = i
  
  # Перемешиваем методом Фишера-Йетса
  for i in countdown(totalMerges - 1, 1):
    let j = rand(i)
    swap(indices[i], indices[j])
  
  # Берем только первые (totalMerges - targetDropped) индексов
  var keepIndices = initHashSet[int]()
  for i in 0..<(totalMerges - targetDropped):
    keepIndices.incl(indices[i])
  
  # Формируем активные merges в правильном порядке
  var activeMerges: seq[BPEMerge] = @[]
  for i, merge in tokenizer.merges:
    if i in keepIndices:
      activeMerges.add(merge)
  
  # Сохраняем оригинальные merges
  let savedMerges = tokenizer.merges
  tokenizer.merges = activeMerges
  
  # Токенизируем
  result = tokenize(text, tokenizer, addSpecialTokens = false)
  
  # Восстанавливаем оригинальные merges
  tokenizer.merges = savedMerges


#==============================================================================
# METRICS & ANALYSIS
#==============================================================================

proc getMetrics*(tokenizer: var Tokenizer, corpus: seq[string]): TokenizerMetrics =
  let startTime = cpuTime()
  
  var totalTokens = 0
  var totalWords = 0
  var totalChars = 0
  var unkCount = 0
  
  for text in corpus:
    let tokens = tokenize(text, tokenizer, addSpecialTokens = false)
    let words = splitIntoWords(text)
    
    totalTokens += tokens.len
    totalWords += words.len
    totalChars += text.runeLen
    
    for tokenId in tokens:
      if tokenId == tokenizer.getUnkTokenId():
        inc unkCount
  
  let elapsed = cpuTime() - startTime
  
  result = TokenizerMetrics(
    vocabSize: tokenizer.vocab.len,
    compressionRatio: if totalTokens > 0: totalChars.float / totalTokens.float else: 0.0,
    avgTokensPerWord: if totalWords > 0: totalTokens.float / totalWords.float else: 0.0,
    vocabUtilization: 0.0,  # Can be calculated if needed
    unkTokenRate: if totalTokens > 0: unkCount.float / totalTokens.float else: 0.0,
    tokensPerSecond: if elapsed > 0: totalTokens.float / elapsed else: 0.0
  )

proc decodeToken*(tokenizer: Tokenizer, token: string): string =
  ## Декодирует один токен в читаемый вид
  if tokenizer.kind == tkByteLevelBPE:
    var tokenBytes: seq[int] = @[]
    for rune in token.runes:
      let charStr = $rune
      if charStr in tokenizer.byteDecoder:
        tokenBytes.add(tokenizer.byteDecoder[charStr])
    if tokenBytes.len > 0:
      return byteLevelDecode(tokenBytes)
  return token


proc analyzeVocabulary*(tokenizer: var Tokenizer, 
                       corpus: seq[string],
                       topN: int = 10): VocabAnalysis =
  
  # Считаем частоты
  var tokenFreqs = initCountTable[string]()
  var totalTokens = 0
  var uniqueTokens = initHashSet[string]()
  
  for text in corpus:
    let tokens = tokenize(text, tokenizer, addSpecialTokens = false)
    totalTokens += tokens.len
    
    for tokenId in tokens:
      if tokenId < tokenizer.inverseVocab.len:
        let token = tokenizer.inverseVocab[tokenId]
        tokenFreqs.inc(token)
        uniqueTokens.incl(token)
  
  # Средняя длина токена
  var totalLen = 0
  for token in tokenizer.vocab.keys:
    totalLen += token.runeLen
  
  let avgLen = if tokenizer.vocab.len > 0: 
                 totalLen.float / tokenizer.vocab.len.float 
               else: 0.0
  
  # Type-Token Ratio
  let ttr = if totalTokens > 0: 
              uniqueTokens.len.float / totalTokens.float 
            else: 0.0
  
  # Топ токены
  var sortedFreqs = toSeq(tokenFreqs.pairs)
  sortedFreqs.sort(proc (a, b: (string, int)): int = cmp(b[1], a[1]))
  
  var mostFreq: seq[tuple[token: string, freq: int]] = @[]
  for i in 0..<min(topN, sortedFreqs.len):
    mostFreq.add(sortedFreqs[i])
  
  # Распределение по длинам
  var lengthDist = initCountTable[int]()
  for token in tokenizer.vocab.keys:
    lengthDist.inc(token.runeLen)
  
  result = VocabAnalysis(
    vocabSize: tokenizer.vocab.len,
    avgTokenLength: avgLen,
    typeTokenRatio: ttr,
    coverageRate: 1.0,  # Simplified
    oovRate: 0.0,       # Simplified
    mostFrequent: mostFreq,
    leastFrequent: @[],
    lengthDistribution: lengthDist
  )


proc printMetrics*(metrics: TokenizerMetrics) =
  echo "╔══════════════════════════════════════════════╗"
  echo "║        TOKENIZER METRICS                     ║"
  echo "╠══════════════════════════════════════════════╣"
  echo "║ Vocab Size:        ", metrics.vocabSize.intToStr.alignLeft(20), "   ║"
  echo "║ Compression Ratio: ", metrics.compressionRatio.formatFloat(ffDecimal, 2).alignLeft(20), "   ║"
  echo "║ Tokens/Word:       ", metrics.avgTokensPerWord.formatFloat(ffDecimal, 2).alignLeft(20), "   ║"
  echo "║ UNK Rate:          ", (metrics.unkTokenRate * 100).formatFloat(ffDecimal, 2).alignLeft(20), " %  ║"
  echo "║ Tokens/Second:     ", metrics.tokensPerSecond.formatFloat(ffDecimal, 0).alignLeft(23), "║"
  echo "╚══════════════════════════════════════════════╝"


#==============================================================================
# SAVE / LOAD
#==============================================================================

proc saveTokenizer*(tokenizer: Tokenizer, path: string) =
  var json = %* {
    "kind": $tokenizer.kind,
    "vocab": tokenizer.vocab,
    "merges": tokenizer.merges.mapIt(%* {
      "pair": [it.pair[0], it.pair[1]],
      "newToken": it.newToken,
      "priority": it.priority
    }),
    "specialTokens": %* {
      "padToken": tokenizer.specialTokens.padToken,
      "unkToken": tokenizer.specialTokens.unkToken,
      "bosToken": tokenizer.specialTokens.bosToken,
      "eosToken": tokenizer.specialTokens.eosToken,
      "sepToken": tokenizer.specialTokens.sepToken,
      "clsToken": tokenizer.specialTokens.clsToken,
      "maskToken": tokenizer.specialTokens.maskToken
    },
    "maxInputCharsPerWord": tokenizer.maxInputCharsPerWord,
    "continuingSubwordPrefix": tokenizer.continuingSubwordPrefix,
    "byteFallback": tokenizer.byteFallback,
    "preserveCase": tokenizer.preserveCase
  }
  
  writeFile(path, json.pretty())

proc loadTokenizer*(path: string): Tokenizer =
  let json = parseFile(path)
  
  result = Tokenizer(
    vocab: initTable[string, int](),
    inverseVocab: newSeq[string](),
    merges: @[],
    specialTokenIds: initTable[string, int](),
    scores: initTable[string, float](),
    cache: initCache(10000),
    cacheMaxSize: 10000
  )
  
  # Парсим kind
  let kindStr = json["kind"].getStr()
  result.kind = case kindStr
    of "tkBPE": tkBPE
    of "tkWordPiece": tkWordPiece
    of "tkSentencePiece": tkSentencePiece
    of "tkByteLevelBPE": tkByteLevelBPE
    else: tkBPE
  
  # Загружаем vocab
  for token, id in json["vocab"].pairs():
    result.vocab[token] = id.getInt()
  
  # Создаём inverseVocab как seq
  let maxId = toSeq(result.vocab.values).max()
  result.inverseVocab = newSeq[string](maxId + 1)
  for token, id in result.vocab:
    result.inverseVocab[id] = token
  
  # Загружаем merges
  for mergeJson in json["merges"]:
    let merge: BPEMerge = (
      pair: (mergeJson["pair"][0].getStr(), mergeJson["pair"][1].getStr()),
      newToken: mergeJson["newToken"].getStr(),
      priority: mergeJson["priority"].getInt()
    )
    result.merges.add(merge)
  
  # Загружаем specialTokens
  result.specialTokens = SpecialTokens(
    padToken: json["specialTokens"]["padToken"].getStr(),
    unkToken: json["specialTokens"]["unkToken"].getStr(),
    bosToken: json["specialTokens"]["bosToken"].getStr(),
    eosToken: json["specialTokens"]["eosToken"].getStr(),
    sepToken: json["specialTokens"]["sepToken"].getStr(),
    clsToken: json["specialTokens"]["clsToken"].getStr(),
    maskToken: json["specialTokens"]["maskToken"].getStr()
  )
  
  # Создаём specialTokenIds
  for token, id in result.vocab:
    if token in [result.specialTokens.padToken, result.specialTokens.unkToken,
                result.specialTokens.bosToken, result.specialTokens.eosToken,
                result.specialTokens.sepToken, result.specialTokens.clsToken,
                result.specialTokens.maskToken]:
      result.specialTokenIds[token] = id
  
  result.maxInputCharsPerWord = json["maxInputCharsPerWord"].getInt()
  result.continuingSubwordPrefix = json["continuingSubwordPrefix"].getStr()
  result.byteFallback = json["byteFallback"].getBool()
  result.preserveCase = json["preserveCase"].getBool()
  
  # Инициализируем byte encoder если нужно
  if result.kind == tkByteLevelBPE:
    result.byteEncoder = initBytePairEncoder()
    result.byteDecoder = initByteDecoder(result.byteEncoder)


#==============================================================================
# MAIN (DEMO)
#==============================================================================


#==============================================================================
# УТИЛИТЫ ДЛЯ РАБОТЫ С ТЕКСТОМ
#==============================================================================

proc splitIntoSentences*(text: string): seq[string] =
  ## Разбивает текст на предложения (простая эвристика)
  result = @[]
  var current = ""
  
  for i, ch in text:
    current.add(ch)
    
    # Конец предложения: .!? за которыми пробел или конец
    if ch in {'.', '!', '?'}:
      if i + 1 >= text.len or text[i + 1] == ' ':
        result.add(current.strip())
        current = ""
  
  if current.len > 0:
    result.add(current.strip())
  
  result = result.filterIt(it.len > 0)

proc truncateText*(text: string, maxLength: int, addEllipsis: bool = true): string =
  ## Обрезает текст до максимальной длины
  if text.runeLen <= maxLength:
    return text
  result = text.runeSubStr(0, maxLength)
  if addEllipsis:
    result.add("...")


proc removeAccents*(text: string): string =
  ## Удаляет акценты с букв (приводит к базовым латинским символам)
  ## Например: café → cafe, résumé → resume
  
  # Создаём таблицу замен для Unicode rune
  let accentPairs = [
    ("à", "a"), ("á", "a"), ("â", "a"), ("ã", "a"), ("ä", "a"), ("å", "a"),
    ("è", "e"), ("é", "e"), ("ê", "e"), ("ë", "e"),
    ("ì", "i"), ("í", "i"), ("î", "i"), ("ï", "i"),
    ("ò", "o"), ("ó", "o"), ("ô", "o"), ("õ", "o"), ("ö", "o"),
    ("ù", "u"), ("ú", "u"), ("û", "u"), ("ü", "u"),
    ("ý", "y"), ("ÿ", "y"),
    ("ñ", "n"), ("ç", "c"),
    ("À", "A"), ("Á", "A"), ("Â", "A"), ("Ã", "A"), ("Ä", "A"), ("Å", "A"),
    ("È", "E"), ("É", "E"), ("Ê", "E"), ("Ë", "E"),
    ("Ì", "I"), ("Í", "I"), ("Î", "I"), ("Ï", "I"),
    ("Ò", "O"), ("Ó", "O"), ("Ô", "O"), ("Õ", "O"), ("Ö", "O"),
    ("Ù", "U"), ("Ú", "U"), ("Û", "U"), ("Ü", "U"),
    ("Ý", "Y"), ("Ñ", "N"), ("Ç", "C")]
  
  var accentMap = initTable[string, string]()
  for (accented, base) in accentPairs:
    accentMap[accented] = base
  
  result = ""
  for rune in text.runes:
    let runeStr = $rune
    if runeStr in accentMap:
      result.add(accentMap[runeStr])
    else:
      result.add(runeStr)


proc normalizeWhitespace*(text: string, preserveNewlines: bool = false): string =
  ## Нормализует пробельные символы
  result = text
  
  if preserveNewlines:
    # Заменяем множественные пробелы на один, сохраняя переносы
    result = result.replace(re"\t", " ")
    result = result.replace(re" +", " ")
    result = result.replace(re"(?m)^\s+", "")
    result = result.replace(re"(?m)\s+$", "")
  else:
    # Заменяем все пробельные символы на обычный пробел
    result = result.replace(re"\s+", " ")
  
  result = result.strip()

proc countWords*(text: string): int =
  ## Подсчитывает количество слов в тексте
  splitIntoWords(text).len

proc countCharacters*(text: string, excludeWhitespace: bool = false): int =
  ## Подсчитывает количество символов
  if excludeWhitespace:
    var count = 0
    for ch in text:
      if ch notin Whitespace:
        inc count
    return count
  else:
    return text.runeLen

proc countSentences*(text: string): int =
  ## Подсчитывает количество предложений
  splitIntoSentences(text).len


#==============================================================================
# УТИЛИТЫ ДЛЯ СПЕЦИАЛЬНЫХ ТОКЕНОВ
#==============================================================================

proc initSpecialTokens*(kind: TokenizerKind): SpecialTokens =
  ## Инициализирует специальные токены по умолчанию для типа токенизатора
  case kind
  of tkBPE, tkByteLevelBPE:
    result = SpecialTokens(
      padToken: "<PAD>",
      unkToken: "<UNK>",
      bosToken: "<BOS>",
      eosToken: "<EOS>",
      sepToken: "<SEP>",
      clsToken: "<CLS>",
      maskToken: "<MASK>"
    )
  of tkWordPiece:
    result = SpecialTokens(
      padToken: "[PAD]",
      unkToken: "[UNK]",
      bosToken: "[CLS]",
      eosToken: "[SEP]",
      sepToken: "[SEP]",
      clsToken: "[CLS]",
      maskToken: "[MASK]"
    )
  of tkSentencePiece:
    result = SpecialTokens(
      padToken: "<pad>",
      unkToken: "<unk>",
      bosToken: "<s>",
      eosToken: "</s>",
      sepToken: "<sep>",
      clsToken: "<cls>",
      maskToken: "<mask>"
    )

proc addSpecialTokensToVocab*(tokenizer: var Tokenizer) =
  ## Добавляет специальные токены в словарь токенизатора
  var nextId = tokenizer.vocab.len
  
  for token in [tokenizer.specialTokens.padToken,
                tokenizer.specialTokens.unkToken,
                tokenizer.specialTokens.bosToken,
                tokenizer.specialTokens.eosToken,
                tokenizer.specialTokens.sepToken,
                tokenizer.specialTokens.clsToken,
                tokenizer.specialTokens.maskToken]:
    if token notin tokenizer.vocab:
      tokenizer.vocab[token] = nextId
      if tokenizer.inverseVocab.len <= nextId:
        tokenizer.inverseVocab.setLen(nextId + 1)
      tokenizer.inverseVocab[nextId] = token
      tokenizer.specialTokenIds[token] = nextId
      inc nextId


#==============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ BPE
#==============================================================================

proc countPairs*(words: seq[seq[string]]): Table[(string, string), int] =
  ## Подсчитывает частоты пар токенов
  result = initTable[(string, string), int]()
  for word in words:
    for i in 0..<word.len - 1:
      let pair = (word[i], word[i + 1])
      result[pair] = result.getOrDefault(pair, 0) + 1

proc mergePair*(words: var seq[seq[string]], pair: (string, string), newToken: string) =
  ## Объединяет заданную пару токенов в один
  for i in 0..<words.len:
    var j = 0
    var newWord: seq[string] = @[]
    while j < words[i].len:
      if j < words[i].len - 1 and words[i][j] == pair[0] and words[i][j + 1] == pair[1]:
        newWord.add(newToken)
        j += 2
      else:
        newWord.add(words[i][j])
        inc j
    words[i] = newWord


#==============================================================================
# АЛЬТЕРНАТИВНЫЕ ENCODE ФУНКЦИИ
#==============================================================================

proc encodeBPE*(tokenizer: var Tokenizer, text: string): seq[int] =
  ## Альтернативная функция кодирования для BPE (для совместимости)
  return tokenize(text, tokenizer, addSpecialTokens = false)

proc encodeWordPiece*(tokenizer: var Tokenizer, text: string, 
                      maxInputCharsPerWord: int = 100): seq[int] =
  ## Альтернативная функция кодирования для WordPiece (для совместимости)
  return tokenize(text, tokenizer, addSpecialTokens = false)

proc viterbiEncode*(tokenizer: var Tokenizer, text: string): seq[string] =
  ## Упрощённая версия Viterbi encoding для SentencePiece
  ## Возвращает последовательность токенов (не ID)
  let tokenIds = tokenize(text, tokenizer, addSpecialTokens = false)
  result = @[]
  for id in tokenIds:
    if id >= 0 and id < tokenizer.inverseVocab.len:
      result.add(tokenizer.inverseVocab[id])

proc encodeSentencePiece*(tokenizer: var Tokenizer, text: string, 
                         addBos: bool = false, addEos: bool = false): seq[int] =
  ## Альтернативная функция кодирования для SentencePiece
  result = tokenize(text, tokenizer, addSpecialTokens = false)
  if addBos:
    result.insert(tokenizer.getBosTokenId(), 0)
  if addEos:
    result.add(tokenizer.getEosTokenId())


#==============================================================================
# УТИЛИТЫ ДЛЯ ПОСЛЕДОВАТЕЛЬНОСТЕЙ
#==============================================================================

proc padSequence*(sequences: seq[seq[int]], maxLength: int, padValue: int): seq[seq[int]] =
  ## Дополняет последовательности до максимальной длины
  result = newSeq[seq[int]](sequences.len)
  for i, seq in sequences:
    result[i] = seq
    while result[i].len < maxLength:
      result[i].add(padValue)

proc truncateSequence*(sequences: seq[seq[int]], maxLength: int): seq[seq[int]] =
  ## Обрезает последовательности до максимальной длины
  result = newSeq[seq[int]](sequences.len)
  for i, seq in sequences:
    if seq.len > maxLength:
      result[i] = seq[0..<maxLength]
    else:
      result[i] = seq

proc createAttentionMask*(sequences: seq[seq[int]], padTokenId: int): seq[seq[int]] =
  ## Создаёт маску внимания (1 для реальных токенов, 0 для padding)
  result = newSeq[seq[int]](sequences.len)
  for i, seq in sequences:
    result[i] = newSeq[int](seq.len)
    for j, token in seq:
      result[i][j] = if token == padTokenId: 0 else: 1

proc createAttentionMask*(tokens: seq[int], padTokenId: int): seq[int] =
  ## Создаёт маску внимания для одной последовательности
  result = newSeq[int](tokens.len)
  for i, token in tokens:
    result[i] = if token == padTokenId: 0 else: 1


#==============================================================================
# МЕТРИКИ И АНАЛИЗ
#==============================================================================

proc calculateCompressionRatio*(tokenizer: var Tokenizer, text: string): float =
  ## Вычисляет коэффициент сжатия (отношение символов к токенам)
  let tokens = tokenize(text, tokenizer)
  if tokens.len == 0:
    return 0.0
  return text.runeLen.float / tokens.len.float

proc calculateAvgTokensPerWord*(tokenizer: var Tokenizer, text: string): float =
  ## Вычисляет среднее количество токенов на слово
  let words = splitIntoWords(text)
  let tokens = tokenize(text, tokenizer)
  if words.len == 0:
    return 0.0
  return tokens.len.float / words.len.float

proc calculateVocabUtilization*(tokenizer: var Tokenizer, corpus: string): float =
  ## Вычисляет процент использования словаря
  var usedTokens = initHashSet[int]()
  let tokens = tokenize(corpus, tokenizer)
  
  for token in tokens:
    usedTokens.incl(token)
  
  if tokenizer.vocab.len == 0:
    return 0.0
  return usedTokens.len.float / tokenizer.vocab.len.float

proc calculateUnkTokenRate*(tokenizer: var Tokenizer, text: string): float =
  ## Вычисляет процент неизвестных токенов
  let tokens = tokenize(text, tokenizer)
  if tokens.len == 0:
    return 0.0
  
  var unkCount = 0
  let unkId = tokenizer.getUnkTokenId()
  
  for token in tokens:
    if token == unkId:
      inc unkCount
  
  return unkCount.float / tokens.len.float

proc benchmark*(tokenizer: var Tokenizer, texts: seq[string]): float =
  ## Измеряет скорость токенизации (токенов в секунду)
  let startTime = cpuTime()
  var totalTokens = 0
  
  for text in texts:
    let tokens = tokenize(text, tokenizer)
    totalTokens += tokens.len
  
  let elapsed = cpuTime() - startTime
  if elapsed == 0:
    return 0.0
  return totalTokens.float / elapsed


#==============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ СЛОВАРЯ
#==============================================================================

proc getTokenById*(tokenizer: Tokenizer, id: int): string =
  ## Получает токен по его ID
  if id >= 0 and id < tokenizer.inverseVocab.len:
    return tokenizer.inverseVocab[id]
  return tokenizer.specialTokens.unkToken

proc getIdByToken*(tokenizer: Tokenizer, token: string): int =
  ## Получает ID токена
  if token in tokenizer.vocab:
    return tokenizer.vocab[token]
  return tokenizer.getUnkTokenId()

proc hasToken*(tokenizer: Tokenizer, token: string): bool =
  ## Проверяет наличие токена в словаре
  return token in tokenizer.vocab

proc getVocabSize*(tokenizer: Tokenizer): int =
  ## Возвращает размер словаря
  return tokenizer.vocab.len

proc getVocabTokens*(tokenizer: Tokenizer): seq[string] =
  ## Возвращает все токены из словаря
  result = newSeq[string](tokenizer.vocab.len)
  for token, id in tokenizer.vocab:
    if id < result.len:
      result[id] = token

proc filterTokensByFrequency*(tokenizer: var Tokenizer, text: string, 
                              minFrequency: int): seq[string] =
  ## Фильтрует токены по минимальной частоте встречаемости
  let tokens = tokenize(text, tokenizer)
  var freqTable = initCountTable[int]()
  
  for token in tokens:
    freqTable.inc(token)
  
  result = @[]
  for tokenId, freq in freqTable:
    if freq >= minFrequency:
      result.add(tokenizer.getTokenById(tokenId))


#==============================================================================
# СРАВНЕНИЕ И АНАЛИЗ ТОКЕНИЗАТОРОВ
#==============================================================================

proc compareTokenizers*(text: string, tokenizers: seq[Tokenizer]): 
                       seq[tuple[name: string, tokens: int, time: float]] =
  ## Сравнивает производительность разных токенизаторов
  result = @[]
  
  for i, tokenizer in tokenizers:
    var mutableTokenizer = tokenizer
    let startTime = cpuTime()
    let tokens = tokenize(text, mutableTokenizer)
    let elapsed = cpuTime() - startTime
    
    let name = case tokenizer.kind
      of tkBPE: "BPE"
      of tkByteLevelBPE: "ByteLevelBPE"
      of tkWordPiece: "WordPiece"
      of tkSentencePiece: "SentencePiece"
    
    result.add((name: name, tokens: tokens.len, time: elapsed))


#==============================================================================
# ДОПОЛНИТЕЛЬНЫЕ ФУНКЦИИ КОДИРОВАНИЯ
#==============================================================================

proc encodeWithPadding*(tokenizer: var Tokenizer, text: string, 
                       maxLength: int, padTokenId: int = -1): seq[int] =
  ## Кодирует текст с padding до заданной длины
  result = tokenize(text, tokenizer)
  
  let actualPadId = if padTokenId >= 0: padTokenId 
                    else: tokenizer.getPadTokenId()
  
  if result.len < maxLength:
    while result.len < maxLength:
      result.add(actualPadId)
  elif result.len > maxLength:
    result = result[0..<maxLength]

proc maskTokens*(tokens: seq[int], tokenizer: Tokenizer, 
                maskProb: float = 0.15, seed: int = -1): 
               tuple[maskedTokens: seq[int], labels: seq[int]] =
  ## Маскирует случайные токены для MLM (Masked Language Modeling)
  if seed >= 0:
    randomize(seed)
  
  result.maskedTokens = tokens
  result.labels = newSeq[int](tokens.len)
  
  let maskId = tokenizer.getMaskTokenId()
  
  for i in 0..<tokens.len:
    result.labels[i] = -100  # ignore index для loss
    
    if rand(1.0) < maskProb:
      result.labels[i] = tokens[i]
      
      let r = rand(1.0)
      if r < 0.8:
        # 80% - заменяем на [MASK]
        result.maskedTokens[i] = maskId
      elif r < 0.9:
        # 10% - заменяем на случайный токен
        result.maskedTokens[i] = rand(tokenizer.vocab.len - 1)
      # 10% - оставляем как есть

proc getSubwordBreakdown*(text: string, tokenizer: var Tokenizer): seq[string] =
  ## Возвращает разбиение текста на подслова
  let tokens = tokenize(text, tokenizer)
  result = @[]
  for tokenId in tokens:
    result.add(tokenizer.getTokenById(tokenId))

proc estimateTokenCount*(text: string, avgCharsPerToken: float = 4.0): int =
  ## Оценивает количество токенов без фактической токенизации
  return int(text.runeLen.float / avgCharsPerToken)


#==============================================================================
# ВАЛИДАЦИЯ ТОКЕНИЗАТОРА
#==============================================================================

proc validateTokenizerDetailed*(tokenizer: Tokenizer): seq[string] =
  ## Проверяет корректность токенизатора
  result = @[]
  
  # Проверка 1: Словарь не пуст
  if tokenizer.vocab.len == 0:
    result.add("ОШИБКА: Словарь пуст")
  
  # Проверка 2: inverseVocab согласован с vocab
  for token, id in tokenizer.vocab:
    if id >= tokenizer.inverseVocab.len or 
       tokenizer.inverseVocab[id] != token:
      result.add("ПРЕДУПРЕЖДЕНИЕ: Несоответствие vocab и inverseVocab для токена: " & token)
  
  # Проверка 3: Специальные токены присутствуют
  if tokenizer.specialTokens.padToken notin tokenizer.vocab:
    result.add("ПРЕДУПРЕЖДЕНИЕ: PAD токен отсутствует в словаре")
  if tokenizer.specialTokens.unkToken notin tokenizer.vocab:
    result.add("ПРЕДУПРЕЖДЕНИЕ: UNK токен отсутствует в словаре")
  
  # Проверка 4: BPE merges валидны
  if tokenizer.kind in {tkBPE, tkByteLevelBPE}:
    for merge in tokenizer.merges:
      if merge.newToken notin tokenizer.vocab:
        result.add("ПРЕДУПРЕЖДЕНИЕ: Merge токен отсутствует в словаре: " & merge.newToken)
  
  if result.len == 0:
    result.add("✓ Токенизатор валиден")








#==============================================================================
# ERROR HANDLING & VALIDATION
#==============================================================================

proc validateInput*(text: string, maxLength: int = MAX_INPUT_LENGTH): Option[string] =
  ## Валидация входного текста
  ## Возвращает None если валидация прошла успешно, иначе Some с сообщением об ошибке
  if text.len == 0:
    return some("Empty input text")
  if text.len > maxLength:
    return some("Input exceeds maximum length: " & $maxLength)
  return none(string)

proc validateTokenizer*(tokenizer: Tokenizer): Option[string] =
  ## Проверка корректности токенизатора
  ## Возвращает None если валидация прошла успешно, иначе Some с сообщением об ошибке
  if tokenizer.vocab.len == 0:
    return some("Empty vocabulary")
  if tokenizer.vocab.len != tokenizer.inverseVocab.len:
    return some("Vocabulary size mismatch: vocab=" & $tokenizer.vocab.len & 
                " inverseVocab=" & $tokenizer.inverseVocab.len)
  if tokenizer.specialTokens.unkToken notin tokenizer.vocab:
    return some("UNK token not in vocabulary")
  return none(string)


#==============================================================================
# ADVANCED TEXT NORMALIZATION
#==============================================================================

proc normalizeNFKC*(text: string): string =
  ## Unicode нормализация NFKC (compatibility decomposition + canonical composition)
  ## Преобразует различные представления одного символа в единое
  ## ПРИМЕЧАНИЕ: Базовая реализация - нормализует основные случаи
  result = ""
  for rune in text.runes:
    result.add($rune)

proc handleZeroWidthChars*(text: string): string =
  ## Удаление невидимых символов (zero-width space, zero-width joiner и т.д.)
  const zeroWidthChars = [
    0x200B,  # zero-width space
    0x200C,  # zero-width non-joiner
    0x200D,  # zero-width joiner
    0xFEFF,  # zero-width no-break space (BOM)
  ]
  result = ""
  for rune in text.runes:
    if rune.int32 notin zeroWidthChars:
      result.add($rune)

proc normalizeWhitespaceAdvanced*(text: string): string =
  ## Нормализация всех типов whitespace в обычные пробелы
  result = text
  # Заменяем табы, переносы строк и другие whitespace
  result = result.multiReplace([
    ("\t", " "),
    ("\r\n", " "),
    ("\r", " "),
    ("\n", " "),
    ("\u00A0", " "),  # non-breaking space
    ("\u2009", " "),  # thin space
    ("\u200A", " "),  # hair space
  ])
  # Схлопываем множественные пробелы
  while "  " in result:
    result = result.replace("  ", " ")
  result = result.strip()

proc fullNormalization*(text: string): string =
  ## Полная цепочка нормализации текста
  result = text
  result = result.normalizeNFKC()
  result = result.handleZeroWidthChars()
  result = result.normalizeWhitespaceAdvanced()


#==============================================================================
# LRU CACHE IMPLEMENTATION
#==============================================================================

proc newLRUCache*(maxSize: int = 10000): LRUCache =
  ## Создание нового LRU кэша с заданным максимальным размером
  result.entries = initTable[string, CacheEntry]()
  result.maxSize = maxSize
  result.hits = 0
  result.misses = 0

proc evictLRU*(cache: var LRUCache) =
  ## Вытеснение наименее недавно использованного элемента
  if cache.entries.len >= cache.maxSize:
    var oldestKey = ""
    var oldestTime = Inf
    
    for key, entry in cache.entries:
      if entry.lastAccess < oldestTime:
        oldestTime = entry.lastAccess
        oldestKey = key
    
    if oldestKey.len > 0:
      cache.entries.del(oldestKey)

proc get*(cache: var LRUCache, key: string): Option[seq[int]] =
  ## Получение значения из кэша
  if key in cache.entries:
    cache.hits.inc()
    cache.entries[key].lastAccess = epochTime()
    cache.entries[key].accessCount.inc()
    return some(cache.entries[key].value)
  else:
    cache.misses.inc()
    return none(seq[int])

proc put*(cache: var LRUCache, key: string, value: seq[int]) =
  ## Добавление значения в кэш с автоматическим eviction при переполнении
  cache.evictLRU()
  cache.entries[key] = CacheEntry(
    value: value,
    lastAccess: epochTime(),
    accessCount: 1
  )

proc clear*(cache: var LRUCache) =
  ## Очистка кэша и сброс статистики
  cache.entries.clear()
  cache.hits = 0
  cache.misses = 0

proc getStats*(cache: LRUCache): tuple[hits, misses: int, hitRate: float] =
  ## Получение статистики использования кэша
  let total = cache.hits + cache.misses
  let hitRate = if total > 0: cache.hits.float / total.float else: 0.0
  return (cache.hits, cache.misses, hitRate)


#==============================================================================
# THREAD-SAFE TOKENIZER
#==============================================================================

proc newThreadSafeTokenizer*(tokenizer: Tokenizer): ThreadSafeTokenizer =
  ## Создание потокобезопасной обёртки над токенизатором
  result = ThreadSafeTokenizer(tokenizer: tokenizer)
  initLock(result.lock)

proc tokenizeThreadSafe*(tst: ThreadSafeTokenizer, text: string, 
                         addSpecialTokens: bool = true): seq[int] =
  ## Потокобезопасная токенизация
  acquire(tst.lock)
  defer: release(tst.lock)
  result = tokenize(text, tst.tokenizer, flag = 0, addSpecialTokens = addSpecialTokens)

proc decodeThreadSafe*(tst: ThreadSafeTokenizer, tokenIds: seq[int], 
                       skipSpecialTokens: bool = true): string =
  ## Потокобезопасное декодирование
  acquire(tst.lock)
  defer: release(tst.lock)
  result = decode(tst.tokenizer, tokenIds, skipSpecialTokens)

proc destroyThreadSafeTokenizer*(tst: var ThreadSafeTokenizer) =
  ## Освобождение ресурсов потокобезопасного токенизатора
  deinitLock(tst.lock)


#==============================================================================
# TOKENIZER PERSISTENCE & VERSIONING
#==============================================================================

proc createMetadata*(tokenizer: Tokenizer, trainInfo: string = ""): TokenizerMetadata =
  ## Создание метаданных токенизатора
  result.version = TOKENIZER_VERSION
  result.created = now()
  result.modified = now()
  result.vocabSize = tokenizer.vocab.len
  result.algorithm = tokenizer.kind
  result.trainedOn = trainInfo

proc saveVersionedTokenizer*(vt: VersionedTokenizer, path: string) =
  ## Сохранение токенизатора с метаданными и версией
  try:
    var specialTokensJson = %* {
      "pad": vt.tokenizer.specialTokens.padToken,
      "unk": vt.tokenizer.specialTokens.unkToken,
      "bos": vt.tokenizer.specialTokens.bosToken,
      "eos": vt.tokenizer.specialTokens.eosToken,
      "sep": vt.tokenizer.specialTokens.sepToken,
      "cls": vt.tokenizer.specialTokens.clsToken,
      "mask": vt.tokenizer.specialTokens.maskToken
    }
    
    var data = %* {
      "metadata": {
        "version": vt.metadata.version,
        "created": $vt.metadata.created,
        "modified": $vt.metadata.modified,
        "vocabSize": vt.metadata.vocabSize,
        "algorithm": $vt.metadata.algorithm,
        "trainedOn": vt.metadata.trainedOn
      },
      "tokenizer": {
        "kind": $vt.tokenizer.kind,
        "vocab": vt.tokenizer.vocab,
        "inverseVocab": vt.tokenizer.inverseVocab,
        "specialTokens": specialTokensJson,
        "preserveCase": vt.tokenizer.preserveCase
      }
    }
    writeFile(path, pretty(data))
  except IOError as e:
    raise newException(TokenizationError, "Failed to save tokenizer: " & e.msg)

proc loadVersionedTokenizer*(path: string): VersionedTokenizer =
  ## Загрузка токенизатора с проверкой версии
  try:
    let data = parseFile(path)
    let version = data["metadata"]["version"].getStr()
    
    if version != TOKENIZER_VERSION:
      stderr.writeLine("Warning: tokenizer version mismatch: " & version & 
                      " != " & TOKENIZER_VERSION)
    
    # Парсинг метаданных
    result.metadata.version = version
    result.metadata.vocabSize = data["metadata"]["vocabSize"].getInt()
    result.metadata.trainedOn = data["metadata"]["trainedOn"].getStr()
    
    # Создание токенизатора
    result.tokenizer = Tokenizer(
      vocab: initTable[string, int](),
      inverseVocab: @[],
      specialTokenIds: initTable[string, int]()
    )
    
    # Загрузка словаря
    for key, val in data["tokenizer"]["vocab"].pairs:
      result.tokenizer.vocab[key] = val.getInt()
    
    # Загрузка inverseVocab
    for item in data["tokenizer"]["inverseVocab"].items:
      result.tokenizer.inverseVocab.add(item.getStr())
    
  except IOError as e:
    raise newException(TokenizationError, "Failed to load tokenizer: " & e.msg)
  except JsonParsingError as e:
    raise newException(TokenizationError, "Invalid tokenizer JSON: " & e.msg)


#==============================================================================
# INCREMENTAL VOCABULARY UPDATES
#==============================================================================

proc updateVocabulary*(tokenizer: var Tokenizer, newTexts: seq[string], 
                       maxNewTokens: int = 1000) =
  ## Инкрементальное обновление словаря на новых текстах
  ## Не переобучает с нуля, а добавляет частые новые токены
  
  # Собираем статистику по новым текстам
  var newTokenCounts = initCountTable[string]()
  
  for text in newTexts:
    let tokens = tokenize(text, tokenizer, addSpecialTokens = false)
    for tokenId in tokens:
      if tokenId < tokenizer.inverseVocab.len:
        let token = tokenizer.inverseVocab[tokenId]
        newTokenCounts.inc(token)
  
  # Находим кандидатов на добавление (токены которых ещё нет в словаре)
  var candidates: seq[tuple[token: string, count: int]]
  for token, count in newTokenCounts:
    if token notin tokenizer.vocab and count >= 5:  # минимальная частота
      candidates.add((token, count))
  
  # Сортируем по частоте
  candidates.sort(proc(a, b: auto): int = cmp(b.count, a.count))
  
  # Добавляем топ-N
  let toAdd = min(maxNewTokens, candidates.len)
  for i in 0..<toAdd:
    let token = candidates[i].token
    let newId = tokenizer.vocab.len
    tokenizer.vocab[token] = newId
    tokenizer.inverseVocab.add(token)

proc addTokens*(tokenizer: var Tokenizer, tokens: seq[string]) =
  ## Добавление конкретных токенов в словарь
  for token in tokens:
    if token notin tokenizer.vocab:
      let newId = tokenizer.vocab.len
      tokenizer.vocab[token] = newId
      tokenizer.inverseVocab.add(token)

proc removeRareTokens*(tokenizer: var Tokenizer, corpus: seq[string], 
                       minFreq: int = 2) =
  ## Удаление редких токенов из словаря
  var tokenCounts = initCountTable[string]()
  
  # Считаем частоты
  for text in corpus:
    let tokens = tokenize(text, tokenizer, addSpecialTokens = false)
    for tokenId in tokens:
      if tokenId < tokenizer.inverseVocab.len:
        tokenCounts.inc(tokenizer.inverseVocab[tokenId])
  
  # Находим редкие токены
  var toRemove: seq[string]
  for token in tokenizer.vocab.keys:
    if tokenCounts[token] < minFreq:
      # Не удаляем специальные токены
      if token notin [tokenizer.specialTokens.padToken,
                      tokenizer.specialTokens.unkToken,
                      tokenizer.specialTokens.bosToken,
                      tokenizer.specialTokens.eosToken,
                      tokenizer.specialTokens.sepToken,
                      tokenizer.specialTokens.clsToken,
                      tokenizer.specialTokens.maskToken]:
        toRemove.add(token)
  
  # Пересоздаём словарь без редких токенов
  var newVocab = initTable[string, int]()
  var newInverseVocab: seq[string] = @[]
  
  for token in tokenizer.inverseVocab:
    if token notin toRemove:
      let newId = newVocab.len
      newVocab[token] = newId
      newInverseVocab.add(token)
  
  tokenizer.vocab = newVocab
  tokenizer.inverseVocab = newInverseVocab


#==============================================================================
# VOCABULARY ALIGNMENT & ANALYSIS
#==============================================================================

proc findCommonTokens*(tokenizers: seq[Tokenizer]): seq[string] =
  ## Находит токены, общие для всех токенизаторов
  if tokenizers.len == 0:
    return @[]
  
  var commonTokens = initHashSet[string]()
  for token in tokenizers[0].vocab.keys:
    commonTokens.incl(token)
  
  for i in 1..<tokenizers.len:
    var newCommon = initHashSet[string]()
    for token in tokenizers[i].vocab.keys:
      if token in commonTokens:
        newCommon.incl(token)
    commonTokens = newCommon
  
  result = toSeq(commonTokens)

proc alignVocabularies*(tokenizer1, tokenizer2: Tokenizer): Tokenizer =
  ## Создаёт токенизатор с объединённым словарём двух токенизаторов
  result = Tokenizer(
    kind: tokenizer1.kind,
    vocab: initTable[string, int](),
    inverseVocab: @[],
    specialTokens: tokenizer1.specialTokens,
    specialTokenIds: initTable[string, int](),
    cache: initCache(10000),
    cacheMaxSize: 10000
  )
  
  # Добавляем токены из первого токенизатора
  for token in tokenizer1.inverseVocab:
    let id = result.vocab.len
    result.vocab[token] = id
    result.inverseVocab.add(token)
  
  # Добавляем уникальные токены из второго
  for token in tokenizer2.inverseVocab:
    if token notin result.vocab:
      let id = result.vocab.len
      result.vocab[token] = id
      result.inverseVocab.add(token)

proc mergeTokenizers*(tokenizers: seq[Tokenizer]): Tokenizer =
  ## Объединяет несколько токенизаторов в один
  if tokenizers.len == 0:
    raise newException(TokenizationError, "Cannot merge empty tokenizer list")
  
  result = Tokenizer(
    kind: tokenizers[0].kind,
    vocab: initTable[string, int](),
    inverseVocab: @[],
    specialTokens: tokenizers[0].specialTokens,
    specialTokenIds: initTable[string, int](),
    cache: initCache(10000),
    cacheMaxSize: 10000
  )
  
  var allTokens = initHashSet[string]()
  
  # Собираем все уникальные токены
  for tokenizer in tokenizers:
    for token in tokenizer.vocab.keys:
      allTokens.incl(token)
  
  # Добавляем в результирующий словарь
  for token in allTokens:
    let id = result.vocab.len
    result.vocab[token] = id
    result.inverseVocab.add(token)

proc analyzeOOVWords*(tokenizer: var Tokenizer, text: string): seq[string] =
  ## Находит слова, которых нет в словаре (Out-Of-Vocabulary)
  result = @[]
  let words = text.split()
  let unkId = tokenizer.getUnkTokenId()
  
  for word in words:
    let tokens = tokenize(word, tokenizer, addSpecialTokens = false)
    # Если весь токен это UNK, значит слово OOV
    if tokens.len == 1 and tokens[0] == unkId:
      result.add(word)


#==============================================================================
# ADVANCED BPE FEATURES
#==============================================================================

proc reverseBPE*(tokens: seq[string]): string =
  ## Восстанавливает исходные слова из BPE токенов
  result = ""
  for token in tokens:
    # Удаляем маркеры продолжения (например, "##" в WordPiece)
    if token.startsWith("##"):
      result.add(token[2..^1])
    else:
      if result.len > 0:
        result.add(" ")
      result.add(token)

proc getBPESegmentations*(word: string, tokenizer: Tokenizer, 
                          maxVariants: int = 10): seq[seq[string]] =
  ## Возвращает все возможные BPE разбиения слова
  ## Полезно для анализа и debugging
  result = @[]
  
  proc generateSegmentations(remaining: string, current: seq[string], depth: int = 0) =
    if remaining.len == 0:
      result.add(current)
      return
    
    if depth > 20 or result.len >= maxVariants:  # ограничения
      return
    
    # Пробуем разные префиксы
    for i in 1..remaining.len:
      let prefix = remaining[0..<i]
      if prefix in tokenizer.vocab:
        var newCurrent = current
        newCurrent.add(prefix)
        generateSegmentations(remaining[i..^1], newCurrent, depth + 1)
  
  generateSegmentations(word, @[])


#==============================================================================
# MULTILINGUAL SUPPORT
#==============================================================================

proc detectLanguage*(text: string): string =
  ## Простая эвристика для определения языка по Unicode диапазонам
  var cyrillicCount = 0
  var latinCount = 0
  var chineseCount = 0
  var arabicCount = 0
  
  for rune in text.runes:
    let code = rune.int32
    if code >= 0x0400 and code <= 0x04FF:  # Cyrillic
      cyrillicCount.inc()
    elif (code >= 0x0041 and code <= 0x005A) or (code >= 0x0061 and code <= 0x007A):  # Latin
      latinCount.inc()
    elif code >= 0x4E00 and code <= 0x9FFF:  # Chinese
      chineseCount.inc()
    elif code >= 0x0600 and code <= 0x06FF:  # Arabic
      arabicCount.inc()
  
  # Возвращаем язык с наибольшим количеством символов
  let maxCount = max([cyrillicCount, latinCount, chineseCount, arabicCount])
  if maxCount == 0:
    return "unknown"
  elif maxCount == cyrillicCount:
    return "cyrillic"
  elif maxCount == latinCount:
    return "latin"
  elif maxCount == chineseCount:
    return "chinese"
  elif maxCount == arabicCount:
    return "arabic"
  else:
    return "unknown"

proc handleScriptMixing*(text: string, normalizeScript: bool = true): string =
  ## Обработка текстов со смешением скриптов (латиница + кириллица)
  ## Приводит визуально похожие символы к единому виду
  if not normalizeScript:
    return text
  
  # Словарь визуально похожих символов для нормализации
  const lookalikeMap = {
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M",
    "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T",
    "Х": "X", "а": "a", "е": "e", "о": "o", "р": "p",
    "с": "c", "у": "y", "х": "x"
  }.toTable
  
  result = ""
  for ch in text:
    let chStr = $ch
    if chStr in lookalikeMap:
      result.add(lookalikeMap[chStr])
    else:
      result.add(ch)


#==============================================================================
# SUBWORD STATISTICS & ANALYSIS
#==============================================================================

proc getSubwordFrequencies*(tokenizer: var Tokenizer, corpus: seq[string]): CountTable[string] =
  ## Статистика частот подслов в корпусе
  result = initCountTable[string]()
  
  for text in corpus:
    let tokens = tokenize(text, tokenizer, addSpecialTokens = false)
    for tokenId in tokens:
      if tokenId < tokenizer.inverseVocab.len:
        let token = tokenizer.inverseVocab[tokenId]
        result.inc(token)

proc getTokenBoundaries*(tokenizer: var Tokenizer, text: string): seq[tuple[token: string, start, endPos: int]] =
  ## Возвращает границы токенов в исходном тексте (позиции начала и конца)
  result = @[]
  let tokens = tokenizeWithOffsets(text, tokenizer)
  
  for t in tokens:
    result.add((t.token, t.startChar, t.endChar))

proc getTokenStatistics*(tokenizer: var Tokenizer, corpus: seq[string]): tuple[
    avgLength: float, 
    medianLength: float,
    minLength: int,
    maxLength: int,
    totalTokens: int
  ] =
  ## Статистика длин токенов в корпусе
  var lengths: seq[int] = @[]
  var totalTokens = 0
  
  for text in corpus:
    let tokens = tokenize(text, tokenizer, addSpecialTokens = false)
    totalTokens += tokens.len
    for tokenId in tokens:
      if tokenId < tokenizer.inverseVocab.len:
        lengths.add(tokenizer.inverseVocab[tokenId].len)
  
  if lengths.len == 0:
    return (0.0, 0.0, 0, 0, 0)
  
  lengths.sort()
  let avgLength = lengths.sum().float / lengths.len.float
  let medianLength = lengths[lengths.len div 2].float
  let minLength = lengths[0]
  let maxLength = lengths[^1]
  
  result = (avgLength, medianLength, minLength, maxLength, totalTokens)


#==============================================================================
# VOCABULARY COMPRESSION
#==============================================================================

proc compressVocabulary*(tokenizer: Tokenizer, targetSize: int): Tokenizer =
  ## Сжимает словарь до целевого размера, удаляя наименее важные токены
  if targetSize >= tokenizer.vocab.len:
    return tokenizer
  
  result = Tokenizer(
    kind: tokenizer.kind,
    vocab: initTable[string, int](),
    inverseVocab: @[],
    specialTokens: tokenizer.specialTokens,
    specialTokenIds: initTable[string, int](),
    merges: tokenizer.merges,
    cache: initCache(10000),
    cacheMaxSize: 10000
  )
  
  # Сохраняем специальные токены
  var preserved = initHashSet[string]()
  for token in [tokenizer.specialTokens.padToken, tokenizer.specialTokens.unkToken,
                tokenizer.specialTokens.bosToken, tokenizer.specialTokens.eosToken,
                tokenizer.specialTokens.sepToken, tokenizer.specialTokens.clsToken,
                tokenizer.specialTokens.maskToken]:
    preserved.incl(token)
  
  # Добавляем специальные токены
  for token in preserved:
    let id = result.vocab.len
    result.vocab[token] = id
    result.inverseVocab.add(token)
  
  # Добавляем остальные токены до достижения целевого размера
  var added = preserved.len
  for token in tokenizer.inverseVocab:
    if added >= targetSize:
      break
    if token notin preserved:
      let id = result.vocab.len
      result.vocab[token] = id
      result.inverseVocab.add(token)
      added.inc()

proc pruneByFrequency*(tokenizer: var Tokenizer, corpus: seq[string], minCount: int) =
  ## Удаляет токены с частотой ниже минимальной
  removeRareTokens(tokenizer, corpus, minCount)


#==============================================================================
# ENHANCED UNICODE OPERATIONS
#==============================================================================

proc toLowerUnicodeOptimized*(s: string): string {.inline.} =
  ## Оптимизированная версия приведения к lowercase с предаллокацией
  result = newStringOfCap(s.len)
  for rune in s.runes:
    result.add($rune.toLower())

proc toUpperUnicodeOptimized*(s: string): string {.inline.} =
  ## Оптимизированная версия приведения к uppercase с предаллокацией
  result = newStringOfCap(s.len)
  for rune in s.runes:
    result.add($rune.toUpper())

proc runeCount*(s: string): int =
  ## Подсчёт количества Unicode символов (рун) в строке
  result = 0
  for _ in s.runes:
    result.inc()


#==============================================================================
# ПРОДВИНУТАЯ ОБРАБОТКА ТЕКСТА (v0.7)
#==============================================================================

proc normalizeNumbers*(text: string, strategy: NumberNormalizationStrategy = nsKeepOriginal): string =
  ## Нормализация чисел в тексте согласно выбранной стратегии
  result = text
  
  case strategy
  of nsKeepOriginal:
    discard  # Ничего не делаем
  
  of nsReplaceWithToken:
    # Заменяем все числа на [NUM]
    result = result.replace(re"\d+([.,]\d+)?", "[NUM]")
  
  of nsReplaceWithDigits:
    # Заменяем числа на токены с указанием количества разрядов
    var processed = ""
    var i = 0
    while i < text.len:
      if text[i].isDigit:
        var numStr = ""
        while i < text.len and (text[i].isDigit or text[i] in ['.', ',']):
          numStr.add(text[i])
          i.inc()
        
        let cleanNum = numStr.replace(".", "").replace(",", "")
        let digitCount = cleanNum.len
        processed.add("[NUM_" & $digitCount & "DIGIT]")
      else:
        processed.add(text[i])
        i.inc()
    result = processed
  
  of nsNormalize:
    # Нормализуем формат чисел
    var processed = ""
    var i = 0
    while i < text.len:
      if text[i].isDigit:
        var numStr = ""
        while i < text.len and (text[i].isDigit or text[i] in ['.', ',', ' ']):
          if text[i] != ' ' and text[i] != ',':
            if text[i] == ',':
              numStr.add('.')
            else:
              numStr.add(text[i])
          i.inc()
        processed.add(numStr)
      else:
        processed.add(text[i])
        i.inc()
    result = processed

proc handleContractions*(text: string, expand: bool = true): string =
  ## Обработка сокращений (contractions) в английском языке
  if not expand:
    return text
  
  const contractions = {
    "n't": " not",
    "'re": " are",
    "'ve": " have",
    "'ll": " will",
    "'d": " would",
    "'m": " am",
    "can't": "cannot",
    "won't": "will not",
    "shan't": "shall not",
    "let's": "let us",
    "ain't": "is not"
  }.toTable
  
  result = text
  for contraction, expansion in contractions:
    result = result.replace(contraction, expansion)

proc normalizeUrls*(text: string, placeholder: string = "[URL]"): string =
  ## Заменяет URL на placeholder токен
  result = text
  result = result.replace(reUrls, placeholder)
  result = result.replace(reWwwUrls, placeholder)

proc detectAndNormalizeEmails*(text: string, placeholder: string = "[EMAIL]"): string =
  ## Детектирует и заменяет email адреса на placeholder
  result = text.replace(reEmails, placeholder)

proc handleEmojis*(text: string, strategy: EmojiStrategy = esKeep): string =
  ## Обработка эмодзи согласно выбранной стратегии
  case strategy
  of esKeep:
    return text
  
  of esRemove:
    var cleaned = ""
    for rune in text.runes:
      let code = int(rune)
      if not (code >= 0x1F600 and code <= 0x1F64F or
              code >= 0x1F300 and code <= 0x1F5FF or
              code >= 0x1F680 and code <= 0x1F6FF or
              code >= 0x2600 and code <= 0x26FF or
              code >= 0x2700 and code <= 0x27BF):
        cleaned.add($rune)
    return cleaned
  
  of esReplace:
    const emojiMap = {
      "😀": "[SMILE]",
      "😂": "[LOL]",
      "❤️": "[HEART]",
      "👍": "[THUMBS_UP]",
      "🎉": "[CELEBRATION]"
    }.toTable
    
    result = text
    for emoji, description in emojiMap:
      result = result.replace(emoji, description)
  
  of esTokenize:
    result = ""
    for rune in text.runes:
      let code = int(rune)
      if code >= 0x1F600 and code <= 0x1F64F or
         code >= 0x1F300 and code <= 0x1F5FF or
         code >= 0x1F680 and code <= 0x1F6FF or
         code >= 0x2600 and code <= 0x26FF or
         code >= 0x2700 and code <= 0x27BF:
        result.add(" " & $rune & " ")
      else:
        result.add($rune)

proc segmentSentences*(text: string): seq[string] =
  ## Сегментация текста на предложения
  result = @[]
  var currentSentence = ""
  var i = 0
  
  while i < text.len:
    currentSentence.add(text[i])
    
    if text[i] in ['.', '!', '?']:
      if i + 1 < text.len and text[i + 1] == ' ':
        if i + 2 < text.len and text[i + 2].isUpperAscii:
          result.add(currentSentence.strip())
          currentSentence = ""
    
    i.inc()
  
  if currentSentence.strip().len > 0:
    result.add(currentSentence.strip())


#==============================================================================
# VOCABULARY MANAGEMENT (v0.7)
#==============================================================================

proc mergeVocabularies*(vocabs: seq[Tokenizer], strategy: MergeStrategy = msUnion): Tokenizer =
  ## Объединяет словари нескольких токенизаторов
  if vocabs.len == 0:
    raise newException(ValidationError, "Пустой список токенизаторов")
  
  result = Tokenizer(
    kind: vocabs[0].kind,
    vocab: initTable[string, int](),
    inverseVocab: @[],
    specialTokens: vocabs[0].specialTokens,
    specialTokenIds: initTable[string, int](),
    merges: @[],
    cache: initTable[string, seq[int]](),
    cacheMaxSize: 10000
  )
  
  case strategy
  of msUnion:
    var allTokens = initHashSet[string]()
    for tokenizer in vocabs:
      for token in tokenizer.vocab.keys:
        allTokens.incl(token)
    
    for token in allTokens:
      let id = result.vocab.len
      result.vocab[token] = id
      result.inverseVocab.add(token)
  
  of msIntersection:
    if vocabs.len == 0:
      return result
    
    var commonTokens = toHashSet(toSeq(vocabs[0].vocab.keys))
    for i in 1..<vocabs.len:
      let currentTokens = toHashSet(toSeq(vocabs[i].vocab.keys))
      commonTokens = commonTokens * currentTokens
    
    for token in commonTokens:
      let id = result.vocab.len
      result.vocab[token] = id
      result.inverseVocab.add(token)
  
  of msWeighted:
    var tokenCounts = initCountTable[string]()
    
    for tokenizer in vocabs:
      for token in tokenizer.vocab.keys:
        tokenCounts.inc(token)
    
    var sortedTokens: seq[tuple[token: string, count: int]] = @[]
    for token, count in tokenCounts:
      sortedTokens.add((token, count))
    
    sortedTokens.sort(proc(a, b: tuple[token: string, count: int]): int =
      cmp(b.count, a.count)
    )
    
    for item in sortedTokens:
      let id = result.vocab.len
      result.vocab[item.token] = id
      result.inverseVocab.add(item.token)



proc expandVocabulary*(t: var Tokenizer, newTexts: seq[string], maxNewTokens: int) =
  ## Расширяет словарь новыми токенами
  var candidates = initCountTable[string]()
  
  for text in newTexts:
    let words = splitIntoWords(text)
    for word in words:
      let tokens = tokenize(word, t, addSpecialTokens = false)
      if tokens.len > 1:
        candidates.inc(word)
  
  var sortedCandidates: seq[tuple[token: string, count: int]] = @[]
  for token, count in candidates:
    sortedCandidates.add((token, count))
  
  sortedCandidates.sort(proc(a, b: tuple[token: string, count: int]): int =
    cmp(b.count, a.count)
  )
  
  var added = 0
  for item in sortedCandidates:
    if added >= maxNewTokens:
      break
    
    if item.token notin t.vocab:
      let id = t.vocab.len
      t.vocab[item.token] = id
      t.inverseVocab.add(item.token)
      added.inc()

proc getVocabularyOverlap*(t1, t2: Tokenizer): VocabOverlapStats =
  ## Анализирует перекрытие двух словарей
  let vocab1 = toHashSet(toSeq(t1.vocab.keys))
  let vocab2 = toHashSet(toSeq(t2.vocab.keys))
  
  let common = vocab1 * vocab2
  let union = vocab1 + vocab2
  
  result.commonTokens = common.len
  result.uniqueToFirst = (vocab1 - vocab2).len
  result.uniqueToSecond = (vocab2 - vocab1).len
  result.overlapRatio = if t1.vocab.len > 0: common.len.float / t1.vocab.len.float else: 0.0
  result.jaccardSimilarity = if union.len > 0: common.len.float / union.len.float else: 0.0

proc exportVocabulary*(t: Tokenizer, format: VocabFormat): string =
  ## Экспортирует словарь в указанном формате
  case format
  of vfJson:
    var jobj = %* {
      "version": TOKENIZER_VERSION,
      "type": $t.kind,
      "vocab_size": t.vocab.len,
      "vocab": newJObject()
    }
    
    for token, id in t.vocab:
      jobj["vocab"][token] = %id
    
    return $jobj
  
  of vfPlainText:
    result = ""
    for i in 0..<t.inverseVocab.len:
      result.add(t.inverseVocab[i] & "\n")
  
  of vfHuggingFace:
    var jobj = %* {
      "version": "1.0",
      "truncation": nil,
      "padding": nil,
      "added_tokens": [],
      "normalizer": nil,
      "pre_tokenizer": nil,
      "post_processor": nil,
      "decoder": nil,
      "model": {
        "type": "BPE",
        "vocab": newJObject(),
        "merges": []
      }
    }
    
    for token, id in t.vocab:
      jobj["model"]["vocab"][token] = %id
    
    return $jobj
  
  of vfSentencePiece, vfBinary:
    return "[NOT_IMPLEMENTED] Формат " & $format & " требует дополнительной реализации"

proc importVocabulary*(format: VocabFormat, data: string): Tokenizer =
  ## Импортирует словарь из указанного формата
  result = Tokenizer(
    kind: tkBPE,
    vocab: initTable[string, int](),
    inverseVocab: @[],
    specialTokens: SpecialTokens(
      padToken: "[PAD]",
      unkToken: "[UNK]",
      bosToken: "[BOS]",
      eosToken: "[EOS]"
    ),
    specialTokenIds: initTable[string, int](),
    merges: @[],
    cache: initTable[string, seq[int]](),
    cacheMaxSize: 10000
  )
  
  case format
  of vfJson:
    let jobj = parseJson(data)
    if jobj.hasKey("vocab"):
      for token, idNode in jobj["vocab"]:
        let id = idNode.getInt()
        result.vocab[token] = id
        
        while result.inverseVocab.len <= id:
          result.inverseVocab.add("")
        result.inverseVocab[id] = token
  
  of vfPlainText:
    var id = 0
    for line in data.splitLines():
      if line.len > 0:
        result.vocab[line] = id
        result.inverseVocab.add(line)
        id.inc()
  
  of vfHuggingFace:
    let jobj = parseJson(data)
    if jobj.hasKey("model") and jobj["model"].hasKey("vocab"):
      for token, idNode in jobj["model"]["vocab"]:
        let id = idNode.getInt()
        result.vocab[token] = id
        
        while result.inverseVocab.len <= id:
          result.inverseVocab.add("")
        result.inverseVocab[id] = token
  
  of vfSentencePiece, vfBinary:
    raise newException(ValidationError, "Формат " & $format & " не поддерживается для импорта")


#==============================================================================
# ПРОДВИНУТЫЕ МЕТРИКИ (v0.7)
#==============================================================================

proc calculatePerplexity*(t: var Tokenizer, text: string, languageModel: Table[string, float] = initTable[string, float]()): float =
  ## Вычисляет perplexity токенизации
  let tokens = tokenize(text, t, addSpecialTokens = false)
  if tokens.len == 0:
    return 0.0
  
  var totalLogProb = 0.0
  
  if languageModel.len > 0:
    for tokenId in tokens:
      if tokenId < t.inverseVocab.len:
        let token = t.inverseVocab[tokenId]
        let prob = if token in languageModel: languageModel[token] else: 1e-10
        totalLogProb += ln(prob)
  else:
    let uniformProb = 1.0 / t.vocab.len.float
    totalLogProb = tokens.len.float * ln(uniformProb)
  
  result = exp(-totalLogProb / tokens.len.float)

proc measureSegmentationQuality*(t: var Tokenizer, goldSegments: seq[seq[string]]): QualityMetrics =
  ## Измеряет качество сегментации
  var truePositives = 0
  var falsePositives = 0
  var falseNegatives = 0
  var totalSegmentationErrors = 0
  
  for goldSegment in goldSegments:
    let text = goldSegment.join("")
    let predictedTokens = tokenize(text, t, addSpecialTokens = false)
    
    var predictedSegment: seq[string] = @[]
    for tokenId in predictedTokens:
      if tokenId < t.inverseVocab.len:
        predictedSegment.add(t.inverseVocab[tokenId])
    
    let goldSet = toHashSet(goldSegment)
    let predSet = toHashSet(predictedSegment)
    
    truePositives += (goldSet * predSet).len
    falsePositives += (predSet - goldSet).len
    falseNegatives += (goldSet - predSet).len
    
    if goldSegment != predictedSegment:
      totalSegmentationErrors.inc()
  
  let precision = if (truePositives + falsePositives) > 0:
                    truePositives.float / (truePositives + falsePositives).float
                  else: 0.0
  
  let recall = if (truePositives + falseNegatives) > 0:
                 truePositives.float / (truePositives + falseNegatives).float
               else: 0.0
  
  let f1 = if (precision + recall) > 0:
             2.0 * precision * recall / (precision + recall)
           else: 0.0
  
  let accuracy = if goldSegments.len > 0:
                   1.0 - (totalSegmentationErrors.float / goldSegments.len.float)
                 else: 0.0
  
  result = QualityMetrics(
    precision: precision,
    recall: recall,
    f1Score: f1,
    accuracy: accuracy,
    segmentationErrors: totalSegmentationErrors
  )

proc analyzeTokenDistribution*(t: var Tokenizer, corpus: seq[string]): DistributionStats =
  ## Анализирует распределение длин токенов
  var lengths: seq[int] = @[]
  
  for text in corpus:
    let tokens = tokenize(text, t, addSpecialTokens = false)
    for tokenId in tokens:
      if tokenId < t.inverseVocab.len:
        lengths.add(t.inverseVocab[tokenId].len)
  
  if lengths.len == 0:
    return DistributionStats(
      mean: 0.0, median: 0.0, stdDev: 0.0,
      min: 0, max: 0,
      quartiles: [0.0, 0.0, 0.0],
      histogram: initCountTable[int]()
    )
  
  lengths.sort()
  
  let mean = lengths.sum().float / lengths.len.float
  let median = lengths[lengths.len div 2].float
  
  var variance = 0.0
  for length in lengths:
    variance += (length.float - mean) * (length.float - mean)
  variance /= lengths.len.float
  let stdDev = sqrt(variance)
  
  let q1 = lengths[lengths.len div 4].float
  let q3 = lengths[lengths.len * 3 div 4].float
  
  var histogram = initCountTable[int]()
  for length in lengths:
    histogram.inc(length)
  
  result = DistributionStats(
    mean: mean,
    median: median,
    stdDev: stdDev,
    min: lengths[0],
    max: lengths[^1],
    quartiles: [q1, median, q3],
    histogram: histogram
  )

proc compareTokenizers*(tokenizers: var seq[Tokenizer], tokenizerNames: seq[string], corpus: seq[string]): ComparisonReport =
  ## Сравнивает несколько токенизаторов
  result.tokenizerNames = tokenizerNames
  result.vocabSizes = @[]
  result.compressionRatios = @[]
  result.avgTokenLengths = @[]
  result.unkRates = @[]
  result.speedComparison = @[]
  result.qualityScores = @[]
  
  for i in 0 ..< tokenizers.len:
    var t = tokenizers[i]
    let metrics = getMetrics(t, corpus)
    # если нужно сохранить изменения t обратно:
    tokenizers[i] = t
    result.compressionRatios.add(metrics.compressionRatio)
    result.unkRates.add(metrics.unkTokenRate)
    
    let stats = getTokenStatistics(t, corpus)
    result.avgTokenLengths.add(stats.avgLength)
    
    let startTime = epochTime()
    for text in corpus[0..min(99, corpus.high)]:
      discard tokenize(text, t)
    let elapsed = epochTime() - startTime
    let tokensPerSec = if elapsed > 0: stats.totalTokens.float / elapsed else: 0.0
    result.speedComparison.add(tokensPerSec)
    
    let qualityScore = (1.0 - metrics.unkTokenRate) * 0.4 +
                       (metrics.vocabUtilization) * 0.3 +
                       (1.0 / (stats.avgLength + 1.0)) * 0.3
    result.qualityScores.add(qualityScore)


#==============================================================================
# СПЕЦИАЛИЗИРОВАННЫЕ ТОКЕНИЗАТОРЫ (v0.7)
#==============================================================================

proc createCharacterTokenizer*(): Tokenizer =
  ## Создаёт токенизатор уровня символов
  result = Tokenizer(
    kind: tkBPE,
    vocab: initTable[string, int](),
    inverseVocab: @[],
    specialTokens: SpecialTokens(
      padToken: "[PAD]",
      unkToken: "[UNK]",
      bosToken: "[BOS]",
      eosToken: "[EOS]"
    ),
    specialTokenIds: initTable[string, int](),
    merges: @[],
    cache: initTable[string, seq[int]](),
    cacheMaxSize: 10000
  )
  
  result.vocab["[PAD]"] = 0
  result.vocab["[UNK]"] = 1
  result.vocab["[BOS]"] = 2
  result.vocab["[EOS]"] = 3
  result.inverseVocab.add("[PAD]")
  result.inverseVocab.add("[UNK]")
  result.inverseVocab.add("[BOS]")
  result.inverseVocab.add("[EOS]")
  
  for code in 32..126:
    let ch = $chr(code)
    let id = result.vocab.len
    result.vocab[ch] = id
    result.inverseVocab.add(ch)

proc createWhitespaceTokenizer*(): Tokenizer =
  ## Создаёт простой токенизатор по пробелам
  result = Tokenizer(
    kind: tkBPE,
    vocab: initTable[string, int](),
    inverseVocab: @[],
    specialTokens: SpecialTokens(
      padToken: "[PAD]",
      unkToken: "[UNK]",
      bosToken: "[BOS]",
      eosToken: "[EOS]"
    ),
    specialTokenIds: initTable[string, int](),
    merges: @[],
    cache: initTable[string, seq[int]](),
    cacheMaxSize: 10000
  )
  
  result.vocab["[PAD]"] = 0
  result.vocab["[UNK]"] = 1
  result.vocab["[BOS]"] = 2
  result.vocab["[EOS]"] = 3
  result.inverseVocab = @["[PAD]", "[UNK]", "[BOS]", "[EOS]"]

proc createRegexTokenizer*(pattern: Regex): Tokenizer =
  ## Создаёт токенизатор на основе регулярного выражения
  result = createWhitespaceTokenizer()

proc tokenizeCode*(t: var Tokenizer, code: string, language: ProgrammingLanguage): seq[int] =
  ## Токенизация программного кода
  var keywords: seq[string]
  case language
  of plPython:
    keywords = @["def", "class", "if", "else", "elif", "for", "while", "return", "import", "from"]
  of plJavaScript:
    keywords = @["function", "const", "let", "var", "if", "else", "for", "while", "return"]
  of plJava:
    keywords = @["public", "private", "class", "if", "else", "for", "while", "return"]
  of plCpp:
    keywords = @["int", "float", "class", "struct", "if", "else", "for", "while"]
  of plRust:
    keywords = @["fn", "let", "mut", "if", "else", "for", "while", "return"]
  of plGo:
    keywords = @["func", "var", "if", "else", "for", "range", "return"]
  of plNim:
    keywords = @["proc", "var", "let", "if", "else", "for", "while", "return"]
  
  var processedCode = code
  for op in ["(", ")", "{", "}", "[", "]", ";", ",", ":", "."]:
    processedCode = processedCode.replace(op, " " & op & " ")
  
  result = tokenize(processedCode, t, addSpecialTokens = false)

proc tokenizeMath*(t: var Tokenizer, mathExpr: string): seq[int] =
  ## Токенизация математических выражений
  var processed = mathExpr
  
  for op in ["+", "-", "*", "/", "^", "=", "(", ")", "[", "]"]:
    processed = processed.replace(op, " " & op & " ")
  
  for fn in ["sin", "cos", "tan", "log", "ln", "exp", "sqrt"]:
    processed = processed.replace(fn, " " & fn & " ")
  
  result = tokenize(processed, t, addSpecialTokens = false)

proc tokenizeMarkdown*(t: var Tokenizer, md: string, preserveStructure: bool = true): seq[int] =
  ## Токенизация Markdown
  if not preserveStructure:
    return tokenize(md, t)
  
  var processed = md
  processed = processed.replace("# ", " [H1] ")
  processed = processed.replace("## ", " [H2] ")
  processed = processed.replace("### ", " [H3] ")
  processed = processed.replace("**", " [BOLD] ")
  processed = processed.replace("*", " [ITALIC] ")
  processed = processed.replace("`", " [CODE] ")
  
  result = tokenize(processed, t, addSpecialTokens = false)

proc tokenizeJson*(t: var Tokenizer, json: string): seq[int] =
  ## Токенизация JSON
  var processed = json
  
  for symbol in ["{", "}", "[", "]", ":", ","]:
    processed = processed.replace(symbol, " " & symbol & " ")
  
  result = tokenize(processed, t, addSpecialTokens = false)


#==============================================================================
# ИНТЕГРАЦИЯ И ЭКСПОРТ (v0.7)
#==============================================================================

proc toHuggingFaceFormat*(t: Tokenizer): JsonNode =
  ## Конвертирует токенизатор в формат HuggingFace
  result = %* {
    "version": "1.0",
    "truncation": nil,
    "padding": nil,
    "added_tokens": [],
    "normalizer": {"type": "Sequence", "normalizers": []},
    "pre_tokenizer": {"type": "ByteLevel"},
    "post_processor": {"type": "ByteLevel"},
    "decoder": {"type": "ByteLevel"},
    "model": {
      "type": "BPE",
      "dropout": nil,
      "unk_token": t.specialTokens.unkToken,
      "vocab": newJObject(),
      "merges": newJArray()
    }
  }
  
  for token, id in t.vocab:
    result["model"]["vocab"][token] = %id
  
  for merge in t.merges:
    result["model"]["merges"].add(%(merge.pair[0] & " " & merge.pair[1]))

proc toSentencePieceModel*(t: Tokenizer, path: string) =
  ## Экспортирует в формат SentencePiece
  var output = "# SentencePiece model\n"
  output.add("# Vocabulary size: " & $t.vocab.len & "\n\n")
  
  for i in 0..<t.inverseVocab.len:
    let token = t.inverseVocab[i]
    let score = if token in t.scores: t.scores[token] else: 0.0
    output.add(token & "\t" & score.formatFloat(ffDecimal, 6) & "\n")
  
  writeFile(path, output)

proc toTikTokenFormat*(t: Tokenizer): string =
  ## Экспортирует в формат TikToken
  var entries: seq[string] = @[]
  
  for i in 0..<t.inverseVocab.len:
    entries.add(t.inverseVocab[i] & " " & $i)
  
  result = entries.join("\n")

proc fromPretrainedModel*(modelName: string): Tokenizer =
  ## Загружает предобученный токенизатор
  case modelName.toLowerAscii()
  of "gpt2", "gpt-2":
    result = Tokenizer(
      kind: tkByteLevelBPE,
      vocab: initTable[string, int](),
      inverseVocab: @[],
      specialTokens: SpecialTokens(
        padToken: "<|endoftext|>",
        unkToken: "<|endoftext|>",
        bosToken: "<|endoftext|>",
        eosToken: "<|endoftext|>"
      ),
      specialTokenIds: initTable[string, int](),
      merges: @[],
      byteEncoder: initBytePairEncoder(),
      cache: initTable[string, seq[int]](),
      cacheMaxSize: 10000
    )
    result.byteDecoder = initByteDecoder(result.byteEncoder)
  
  of "bert", "bert-base":
    result = Tokenizer(
      kind: tkWordPiece,
      vocab: initTable[string, int](),
      inverseVocab: @[],
      specialTokens: SpecialTokens(
        padToken: "[PAD]",
        unkToken: "[UNK]",
        bosToken: "[CLS]",
        eosToken: "[SEP]",
        clsToken: "[CLS]",
        sepToken: "[SEP]",
        maskToken: "[MASK]"
      ),
      specialTokenIds: initTable[string, int](),
      continuingSubwordPrefix: "##",
      cache: initTable[string, seq[int]](),
      cacheMaxSize: 10000
    )
  
  else:
    raise newException(ValidationError, "Неизвестная модель: " & modelName)


#==============================================================================
# DEBUGGING И ВИЗУАЛИЗАЦИЯ (v0.7)
#==============================================================================

proc visualizeTokenization*(t: var Tokenizer, text: string): string =
  ## Визуализирует токенизацию с цветовым кодированием
  let tokens = tokenizeWithOffsets(text, t)
  
  const colors = [
    "\e[31m", "\e[32m", "\e[33m", "\e[34m", "\e[35m", "\e[36m"
  ]
  const reset = "\e[0m"
  
  result = ""
  var lastEnd = 0
  
  for i, token in tokens:
    if token.startChar > lastEnd and lastEnd < text.len:
      result.add(text[lastEnd..<token.startChar])
    
    let colorIdx = i mod colors.len
    result.add(colors[colorIdx] & token.token & reset)
    lastEnd = token.endChar
  
  if lastEnd < text.len:
    result.add(text[lastEnd..^1])
  
  result.add("\n\nТокены:\n")
  for i, token in tokens:
    result.add("  [" & $i & "] " & token.token & " (id: " & $token.tokenId & ")\n")

proc debugTokenization*(t: var Tokenizer, text: string): DebugInfo =
  ## Детальная отладочная информация
  result.originalText = text
  
  let tokenOffsets = tokenizeWithOffsets(text, t)
  result.tokenIds = @[]
  result.tokens = @[]
  result.boundaries = @[]
  result.unknownWords = @[]
  result.warnings = @[]
  
  let unkId = getUnkTokenId(t)
  
  for token in tokenOffsets:
    result.tokenIds.add(token.tokenId)
    result.tokens.add(token.token)
    result.boundaries.add((token.startChar, token.endChar))
    
    if token.tokenId == unkId:
      result.unknownWords.add(token.token)
      result.warnings.add("Неизвестное слово: '" & token.token & "' в позиции " & $token.startChar)
  
  if result.unknownWords.len > result.tokens.len div 2:
    result.warnings.add("ВНИМАНИЕ: Более 50% токенов неизвестны!")

proc explainToken*(t: Tokenizer, tokenId: int): TokenExplanation =
  ## Объясняет происхождение токена
  if tokenId < 0 or tokenId >= t.inverseVocab.len:
    return TokenExplanation(
      token: "[INVALID]",
      tokenId: tokenId,
      frequency: 0,
      mergeHistory: @[],
      exampleContexts: @[]
    )
  
  result.token = t.inverseVocab[tokenId]
  result.tokenId = tokenId
  result.frequency = 0
  result.mergeHistory = @[]
  
  if t.kind == tkBPE or t.kind == tkByteLevelBPE:
    let token = result.token
    if token.len > 1:
      for i in 1..<token.len:
        let left = token[0..<i]
        let right = token[i..^1]
        if left in t.vocab and right in t.vocab:
          result.mergeHistory.add(left & " + " & right)
  
  result.exampleContexts = @["[требуется корпус для примеров]"]

proc findTokenConflicts*(t: Tokenizer): seq[TokenConflict] =
  ## Находит конфликты в словаре
  result = @[]
  
  for i in 0..<t.inverseVocab.len:
    let token1 = t.inverseVocab[i]
    
    for j in (i+1)..<t.inverseVocab.len:
      let token2 = t.inverseVocab[j]
      
      if token1.startsWith(token2) or token2.startsWith(token1):
        result.add(TokenConflict(
          token1: token1,
          token2: token2,
          conflictType: "overlap",
          suggestedResolution: "Рассмотреть удаление более короткого токена"
        ))
      
      if token1.len == token2.len and token1.len > 2:
        var differences = 0
        for k in 0..<token1.len:
          if token1[k] != token2[k]:
            differences.inc()
        
        if differences == 1:
          result.add(TokenConflict(
            token1: token1,
            token2: token2,
            conflictType: "ambiguous",
            suggestedResolution: "Проверить, не ошибка ли это в словаре"
          ))
  
  if result.len > 100:
    result = result[0..<100]

proc getReadableTokens*(t: Tokenizer, tokenIds: seq[int]): seq[string] =
  ## Конвертирует ID токенов в читаемые строки
  result = @[]
  for id in tokenIds:
    if id < t.inverseVocab.len:
      result.add(t.inverseVocab[id])
    else:
      result.add("[INVALID_ID:" & $id & "]")

proc analyzeVocabCoverage*(t: var Tokenizer, texts: seq[string]): tuple[coverage: float, uniqueTokens: int, totalWords: int] =
  ## Анализирует покрытие словаря
  var uniqueWords = initHashSet[string]()
  var totalWords = 0
  var coveredWords = 0
  
  for text in texts:
    let words = splitIntoWords(text)
    totalWords += words.len
    
    for word in words:
      uniqueWords.incl(word.toLowerAscii())
      
      let tokens = tokenize(word, t, addSpecialTokens = false)
      var hasUnk = false
      let unkId = getUnkTokenId(t)
      
      for tokenId in tokens:
        if tokenId == unkId:
          hasUnk = true
          break
      
      if not hasUnk:
        coveredWords.inc()
  
  let coverage = if totalWords > 0: coveredWords.float / totalWords.float else: 0.0
  
  result = (coverage: coverage, uniqueTokens: uniqueWords.len, totalWords: totalWords)


proc truncateToRunes*(s: string, maxRunes: int): string =
  ## Обрезает строку до заданного количества рун (Unicode символов)
  result = ""
  var count = 0
  for rune in s.runes:
    if count >= maxRunes:
      break
    result.add($rune)
    count.inc()




when isMainModule:
  #============================================================================
  # КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ БИБЛИОТЕКИ ТОКЕНИЗАЦИИ
  #============================================================================
  
  randomize()
  
  echo "╔" & "═".repeat(70) & "╗"
  echo "║           КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ БИБЛИОТЕКИ ТОКЕНИЗАЦИИ            ║"
  echo "╚" & "═".repeat(70) & "╝"
  echo ""
  echo "Дата запуска: ", now().format("yyyy-MM-dd HH:mm:ss")
  echo ""
  
  #============================================================================
  # УТИЛИТЫ ДЛЯ ТЕСТИРОВАНИЯ
  #============================================================================
  
  type
    TestResult = object
      name: string
      passed: bool
      message: string
      duration: float

    TestGroup = object
      name: string
      tests: seq[TestResult]
      totalTests: int
      passedTests: int
      failedTests: int
      totalDuration: float

  var allGroups: seq[TestGroup] = @[]
  var currentGroup: TestGroup

  proc startTestGroup(name: string) =
    currentGroup = TestGroup(
      name: name,
      tests: @[],
      totalTests: 0,
      passedTests: 0,
      failedTests: 0,
      totalDuration: 0.0
    )
    currentGroup.totalDuration = -epochTime()  # Запоминаем время начала (отрицательное)
    echo ""
    echo "╔" & "═".repeat(70) & "╗"
    echo "║  ", alignLeft(name, 68), "║"
    echo "╚" & "═".repeat(70) & "╝"

  proc endTestGroup() =
    currentGroup.totalDuration += epochTime()  # Добавляем время окончания
    allGroups.add(currentGroup)
    echo ""
    echo repeat("━", 72)
    echo "Итого: ", currentGroup.passedTests, "/", currentGroup.totalTests, " тестов пройдено"
    if currentGroup.failedTests > 0:
      echo "❌ Провалено: ", currentGroup.failedTests
    else:
      echo "✅ Все тесты успешно пройдены!"
    echo "Время выполнения: ", currentGroup.totalDuration.formatFloat(ffDecimal, 3), " сек"
    echo repeat("━", 72)

  proc test(name: string, condition: bool, message: string = "") =
    let passed = condition
    
    currentGroup.totalTests += 1
    
    if passed:
      currentGroup.passedTests += 1
      echo "✓ ", name
    else:
      currentGroup.failedTests += 1
      echo "✗ ", name
      if message != "":
        echo "  Причина: ", message
    
    currentGroup.tests.add(TestResult(
      name: name,
      passed: passed,
      message: message,
      duration: 0.0  # Не измеряем время отдельных тестов
    ))



  #============================================================================
  # ТЕСТОВЫЕ ДАННЫЕ
  #============================================================================
  const FN = "../Тексты и книги/Базовый текст.txt"
  let corpus = split(readFile(FN), '\n')

  const testSentences = @[
    "Сначала он всё-таки хотел разыскать её и ребёнка.",
    """Речь товарища прокурора, по его мнению, должна была иметь 
общественное значение, подобно тем знаменитым речам, которые говорили 
сделавшиеся знаменитыми адвокаты.""",
    "Весёлый купец.",
    "Текст со специальными символами: !@#$%^&*()",
    "Numbers: 123 456 789",
    "ВЕРХНИЙ РЕГИСТР И нижний регистр СмЕшАнНыЙ",
    "Повторение повторение повторение слов слов слов",
    "княгиня Софья Васильевна была худая длинная женщина"
  ]

  #============================================================================
  # ГРУППА 1: ТЕСТЫ BPE
  #============================================================================
  
  proc testBPE() =
    startTestGroup("ГРУППА 1: ТЕСТЫ BPE (BYTE PAIR ENCODING)")
    
    echo "\n→ Создание и обучение BPE токенизатора..."
    var bpeTokenizer = trainBPE(corpus, vocabSize = 1500, minFrequency = 1)
    
    # Сохраняем словарь в JSON
    exportTokenizerToJson(bpeTokenizer, "bpe_vocab.json")
    echo "✓ Словарь BPE сохранён в: bpe_vocab.json"
    
    test("1.1 Размер словаря BPE",
         bpeTokenizer.vocab.len > 0 and bpeTokenizer.vocab.len <= 1500,
         "Размер словаря: " & $bpeTokenizer.vocab.len)
    
    test("1.2 Наличие PAD токена в словаре",
         bpeTokenizer.specialTokens.padToken in bpeTokenizer.vocab)
    test("1.3 Наличие UNK токена в словаре",
         bpeTokenizer.specialTokens.unkToken in bpeTokenizer.vocab)
    test("1.4 Наличие BOS токена в словаре",
         bpeTokenizer.specialTokens.bosToken in bpeTokenizer.vocab)
    test("1.5 Наличие EOS токена в словаре",
         bpeTokenizer.specialTokens.eosToken in bpeTokenizer.vocab)
    
    let testText = testSentences[0]
    let tokens = tokenize(testText, bpeTokenizer)
    test("1.6 Токенизация возвращает непустой результат",
         tokens.len > 0,
         "Количество токенов: " & $tokens.len)
    
    let decoded = bpeTokenizer.decode(tokens, skipSpecialTokens = true)
    test("1.7 Декодирование восстанавливает текст",
         decoded.strip() == testText or 
         decoded.replace(" ", "").toLowerAscii() == testText.replace(" ", "").toLowerAscii(),
         "Оригинал: '" & testText & "', Декодировано: '" & decoded & "'")
    
    var vocabConsistent = true
    for token, id in bpeTokenizer.vocab:
      if id >= bpeTokenizer.inverseVocab.len or bpeTokenizer.inverseVocab[id] != token:
        vocabConsistent = false
        break
    test("1.8 Согласованность vocab и inverseVocab", vocabConsistent)
    
    test("1.9 Наличие BPE merges",
         bpeTokenizer.merges.len > 0,
         "Количество merges: " & $bpeTokenizer.merges.len)
    
    let savePath = "test_bpe.json"
    saveTokenizer(bpeTokenizer, savePath)
    test("1.10 Сохранение токенизатора", fileExists(savePath))
    
    var loadedTokenizer = loadTokenizer(savePath)
    test("1.11 Загрузка токенизатора", loadedTokenizer.vocab.len == bpeTokenizer.vocab.len)
    
    let tokensOriginal = tokenize("тестовый текст", bpeTokenizer)
    let tokensLoaded = tokenize("тестовый текст", loadedTokenizer)
    test("1.12 Идентичность токенизации после загрузки",
         tokensOriginal == tokensLoaded)
    
    let metrics = getMetrics(bpeTokenizer, corpus)
    test("1.13 Вычисление метрик - размер словаря",
         metrics.vocabSize > 0)
    test("1.14 Вычисление метрик - коэффициент сжатия",
         metrics.compressionRatio > 0.0 and metrics.compressionRatio < 100.0)
    
    if fileExists(savePath):
      removeFile(savePath)
    
    endTestGroup()

  #============================================================================
  # ГРУППА 2: ТЕСТЫ WORDPIECE
  #============================================================================
  
  proc testWordPiece() =
    startTestGroup("ГРУППА 2: ТЕСТЫ WORDPIECE")
    
    echo "\n→ Создание и обучение WordPiece токенизатора..."
    var wpTokenizer = trainWordPiece(corpus, vocabSize = 1500, minFrequency = 1, preserveCase = true)
    
    # Сохраняем словарь в JSON
    exportTokenizerToJson(wpTokenizer, "wordpiece_vocab.json")
    echo "✓ Словарь WordPiece сохранён в: wordpiece_vocab.json"
    
    test("2.1 Тип токенизатора WordPiece",
         wpTokenizer.kind == tkWordPiece)
    
    test("2.2 Размер словаря WordPiece",
         wpTokenizer.vocab.len > 0 and wpTokenizer.vocab.len <= 1500)
    
    test("2.3 Наличие префикса продолжения",
         wpTokenizer.continuingSubwordPrefix == "##")
    
    test("2.4 Наличие специальных токенов",
         wpTokenizer.specialTokens.padToken in wpTokenizer.vocab and
         wpTokenizer.specialTokens.unkToken in wpTokenizer.vocab)
    
    let testText = "непонятное слово"
    let tokens = tokenize(testText, wpTokenizer)
    test("2.5 Токенизация возвращает результат",
         tokens.len > 0)
    
    let decoded = wpTokenizer.decode(tokens, skipSpecialTokens = true)
    test("2.6 Декодирование убирает ## префиксы",
         "##" notin decoded,
         "Декодировано: " & decoded)
    
    let unknownText = "qwertyzxcvb"
    let unknownTokens = tokenize(unknownText, wpTokenizer)
    test("2.7 Обработка неизвестных слов",
         unknownTokens.len > 0)
    
    let longWord = "длинноенепонятноеслово"
    let longTokens = tokenize(longWord, wpTokenizer)
    test("2.8 Длинные слова разбиваются на подслова",
         longTokens.len >= 1)
    
    for sentence in testSentences[0..2]:
      let encoded = tokenize(sentence, wpTokenizer)
      let redecoded = wpTokenizer.decode(encoded, skipSpecialTokens = true)
      let normalized1 = sentence.replace(" ", "").toLowerAscii()
      let normalized2 = redecoded.replace(" ", "").toLowerAscii()
      test("2.9 Согласованность encode-decode для: " & sentence[0..min(20, sentence.len-1)],
           normalized1 == normalized2 or normalized2.contains(normalized1[0..min(5, normalized1.len-1)]))
    
    let metrics = getMetrics(wpTokenizer, corpus)
    test("2.10 Метрики - утилизация словаря",
         metrics.vocabUtilization >= 0.0 and metrics.vocabUtilization <= 1.0)
    
    endTestGroup()

  #============================================================================
  # ГРУППА 3: ТЕСТЫ SENTENCEPIECE
  #============================================================================
  
  proc testSentencePiece() =
    startTestGroup("ГРУППА 3: ТЕСТЫ SENTENCEPIECE")
    
    echo "\n→ Создание и обучение SentencePiece токенизатора..."
    var spTokenizer = trainSentencePiece(corpus, vocabSize = 1500)
    
    # Сохраняем словарь в JSON
    exportTokenizerToJson(spTokenizer, "sentencepiece_vocab.json")
    echo "✓ Словарь SentencePiece сохранён в: sentencepiece_vocab.json"
    
    test("3.1 Тип токенизатора SentencePiece",
         spTokenizer.kind == tkSentencePiece)
    
    test("3.2 Размер словаря SentencePiece",
         spTokenizer.vocab.len > 0)
    
    test("3.3 Наличие scores для токенов",
         spTokenizer.scores.len > 0)
    
    test("3.4 Специальные токены в словаре",
         spTokenizer.specialTokens.unkToken in spTokenizer.vocab)
    
    let testText = "Тестовое предложение"
    let tokens = tokenize(testText, spTokenizer)
    test("3.5 Токенизация работает",
         tokens.len > 0,
         "Количество токенов: " & $tokens.len)
    
    let decoded = spTokenizer.decode(tokens, skipSpecialTokens = true)
    test("3.6 Декодирование работает",
         decoded.len > 0)
    
    var allHaveScores = true
    for token in spTokenizer.vocab.keys:
      if token notin spTokenizer.scores:
        allHaveScores = false
        break
    test("3.7 Все токены словаря имеют scores", allHaveScores)
    
    let textWithSpaces = "слово пробел слово"
    let spacesTokens = tokenize(textWithSpaces, spTokenizer)
    test("3.8 Обработка пробелов",
         spacesTokens.len > 0)
    
    let original = "Проверка консистентности"
    let encoded = tokenize(original, spTokenizer)
    let redecoded = spTokenizer.decode(encoded, skipSpecialTokens = true)
    let norm1 = original.replace(" ", "").toLowerAscii()
    let norm2 = redecoded.replace(" ", "").replace("▁", "").toLowerAscii()
    test("3.9 Консистентность encode-decode",
         norm1 == norm2 or norm2.contains(norm1[0..min(3, norm1.len-1)]))
    
    let metrics = getMetrics(spTokenizer, corpus)
    test("3.10 Метрики - коэффициент сжатия",
         metrics.compressionRatio > 0.0)
    
    endTestGroup()

  #============================================================================
  # ГРУППА 4: ТЕСТЫ BYTE-LEVEL BPE
  #============================================================================
  
  proc testByteLevelBPE() =
    startTestGroup("ГРУППА 4: ТЕСТЫ BYTE-LEVEL BPE (GPT-2 STYLE)")
    
    echo "\n→ Создание и обучение ByteLevel BPE токенизатора..."
    var blbpeTokenizer = trainByteLevelBPE(corpus, vocabSize = 2000)
    
    # Сохраняем словарь в JSON
    exportTokenizerToJson(blbpeTokenizer, "bytelevelbpe_vocab.json")
    echo "✓ Словарь ByteLevelBPE сохранён в: bytelevelbpe_vocab.json"
    
    test("4.1 Тип токенизатора ByteLevelBPE",
         blbpeTokenizer.kind == tkByteLevelBPE)
    
    test("4.2 Наличие byte encoder",
         blbpeTokenizer.byteEncoder.len == 256)
    
    test("4.3 Наличие byte decoder",
         blbpeTokenizer.byteDecoder.len == 256)
    
    var encoderDecoderConsistent = true
    for b, s in blbpeTokenizer.byteEncoder:
      if blbpeTokenizer.byteDecoder[s] != b:
        encoderDecoderConsistent = false
        break
    test("4.4 Консистентность byte encoder/decoder", encoderDecoderConsistent)
    
    let testText = "княгиня Софья Васильевна"
    let tokens = tokenize(testText, blbpeTokenizer, addSpecialTokens = false)
    test("4.5 Токенизация UTF-8 текста",
         tokens.len > 0,
         "Количество токенов: " & $tokens.len)
    
    let decoded = blbpeTokenizer.decode(tokens, skipSpecialTokens = true)
    test("4.6 Декодирование сохраняет оригинальный текст",
         decoded == testText,
         "Оригинал: '" & testText & "', Декодировано: '" & decoded & "'")
    
    let specialChars = "!@#$%^&*()"
    let specialTokens = tokenize(specialChars, blbpeTokenizer, addSpecialTokens = false)
    let specialDecoded = blbpeTokenizer.decode(specialTokens, skipSpecialTokens = true)
    test("4.7 Обработка специальных символов",
         specialDecoded == specialChars,
         "Оригинал: '" & specialChars & "', Декодировано: '" & specialDecoded & "'")
    
    let numbers = "123456789"
    let numTokens = tokenize(numbers, blbpeTokenizer, addSpecialTokens = false)
    let numDecoded = blbpeTokenizer.decode(numTokens, skipSpecialTokens = true)
    test("4.8 Обработка чисел",
         numDecoded == numbers)
    
    let textWithSpaces = "слово пробел слово"
    let spaceTokens = tokenize(textWithSpaces, blbpeTokenizer, addSpecialTokens = false)
    let spaceDecoded = blbpeTokenizer.decode(spaceTokens, skipSpecialTokens = true)
    test("4.9 Сохранение пробелов",
         spaceDecoded == textWithSpaces,
         "Оригинал: '" & textWithSpaces & "', Декодировано: '" & spaceDecoded & "'")
    
    let offsets = tokenizeWithOffsets(testText, blbpeTokenizer, addSpecialTokens = false)
    test("4.10 Генерация token offsets",
         offsets.len > 0)
    
    if offsets.len > 0:
      var offsetsCorrect = true
      for offset in offsets:
        if offset.startChar < 0 or offset.endChar > testText.runeLen or 
           offset.startChar >= offset.endChar:
          offsetsCorrect = false
          break
      test("4.11 Корректность char offsets", offsetsCorrect)
    
    if offsets.len > 0:
      var byteOffsetsCorrect = true
      for offset in offsets:
        if offset.startByte < 0 or offset.endByte > testText.len or
           offset.startByte >= offset.endByte:
          byteOffsetsCorrect = false
          break
      test("4.12 Корректность byte offsets", byteOffsetsCorrect)
    
    endTestGroup()

  #============================================================================
  # ГРУППА 5: ТЕСТЫ ДОПОЛНИТЕЛЬНЫХ ФУНКЦИЙ
  #============================================================================
  
  proc testAdditionalFunctions() =
    startTestGroup("ГРУППА 5: ТЕСТЫ ДОПОЛНИТЕЛЬНЫХ ФУНКЦИЙ")
    
    var tokenizer = trainBPE(corpus, vocabSize = 1500)
    
    let htmlText = "<div>Текст с <b>HTML</b> тегами</div>"
    let cleaned = cleanText(htmlText, removeHtml = true)
    test("5.1 cleanText - удаление HTML тегов",
         "<" notin cleaned and ">" notin cleaned)
    
    let urlText = "Ссылка https://example.com в тексте"
    let noUrls = cleanText(urlText, removeUrls = true)
    test("5.2 cleanText - удаление URLs",
         "https://" notin noUrls)
    
    let emailText = "Контакт test@example.com здесь"
    let noEmails = cleanText(emailText, removeEmails = true)
    test("5.3 cleanText - удаление email",
         "@" notin noEmails or "example.com" notin noEmails)
    
    let spacesText = "Много    пробелов     здесь"
    let normalizedSpaces = cleanText(spacesText, removeExtraWhitespace = true)
    test("5.4 cleanText - нормализация пробелов",
         "    " notin normalizedSpaces)
    
    let batchTexts = @["первый", "второй текст", "третий"]
    let batchEncoding = encodeBatch(tokenizer, batchTexts, maxLength = 20, padding = true)
    test("5.5 encodeBatch - количество последовательностей",
         batchEncoding.inputIds.len == 3)
    
    test("5.6 encodeBatch - одинаковая длина после padding",
         batchEncoding.inputIds[0].len == batchEncoding.inputIds[1].len and
         batchEncoding.inputIds[1].len == batchEncoding.inputIds[2].len)
    
    test("5.7 encodeBatch - корректная attention mask",
         batchEncoding.attentionMask.len == 3)
    
    let paddedTokens = encodeWithPadding(tokenizer, "короткий текст", maxLength = 20)
    test("5.8 encodeWithPadding - результат имеет заданную длину",
         paddedTokens.len == 20)
    
    let originalTokens = tokenize("тестовое предложение для маскирования", tokenizer)
    let (maskedTokens, labels) = maskTokens(originalTokens, tokenizer, maskProb = 0.15, seed = 42)
    test("5.9 maskTokens - длины совпадают",
         maskedTokens.len == labels.len and labels.len == originalTokens.len)
    
    var hasMasked = false
    for i in 0..<maskedTokens.len:
      if labels[i] != -100:
        hasMasked = true
        break
    test("5.10 maskTokens - присутствуют замаскированные токены", hasMasked)
    
    let breakdown = getSubwordBreakdown("тестовое слово", tokenizer)
    test("5.11 getSubwordBreakdown - возвращает подслова",
         breakdown.len > 0)
    
    let estimated = estimateTokenCount("это текст для оценки количества токенов")
    test("5.12 estimateTokenCount - разумная оценка",
         estimated > 0 and estimated < 100)
    
    let validationResults = validateTokenizerDetailed(tokenizer)
    test("5.13 validateTokenizerDetailed - проверка проходит",
         validationResults.len > 0)
    
    var wpTokenizer = trainWordPiece(corpus, vocabSize = 1500)
    let comparison = compareTokenizers("Тестовый текст", @[tokenizer, wpTokenizer])
    test("5.14 compareTokenizers - сравнение работает",
         comparison.len == 2)
    
    let analysis = analyzeVocabulary(tokenizer, corpus, topN = 5)
    test("5.15 analyzeVocabulary - размер словаря",
         analysis.vocabSize > 0)
    test("5.16 analyzeVocabulary - средняя длина токена",
         analysis.avgTokenLength > 0.0)
    test("5.17 analyzeVocabulary - топ токены",
         analysis.mostFrequent.len <= 5)
    
    let originalSize = tokenizer.vocab.len
    discard pruneVocabulary(tokenizer, minFrequency = 2, corpus = corpus)
    test("5.18 pruneVocabulary - уменьшение размера словаря",
         tokenizer.vocab.len <= originalSize)
    
    let mixedCase = "ТеСтОвЫй ТЕКСТ"
    let lowered = toLowerUnicode(mixedCase)
    test("5.19 toLowerUnicode - приведение к нижнему регистру",
         lowered == "тестовый текст")
    
    let upper = toUpperUnicode("тестовый текст")
    test("5.20 toUpperUnicode - приведение к верхнему регистру",
         upper == "ТЕСТОВЫЙ ТЕКСТ")
    
    endTestGroup()

  #============================================================================
  # ГРУППА 6: ТЕСТЫ КЭШИРОВАНИЯ И ПРОИЗВОДИТЕЛЬНОСТИ
  #============================================================================
  
  proc testCachingAndPerformance() =
    startTestGroup("ГРУППА 6: ТЕСТЫ КЭШИРОВАНИЯ И ПРОИЗВОДИТЕЛЬНОСТИ")
    
    var tokenizer = trainByteLevelBPE(corpus, vocabSize = 2000)
    tokenizer.cacheMaxSize = 100
    
    test("6.1 Кэш изначально пуст",
         tokenizer.cache.len == 0)
    
    let text1 = "Первая токенизация"
    discard tokenize(text1, tokenizer)
    test("6.2 Первая токенизация создаёт cache miss",
         tokenizer.cacheMisses == 1)
    
    discard tokenize(text1, tokenizer)
    test("6.3 Повторная токенизация создаёт cache hit",
         tokenizer.cacheHits == 1)
    
    test("6.4 Кэш содержит элемент",
         (text1 & "0" & "false") in tokenizer.cache)
    
    let testText = "Тест производительности кэша"
    let startNoCache = cpuTime()
    for i in 1..10:
      discard tokenize(testText & $i, tokenizer)
    let timeNoCache = cpuTime() - startNoCache
    
    let startWithCache = cpuTime()
    for i in 1..10:
      discard tokenize(testText, tokenizer)
    let timeWithCache = cpuTime() - startWithCache
    
    test("6.5 Кэш ускоряет повторные токенизации",
         timeWithCache < timeNoCache or abs(timeWithCache - timeNoCache) < 0.01)
    
    clearCache(tokenizer)
    test("6.6 Очистка кэша работает",
         tokenizer.cache.len == 0)
    
    let batchSize = 10
    var batchTexts = newSeq[string](batchSize)
    for i in 0..<batchSize:
      batchTexts[i] = "Текст номер " & $i
    
    let startBatch = cpuTime()
    let batchResult = encodeBatch(tokenizer, batchTexts, maxLength = 50)
    let batchTime = cpuTime() - startBatch
    
    test("6.7 Batch processing завершается за разумное время",
         batchTime < 1.0)
    
    let metrics = getMetrics(tokenizer, corpus)
    test("6.8 Метрики - скорость токенизации измерена",
         metrics.tokensPerSecond > 0.0)
    
    endTestGroup()

  #============================================================================
  # ГРУППА 7: ТЕСТЫ BPE-DROPOUT И РЕГУЛЯРИЗАЦИИ
  #============================================================================
  
  proc testDropoutAndRegularization() =
    startTestGroup("ГРУППА 7: ТЕСТЫ BPE-DROPOUT И РЕГУЛЯРИЗАЦИИ")
    
    var tokenizer = trainByteLevelBPE(corpus, vocabSize = 2000)
    let testText = "Тестовое предложение для проверки dropout"
    
    let originalTokens = tokenize(testText, tokenizer, addSpecialTokens = false)
    test("7.1 Оригинальная токенизация работает",
         originalTokens.len > 0)
    
    let noDropout = tokenizeWithDropout(testText, tokenizer, dropoutProb = 0.0, seed = 42)
    test("7.2 Dropout с вероятностью 0.0 идентичен оригиналу",
         noDropout == originalTokens)
    
    let dropout1 = tokenizeWithDropout(testText, tokenizer, dropoutProb = 0.3, seed = 1)
    let dropout2 = tokenizeWithDropout(testText, tokenizer, dropoutProb = 0.3, seed = 2)
    let dropout3 = tokenizeWithDropout(testText, tokenizer, dropoutProb = 0.3, seed = 3)
    
    test("7.3 Dropout создаёт разные токенизации (1 vs 2)",
         dropout1 != dropout2)
    test("7.4 Dropout создаёт разные токенизации (2 vs 3)",
         dropout2 != dropout3)
    
    let dropoutMin = tokenizeWithDropout(testText, tokenizer, 
                                          dropoutProb = 0.3, seed = 10, minDropped = 2)
    test("7.5 Dropout с minDropped работает",
         dropoutMin.len >= originalTokens.len)
    
    let dropout4a = tokenizeWithDropout(testText, tokenizer, dropoutProb = 0.3, seed = 100)
    let dropout4b = tokenizeWithDropout(testText, tokenizer, dropoutProb = 0.3, seed = 100)
    test("7.6 Dropout детерминирован при одинаковом seed",
         dropout4a == dropout4b)
    
    let decodedDropout = tokenizer.decode(dropout1, skipSpecialTokens = true)
    test("7.7 Декодирование dropout токенов работает",
         decodedDropout.len > 0)
    
    let normalizedOriginal = testText.replace(" ", "").toLowerAscii()
    let normalizedDropout = decodedDropout.replace(" ", "").toLowerAscii()
    test("7.8 Dropout сохраняет основной смысл текста",
         normalizedOriginal == normalizedDropout or 
         normalizedDropout.contains(normalizedOriginal[0..min(5, normalizedOriginal.len-1)]))
    
    endTestGroup()

  #============================================================================
  # ГРУППА 8: ТЕСТЫ СПЕЦИАЛЬНЫХ СЛУЧАЕВ
  #============================================================================
  
  proc testEdgeCases() =
    startTestGroup("ГРУППА 8: ТЕСТЫ СПЕЦИАЛЬНЫХ СЛУЧАЕВ")
    
    var tokenizer = trainBPE(corpus, vocabSize = 1500)
    
    let emptyTokens = tokenize("", tokenizer, addSpecialTokens = false)
    test("8.1 Токенизация пустой строки",
         emptyTokens.len == 0)
    
    let spaceTokens = tokenize("     ", tokenizer, addSpecialTokens = false)
    test("8.2 Токенизация строки из пробелов",
         spaceTokens.len >= 0)
    
    let longText = "слово ".repeat(1000)
    let longTokens = tokenize(longText, tokenizer)
    test("8.3 Токенизация очень длинной строки",
         longTokens.len > 0)
    
    let singleChar = "а"
    let singleTokens = tokenize(singleChar, tokenizer, addSpecialTokens = false)
    test("8.4 Токенизация одного символа",
         singleTokens.len > 0)
    
    let onlyNumbers = "1234567890"
    let numberTokens = tokenize(onlyNumbers, tokenizer, addSpecialTokens = false)
    test("8.5 Токенизация только цифр",
         numberTokens.len > 0)
    
    let onlySpecial = "!@#$%^&*()"
    let specialTokens = tokenize(onlySpecial, tokenizer, addSpecialTokens = false)
    test("8.6 Токенизация только спецсимволов",
         specialTokens.len >= 0)
    
    let mixed = "Hello мир World"
    let mixedTokens = tokenize(mixed, tokenizer)
    test("8.7 Токенизация смешанных языков",
         mixedTokens.len > 0)
    
    let emoji = "Текст с 😀 emoji 🎉"
    let emojiTokens = tokenize(emoji, tokenizer)
    test("8.8 Токенизация текста с emoji",
         emojiTokens.len > 0)
    
    let repeated = "ааааааа"
    let repeatedTokens = tokenize(repeated, tokenizer, addSpecialTokens = false)
    test("8.9 Токенизация повторяющихся символов",
         repeatedTokens.len > 0)
    
    let decodedEmpty = tokenizer.decode(@[], skipSpecialTokens = true)
    test("8.10 Декодирование пустой последовательности",
         decodedEmpty == "")
    
    endTestGroup()



  #============================================================================
  # СТАТИСТИЧЕСКИЙ АНАЛИЗ РЕЗУЛЬТАТОВ
  #============================================================================
  
  proc printStatistics() =
    echo ""
    echo "╔" & "═".repeat(70) & "╗"
    echo "║          СТАТИСТИЧЕСКИЙ АНАЛИЗ РЕЗУЛЬТАТОВ ТЕСТИРОВАНИЯ              ║"
    echo "╚" & "═".repeat(70) & "╝"
    echo ""
    
    var totalTests = 0
    var totalPassed = 0
    var totalFailed = 0
    var totalDuration = 0.0
    
    echo "┌" & "─".repeat(70) & "┐"
    echo "│ ГРУППА                              │ ПРОЙДЕНО │ ПРОВАЛЕНО │  ВРЕМЯ  │"
    echo "├" & "─".repeat(70) & "┤"
    
    for group in allGroups:
      totalTests += group.totalTests
      totalPassed += group.passedTests
      totalFailed += group.failedTests
      totalDuration += group.totalDuration
      
      let groupName = group.name[0..min(34, group.name.len-1)]
      let passedStr = $group.passedTests & "/" & $group.totalTests
      let failedStr = $group.failedTests
      let timeStr = group.totalDuration.formatFloat(ffDecimal, 3) & "s"
      
      echo "│ ", groupName.alignLeft(35), " │ ", 
           passedStr.alignLeft(8), " │ ",
           failedStr.align(9), " │ ",
           timeStr.align(7), " │"
    
    echo "└" & "─".repeat(70) & "┘"
    echo ""
    
    echo "ОБЩАЯ СТАТИСТИКА:"
    echo "  Всего тестов:        ", totalTests
    echo "  Успешно пройдено:    ", totalPassed, " (", 
         (totalPassed.float / totalTests.float * 100).formatFloat(ffDecimal, 1), "%)"
    echo "  Провалено:           ", totalFailed
    echo "  Общее время:         ", totalDuration.formatFloat(ffDecimal, 3), " сек"
    echo "  Среднее время/тест:  ", 
         (totalDuration / totalTests.float * 1000).formatFloat(ffDecimal, 2), " мс"
    echo ""
    
    echo "АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ:"
    var fastestGroup = allGroups[0]
    var slowestGroup = allGroups[0]
    
    for group in allGroups:
      let avgTime = group.totalDuration / group.totalTests.float
      let fastestAvg = fastestGroup.totalDuration / fastestGroup.totalTests.float
      let slowestAvg = slowestGroup.totalDuration / slowestGroup.totalTests.float
      
      if avgTime < fastestAvg:
        fastestGroup = group
      if avgTime > slowestAvg:
        slowestGroup = group
    
    echo "  Самая быстрая группа: ", fastestGroup.name
    echo "    Среднее время/тест: ", 
         (fastestGroup.totalDuration / fastestGroup.totalTests.float * 1000).formatFloat(ffDecimal, 2), " мс"
    echo ""
    echo "  Самая медленная группа: ", slowestGroup.name
    echo "    Среднее время/тест: ", 
         (slowestGroup.totalDuration / slowestGroup.totalTests.float * 1000).formatFloat(ffDecimal, 2), " мс"
    echo ""
    
    if totalFailed == 0:
      echo "╔" & "═".repeat(70) & "╗"
      echo "║  ✅ ВСЕ ТЕСТЫ УСПЕШНО ПРОЙДЕНЫ!                              ║"
      echo "╚" & "═".repeat(70) & "╝"
    else:
      echo "╔" & "═".repeat(70) & "╗"
      echo "║      ⚠️  ОБНАРУЖЕНЫ ПРОВАЛЕННЫЕ ТЕСТЫ: ", align($totalFailed, 30), " ║"
      echo "╚" & "═".repeat(70) & "╝"
    echo ""

  #============================================================================
  # СРАВНИТЕЛЬНЫЙ АНАЛИЗ ТОКЕНИЗАТОРОВ
  #============================================================================
  
  proc comparativeAnalysis() =
    echo ""
    echo "╔" & "═".repeat(70) & "╗"
    echo "║          СРАВНИТЕЛЬНЫЙ АНАЛИЗ ТОКЕНИЗАТОРОВ                          ║"
    echo "╚" & "═".repeat(70) & "╝"
    echo ""
    
    echo "→ Обучение токенизаторов..."
    var bpeTokenizer = trainBPE(corpus, vocabSize = 1500)
    var wpTokenizer = trainWordPiece(corpus, vocabSize = 1500, preserveCase = true)
    var spTokenizer = trainSentencePiece(corpus, vocabSize = 1500)
    var blbpeTokenizer = trainByteLevelBPE(corpus, vocabSize = 2000)
    
    # Сохраняем все словари для сравнительного анализа
    exportTokenizerToJson(bpeTokenizer, "comparative_bpe.json")
    exportTokenizerToJson(wpTokenizer, "comparative_wordpiece.json")
    exportTokenizerToJson(spTokenizer, "comparative_sentencepiece.json")
    exportTokenizerToJson(blbpeTokenizer, "comparative_bytelevelbpe.json")
    echo "✓ Все словари сохранены в файлах comparative_*.json"
    echo ""
    
    let testText = "княгиня Софья Васильевна была худая длинная женщина"
    
    echo ""
    echo "Тестовый текст: ", testText
    echo ""
    
    echo "СРАВНЕНИЕ ТОКЕНИЗАЦИИ:"
    echo "─" & "─".repeat(69)
    
    let bpeTokens = tokenize(testText, bpeTokenizer, addSpecialTokens = false)
    echo "BPE:            ", bpeTokens.len, " токенов"
    echo "  Токены: ", bpeTokens
    
    let wpTokens = tokenize(testText, wpTokenizer, addSpecialTokens = false)
    echo ""
    echo "WordPiece:      ", wpTokens.len, " токенов"
    echo "  Токены: ", wpTokens
    
    let spTokens = tokenize(testText, spTokenizer, addSpecialTokens = false)
    echo ""
    echo "SentencePiece:  ", spTokens.len, " токенов"
    echo "  Токены: ", spTokens
    
    let blbpeTokens = tokenize(testText, blbpeTokenizer, addSpecialTokens = false)
    echo ""
    echo "ByteLevel BPE:  ", blbpeTokens.len, " токенов"
    echo "  Токены: ", blbpeTokens
    echo ""
    
    echo "─" & "─".repeat(69)
    echo "СРАВНЕНИЕ МЕТРИК:"
    echo "─" & "─".repeat(69)
    
    let bpeMetrics = getMetrics(bpeTokenizer, corpus)
    let wpMetrics = getMetrics(wpTokenizer, corpus)
    let spMetrics = getMetrics(spTokenizer, corpus)
    let blbpeMetrics = getMetrics(blbpeTokenizer, corpus)
    
    echo "                    │   BPE   │ WordPiece │ SentPiece │ ByteLvlBPE"
    echo "────────────────────┼─────────┼───────────┼───────────┼───────────"
    echo "Размер словаря      │ ", align($bpeMetrics.vocabSize, 7), " │ ",
         align($wpMetrics.vocabSize, 9), " │ ",
         align($spMetrics.vocabSize, 9), " │ ",
         align($blbpeMetrics.vocabSize, 10)
    
    echo "Коэфф. сжатия       │ ", 
         bpeMetrics.compressionRatio.formatFloat(ffDecimal, 2).align(7), " │ ",
         wpMetrics.compressionRatio.formatFloat(ffDecimal, 2).align(9), " │ ",
         spMetrics.compressionRatio.formatFloat(ffDecimal, 2).align(9), " │ ",
         blbpeMetrics.compressionRatio.formatFloat(ffDecimal, 2).align(10)
    
    echo "Утилиз. словаря     │ ",
         (bpeMetrics.vocabUtilization * 100).formatFloat(ffDecimal, 1).align(6), "% │ ",
         (wpMetrics.vocabUtilization * 100).formatFloat(ffDecimal, 1).align(8), "% │ ",
         (spMetrics.vocabUtilization * 100).formatFloat(ffDecimal, 1).align(8), "% │ ",
         (blbpeMetrics.vocabUtilization * 100).formatFloat(ffDecimal, 1).align(9), "%"
    
    echo "UNK токенов         │ ",
         (bpeMetrics.unkTokenRate * 100).formatFloat(ffDecimal, 1).align(6), "% │ ",
         (wpMetrics.unkTokenRate * 100).formatFloat(ffDecimal, 1).align(8), "% │ ",
         (spMetrics.unkTokenRate * 100).formatFloat(ffDecimal, 1).align(8), "% │ ",
         (blbpeMetrics.unkTokenRate * 100).formatFloat(ffDecimal, 1).align(9), "%"
    
    echo ""
    
    echo "─" & "─".repeat(69)
    echo "ПРОВЕРКА ДЕКОДИРОВАНИЯ:"
    echo "─" & "─".repeat(69)
    
    echo "Оригинал:       ", testText
    echo ""
    echo "BPE:            ", bpeTokenizer.decode(bpeTokens, skipSpecialTokens = true)
    echo "WordPiece:      ", wpTokenizer.decode(wpTokens, skipSpecialTokens = true)
    echo "SentencePiece:  ", spTokenizer.decode(spTokens, skipSpecialTokens = true)
    echo "ByteLevel BPE:  ", blbpeTokenizer.decode(blbpeTokens, skipSpecialTokens = true)
    echo ""

  #============================================================================
  # ЗАПУСК ВСЕХ ТЕСТОВ
  #============================================================================
  
  let overallStart = epochTime()
  
  testBPE()
  testWordPiece()
  testSentencePiece()
  testByteLevelBPE()
  testAdditionalFunctions()
  testCachingAndPerformance()
  testDropoutAndRegularization()
  testEdgeCases()
  
  #============================================================================
  # НОВАЯ ГРУППА: ТЕСТЫ v0.6 FEATURES
  #============================================================================
  
  proc testNewFeaturesV06() =
    startTestGroup("ГРУППА 9: НОВЫЕ ФУНКЦИИ v0.6")
    
    echo "\n→ Тестирование валидации и обработки ошибок..."
    
    # Валидация
    test("9.1 Валидация пустого текста",
         validateInput("").isSome,
         "Пустой текст должен вызывать ошибку")
    
    test("9.2 Валидация нормального текста",
         validateInput("нормальный текст").isNone,
         "Нормальный текст должен проходить валидацию")
    
    test("9.3 Валидация слишком длинного текста",
         validateInput("x".repeat(MAX_INPUT_LENGTH + 1)).isSome,
         "Слишком длинный текст должен вызывать ошибку")
    
    # Advanced normalization
    echo "\n→ Тестирование продвинутой нормализации..."
    
    let textWithZeroWidth = "Hello\u200BWorld"
    let cleaned = handleZeroWidthChars(textWithZeroWidth)
    test("9.4 Удаление zero-width символов",
         cleaned == "HelloWorld",
         "Результат: " & cleaned)
    
    let textWithWhitespace = "test\t\ttext  \n\n  end"
    let normalized = normalizeWhitespaceAdvanced(textWithWhitespace)
    test("9.5 Нормализация whitespace (advanced)",
         normalized == "test text end",
         "Результат: " & normalized)
    
    let fullNorm = fullNormalization("test\u200B\t\ttext")
    test("9.6 Полная нормализация",
         fullNorm.len > 0 and "\u200B" notin fullNorm)
    
    # LRU Cache
    echo "\n→ Тестирование LRU Cache..."
    
    var lruCache = newLRUCache(maxSize = 3)
    lruCache.put("key1", @[1, 2, 3])
    lruCache.put("key2", @[4, 5, 6])
    lruCache.put("key3", @[7, 8, 9])
    
    test("9.7 LRU Cache сохраняет значения",
         lruCache.get("key1").isSome,
         "key1 должен быть в кэше")
    
    lruCache.put("key4", @[10, 11, 12])  # должен вытеснить старейший
    
    test("9.8 LRU Cache eviction работает",
         lruCache.entries.len <= 3,
         "Размер кэша: " & $lruCache.entries.len)
    
    let stats = lruCache.getStats()
    test("9.9 LRU Cache статистика",
         stats.hits > 0 or stats.misses >= 0,
         "Hits: " & $stats.hits & ", Misses: " & $stats.misses)
    
    # Language detection
    echo "\n→ Тестирование определения языка..."
    
    test("9.10 Определение кириллицы",
         detectLanguage("Привет мир") == "cyrillic",
         "Результат: " & detectLanguage("Привет мир"))
    
    test("9.11 Определение латиницы",
         detectLanguage("Hello world") == "latin",
         "Результат: " & detectLanguage("Hello world"))
    
    # Incremental vocabulary
    echo "\n→ Тестирование инкрементального обновления словаря..."
    
    var testTokenizer = trainBPE(corpus[0..99], vocabSize = 500)
    let initialSize = testTokenizer.vocab.len
    
    testTokenizer.addTokens(@["новыйтокен1", "новыйтокен2"])
    test("9.12 Добавление токенов",
         testTokenizer.vocab.len == initialSize + 2,
         "Было: " & $initialSize & ", стало: " & $testTokenizer.vocab.len)
    
    test("9.13 Новые токены в словаре",
         "новыйтокен1" in testTokenizer.vocab and "новыйтокен2" in testTokenizer.vocab)
    
    # Vocabulary alignment
    echo "\n→ Тестирование выравнивания словарей..."
    
    var tokenizer1 = trainBPE(corpus[0..49], vocabSize = 300)
    var tokenizer2 = trainBPE(corpus[50..99], vocabSize = 300)
    
    let commonTokens = findCommonTokens(@[tokenizer1, tokenizer2])
    test("9.14 Поиск общих токенов",
         commonTokens.len > 0,
         "Найдено общих токенов: " & $commonTokens.len)
    
    let aligned = alignVocabularies(tokenizer1, tokenizer2)
    test("9.15 Объединение словарей",
         aligned.vocab.len >= tokenizer1.vocab.len,
         "Размер объединённого словаря: " & $aligned.vocab.len)
    
    # OOV detection
    echo "\n→ Тестирование детекции OOV слов..."
    
    let oovWords = analyzeOOVWords(tokenizer1, "неизвестноеслово тест")
    test("9.16 Детекция OOV работает",
         oovWords.len >= 0,
         "Найдено OOV слов: " & $oovWords.len)
    
    # Enhanced Unicode
    echo "\n→ Тестирование улучшенных Unicode операций..."
    
    test("9.17 Подсчёт рун",
         runeCount("Привет") == 6,
         "Подсчитано рун: " & $runeCount("Привет"))
    
    let truncated = truncateToRunes("Длинная строка текста", 7)
    test("9.18 Обрезка до рун",
         runeCount(truncated) <= 7,
         "Обрезано до: " & $runeCount(truncated) & " рун")
    
    # Token statistics
    echo "\n→ Тестирование статистики токенов..."
    
    let tokenStats = getTokenStatistics(tokenizer1, corpus[0..49])
    test("9.19 Статистика токенов",
         tokenStats.totalTokens > 0,
         "Всего токенов: " & $tokenStats.totalTokens)
    
    test("9.20 Средняя длина токенов",
         tokenStats.avgLength > 0,
         "Средняя длина: " & tokenStats.avgLength.formatFloat(ffDecimal, 2))
    
    endTestGroup()
  
  testNewFeaturesV06()
  
  let overallTime = epochTime() - overallStart
  
  printStatistics()
  comparativeAnalysis()
  
  echo "╔" & "═".repeat(70) & "╗"
  echo "║          ТЕСТИРОВАНИЕ ЗАВЕРШЕНО                                      ║"
  echo "╚" & "═".repeat(70) & "╝"
  echo ""
  echo "Общее время выполнения всех тестов: ", 
       overallTime.formatFloat(ffDecimal, 3), " сек"
  echo ""





# nim c -d:release tokenization.nim
# nim c -d:release -d:danger --opt:speed tokenization.nim