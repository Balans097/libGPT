################################################################
##           ТОКЕНИЗАЦИЯ И ОБРАБОТКА ТЕКСТА
## 
##          Tokenization and text processing
## 
## Версия:   0.3.1
## Дата:     2026-01-30
## Автор:    github.com/Balans097
################################################################

# 0.3.1 — исправлены критические баги с обработкой Unicode:
#         некорректная работа cleanText с кириллицей (regex),
#         ложные предупреждения валидации (2026-01-30)
# 0.3 — добавлены расширенные функции очистки текста (cleanText),
#       дополнительные утилиты токенизации, маскирование токенов,
#       сравнение токенизаторов, валидация (2026-01-30)
# 0.2 — добавлены пакетная обработка, специальные токены
#       метрики (2026-01-30)
# 0.1 — начальная реализация токенизаторов:
#       BPE, WordPiece, SentencePiece (2026-01-30)




# nim c -d:release tokenization.nim






import math, times, random
import std/[tables, sequtils, strutils, algorithm, sets, unicode, json, os, re]





#==============================================================================
# БАЗОВЫЕ ТИПЫ
#==============================================================================

type
  TokenizerKind* = enum
    tkBPE = 0
    tkWordPiece = 1
    tkSentencePiece = 2

  BPEMerge = tuple[pair: (string, string), newToken: string, priority: int]

  SpecialTokens* = object
    padToken*: string
    unkToken*: string
    bosToken*: string  # beginning of sequence
    eosToken*: string  # end of sequence
    sepToken*: string  # Separator
    clsToken*: string  # classification
    maskToken*: string # mask token

  Tokenizer* = ref object
    kind*: TokenizerKind
    vocab*: Table[string, int]
    inverseVocab*: Table[int, string]
    merges*: seq[BPEMerge]
    specialTokens*: SpecialTokens
    specialTokenIds*: Table[string, int]
    maxInputCharsPerWord*: int
    continuingSubwordPrefix*: string
    scores*: Table[string, float]
    byteFallback*: bool
    preserveCase*: bool  # сохранять регистр
  
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

#==============================================================================
# ПРЕДОБРАБОТКА ТЕКСТА (С ПОДДЕРЖКОЙ UNICODE)
#==============================================================================

proc toLowerUnicode(s: string): string =
  ## Приводит к lowercase с поддержкой Unicode
  result = ""
  for rune in s.runes:
    result.add($rune.toLower())

proc toUpperUnicode(s: string): string =
  ## Приводит к uppercase с поддержкой Unicode
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
  ## Универсальная очистка текста от мусора и нормализация
  ## 
  ## Параметры:
  ##   removeHtml - удалить HTML теги
  ##   removeUrls - удалить URL адреса
  ##   removeEmails - удалить email адреса
  ##   removeExtraWhitespace - удалить лишние пробелы
  ##   removeEmoji - удалить emoji символы
  ##   removeNumbers - удалить цифры
  ##   removePunctuation - удалить пунктуацию
  ##   normalizeQuotes - нормализовать кавычки к стандартным " '
  ##   normalizeDashes - нормализовать тире к стандартному -
  ##   removeControlChars - удалить управляющие символы
  result = text
  
  # Удаляем HTML теги
  if removeHtml:
    result = result.replace(re"<[^>]+>", "")
    result = result.replace(re"&[a-z]+;", " ")  # HTML entities
  
  # Удаляем URLs
  if removeUrls:
    result = result.replace(re"https?://[^\s]+", "")
    result = result.replace(re"www\.[^\s]+", "")
  
  # Удаляем email адреса
  if removeEmails:
    result = result.replace(re"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "")
  
  # Удаляем управляющие символы (кроме \n, \t)
  if removeControlChars:
    var cleaned = ""
    for rune in result.runes:
      let code = int(rune)
      # Оставляем: печатные символы, пробелы, табы, переносы строк
      # ASCII печатные: >= 32
      # Unicode: >= 128 (включая кириллицу и другие языки)
      if code >= 32 or code == 9 or code == 10:
        cleaned.add($rune)
    result = cleaned
  
  # Нормализуем кавычки
  if normalizeQuotes:
    var cleaned = ""
    for rune in result.runes:
      let code = int(rune)
      # Различные типы кавычек → стандартные двойные кавычки
      if code in [0x00AB, 0x00BB, 0x201E, 0x201C, 0x201D, 0x0060, 0x00B4, 
                  0x2032, 0x2033, 0x2034]:  # « » „ " " ` ´ ′ ″ ‴
        cleaned.add("\"")
      # Апострофы и одинарные кавычки → стандартные одинарные
      elif code in [0x2018, 0x2019, 0x201A, 0x201B, 0x02BB, 0x02BC, 
                    0x02CA, 0x02CB]:  # ' ' ‚ ‛ ʻ ʼ ˊ ˋ
        cleaned.add("'")
      else:
        cleaned.add($rune)
    result = cleaned
  
  # Нормализуем тире и дефисы
  if normalizeDashes:
    var cleaned = ""
    for rune in result.runes:
      let code = int(rune)
      # Все виды тире → стандартный дефис
      if code in [0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2015, 0x2212]:  # ‐ ‑ ‒ – — ― −
        cleaned.add("-")
      else:
        cleaned.add($rune)
    result = cleaned
  
  # Удаляем emoji (основные диапазоны Unicode)
  if removeEmoji:
    var cleaned = ""
    for rune in result.runes:
      let code = int(rune)
      # Пропускаем основные диапазоны emoji
      if not ((code >= 0x1F600 and code <= 0x1F64F) or  # Emoticons
              (code >= 0x1F300 and code <= 0x1F5FF) or  # Misc Symbols and Pictographs
              (code >= 0x1F680 and code <= 0x1F6FF) or  # Transport and Map
              (code >= 0x2600 and code <= 0x26FF) or    # Misc symbols
              (code >= 0x2700 and code <= 0x27BF) or    # Dingbats
              (code >= 0xFE00 and code <= 0xFE0F) or    # Variation Selectors
              (code >= 0x1F900 and code <= 0x1F9FF) or  # Supplemental Symbols
              (code >= 0x1FA70 and code <= 0x1FAFF)):   # Symbols and Pictographs Extended-A
        cleaned.add($rune)
    result = cleaned
  
  # Удаляем цифры
  if removeNumbers:
    result = result.replace(re"\d+", "")
  
  # Удаляем пунктуацию
  if removePunctuation:
    # Список пунктуационных символов
    const punctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    var cleaned = ""
    for ch in result:
      if ch notin punctuation:
        cleaned.add(ch)
      else:
        cleaned.add(' ')
    result = cleaned
  
  # Удаляем лишние пробелы
  if removeExtraWhitespace:
    result = result.strip()
    result = result.replace(re"\s+", " ")
    result = result.replace(re"\n\s*\n+", "\n\n")  # Множественные переносы → двойной

proc normalizeText*(text: string, lowercase: bool = false): string =
  ## Нормализует текст с поддержкой Unicode
  result = text
  result = result.strip()
  result = result.replace(re"\s+", " ")
  
  if lowercase:
    result = toLowerUnicode(result)

proc splitIntoWords*(text: string): seq[string] =
  ## Разбивает текст на слова
  result = text.split(Whitespace)
  result = result.filterIt(it.len > 0)

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
  ## Обрезает текст до указанной длины
  if text.len <= maxLength:
    return text
  
  if addEllipsis and maxLength >= 3:
    return text[0..<(maxLength - 3)] & "..."
  else:
    return text[0..<maxLength]

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
    ("Ý", "Y"), ("Ñ", "N"), ("Ç", "C")
  ]
  
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
    # Заменяем все пробелы (кроме \n) на обычные пробелы
    result = result.replace('\t', ' ')
    result = result.replace('\r', ' ')
    # Убираем множественные пробелы
    result = result.replace(re" +", " ")
    # Убираем пробелы в начале/конце строк
    var lines = result.split('\n')
    for line in lines.mitems:
      line = line.strip()
    result = lines.join("\n")
  else:
    # Все пробелы → один обычный пробел
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
      if not ch.isSpaceAscii():
        inc count
    return count
  else:
    return text.len

proc countSentences*(text: string): int =
  ## Подсчитывает количество предложений
  splitIntoSentences(text).len

#==============================================================================
# СПЕЦИАЛЬНЫЕ ТОКЕНЫ
#==============================================================================

proc initSpecialTokens*(kind: TokenizerKind): SpecialTokens =
  ## Инициализирует специальные токены в зависимости от типа токенизатора
  case kind
  of tkBPE:
    result = SpecialTokens(
      padToken: "<pad>",
      unkToken: "<unk>",
      bosToken: "<s>",
      eosToken: "</s>",
      sepToken: "<sep>",
      clsToken: "<cls>",
      maskToken: "<mask>"
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

proc addSpecialTokensToVocab(tokenizer: var Tokenizer) =
  ## Добавляет специальные токены в словарь
  var tokenId = 0

  let tokens = [
    tokenizer.specialTokens.padToken,
    tokenizer.specialTokens.unkToken,
    tokenizer.specialTokens.bosToken,
    tokenizer.specialTokens.eosToken,
    tokenizer.specialTokens.sepToken,
    tokenizer.specialTokens.clsToken,
    tokenizer.specialTokens.maskToken
  ]

  for token in tokens:
    if token notin tokenizer.vocab:
      tokenizer.vocab[token] = tokenId
      tokenizer.inverseVocab[tokenId] = token
      tokenizer.specialTokenIds[token] = tokenId
      inc tokenId

proc getPadTokenId*(tokenizer: Tokenizer): int =
  tokenizer.specialTokenIds[tokenizer.specialTokens.padToken]

proc getUnkTokenId*(tokenizer: Tokenizer): int =
  tokenizer.specialTokenIds[tokenizer.specialTokens.unkToken]

proc getBosTokenId*(tokenizer: Tokenizer): int =
  tokenizer.specialTokenIds[tokenizer.specialTokens.bosToken]

proc getEosTokenId*(tokenizer: Tokenizer): int =
  tokenizer.specialTokenIds[tokenizer.specialTokens.eosToken]

proc getMaskTokenId*(tokenizer: Tokenizer): int =
  tokenizer.specialTokenIds[tokenizer.specialTokens.maskToken]

#==============================================================================
# BPE (BYTE PAIR ENCODING) - С СОХРАНЕНИЕМ ПРОБЕЛОВ
#==============================================================================

proc countPairs(words: seq[seq[string]]): Table[(string, string), int] =
  result = initTable[(string, string), int]()
  for word in words:
    for i in 0..<(word.len - 1):
      let pair = (word[i], word[i + 1])
      result[pair] = result.getOrDefault(pair, 0) + 1

proc mergePair(words: var seq[seq[string]], pair: (string, string), newToken: string) =
  for word in words.mitems:
    var i = 0
    while i < word.len - 1:
      if word[i] == pair[0] and word[i + 1] == pair[1]:
        word[i] = newToken
        word.delete(i + 1)
      else:
        inc i

proc trainBPE*(corpus: string, vocabSize: int, 
               customSpecialTokens: seq[string] = @[],
               preserveCase: bool = false): Tokenizer =
  result = Tokenizer(kind: tkBPE)
  result.vocab = initTable[string, int]()
  result.inverseVocab = initTable[int, string]()
  result.merges = @[]
  result.specialTokenIds = initTable[string, int]()
  result.specialTokens = initSpecialTokens(tkBPE)
  result.continuingSubwordPrefix = ""
  result.maxInputCharsPerWord = 100
  result.preserveCase = preserveCase
  
  # Добавляем специальные токены
  addSpecialTokensToVocab(result)
  var tokenId = result.vocab.len
  
  # Добавляем пользовательские специальные токены
  for token in customSpecialTokens:
    if token notin result.vocab:
      result.vocab[token] = tokenId
      result.inverseVocab[tokenId] = token
      result.specialTokenIds[token] = tokenId
      inc tokenId
  
  # Заменяем пробелы на специальный символ Ġ (GPT-2 style)
  let text = normalizeText(corpus, lowercase = not preserveCase).replace(" ", "Ġ")
  var words = text.split("Ġ").filterIt(it.len > 0)
  
  # Инициализируем символами
  var charWords = newSeq[seq[string]]()
  var charSet = initHashSet[string]()
  
  # Добавляем Ġ в набор символов
  incl(charSet, "Ġ")
  
  for word in words:
    var charSeq = @["Ġ"]  # Начинаем с маркера пробела
    for rune in runes(word):
      let ch = $rune
      charSeq.add(ch)
      charSet.incl(ch)
    if charSeq.len > 1:  # Только если есть что-то кроме Ġ
      charWords.add(charSeq)
  
  # Добавляем символы в словарь
  for ch in charSet:
    if ch notin result.vocab:
      result.vocab[ch] = tokenId
      result.inverseVocab[tokenId] = ch
      inc tokenId
  
  echo "BPE: Начальный размер словаря: ", len(result.vocab)
  echo "BPE: Целевой размер: ", vocabSize
  
  var iteration = 0
  while result.vocab.len < vocabSize:
    let pairCounts = countPairs(charWords)
    
    if pairCounts.len == 0:
      break
    
    var maxPair: (string, string)
    var maxCount = 0
    for pair, count in pairCounts:
      if count > maxCount:
        maxCount = count
        maxPair = pair
    
    if maxCount == 0:
      break
    
    let newToken = maxPair[0] & maxPair[1]
    
    if newToken in result.vocab:
      continue
    
    result.vocab[newToken] = tokenId
    result.inverseVocab[tokenId] = newToken
    result.merges.add((pair: maxPair, newToken: newToken, priority: iteration))
    
    mergePair(charWords, maxPair, newToken)
    
    inc tokenId
    inc iteration
    
    if iteration mod 100 == 0:
      echo "BPE: Итерация ", iteration, ", размер словаря: ", result.vocab.len
  
  echo "BPE: Обучение завершено. Финальный размер: ", result.vocab.len

proc encodeBPE*(tokenizer: Tokenizer, text: string): seq[int] =
  result = @[]
  
  let normalizedText = normalizeText(text, lowercase = not tokenizer.preserveCase)
  
  # Заменяем пробелы на Ġ
  let processedText = normalizedText.replace(" ", "Ġ")
  var words = processedText.split("Ġ").filterIt(it.len > 0)
  
  for word in words:
    var tokens = @["Ġ"]  # Начинаем с маркера пробела
    for rune in word.runes:
      tokens.add($rune)
    
    # Применяем все слияния
    for merge in tokenizer.merges:
      var i = 0
      while i < tokens.len - 1:
        if tokens[i] == merge.pair[0] and tokens[i + 1] == merge.pair[1]:
          tokens[i] = merge.newToken
          tokens.delete(i + 1)
        else:
          inc i
    
    # Конвертируем в ID
    for token in tokens:
      if token in tokenizer.vocab:
        result.add(tokenizer.vocab[token])
      else:
        result.add(tokenizer.getUnkTokenId())

#==============================================================================
# WORDPIECE - С ОПЦИЕЙ СОХРАНЕНИЯ РЕГИСТРА
#==============================================================================

proc trainWordPiece*(corpus: string, vocabSize: int, 
                     customSpecialTokens: seq[string] = @[],
                     preserveCase: bool = false): Tokenizer =
  result = Tokenizer(kind: tkWordPiece)
  result.vocab = initTable[string, int]()
  result.inverseVocab = initTable[int, string]()
  result.specialTokenIds = initTable[string, int]()
  result.specialTokens = initSpecialTokens(tkWordPiece)
  result.continuingSubwordPrefix = "##"
  result.maxInputCharsPerWord = 100
  result.preserveCase = preserveCase
  
  # Добавляем специальные токены
  addSpecialTokensToVocab(result)
  var tokenId = result.vocab.len
  
  # Пользовательские специальные токены
  for token in customSpecialTokens:
    if token notin result.vocab:
      result.vocab[token] = tokenId
      result.inverseVocab[tokenId] = token
      result.specialTokenIds[token] = tokenId
      inc tokenId
  
  let text = normalizeText(corpus, lowercase = not preserveCase)
  let words = splitIntoWords(text)
  
  var charSet = initHashSet[string]()
  for word in words:
    for rune in word.runes:
      charSet.incl($rune)
  
  # Добавляем символы
  for ch in charSet:
    if ch notin result.vocab:
      result.vocab[ch] = tokenId
      result.inverseVocab[tokenId] = ch
      inc tokenId
  
  var subwordCounts = initTable[string, int]()
  
  for word in words:
    let runes = toSeq(word.runes)
    
    if runes.len > 0:
      let first = $runes[0]
      subwordCounts[first] = subwordCounts.getOrDefault(first, 0) + 1
    
    for start in 1..<runes.len:
      for length in 1..min(6, runes.len - start):
        var subword = result.continuingSubwordPrefix
        for i in 0..<length:
          subword.add($runes[start + i])
        subwordCounts[subword] = subwordCounts.getOrDefault(subword, 0) + 1
  
  var subwordList = newSeq[tuple[word: string, count: int]]()
  for word, count in subwordCounts:
    subwordList.add((word: word, count: count))
  
  subwordList.sort(proc(a, b: auto): int = cmp(b.count, a.count))
  
  for item in subwordList:
    if result.vocab.len >= vocabSize:
      break
    if item.word notin result.vocab:
      result.vocab[item.word] = tokenId
      result.inverseVocab[tokenId] = item.word
      inc tokenId
  
  echo "WordPiece: Обучен. Размер словаря: ", result.vocab.len

proc encodeWordPiece*(tokenizer: Tokenizer, text: string, 
                      addSpecialTokens: bool = false): seq[int] =
  result = @[]
  
  if addSpecialTokens:
    result.add(tokenizer.getBosTokenId())
  
  let normalizedText = normalizeText(text, lowercase = not tokenizer.preserveCase)
  let words = splitIntoWords(normalizedText)
  
  for word in words:
    if word.runeLen > tokenizer.maxInputCharsPerWord:
      result.add(tokenizer.getUnkTokenId())
      continue
    
    var isBad = false
    var start = 0
    var subTokens: seq[string] = @[]
    let runes = toSeq(word.runes)
    
    while start < runes.len:
      var endIdx = runes.len
      var curSubstr = ""
      var found = false
      
      while start < endIdx:
        var substr = if start > 0: tokenizer.continuingSubwordPrefix else: ""
        for i in start..<endIdx:
          substr.add($runes[i])
        
        if substr in tokenizer.vocab:
          curSubstr = substr
          found = true
          break
        
        dec endIdx
      
      if not found:
        isBad = true
        break
      
      subTokens.add(curSubstr)
      start = endIdx
    
    if isBad:
      result.add(tokenizer.getUnkTokenId())
    else:
      for token in subTokens:
        result.add(tokenizer.vocab[token])
  
  if addSpecialTokens:
    result.add(tokenizer.getEosTokenId())

#==============================================================================
# SENTENCEPIECE
#==============================================================================

proc trainSentencePiece*(corpus: string, vocabSize: int, 
                         customSpecialTokens: seq[string] = @[]): Tokenizer =
  result = Tokenizer(kind: tkSentencePiece)
  result.vocab = initTable[string, int]()
  result.inverseVocab = initTable[int, string]()
  result.specialTokenIds = initTable[string, int]()
  result.scores = initTable[string, float]()
  result.specialTokens = initSpecialTokens(tkSentencePiece)
  result.byteFallback = true
  result.preserveCase = true  # SentencePiece всегда сохраняет регистр
  
  # Добавляем специальные токены
  addSpecialTokensToVocab(result)
  var tokenId = result.vocab.len
  
  for token in customSpecialTokens:
    if token notin result.vocab:
      result.vocab[token] = tokenId
      result.inverseVocab[tokenId] = token
      result.specialTokenIds[token] = tokenId
      result.scores[token] = 0.0
      inc tokenId

  # Заменяем пробелы на ▁
  let text = corpus.replace(" ", "▁")

  var substringCounts = initTable[string, int]()

  let maxSubstringLen = 10
  let runes = toSeq(text.runes)

  for i in 0..<runes.len:
    for length in 1..min(maxSubstringLen, runes.len - i):
      var substring = ""
      for j in 0..<length:
        if i + j < runes.len:
          substring.add($runes[i + j])
      
      if substring.len > 0:
        substringCounts[substring] = substringCounts.getOrDefault(substring, 0) + 1

  let totalCount = substringCounts.values.toSeq.foldl(a + b, 0)

  var candidates = newSeq[tuple[token: string, score: float]]()
  for token, count in substringCounts:
    let score = ln(count.float / totalCount.float)
    candidates.add((token: token, score: score))

  candidates.sort(proc(a, b: auto): int = cmp(b.score, a.score))

  for item in candidates:
    if result.vocab.len >= vocabSize:
      break
    if item.token notin result.vocab:
      result.vocab[item.token] = tokenId
      result.inverseVocab[tokenId] = item.token
      result.scores[item.token] = item.score
      inc tokenId

  echo "SentencePiece: Обучен. Размер словаря: ", result.vocab.len

proc viterbiEncode(tokenizer: Tokenizer, text: string): seq[string] =
  let runes = toSeq(text.runes)
  let n = runes.len

  if n == 0:
    return @[]

  var bestScore = newSeq[float](n + 1)
  var bestToken = newSeq[string](n + 1)

  for i in 0..n:
    bestScore[i] = -1e10

  bestScore[0] = 0.0

  for i in 0..<n:
    for length in 1..min(10, n - i):
      var token = ""
      for j in 0..<length:
        token.add($runes[i + j])
      
      if token in tokenizer.vocab:
        let score = bestScore[i] + tokenizer.scores.getOrDefault(token, -10.0)
        if score > bestScore[i + length]:
          bestScore[i + length] = score
          bestToken[i + length] = token

  result = @[]
  var pos = n
  while pos > 0:
    let token = bestToken[pos]
    if token.len == 0:
      if pos > 0:
        result.insert($runes[pos - 1], 0)
        dec pos
    else:
      result.insert(token, 0)
      pos -= token.runeLen

proc encodeSentencePiece*(tokenizer: Tokenizer, text: string, 
                          addSpecialTokens: bool = false): seq[int] =
  result = @[]

  if addSpecialTokens:
    result.add(tokenizer.getBosTokenId())

  let processedText = text.replace(" ", "▁")
  let tokens = viterbiEncode(tokenizer, processedText)

  for token in tokens:
    if token in tokenizer.vocab:
      result.add(tokenizer.vocab[token])
    else:
      result.add(tokenizer.getUnkTokenId())

  if addSpecialTokens:
    result.add(tokenizer.getEosTokenId())

#==============================================================================
# УНИВЕРСАЛЬНАЯ ФУНКЦИЯ ТОКЕНИЗАЦИИ
#==============================================================================

proc tokenize*(text: string, tokenizer: Tokenizer, flag: int = 2, 
               addSpecialTokens: bool = false): seq[int] =
  case flag
  of 0:
    if tokenizer.kind != tkBPE:
      raise newException(ValueError, "Токенизатор должен быть обучен как BPE")
    result = encodeBPE(tokenizer, text)
    if addSpecialTokens:
      result.insert(tokenizer.getBosTokenId(), 0)
      result.add(tokenizer.getEosTokenId())
  of 1:
    if tokenizer.kind != tkWordPiece:
      raise newException(ValueError, "Токенизатор должен быть обучен как WordPiece")
    return encodeWordPiece(tokenizer, text, addSpecialTokens)
  of 2:
    if tokenizer.kind != tkSentencePiece:
      raise newException(ValueError, "Токенизатор должен быть обучен как SentencePiece")
    return encodeSentencePiece(tokenizer, text, addSpecialTokens)
  else:
    raise newException(ValueError, "Неверный flag: 0 (BPE), 1 (WordPiece), 2 (SentencePiece)")

#==============================================================================
# БАТЧ-ОБРАБОТКА
#==============================================================================

proc padSequence*(sequences: seq[seq[int]], maxLength: int, padValue: int): seq[seq[int]] =
  ## Дополняет последовательности до maxLength
  result = newSeq[seq[int]]()
  for seq in sequences:
    var paddedSeq = seq
    while paddedSeq.len < maxLength:
      paddedSeq.add(padValue)
    if paddedSeq.len > maxLength:
      paddedSeq = paddedSeq[0..<maxLength]
    result.add(paddedSeq)

proc truncateSequence*(sequences: seq[seq[int]], maxLength: int): seq[seq[int]] =
  ## Обрезает последовательности до maxLength
  result = newSeq[seq[int]]()
  for seq in sequences:
    if seq.len > maxLength:
      result.add(seq[0..<maxLength])
    else:
      result.add(seq)

proc createAttentionMask*(sequences: seq[seq[int]], padTokenId: int): seq[seq[int]] =
  ## Создаёт маску внимания (1 для реальных токенов, 0 для padding)
  result = newSeq[seq[int]]()
  for seq in sequences:
    var mask = newSeq[int]()
    for token in seq:
      if token == padTokenId:
        mask.add(0)
      else:
        mask.add(1)
    result.add(mask)

proc encodeBatch*(tokenizer: Tokenizer, texts: seq[string], 
                  maxLength: int = 512, 
                  padding: bool = true,
                  truncation: bool = true,
                  addSpecialTokens: bool = true,
                  returnAttentionMask: bool = true,
                  returnTokenTypeIds: bool = false): BatchEncoding =
  ## Батч-кодирование текстов
  result.inputIds = newSeq[seq[int]]()
  result.lengths = newSeq[int]()
  
  # Кодируем все тексты
  for text in texts:
    let tokens = tokenize(text, tokenizer, ord(tokenizer.kind), addSpecialTokens)
    result.inputIds.add(tokens)
    result.lengths.add(tokens.len)
  
  # Обрезка
  if truncation:
    result.inputIds = truncateSequence(result.inputIds, maxLength)
    for i in 0..<result.lengths.len:
      if result.lengths[i] > maxLength:
        result.lengths[i] = maxLength
  
  # Padding
  if padding:
    let padTokenId = tokenizer.getPadTokenId()
    result.inputIds = padSequence(result.inputIds, maxLength, padTokenId)
  
  # Маска внимания
  if returnAttentionMask:
    result.attentionMask = createAttentionMask(result.inputIds, tokenizer.getPadTokenId())
  
  # Token type IDs (для задач с парами предложений)
  if returnTokenTypeIds:
    result.tokenTypeIds = newSeq[seq[int]]()
    for seq in result.inputIds:
      result.tokenTypeIds.add(newSeq[int](seq.len))  # Все нули для одного предложения


#==============================================================================
# ДЕКОДИРОВАНИЕ
#==============================================================================

proc decode*(tokenizer: Tokenizer, tokenIds: seq[int], 
             skipSpecialTokens: bool = false): string =
  ## Декодирует последовательность ID обратно в текст
  result = ""
  
  for id in tokenIds:
    if id in tokenizer.inverseVocab:
      let token = tokenizer.inverseVocab[id]
      
      # Пропускаем специальные токены если нужно
      if skipSpecialTokens:
        var isSpecial = false
        for _, specialId in tokenizer.specialTokenIds:
          if id == specialId:
            isSpecial = true
            break
        if isSpecial:
          continue
      
      case tokenizer.kind
      of tkWordPiece:
        if token.startsWith(tokenizer.continuingSubwordPrefix):
          result.add(token[2..^1])
        else:
          if result.len > 0 and not result.endsWith(" "):
            result.add(" ")
          result.add(token)
      
      of tkSentencePiece:
        result.add(token.replace("▁", " "))
      
      of tkBPE:
        result.add(token.replace("Ġ", " "))
    else:
      if not skipSpecialTokens:
        result.add("<unk>")
  
  # Очистка
  if tokenizer.kind == tkSentencePiece or tokenizer.kind == tkBPE:
    result = result.strip()


proc decodeBatch*(tokenizer: Tokenizer, batchEncoding: BatchEncoding, 
                  skipSpecialTokens: bool = true): seq[string] =
  ## Декодирует батч токенов обратно в тексты
  result = newSeq[string]()
  for tokens in batchEncoding.inputIds:
    result.add(tokenizer.decode(tokens, skipSpecialTokens))



#==============================================================================
# МЕТРИКИ КАЧЕСТВА
#==============================================================================

proc calculateCompressionRatio*(tokenizer: Tokenizer, text: string): float =
  ## Вычисляет коэффициент сжатия (символы / токены)
  let tokens = tokenize(text, tokenizer, ord(tokenizer.kind))
  if tokens.len == 0:
    return 0.0
  result = text.len.float / tokens.len.float

proc calculateAvgTokensPerWord*(tokenizer: Tokenizer, text: string): float =
  ## Средее количество токенов на слово
  let words = splitIntoWords(text)
  if words.len == 0:
    return 0.0
  let tokens = tokenize(text, tokenizer, ord(tokenizer.kind))
  result = tokens.len.float / words.len.float

proc calculateVocabUtilization*(tokenizer: Tokenizer, corpus: string): float =
  ## Процент использованного словаря на корпусе
  var usedTokens = initHashSet[int]()
  let tokens = tokenize(corpus, tokenizer, ord(tokenizer.kind))
  for token in tokens:
    usedTokens.incl(token)
  result = (usedTokens.len.float / tokenizer.vocab.len.float) * 100.0

proc calculateUnkTokenRate*(tokenizer: Tokenizer, text: string): float =
  ## Процент неизвестных токенов
  let tokens = tokenize(text, tokenizer, ord(tokenizer.kind))
  if tokens.len == 0:
    return 0.0
  
  let unkId = tokenizer.getUnkTokenId()
  var unkCount = 0
  for token in tokens:
    if token == unkId:
      inc unkCount
  
  result = (unkCount.float / tokens.len.float) * 100.0

proc benchmark*(tokenizer: Tokenizer, texts: seq[string]): float =
  ## Измеряет скорость токенизации (токенов в секунду)
  let startTime = cpuTime()
  var totalTokens = 0
  
  for text in texts:
    let tokens = tokenize(text, tokenizer, ord(tokenizer.kind))
    totalTokens += tokens.len
  
  let elapsed = cpuTime() - startTime
  result = totalTokens.float / elapsed

proc getMetrics*(tokenizer: Tokenizer, testCorpus: string, 
                 benchmarkTexts: seq[string] = @[]): TokenizerMetrics =
  ## Собирает все метрики токенизатора
  result.vocabSize = tokenizer.vocab.len
  result.compressionRatio = calculateCompressionRatio(tokenizer, testCorpus)
  result.avgTokensPerWord = calculateAvgTokensPerWord(tokenizer, testCorpus)
  result.vocabUtilization = calculateVocabUtilization(tokenizer, testCorpus)
  result.unkTokenRate = calculateUnkTokenRate(tokenizer, testCorpus)
  
  if benchmarkTexts.len > 0:
    result.tokensPerSecond = benchmark(tokenizer, benchmarkTexts)
  else:
    result.tokensPerSecond = 0.0

proc printMetrics*(metrics: TokenizerMetrics) =
  ## Красивый вывод метрик
  echo repeat("=", 60)
  echo "МЕТРИКИ ТОКЕНИЗАТОРА"
  echo repeat("=", 60)
  echo "Размер словаря:           ", metrics.vocabSize
  echo "Коэффициент сжатия:       ", metrics.compressionRatio.formatFloat(ffDecimal, 2), " символов/токен"
  echo "Токенов на слово:         ", metrics.avgTokensPerWord.formatFloat(ffDecimal, 2)
  echo "Использование словаря:    ", metrics.vocabUtilization.formatFloat(ffDecimal, 2), "%"
  echo "Процент <unk> токенов:    ", metrics.unkTokenRate.formatFloat(ffDecimal, 2), "%"
  if metrics.tokensPerSecond > 0:
    echo "Скорость токенизации:     ", metrics.tokensPerSecond.formatFloat(ffDecimal, 0), " токенов/сек"
  echo repeat("=", 60)

#==============================================================================
# ДОПОЛНИТЕЛЬНЫЕ ФУНКЦИИ ТОКЕНИЗАЦИИ
#==============================================================================

proc getTokenById*(tokenizer: Tokenizer, id: int): string =
  ## Возвращает токен по его ID
  if id in tokenizer.inverseVocab:
    return tokenizer.inverseVocab[id]
  else:
    return tokenizer.specialTokens.unkToken

proc getIdByToken*(tokenizer: Tokenizer, token: string): int =
  ## Возвращает ID токена
  if token in tokenizer.vocab:
    return tokenizer.vocab[token]
  else:
    return tokenizer.getUnkTokenId()

proc hasToken*(tokenizer: Tokenizer, token: string): bool =
  ## Проверяет, есть ли токен в словаре
  token in tokenizer.vocab

proc getVocabSize*(tokenizer: Tokenizer): int =
  ## Возвращает размер словаря
  tokenizer.vocab.len

proc getVocabTokens*(tokenizer: Tokenizer): seq[string] =
  ## Возвращает все токены из словаря
  result = @[]
  for token in tokenizer.vocab.keys:
    result.add(token)

proc filterTokensByFrequency*(tokenizer: Tokenizer, text: string, 
                               minFrequency: int = 1): seq[(string, int)] =
  ## Возвращает токены и их частоты в тексте
  let tokens = tokenize(text, tokenizer, flag = ord(tokenizer.kind))
  var freqs = initTable[string, int]()
  
  for tokenId in tokens:
    let token = tokenizer.getTokenById(tokenId)
    freqs[token] = freqs.getOrDefault(token, 0) + 1
  
  result = @[]
  for token, freq in freqs:
    if freq >= minFrequency:
      result.add((token, freq))
  
  result.sort(proc(a, b: (string, int)): int = b[1] - a[1])

proc compareTokenizers*(text: string, tokenizers: seq[Tokenizer]): 
                       seq[tuple[name: string, tokens: int, avgLen: float]] =
  ## Сравнивает эффективность различных токенизаторов
  result = @[]
  
  for i, tok in tokenizers:
    let tokens = tokenize(text, tok, flag = ord(tok.kind))
    let avgLen = if tokens.len > 0: float(text.len) / float(tokens.len) else: 0.0
    
    let name = case tok.kind
      of tkBPE: "BPE"
      of tkWordPiece: "WordPiece"
      of tkSentencePiece: "SentencePiece"
    
    result.add((name: name, tokens: tokens.len, avgLen: avgLen))

proc tokenizeWithOffsets*(text: string, tokenizer: Tokenizer): 
                         seq[tuple[token: string, start: int, stop: int]] =
  ## Токенизирует текст с возвращением позиций каждого токена
  ## УПРОЩЁННАЯ ВЕРСИЯ - работает только для базовых случаев
  result = @[]
  var position = 0
  
  let tokens = tokenize(text, tokenizer, flag = ord(tokenizer.kind), 
                        addSpecialTokens = false)
  
  for tokenId in tokens:
    let token = tokenizer.getTokenById(tokenId)
    var tokenText = token
    
    # Убираем префиксы для отображения
    if tokenizer.kind == tkBPE:
      tokenText = tokenText.replace("Ġ", " ")
    elif tokenizer.kind == tkWordPiece:
      tokenText = tokenText.replace("##", "")
    elif tokenizer.kind == tkSentencePiece:
      tokenText = tokenText.replace("▁", " ")
    
    let start = position
    let stop = position + tokenText.len
    result.add((token: token, start: start, stop: stop))
    position = stop

proc encodeWithPadding*(tokenizer: Tokenizer, text: string, 
                        maxLength: int, truncate: bool = true): seq[int] =
  ## Кодирует текст с паддингом до указанной длины
  var tokens = tokenize(text, tokenizer, flag = ord(tokenizer.kind), 
                        addSpecialTokens = true)
  
  # Обрезка если нужно
  if truncate and tokens.len > maxLength:
    tokens = tokens[0..<maxLength]
  
  # Паддинг
  let padId = tokenizer.getPadTokenId()
  while tokens.len < maxLength:
    tokens.add(padId)
  
  return tokens

proc createAttentionMask*(tokens: seq[int], padTokenId: int): seq[int] =
  ## Создаёт маску внимания (1 для реальных токенов, 0 для padding)
  result = newSeq[int](tokens.len)
  for i, token in tokens:
    result[i] = if token == padTokenId: 0 else: 1

proc maskTokens*(tokens: seq[int], tokenizer: Tokenizer, 
                 maskProb: float = 0.15): seq[int] =
  ## Маскирует случайные токены (для MLM задач)
  randomize()
  
  result = tokens
  let maskId = tokenizer.getMaskTokenId()
  
  for i in 0..<result.len:
    # Не маскируем специальные токены
    if result[i] in [tokenizer.getPadTokenId(), tokenizer.getBosTokenId(), 
                     tokenizer.getEosTokenId()]:
      continue
    
    if rand(1.0) < maskProb:
      let choice = rand(1.0)
      if choice < 0.8:
        result[i] = maskId  # 80% - заменяем на [MASK]
      elif choice < 0.9:
        result[i] = rand(tokenizer.getVocabSize() - 1)  # 10% - случайный токен
      # 10% - оставляем как есть

proc getSubwordBreakdown*(text: string, tokenizer: Tokenizer): seq[string] =
  ## Показывает разбиение на подслова для анализа
  let tokens = tokenize(text, tokenizer, flag = ord(tokenizer.kind), 
                        addSpecialTokens = false)
  result = @[]
  
  for tokenId in tokens:
    result.add(tokenizer.getTokenById(tokenId))

proc estimateTokenCount*(text: string, avgCharsPerToken: float = 4.0): int =
  ## Оценивает количество токенов без фактической токенизации
  ## avgCharsPerToken зависит от языка: ~4 для английского, ~2-3 для русского
  int(float(text.len) / avgCharsPerToken)

proc validateTokenizer*(tokenizer: Tokenizer): seq[string] =
  ## Проверяет токенизатор на проблемы и возвращает список предупреждений
  result = @[]
  
  # Проверка размера словаря
  if tokenizer.vocab.len == 0:
    result.add("ОШИБКА: Словарь пустой")
  elif tokenizer.vocab.len < 100:
    result.add("ПРЕДУПРЕЖДЕНИЕ: Слишком маленький словарь (< 100 токенов)")
  
  # Проверка специальных токенов
  if tokenizer.getPadTokenId() notin tokenizer.inverseVocab:
    result.add("ОШИБКА: PAD токен не найден в словаре")
  if tokenizer.getUnkTokenId() notin tokenizer.inverseVocab:
    result.add("ОШИБКА: UNK токен не найден в словаре")
  
  # Проверка обратимости (упрощённая - только базовая проверка)
  let testText = "Test text 123"
  let tokens = tokenize(testText, tokenizer, flag = ord(tokenizer.kind))
  let decoded = tokenizer.decode(tokens, skipSpecialTokens = true)
  
  # Нормализуем оба текста для сравнения (убираем маркеры пробелов)
  var normalizedOriginal = testText.replace(" ", "")
  var normalizedDecoded = decoded.replace(" ", "").replace("Ġ", "").replace("▁", "")
  
  # Проверяем только базовую обратимость - все символы на месте
  if normalizedOriginal.len > 0 and normalizedDecoded.len > 0:
    let ratio = float(normalizedDecoded.len) / float(normalizedOriginal.len)
    if ratio < 0.8 or ratio > 1.2:
      result.add("ПРЕДУПРЕЖДЕНИЕ: Токенизация может быть необратимой (длина изменилась значительно)")
  
  if result.len == 0:
    result.add("✓ Токенизатор прошёл все проверки")


#==============================================================================
# СОХРАНЕНИЕ И ЗАГРУЗКА
#==============================================================================

proc saveTokenizer*(tokenizer: Tokenizer, filepath: string) =
  var j = %* {
    "kind": ord(tokenizer.kind),
    "vocab": tokenizer.vocab,
    "specialTokens": {
      "padToken": tokenizer.specialTokens.padToken,
      "unkToken": tokenizer.specialTokens.unkToken,
      "bosToken": tokenizer.specialTokens.bosToken,
      "eosToken": tokenizer.specialTokens.eosToken,
      "sepToken": tokenizer.specialTokens.sepToken,
      "clsToken": tokenizer.specialTokens.clsToken,
      "maskToken": tokenizer.specialTokens.maskToken
    },
    "specialTokenIds": tokenizer.specialTokenIds,
    "continuingSubwordPrefix": tokenizer.continuingSubwordPrefix,
    "maxInputCharsPerWord": tokenizer.maxInputCharsPerWord,
    "preserveCase": tokenizer.preserveCase
  }
  
  if tokenizer.kind == tkBPE:
    var mergesJson = newJArray()
    for merge in tokenizer.merges:
      mergesJson.add(%* {
        "pair": [merge.pair[0], merge.pair[1]],
        "newToken": merge.newToken,
        "priority": merge.priority
      })
    j["merges"] = mergesJson
  
  if tokenizer.kind == tkSentencePiece:
    j["scores"] = %tokenizer.scores
  
  writeFile(filepath, j.pretty())
  echo "Токенизатор сохранён в: ", filepath

proc loadTokenizer*(filepath: string): Tokenizer =
  let content = readFile(filepath)
  let j = parseJson(content)
  
  result = Tokenizer()
  result.kind = TokenizerKind(j["kind"].getInt())
  
  # Загружаем специальные токены
  let st = j["specialTokens"]
  result.specialTokens = SpecialTokens(
    padToken: st["padToken"].getStr(),
    unkToken: st["unkToken"].getStr(),
    bosToken: st["bosToken"].getStr(),
    eosToken: st["eosToken"].getStr(),
    sepToken: st["sepToken"].getStr(),
    clsToken: st["clsToken"].getStr(),
    maskToken: st["maskToken"].getStr()
  )
  
  result.continuingSubwordPrefix = j["continuingSubwordPrefix"].getStr()
  result.maxInputCharsPerWord = j["maxInputCharsPerWord"].getInt()
  result.preserveCase = j["preserveCase"].getBool()
  
  # Загружаем vocab
  result.vocab = initTable[string, int]()
  result.inverseVocab = initTable[int, string]()
  result.specialTokenIds = initTable[string, int]()
  
  for key, val in j["vocab"]:
    let id = val.getInt()
    result.vocab[key] = id
    result.inverseVocab[id] = key
  
  for key, val in j["specialTokenIds"]:
    result.specialTokenIds[key] = val.getInt()
  
  # Загружаем merges для BPE
  if result.kind == tkBPE and j.hasKey("merges"):
    result.merges = @[]
    for item in j["merges"]:
      let pair = (item["pair"][0].getStr(), item["pair"][1].getStr())
      let newToken = item["newToken"].getStr()
      let priority = item["priority"].getInt()
      result.merges.add((pair: pair, newToken: newToken, priority: priority))
  
  # Загружаем scores для SentencePiece
  if result.kind == tkSentencePiece and j.hasKey("scores"):
    result.scores = initTable[string, float]()
    for key, val in j["scores"]:
      result.scores[key] = val.getFloat()
  
  echo "Токенизатор загружен из: ", filepath








#==============================================================================
# ДЕМОНСТРАЦИЯ
#==============================================================================

when isMainModule:
  const FN = "../Тексты и книги/Воскресение (1899).txt"
  # const FN = "../Тексты и книги/Базовый текст.txt"

  echo "╔═══════════════════════════════════════════════════════════╗"
  echo "║       РАСШИРЕННАЯ БИБЛИОТЕКА ТОКЕНИЗАЦИИ ДЛЯ NIM          ║"
  echo "╚═══════════════════════════════════════════════════════════╝"
  echo ""

  let corpus = readFile(FN)
  let testText = "княгиня Софья Васильевна была худая длинная"


  # ========== BPE С СОХРАНЕНИЕМ ПРОБЕЛОВ И РЕГИСТРА ==========
  echo "┌─────────────────────────────────────────────────────────┐"
  echo "│  1. BPE (Byte Pair Encoding) - С ПРОБЕЛАМИ И РЕГИСТРОМ  │"
  echo "└─────────────────────────────────────────────────────────┘"
  let bpeTokenizer = trainBPE(corpus, vocabSize = 500, preserveCase = true)
  let bpeTokens = tokenize(testText, bpeTokenizer, flag = 0, addSpecialTokens = true)
  echo "Текст:         ", testText
  echo "Токены (BPE):  ", bpeTokens
  echo "Декодирование: ", bpeTokenizer.decode(bpeTokens, skipSpecialTokens = true)
  echo ""

  # Метрики BPE
  let bpeMetrics = getMetrics(bpeTokenizer, corpus)
  printMetrics(bpeMetrics)
  echo ""

  # ========== WORDPIECE С РЕГИСТРОМ ==========
  echo "┌─────────────────────────────────────────────────────────┐"
  echo "│  2. WordPiece (BERT-style) - С СОХРАНЕНИЕМ РЕГИСТРА     │"
  echo "└─────────────────────────────────────────────────────────┘"
  let wpTokenizer = trainWordPiece(corpus, vocabSize = 500, preserveCase = true)
  let wpTokens = tokenize(testText, wpTokenizer, flag = 1, addSpecialTokens = true)
  echo "Текст:              ", testText
  echo "Токены (WordPiece): ", wpTokens
  echo "Декодирование:      ", wpTokenizer.decode(wpTokens, skipSpecialTokens = true)
  echo ""

  # Метрики WordPiece
  let wpMetrics = getMetrics(wpTokenizer, corpus)
  printMetrics(wpMetrics)
  echo ""

  # ========== SENTENCEPIECE ==========
  echo "┌─────────────────────────────────────────────────────────┐"
  echo "│  3. SentencePiece (Unigram) - С ПРОБЕЛАМИ И РЕГИСТРОМ   │"
  echo "└─────────────────────────────────────────────────────────┘"
  let spTokenizer = trainSentencePiece(corpus, vocabSize = 500)
  let spTokens = tokenize(testText, spTokenizer, flag = 2, addSpecialTokens = true)
  echo "Текст:                 ", testText
  echo "Токены (SentencePiece):", spTokens
  echo "Декодирование:         ", spTokenizer.decode(spTokens, skipSpecialTokens = true)
  echo ""

  # Метрики SentencePiece
  let spMetrics = getMetrics(spTokenizer, corpus)
  printMetrics(spMetrics)
  echo ""

  # ========== БАТЧ-ОБРАБОТКА ==========
  echo "┌─────────────────────────────────────────────────────────┐"
  echo "│ 4. БАТЧ-ОБРАБОТКА                                       │"
  echo "└─────────────────────────────────────────────────────────┘"
  let batchTexts = @[
    "княгиня Софья Васильевна",
    "доктор с намасленной бородой",
    "Колосов у столика"
  ]

  let batchEncoding = encodeBatch(bpeTokenizer, batchTexts, 
                                   maxLength = 20, 
                                   padding = true,
                                   addSpecialTokens = true,
                                   returnAttentionMask = true)

  echo "Батч текстов:"
  for i, text in batchTexts:
    echo "  [", i, "] ", text
  echo ""
  echo "Input IDs:"
  for i, ids in batchEncoding.inputIds:
    echo "  [", i, "] ", ids
  echo ""
  echo "Attention Mask:"
  for i, mask in batchEncoding.attentionMask:
    echo "  [", i, "] ", mask
  echo ""
  echo "Lengths: ", batchEncoding.lengths
  echo ""

  # Декодирование батча
  echo "Декодированные тексты:"
  let decodedTexts = decodeBatch(bpeTokenizer, batchEncoding, skipSpecialTokens = true)
  for i, text in decodedTexts:
    echo "  [", i, "] ", text
  echo ""


  # ========== СПЕЦИАЛЬНЫЕ ТОКЕНЫ ==========
  echo "┌─────────────────────────────────────────────────────────┐"
  echo "│ 5. СПЕЦИАЛЬНЫЕ ТОКЕНЫ                                   │"
  echo "└─────────────────────────────────────────────────────────┘"
  echo "PAD token:  ", bpeTokenizer.specialTokens.padToken, " (ID: ", bpeTokenizer.getPadTokenId(), ")"
  echo "UNK token:  ", bpeTokenizer.specialTokens.unkToken, " (ID: ", bpeTokenizer.getUnkTokenId(), ")"
  echo "BOS token:  ", bpeTokenizer.specialTokens.bosToken, " (ID: ", bpeTokenizer.getBosTokenId(), ")"
  echo "EOS token:  ", bpeTokenizer.specialTokens.eosToken, " (ID: ", bpeTokenizer.getEosTokenId(), ")"
  echo "MASK token: ", bpeTokenizer.specialTokens.maskToken, " (ID: ", bpeTokenizer.getMaskTokenId(), ")"
  echo ""

  # Сохранение
  saveTokenizer(bpeTokenizer, "bpe_enhanced.json")
  saveTokenizer(wpTokenizer, "wordpiece_enhanced.json")
  saveTokenizer(spTokenizer, "sentencepiece_enhanced.json")

  echo ""
  echo ""

  # ========== ОЧИСТКА ТЕКСТА ==========
  echo "┌─────────────────────────────────────────────────────────┐"
  echo "│ 6. ОЧИСТКА И НОРМАЛИЗАЦИЯ ТЕКСТА                        │"
  echo "└─────────────────────────────────────────────────────────┘"
  
  let dirtyText = """
    Привет! Это <b>HTML</b> текст с URL: https://example.com 
    и email: test@example.com 🎉 с emoji и   лишними   пробелами...
    Также есть "кавычки" и тире—вот так.
  """
  
  echo "Исходный текст:"
  echo dirtyText
  echo ""
  
  echo "После базовой очистки:"
  let cleaned1 = cleanText(dirtyText, 
                          removeHtml = true,
                          removeUrls = true,
                          removeEmails = true,
                          removeEmoji = true)
  echo cleaned1
  echo ""
  
  echo "После полной очистки (без пунктуации):"
  let cleaned2 = cleanText(dirtyText,
                          removeHtml = true,
                          removeUrls = true,
                          removeEmails = true,
                          removeEmoji = true,
                          removePunctuation = true)
  echo cleaned2
  echo ""
  
  # Демонстрация других функций очистки
  echo "Разбиение на предложения:"
  let sentences = splitIntoSentences("Первое предложение. Второе предложение! Третье?")
  for i, sent in sentences:
    echo "  [", i, "] ", sent
  echo ""
  
  echo "Удаление акцентов:"
  echo "  café → ", removeAccents("café")
  echo "  résumé → ", removeAccents("résumé")
  echo ""
  
  echo "Статистика текста:"
  let statsText = "Это пример текста. Он содержит несколько предложений."
  echo "  Слов: ", countWords(statsText)
  echo "  Символов: ", countCharacters(statsText)
  echo "  Символов без пробелов: ", countCharacters(statsText, excludeWhitespace = true)
  echo "  Предложений: ", countSentences(statsText)
  echo ""

  # ========== ДОПОЛНИТЕЛЬНЫЕ УТИЛИТЫ ТОКЕНИЗАЦИИ ==========
  echo "┌─────────────────────────────────────────────────────────┐"
  echo "│ 7. ДОПОЛНИТЕЛЬНЫЕ УТИЛИТЫ ТОКЕНИЗАЦИИ                   │"
  echo "└─────────────────────────────────────────────────────────┘"
  
  echo "Размер словаря BPE: ", bpeTokenizer.getVocabSize()
  echo ""
  
  echo "Разбиение на подслова:"
  let subwords = getSubwordBreakdown(testText, bpeTokenizer)
  echo "  ", testText
  echo "  → ", subwords
  echo ""
  
  echo "Частотный анализ токенов:"
  let freqs = filterTokensByFrequency(bpeTokenizer, corpus, minFrequency = 50)
  echo "  Топ-10 токенов:"
  for i in 0..<min(10, freqs.len):
    echo "    ", i + 1, ". '", freqs[i][0], "' - ", freqs[i][1], " раз"
  echo ""
  
  echo "Сравнение токенизаторов:"
  let comparison = compareTokenizers(testText, @[bpeTokenizer, wpTokenizer, spTokenizer])
  for comp in comparison:
    echo "  ", comp.name, ": ", comp.tokens, " токенов (", 
         comp.avgLen.formatFloat(ffDecimal, 2), " символов/токен)"
  echo ""
  
  echo "Оценка количества токенов (без токенизации):"
  echo "  Текст: ", testText.len, " символов"
  echo "  Оценка: ~", estimateTokenCount(testText, avgCharsPerToken = 3.0), " токенов"
  echo "  Реально: ", bpeTokens.len, " токенов"
  echo ""
  
  echo "Валидация токенизатора:"
  let warnings = validateTokenizer(bpeTokenizer)
  for warning in warnings:
    echo "  • ", warning
  echo ""

  # ========== МАСКИРОВАНИЕ ТОКЕНОВ (MLM) ==========
  echo "┌─────────────────────────────────────────────────────────┐"
  echo "│ 8. МАСКИРОВАНИЕ ТОКЕНОВ (для MLM задач)                 │"
  echo "└─────────────────────────────────────────────────────────┘"
  
  let originalTokens = tokenize(testText, bpeTokenizer, flag = 0, addSpecialTokens = true)
  echo "Исходные токены:"
  echo "  ", originalTokens
  echo "  ", bpeTokenizer.decode(originalTokens, skipSpecialTokens = true)
  echo ""
  
  let maskedTokens = maskTokens(originalTokens, bpeTokenizer, maskProb = 0.3)
  echo "Маскированные токены (30% вероятность):"
  echo "  ", maskedTokens
  echo "  ", bpeTokenizer.decode(maskedTokens, skipSpecialTokens = true)
  echo ""

  echo ""

  echo "╔═══════════════════════════════════════════════════════════╗"
  echo "║               ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!                     ║"
  echo "╚═══════════════════════════════════════════════════════════╝"








# nim c -d:release tokenization.nim
