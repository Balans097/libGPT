# Библиотека TOKENIZATION - Справочник API

**Версия:** 0.7  
**Дата:** 2026-02-01  
**Автор:** github.com/Balans097  

---

## Содержание

1. [Обзор](#обзор)
2. [Инициализация и обучение](#инициализация-и-обучение)
3. [Токенизация](#токенизация)
4. [Декодирование](#декодирование)
5. [Предобработка текста](#предобработка-текста)
6. [Управление словарем](#управление-словарем)
7. [Метрики и анализ](#метрики-и-анализ)
8. [Сохранение и загрузка](#сохранение-и-загрузка)
9. [Пакетная обработка](#пакетная-обработка)
10. [Кэширование](#кэширование)
11. [Потокобезопасность](#потокобезопасность)
12. [Отладка и визуализация](#отладка-и-визуализация)
13. [Вспомогательные функции](#вспомогательные-функции)

---

## Обзор

Библиотека предоставляет полнофункциональный инструментарий для токенизации текста с поддержкой:
- **4 основных алгоритма**: BPE, WordPiece, SentencePiece, Byte-Level BPE
- **130+ функций** для работы с текстом
- **Unicode** полная поддержка
- **Thread-safe** операции
- **Специализированные токенизаторы** для кода, математики, Markdown, JSON
- **Интеграция** с HuggingFace, SentencePiece, TikToken

---

## Инициализация и обучение

### trainBPE
```nim
proc trainBPE*(corpus: seq[string], vocabSize: int = 10000, minFrequency: int = 2): Tokenizer
```
Обучает BPE токенизатор.

**Пример:**
```nim
let corpus = @["Привет мир", "Токенизация текста"]
var bpe = trainBPE(corpus, vocabSize = 5000)
```

---

### trainWordPiece
```nim
proc trainWordPiece*(corpus: seq[string], vocabSize: int = 10000, 
                     continuingSubwordPrefix: string = "##",
                     maxInputCharsPerWord: int = 100): Tokenizer
```
Обучает WordPiece токенизатор (BERT-style).

**Пример:**
```nim
var wp = trainWordPiece(corpus, vocabSize = 5000)
```

---

### trainSentencePiece
```nim
proc trainSentencePiece*(corpus: seq[string], vocabSize: int = 8000,
                          characterCoverage: float = 0.9995): Tokenizer
```
Обучает SentencePiece токенизатор.

**Пример:**
```nim
var sp = trainSentencePiece(corpus, vocabSize = 8000)
```

---

### trainByteLevelBPE
```nim
proc trainByteLevelBPE*(corpus: seq[string], vocabSize: int = 50000,
                         minFrequency: int = 2): Tokenizer
```
Обучает Byte-Level BPE (GPT-2/3 compatible).

**Пример:**
```nim
var blbpe = trainByteLevelBPE(corpus, vocabSize = 50000)
```

---

### createCharacterTokenizer
```nim
proc createCharacterTokenizer*(corpus: seq[string]): Tokenizer
```
Создает символьный токенизатор.

---

### createWhitespaceTokenizer
```nim
proc createWhitespaceTokenizer*(): Tokenizer
```
Создает токенизатор по пробелам.

---

### fromPretrainedModel
```nim
proc fromPretrainedModel*(modelName: string): Tokenizer
```
Загружает предобученную модель ("gpt2", "bert", и т.д.).

**Пример:**
```nim
var bertTok = fromPretrainedModel("bert")
```

---

## Токенизация

### tokenize
```nim
proc tokenize*(text: string, tokenizer: var Tokenizer, 
               flag: int = 0, addSpecialTokens: bool = true): seq[int]
```
Основная функция токенизации.

**Пример:**
```nim
let tokens = tokenize("Привет мир", tok, addSpecialTokens = true)
```

---

### tokenizeWithOffsets
```nim
proc tokenizeWithOffsets*(text: string, tokenizer: var Tokenizer,
                           addSpecialTokens: bool = true): seq[TokenOffset]
```
Токенизация с позициями (для NER/QA).

**Пример:**
```nim
let offsets = tokenizeWithOffsets("текст", tok)
for offset in offsets:
  echo "Token: ", offset.token, " at ", offset.startChar, "-", offset.endChar
```

---

### streamTokenize
```nim
iterator streamTokenize*(filePath: string, tokenizer: var Tokenizer,
                         chunkSize: int = 8192, addSpecialTokens: bool = true): seq[int]
```
Потоковая токенизация больших файлов.

**Пример:**
```nim
for chunk in streamTokenize("large.txt", tok):
  processChunk(chunk)
```

---

### tokenizeWithDropout
```nim
proc tokenizeWithDropout*(text: string, tokenizer: var Tokenizer,
                           dropoutProb: float = 0.1, seed: int = -1,
                           minDropped: int = 1): seq[int]
```
BPE-dropout для data augmentation.

**Пример:**
```nim
let tokens1 = tokenizeWithDropout("текст", bpe, dropoutProb = 0.2)
let tokens2 = tokenizeWithDropout("текст", bpe, dropoutProb = 0.2)
# tokens1 != tokens2 - разные результаты для аугментации
```

---

### tokenizeCode
```nim
proc tokenizeCode*(code: string, tokenizer: var Tokenizer,
                    language: ProgrammingLanguage = plPython): seq[int]
```
Токенизация кода с учетом синтаксиса.

**Пример:**
```nim
let tokens = tokenizeCode(pythonCode, tok, language = plPython)
```

---

### tokenizeMath / tokenizeMarkdown / tokenizeJson
```nim
proc tokenizeMath*(mathExpr: string, tokenizer: var Tokenizer): seq[int]
proc tokenizeMarkdown*(markdown: string, tokenizer: var Tokenizer): seq[int]
proc tokenizeJson*(jsonStr: string, tokenizer: var Tokenizer): seq[int]
```
Специализированные токенизаторы.

---

### encodeWithPadding
```nim
proc encodeWithPadding*(tokenizer: var Tokenizer, text: string,
                         maxLength: int, padTokenId: int = -1): seq[int]
```
Кодирование с padding.

**Пример:**
```nim
let padded = encodeWithPadding(tok, "короткий текст", maxLength = 128)
```

---

### maskTokens
```nim
proc maskTokens*(tokens: seq[int], tokenizer: Tokenizer,
                  maskProb: float = 0.15, seed: int = -1):
                tuple[maskedTokens: seq[int], labels: seq[int]]
```
Маскирование для MLM (BERT-style).

**Пример:**
```nim
let (masked, labels) = maskTokens(tokens, tok, maskProb = 0.15)
```

---

## Декодирование

### decode
```nim
proc decode*(tokenizer: Tokenizer, tokens: seq[int],
              skipSpecialTokens: bool = false): string
```
Декодирование токенов в текст.

**Пример:**
```nim
let text = decode(tok, tokens, skipSpecialTokens = true)
```

---

### decodeBatch
```nim
proc decodeBatch*(tokenizer: Tokenizer, encoding: BatchEncoding,
                   skipSpecialTokens: bool = false): seq[string]
```
Декодирование батча.

---

### byteLevelDecode
```nim
proc byteLevelDecode*(bytes: seq[int]): string
```
Декодирование байтов в текст.

---

## Предобработка текста

### cleanText
```nim
proc cleanText*(text: string, removeHtml: bool = true,
                 removeUrls: bool = true, removeEmails: bool = true,
                 removeExtraWhitespace: bool = true, removeEmoji: bool = false,
                 removeNumbers: bool = false, removePunctuation: bool = false,
                 normalizeQuotes: bool = true, normalizeDashes: bool = true,
                 removeControlChars: bool = true): string
```
Универсальная очистка текста.

**Пример:**
```nim
let clean = cleanText(dirty, removeHtml = true, removeUrls = true,
                       removeEmoji = true)
```

---

### normalizeText
```nim
proc normalizeText*(text: string): string
```
Базовая нормализация (HTML, кавычки, пробелы).

---

### fullNormalization
```nim
proc fullNormalization*(text: string): string
```
Полная цепочка нормализации (NFKC + zero-width + whitespace).

---

### normalizeNumbers
```nim
proc normalizeNumbers*(text: string, strategy: NumberNormalizationStrategy): string
```
Нормализация чисел:
- `nsKeepOriginal` - оставить как есть
- `nsReplaceWithToken` - заменить на [NUM]
- `nsReplaceWithDigits` - заменить на разряды
- `nsNormalize` - нормализовать (1,234.56 -> 1234.56)

---

### handleEmojis
```nim
proc handleEmojis*(text: string, strategy: EmojiStrategy): string
```
Обработка эмодзи:
- `esKeep` - сохранить
- `esRemove` - удалить
- `esReplace` - заменить на текст
- `esTokenize` - токенизировать отдельно

---

### removeAccents
```nim
proc removeAccents*(text: string): string
```
Удаление диакритических знаков (café -> cafe).

---

### segmentSentences / splitIntoWords
```nim
proc segmentSentences*(text: string): seq[string]
proc splitIntoWords*(text: string): seq[string]
```
Сегментация текста.

---

### truncateToRunes
```nim
proc truncateToRunes*(text: string, maxRunes: int): string
```
Обрезка до N Unicode символов.

---

## Управление словарем

### pruneVocabulary
```nim
proc pruneVocabulary*(tokenizer: var Tokenizer, corpus: seq[string],
                       minFrequency: int = 5, keepTopN: int = -1,
                       keepSpecialTokens: bool = true): int
```
Удаляет редкие токены.

**Пример:**
```nim
let removed = pruneVocabulary(tok, corpus, minFrequency = 10, keepTopN = 5000)
```

---

### incrementalTrain
```nim
proc incrementalTrain*(tokenizer: var Tokenizer, newCorpus: seq[string],
                        maxNewTokens: int = 1000, minFrequency: int = 2): int
```
Дообучение токенизатора.

---

### addTokens
```nim
proc addTokens*(tokenizer: var Tokenizer, tokens: seq[string])
```
Добавление конкретных токенов.

**Пример:**
```nim
addTokens(tok, @["<SPECIAL>", "<CUSTOM>"])
```

---

### mergeVocabularies
```nim
proc mergeVocabularies*(vocabs: seq[Tokenizer], strategy: MergeStrategy): Tokenizer
```
Объединение словарей:
- `msUnion` - объединить все
- `msIntersection` - только общие
- `msWeighted` - взвешенное объединение

---

### getVocabularyOverlap
```nim
proc getVocabularyOverlap*(t1, t2: Tokenizer): VocabOverlapStats
```
Анализ перекрытия словарей.

**Пример:**
```nim
let overlap = getVocabularyOverlap(tok1, tok2)
echo "Общих: ", overlap.commonTokens
echo "Jaccard: ", overlap.jaccardSimilarity
```

---

### exportVocabulary / importVocabulary
```nim
proc exportVocabulary*(t: Tokenizer, format: VocabFormat): string
proc importVocabulary*(format: VocabFormat, data: string): Tokenizer
```
Экспорт/импорт словаря в различных форматах.

---

## Метрики и анализ

### getMetrics
```nim
proc getMetrics*(tokenizer: var Tokenizer, corpus: seq[string]): TokenizerMetrics
```
Вычисляет метрики токенизатора.

**Пример:**
```nim
let m = getMetrics(tok, corpus)
echo "Vocab size: ", m.vocabSize
echo "Compression ratio: ", m.compressionRatio
echo "Avg tokens/word: ", m.avgTokensPerWord
echo "Vocab utilization: ", m.vocabUtilization
echo "UNK rate: ", m.unkTokenRate
echo "Tokens/sec: ", m.tokensPerSecond
```

---

### analyzeVocabulary
```nim
proc analyzeVocabulary*(tokenizer: Tokenizer, corpus: seq[string]): VocabAnalysis
```
Детальный анализ словаря.

---

### calculatePerplexity
```nim
proc calculatePerplexity*(t: var Tokenizer, text: string,
                           languageModel: Table[string, float]): float
```
Вычисляет perplexity токенизации.

---

### measureSegmentationQuality
```nim
proc measureSegmentationQuality*(t: var Tokenizer,
                                   goldSegments: seq[seq[string]]): QualityMetrics
```
Измеряет качество сегментации (precision, recall, F1).

---

### compareTokenizers
```nim
proc compareTokenizers*(text: string, tokenizers: seq[Tokenizer]):
                         seq[tuple[name: string, tokens: int, time: float]]
```
Сравнивает производительность токенизаторов.

---

### benchmark
```nim
proc benchmark*(tokenizer: var Tokenizer, texts: seq[string]): float
```
Измеряет скорость (токенов/сек).

---

## Сохранение и загрузка

### saveTokenizer / loadTokenizer
```nim
proc saveTokenizer*(tokenizer: Tokenizer, filepath: string)
proc loadTokenizer*(filepath: string): Tokenizer
```
Сохранение/загрузка токенизатора.

---

### saveVersionedTokenizer / loadVersionedTokenizer
```nim
proc saveVersionedTokenizer*(vt: VersionedTokenizer, path: string)
proc loadVersionedTokenizer*(path: string): VersionedTokenizer
```
Версионированное сохранение с метаданными.

**Пример:**
```nim
var vt = VersionedTokenizer(
  tokenizer: tok,
  metadata: createMetadata(tok, trainInfo = "Корпус новостей")
)
saveVersionedTokenizer(vt, "tokenizer.json")
```

---

### toHuggingFaceFormat
```nim
proc toHuggingFaceFormat*(tokenizer: Tokenizer): string
```
Экспорт в формат HuggingFace tokenizer.json.

---

### toSentencePieceModel / toTikTokenFormat
```nim
proc toSentencePieceModel*(tokenizer: Tokenizer): string
proc toTikTokenFormat*(tokenizer: Tokenizer): string
```
Экспорт в другие форматы.

---

## Пакетная обработка

### encodeBatch
```nim
proc encodeBatch*(tokenizer: var Tokenizer, texts: seq[string],
                   maxLength: int = 512, padding: bool = true,
                   truncation: bool = true, addSpecialTokens: bool = true,
                   returnAttentionMask: bool = true,
                   returnTokenTypeIds: bool = false): BatchEncoding
```
Батч-кодирование с padding/truncation.

**Пример:**
```nim
let batch = encodeBatch(tok, @["текст 1", "текст 2"],
                         maxLength = 128, padding = true)
echo "Input IDs: ", batch.inputIds
echo "Attention mask: ", batch.attentionMask
```

---

### padSequence / truncateSequence
```nim
proc padSequence*(sequence: seq[int], maxLength: int, padValue: int = 0): seq[int]
proc truncateSequence*(sequence: seq[int], maxLength: int): seq[int]
```
Утилиты для padding/truncation.

---

### createAttentionMask
```nim
proc createAttentionMask*(inputIds: seq[int], padTokenId: int): seq[int]
```
Создает attention mask.

---

## Кэширование

### initCache / clearCache
```nim
proc initCache*(tokenizer: var Tokenizer, maxSize: int = 10000)
proc clearCache*(tokenizer: var Tokenizer)
```
Управление встроенным кэшем.

---

### newLRUCache
```nim
proc newLRUCache*(maxSize: int = 10000): LRUCache
```
Создание LRU кэша.

**Пример:**
```nim
var cache = newLRUCache(maxSize = 5000)
cache.put("текст", @[1, 2, 3])
let tokens = cache.get("текст")
let stats = cache.getStats()
echo "Hit rate: ", stats.hitRate
```

---

## Потокобезопасность

### newThreadSafeTokenizer
```nim
proc newThreadSafeTokenizer*(tokenizer: Tokenizer): ThreadSafeTokenizer
```
Создает потокобезопасную обертку.

**Пример:**
```nim
let safeTok = newThreadSafeTokenizer(tok)
# Можно безопасно использовать из разных потоков
let tokens = tokenizeThreadSafe(safeTok, "текст")
```

---

### tokenizeThreadSafe / decodeThreadSafe
```nim
proc tokenizeThreadSafe*(tst: ThreadSafeTokenizer, text: string,
                          addSpecialTokens: bool = true): seq[int]
proc decodeThreadSafe*(tst: ThreadSafeTokenizer, tokenIds: seq[int],
                        skipSpecialTokens: bool = true): string
```
Потокобезопасные операции.

---

## Отладка и визуализация

### visualizeTokenization
```nim
proc visualizeTokenization*(text: string, tokenizer: var Tokenizer)
```
Визуализирует токенизацию с цветовым кодированием.

---

### debugTokenization
```nim
proc debugTokenization*(text: string, tokenizer: var Tokenizer): DebugInfo
```
Детальная отладочная информация.

**Пример:**
```nim
let debug = debugTokenization("текст", tok)
echo "Tokens: ", debug.tokens
echo "Boundaries: ", debug.boundaries
echo "Unknown words: ", debug.unknownWords
echo "Warnings: ", debug.warnings
```

---

### explainToken
```nim
proc explainToken*(tokenizer: Tokenizer, token: string): TokenExplanation
```
Объясняет происхождение токена.

---

### findTokenConflicts
```nim
proc findTokenConflicts*(tokenizer: Tokenizer): seq[TokenConflict]
```
Находит конфликты в словаре.

---

### validateTokenizer / validateInput
```nim
proc validateTokenizer*(tokenizer: Tokenizer): Option[string]
proc validateInput*(text: string, maxLength: int = MAX_INPUT_LENGTH): Option[string]
```
Валидация токенизатора и входных данных.

---

## Вспомогательные функции

### getTokenById / getIdByToken / hasToken
```nim
proc getTokenById*(tokenizer: Tokenizer, id: int): string
proc getIdByToken*(tokenizer: Tokenizer, token: string): int
proc hasToken*(tokenizer: Tokenizer, token: string): bool
```
Работа со словарем.

---

### getVocabSize / getVocabTokens
```nim
proc getVocabSize*(tokenizer: Tokenizer): int
proc getVocabTokens*(tokenizer: Tokenizer): seq[string]
```
Информация о словаре.

---

### detectLanguage
```nim
proc detectLanguage*(text: string): string
```
Детекция языка/скрипта ("latin", "cyrillic", "mixed").

---

### countWords / countCharacters / countSentences / runeCount
```nim
proc countWords*(text: string): int
proc countCharacters*(text: string): int
proc countSentences*(text: string): int
proc runeCount*(text: string): int
```
Подсчет различных единиц текста.

---

## Константы и типы

### TokenizerKind
```nim
type TokenizerKind* = enum
  tkBPE = 0
  tkWordPiece = 1
  tkSentencePiece = 2
  tkByteLevelBPE = 3
```

### Основные типы
```nim
type
  Tokenizer* = ref object
    kind*: TokenizerKind
    vocab*: Table[string, int]
    inverseVocab*: seq[string]
    merges*: seq[BPEMerge]
    specialTokens*: SpecialTokens
    # ...
  
  TokenOffset* = object
    token*: string
    tokenId*: int
    startChar*, endChar*: int
    startByte*, endByte*: int
  
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
```

---

## Примеры использования

### Базовый пример
```nim
# Обучение
let corpus = @["Привет мир", "Токенизация текста", "BPE алгоритм"]
var tok = trainBPE(corpus, vocabSize = 5000)

# Токенизация
let tokens = tokenize("Новый текст", tok)
echo tokens

# Декодирование
let text = decode(tok, tokens)
echo text

# Сохранение
saveTokenizer(tok, "my_tokenizer.json")
```

### Продвинутый пример
```nim
# Обучение с метриками
var tok = trainBPE(corpus, vocabSize = 10000)

# Анализ качества
let metrics = getMetrics(tok, testCorpus)
echo "UNK rate: ", metrics.unkTokenRate
echo "Compression: ", metrics.compressionRatio

# Pruning редких токенов
discard pruneVocabulary(tok, corpus, minFrequency = 5)

# Батч-обработка
let batch = encodeBatch(tok, texts, maxLength = 128, padding = true)

# Экспорт в HuggingFace
let hfJson = toHuggingFaceFormat(tok)
writeFile("tokenizer.json", hfJson)
```

### Потокобезопасная обработка
```nim
# Создание потокобезопасного токенизатора
let safeTok = newThreadSafeTokenizer(tok)

# Использование в параллельных потоках
import threadpool

proc processText(text: string) =
  let tokens = tokenizeThreadSafe(safeTok, text)
  # обработка tokens...

parallel:
  for text in largeCorpus:
    spawn processText(text)
```

---

## Лицензия

MIT License

## Контакты

**E-mail**: vasil.minsk@yahoo.com
**GitHub**: github.com/Balans097

---






