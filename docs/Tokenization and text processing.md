
# API Документация: библиотека токенизации и обработки текстов

**Версия:** 0.5  
**Дата:** 2026-01-31  
**Автор:** github.com/Balans097

---

## Содержание

1. [Введение](#введение)
2. [Типы данных](#типы-данных)
3. [Обучение токенизаторов](#обучение-токенизаторов)
4. [Токенизация и декодирование](#токенизация-и-декодирование)
5. [Пакетная обработка](#пакетная-обработка)
6. [Работа с текстом](#работа-с-текстом)
7. [Метрики и анализ](#метрики-и-анализ)
8. [Сохранение и загрузка](#сохранение-и-загрузка)
9. [Утилиты](#утилиты)
10. [Продвинутые функции](#продвинутые-функции)

---

## Введение

Библиотека предоставляет полный набор инструментов для токенизации текста с поддержкой четырёх основных алгоритмов:
- **BPE** (Byte Pair Encoding) - классический алгоритм
- **WordPiece** - используется в BERT
- **SentencePiece** - универсальный алгоритм
- **ByteLevel BPE** - используется в GPT-2/3

### Основные возможности:
- ✅ Полная поддержка Unicode и кириллицы
- ✅ Кэширование для ускорения повторных токенизаций
- ✅ Потоковая обработка больших файлов
- ✅ Отслеживание позиций токенов (для NER/QA)
- ✅ BPE-dropout для регуляризации
- ✅ Экспорт словарей в JSON
- ✅ Пакетная обработка с padding/truncation

---

## Типы данных

### TokenizerKind
```nim
type TokenizerKind* = enum
  tkBPE = 0              # Byte Pair Encoding
  tkWordPiece = 1        # WordPiece (BERT-style)
  tkSentencePiece = 2    # SentencePiece (универсальный)
  tkByteLevelBPE = 3     # ByteLevel BPE (GPT-2/3)
```

### SpecialTokens
```nim
type SpecialTokens* = object
  padToken*: string      # Токен заполнения (padding)
  unkToken*: string      # Неизвестный токен
  bosToken*: string      # Токен начала последовательности
  eosToken*: string      # Токен конца последовательности
  sepToken*: string      # Токен разделителя
  clsToken*: string      # Токен классификации
  maskToken*: string     # Токен маскирования
```

### Tokenizer
```nim
type Tokenizer* = ref object
  kind*: TokenizerKind                    # Тип токенизатора
  vocab*: Table[string, int]              # Словарь: токен -> ID
  inverseVocab*: seq[string]              # Обратный словарь: ID -> токен
  merges*: seq[BPEMerge]                  # BPE merges (только для BPE)
  specialTokens*: SpecialTokens           # Специальные токены
  specialTokenIds*: Table[string, int]    # ID специальных токенов
  maxInputCharsPerWord*: int              # Макс. длина слова (WordPiece)
  continuingSubwordPrefix*: string        # Префикс подслова (обычно "##")
  scores*: Table[string, float]           # Scores (SentencePiece)
  preserveCase*: bool                     # Сохранять регистр
  cache*: Table[string, seq[int]]         # Кэш токенизаций
  cacheMaxSize*: int                      # Размер кэша
  cacheHits*: int                         # Попадания в кэш
  cacheMisses*: int                       # Промахи кэша
  byteEncoder*: Table[int, string]        # Byte encoder (ByteLevel BPE)
  byteDecoder*: Table[string, int]        # Byte decoder (ByteLevel BPE)
```

### TokenOffset
```nim
type TokenOffset* = object
  token*: string         # Текст токена
  tokenId*: int          # ID токена
  startChar*: int        # Начальная позиция (в символах)
  endChar*: int          # Конечная позиция (в символах)
  startByte*: int        # Начальная позиция (в байтах)
  endByte*: int          # Конечная позиция (в байтах)
```

### BatchEncoding
```nim
type BatchEncoding* = object
  inputIds*: seq[seq[int]]        # ID токенов для каждого текста
  attentionMask*: seq[seq[int]]   # Маски внимания
  tokenTypeIds*: seq[seq[int]]    # ID типов токенов
  lengths*: seq[int]              # Длины последовательностей
```

### TokenizerMetrics
```nim
type TokenizerMetrics* = object
  vocabSize*: int              # Размер словаря
  compressionRatio*: float     # Коэффициент сжатия
  avgTokensPerWord*: float     # Среднее кол-во токенов на слово
  vocabUtilization*: float     # Утилизация словаря (0.0-1.0)
  unkTokenRate*: float         # Доля неизвестных токенов
  tokensPerSecond*: float      # Скорость токенизации
```

### VocabAnalysis
```nim
type VocabAnalysis* = object
  vocabSize*: int                                        # Размер словаря
  avgTokenLength*: float                                 # Средняя длина токена
  typeTokenRatio*: float                                 # Type/Token ratio
  coverageRate*: float                                   # Покрытие корпуса
  oovRate*: float                                        # Out-of-vocabulary rate
  mostFrequent*: seq[tuple[token: string, freq: int]]    # Топ токены
  leastFrequent*: seq[tuple[token: string, freq: int]]   # Редкие токены
  lengthDistribution*: CountTable[int]                   # Распределение длин
```

---

## Обучение токенизаторов

### trainBPE
Обучение BPE токенизатора (Byte Pair Encoding).

```nim
proc trainBPE*(
  corpus: seq[string],
  vocabSize: int = 8000,
  minFrequency: int = 2,
  preserveCase: bool = false
): Tokenizer
```

**Параметры:**
- `corpus` - корпус текстов для обучения
- `vocabSize` - желаемый размер словаря (по умолчанию: 8000)
- `minFrequency` - минимальная частота для включения токена (по умолчанию: 2)
- `preserveCase` - сохранять регистр букв (по умолчанию: false)

**Возвращает:** обученный токенизатор

**Пример:**
```nim
let corpus = @["Привет мир", "Токенизация текста"]
var tokenizer = trainBPE(corpus, vocabSize = 1000)
```

---

### trainWordPiece
Обучение WordPiece токенизатора (используется в BERT).

```nim
proc trainWordPiece*(
  corpus: seq[string],
  vocabSize: int = 8000,
  minFrequency: int = 2,
  continuingSubwordPrefix: string = "##",
  preserveCase: bool = false
): Tokenizer
```

**Параметры:**
- `corpus` - корпус текстов для обучения
- `vocabSize` - желаемый размер словаря (по умолчанию: 8000)
- `minFrequency` - минимальная частота (по умолчанию: 2)
- `continuingSubwordPrefix` - префикс для продолжений слов (по умолчанию: "##")
- `preserveCase` - сохранять регистр букв (по умолчанию: false)

**Возвращает:** обученный токенизатор

**Особенности:**
- Поддерживает fallback на lowercase для заглавных букв
- Сохраняет пробелы как отдельные токены
- Оптимизирован для длинных кириллических слов (n-граммы до 15 символов)

**Пример:**
```nim
var tokenizer = trainWordPiece(corpus, vocabSize = 1500, preserveCase = true)
```

---

### trainSentencePiece
Обучение SentencePiece токенизатора.

```nim
proc trainSentencePiece*(
  corpus: seq[string],
  vocabSize: int = 8000,
  characterCoverage: float = 0.9995,
  preserveCase: bool = false
): Tokenizer
```

**Параметры:**
- `corpus` - корпус текстов для обучения
- `vocabSize` - желаемый размер словаря (по умолчанию: 8000)
- `characterCoverage` - покрытие символов (по умолчанию: 0.9995)
- `preserveCase` - сохранять регистр букв (по умолчанию: false)

**Возвращает:** обученный токенизатор

**Особенности:**
- Использует символ ▁ для обозначения пробелов
- Присваивает scores каждому токену
- Хорошо работает с языками без явных границ слов

**Пример:**
```nim
var tokenizer = trainSentencePiece(corpus, vocabSize = 2000)
```

---

### trainByteLevelBPE
Обучение ByteLevel BPE токенизатора (GPT-2/3 стиль).

```nim
proc trainByteLevelBPE*(
  corpus: seq[string],
  vocabSize: int = 8000,
  minFrequency: int = 2
): Tokenizer
```

**Параметры:**
- `corpus` - корпус текстов для обучения
- `vocabSize` - желаемый размер словаря (по умолчанию: 8000)
- `minFrequency` - минимальная частота (по умолчанию: 2)

**Возвращает:** обученный токенизатор

**Особенности:**
- Каждый байт маппится на уникальный Unicode символ
- Гарантирует обработку любого UTF-8 текста без UNK токенов
- Сохраняет пробелы корректно
- Совместим с GPT-2/3

**Пример:**
```nim
var tokenizer = trainByteLevelBPE(corpus, vocabSize = 2000)
```

---

## Токенизация и декодирование

### tokenize
Токенизация текста в последовательность ID токенов.

```nim
proc tokenize*(
  text: string,
  tokenizer: var Tokenizer,
  flag: int = 0,
  addSpecialTokens: bool = false
): seq[int]
```

**Параметры:**
- `text` - текст для токенизации
- `tokenizer` - обученный токенизатор
- `flag` - дополнительный флаг (не используется)
- `addSpecialTokens` - добавлять BOS/EOS токены (по умолчанию: false)

**Возвращает:** последовательность ID токенов

**Особенности:**
- Использует кэширование для ускорения повторных токенизаций
- Для WordPiece поддерживает fallback на lowercase
- Корректно обрабатывает пробелы

**Пример:**
```nim
let text = "Привет мир"
let tokens = tokenize(text, tokenizer, addSpecialTokens = true)
# Результат: @[1, 145, 289, 2]  # [BOS, "Привет", "мир", EOS]
```

---

### decode
Декодирование последовательности ID токенов обратно в текст.

```nim
proc decode*(
  tokenizer: Tokenizer,
  tokens: seq[int],
  skipSpecialTokens: bool = false
): string
```

**Параметры:**
- `tokenizer` - токенизатор
- `tokens` - последовательность ID токенов
- `skipSpecialTokens` - пропускать специальные токены (по умолчанию: false)

**Возвращает:** декодированный текст

**Особенности:**
- Для WordPiece удаляет префикс "##"
- Для SentencePiece заменяет ▁ на пробелы
- Для ByteLevel BPE корректно декодирует UTF-8

**Пример:**
```nim
let decoded = tokenizer.decode(tokens, skipSpecialTokens = true)
# Результат: "Привет мир"
```

---

### tokenizeWithOffsets
Токенизация с отслеживанием позиций токенов.

```nim
proc tokenizeWithOffsets*(
  text: string,
  tokenizer: var Tokenizer,
  addSpecialTokens: bool = true
): seq[TokenOffset]
```

**Параметры:**
- `text` - текст для токенизации
- `tokenizer` - токенизатор
- `addSpecialTokens` - добавлять специальные токены (по умолчанию: true)

**Возвращает:** последовательность `TokenOffset` объектов с позициями

**Применение:**
- Named Entity Recognition (NER)
- Question Answering (QA)
- Извлечение информации
- Выравнивание токенов с исходным текстом

**Пример:**
```nim
let offsets = tokenizeWithOffsets("Москва - столица", tokenizer)
for offset in offsets:
  echo offset.token, " -> chars: ", offset.startChar, "..", offset.endChar
# Результат:
# "Москва" -> chars: 0..6
# " " -> chars: 6..7
# "-" -> chars: 7..8
# " " -> chars: 8..9
# "столица" -> chars: 9..16
```

---

### streamTokenize
Потоковая токенизация для больших файлов.

```nim
iterator streamTokenize*(
  filePath: string,
  tokenizer: var Tokenizer,
  chunkSize: int = 8192,
  addSpecialTokens: bool = true
): seq[int]
```

**Параметры:**
- `filePath` - путь к файлу
- `tokenizer` - токенизатор
- `chunkSize` - размер чанка для чтения (по умолчанию: 8192 байт)
- `addSpecialTokens` - добавлять специальные токены (по умолчанию: true)

**Возвращает:** итератор по чанкам токенов

**Применение:**
- Обработка файлов > 1GB
- Экономия памяти
- Пайплайны обработки данных

**Пример:**
```nim
for chunk in streamTokenize("large_file.txt", tokenizer):
  # Обрабатываем каждый чанк токенов
  processTokens(chunk)
```

---

## Пакетная обработка

### encodeBatch
Пакетная токенизация с padding и truncation.

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

**Параметры:**
- `tokenizer` - токенизатор
- `texts` - список текстов
- `maxLength` - максимальная длина последовательности (по умолчанию: 512)
- `padding` - дополнять до maxLength (по умолчанию: true)
- `truncation` - обрезать длинные последовательности (по умолчанию: true)
- `addSpecialTokens` - добавлять BOS/EOS (по умолчанию: true)
- `returnAttentionMask` - возвращать маску внимания (по умолчанию: true)
- `returnTokenTypeIds` - возвращать token type IDs (по умолчанию: false)

**Возвращает:** `BatchEncoding` объект

**Пример:**
```nim
let texts = @["Первый текст", "Второй длинный текст"]
let batch = encodeBatch(tokenizer, texts, maxLength = 10)
echo batch.inputIds        # Токены с padding
echo batch.attentionMask   # Маски внимания
echo batch.lengths         # Реальные длины
```

---

### decodeBatch
Пакетное декодирование.

```nim
proc decodeBatch*(
  tokenizer: Tokenizer,
  tokensBatch: seq[seq[int]],
  skipSpecialTokens: bool = true
): seq[string]
```

**Параметры:**
- `tokenizer` - токенизатор
- `tokensBatch` - список последовательностей токенов
- `skipSpecialTokens` - пропускать специальные токены (по умолчанию: true)

**Возвращает:** список декодированных текстов

**Пример:**
```nim
let decoded = decodeBatch(tokenizer, batch.inputIds)
```

---

## Работа с текстом

### cleanText
Универсальная очистка текста.

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

**Параметры:** множественные флаги для различных видов очистки

**Возвращает:** очищенный текст

**Пример:**
```nim
let cleaned = cleanText(
  "<p>Email: test@example.com</p>",
  removeHtml = true,
  removeEmails = true
)
# Результат: ""
```

---

### toLowerUnicode / toUpperUnicode
Приведение к нижнему/верхнему регистру с поддержкой Unicode.

```nim
proc toLowerUnicode*(s: string): string
proc toUpperUnicode*(s: string): string
```

**Пример:**
```nim
echo toLowerUnicode("ПРИВЕТ")  # "привет"
echo toUpperUnicode("hello")   # "HELLO"
```

---

### normalizeText
Базовая нормализация текста.

```nim
proc normalizeText*(text: string): string
```

Выполняет:
- Удаление лишних пробелов
- Нормализацию Unicode
- Приведение к lowercase

---

### splitIntoWords
Разбиение текста на слова.

```nim
proc splitIntoWords*(text: string): seq[string]
```

**Особенности:**
- Корректно обрабатывает Unicode
- Учитывает пунктуацию
- Оптимизировано для кириллицы

---

### splitIntoSentences
Разбиение текста на предложения.

```nim
proc splitIntoSentences*(text: string): seq[string]
```

**Пример:**
```nim
let sentences = splitIntoSentences("Первое. Второе! Третье?")
# Результат: @["Первое.", "Второе!", "Третье?"]
```

---

### normalizeWhitespace
Нормализация пробелов.

```nim
proc normalizeWhitespace*(
  text: string,
  preserveNewlines: bool = false
): string
```

**Параметры:**
- `text` - текст
- `preserveNewlines` - сохранять переводы строк (по умолчанию: false)

---

### removeAccents
Удаление диакритических знаков.

```nim
proc removeAccents*(text: string): string
```

**Пример:**
```nim
echo removeAccents("café")  # "cafe"
```

---

### truncateText
Обрезка текста до максимальной длины.

```nim
proc truncateText*(
  text: string,
  maxLength: int,
  addEllipsis: bool = true
): string
```

**Пример:**
```nim
echo truncateText("Длинный текст", 7)  # "Длинный..."
```

---

### countWords / countCharacters / countSentences
Подсчёт слов, символов, предложений.

```nim
proc countWords*(text: string): int
proc countCharacters*(text: string, excludeWhitespace: bool = false): int
proc countSentences*(text: string): int
```

---

## Метрики и анализ

### getMetrics
Вычисление метрик токенизатора.

```nim
proc getMetrics*(
  tokenizer: var Tokenizer,
  corpus: seq[string]
): TokenizerMetrics
```

**Возвращает:** объект с метриками:
- `vocabSize` - размер словаря
- `compressionRatio` - коэффициент сжатия
- `avgTokensPerWord` - среднее количество токенов на слово
- `vocabUtilization` - утилизация словаря
- `unkTokenRate` - доля неизвестных токенов
- `tokensPerSecond` - скорость токенизации

**Пример:**
```nim
let metrics = getMetrics(tokenizer, corpus)
echo "Compression ratio: ", metrics.compressionRatio
echo "Vocab size: ", metrics.vocabSize
```

---

### analyzeVocabulary
Анализ словаря токенизатора.

```nim
proc analyzeVocabulary*(
  tokenizer: var Tokenizer,
  corpus: seq[string],
  topN: int = 10
): VocabAnalysis
```

**Параметры:**
- `tokenizer` - токенизатор
- `corpus` - корпус для анализа
- `topN` - сколько топ-токенов вернуть (по умолчанию: 10)

**Возвращает:** объект `VocabAnalysis` с детальной статистикой

**Пример:**
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
Вывод метрик в читаемом формате.

```nim
proc printMetrics*(metrics: TokenizerMetrics)
```

**Пример:**
```nim
printMetrics(metrics)
# Выводит:
# Размер словаря: 8000
# Коэффициент сжатия: 2.45
# ...
```

---

### compareTokenizers
Сравнение нескольких токенизаторов на одном тексте.

```nim
proc compareTokenizers*(
  text: string,
  tokenizers: seq[Tokenizer]
): seq[tuple[kind: TokenizerKind, tokens: seq[int], decoded: string]]
```

**Пример:**
```nim
let results = compareTokenizers("Тестовый текст", @[bpe, wordpiece, sp])
for result in results:
  echo result.kind, ": ", result.tokens.len, " tokens"
```

---

### benchmark
Измерение скорости токенизации.

```nim
proc benchmark*(
  tokenizer: var Tokenizer,
  texts: seq[string]
): float
```

**Возвращает:** токенов в секунду

**Пример:**
```nim
let speed = benchmark(tokenizer, corpus)
echo "Speed: ", speed, " tokens/sec"
```

---

## Сохранение и загрузка

### saveTokenizer
Сохранение токенизатора на диск.

```nim
proc saveTokenizer*(tokenizer: Tokenizer, path: string)
```

**Формат:** JSON

**Сохраняет:**
- Тип токенизатора
- Полный словарь
- Специальные токены
- Merges (для BPE)
- Scores (для SentencePiece)
- Все параметры

**Пример:**
```nim
saveTokenizer(tokenizer, "my_tokenizer.json")
```

---

### loadTokenizer
Загрузка токенизатора с диска.

```nim
proc loadTokenizer*(path: string): Tokenizer
```

**Пример:**
```nim
let tokenizer = loadTokenizer("my_tokenizer.json")
```

---

### exportTokenizerToJson
Экспорт частичных данных токенизатора в JSON (для просмотра).

```nim
proc exportTokenizerToJson*(tokenizer: Tokenizer, filepath: string)
```

**Сохраняет:**
- Первые 100 элементов словаря
- Первые 50 merges/scores
- Все параметры

**Применение:** для просмотра и отладки словаря

**Пример:**
```nim
exportTokenizerToJson(tokenizer, "vocab_preview.json")
```

---

## Утилиты

### Получение специальных токенов

```nim
proc getUnkTokenId*(tokenizer: Tokenizer): int
proc getPadTokenId*(tokenizer: Tokenizer): int
proc getBosTokenId*(tokenizer: Tokenizer): int
proc getEosTokenId*(tokenizer: Tokenizer): int
proc getSepTokenId*(tokenizer: Tokenizer): int
proc getClsTokenId*(tokenizer: Tokenizer): int
proc getMaskTokenId*(tokenizer: Tokenizer): int
```

**Пример:**
```nim
let unkId = tokenizer.getUnkTokenId()
let padId = tokenizer.getPadTokenId()
```

---

### Работа со словарём

```nim
proc getTokenById*(tokenizer: Tokenizer, id: int): string
proc getIdByToken*(tokenizer: Tokenizer, token: string): int
proc hasToken*(tokenizer: Tokenizer, token: string): bool
proc getVocabSize*(tokenizer: Tokenizer): int
proc getVocabTokens*(tokenizer: Tokenizer): seq[string]
```

**Пример:**
```nim
if tokenizer.hasToken("hello"):
  let id = tokenizer.getIdByToken("hello")
  echo "Token 'hello' has ID: ", id
```

---

### Кэширование

```nim
proc initCache*(maxSize: int): Table[string, seq[int]]
proc clearCache*(tokenizer: var Tokenizer)
```

**Пример:**
```nim
tokenizer.clearCache()
echo "Cache hits: ", tokenizer.cacheHits
echo "Cache misses: ", tokenizer.cacheMisses
```

---

## Продвинутые функции

### tokenizeWithDropout
BPE-dropout для регуляризации модели.

```nim
proc tokenizeWithDropout*(
  text: string,
  tokenizer: var Tokenizer,
  dropoutProb: float = 0.1,
  minDropped: int = 0,
  seed: int = -1
): seq[int]
```

**Параметры:**
- `text` - текст для токенизации
- `tokenizer` - BPE токенизатор
- `dropoutProb` - вероятность пропуска merge (0.0-1.0)
- `minDropped` - минимум пропущенных merges
- `seed` - random seed (-1 = случайный)

**Применение:**
- Регуляризация при обучении моделей
- Аугментация данных
- Улучшение обобщения

**Пример:**
```nim
# Обычная токенизация
let normal = tokenize("текст", tokenizer)

# С dropout - каждый раз разная токенизация
let dropout1 = tokenizeWithDropout("текст", tokenizer, 0.3)
let dropout2 = tokenizeWithDropout("текст", tokenizer, 0.3)
# dropout1 != dropout2 (скорее всего)
```

---

### incrementalTrain
Дообучение токенизатора на новых данных.

```nim
proc incrementalTrain*(
  tokenizer: var Tokenizer,
  newCorpus: seq[string],
  maxNewTokens: int = 1000,
  minFrequency: int = 2
)
```

**Параметры:**
- `tokenizer` - существующий токенизатор
- `newCorpus` - новые тексты
- `maxNewTokens` - максимум новых токенов для добавления
- `minFrequency` - минимальная частота

**Применение:**
- Адаптация к новым доменам
- Добавление терминологии
- Онлайн обучение

**Пример:**
```nim
incrementalTrain(tokenizer, newDomainTexts, maxNewTokens = 500)
```

---

### pruneVocabulary
Удаление редких токенов из словаря.

```nim
proc pruneVocabulary*(
  tokenizer: var Tokenizer,
  corpus: seq[string],
  minFrequency: int = 2,
  keepTopN: int = -1
)
```

**Параметры:**
- `tokenizer` - токенизатор
- `corpus` - корпус для подсчёта частот
- `minFrequency` - минимальная частота для сохранения
- `keepTopN` - сохранить топ-N токенов (-1 = все)

**Применение:**
- Уменьшение размера словаря
- Удаление шума
- Оптимизация для production

**Пример:**
```nim
# До: 10000 токенов
pruneVocabulary(tokenizer, corpus, minFrequency = 5)
# После: ~7000 токенов (удалены редкие)
```

---

### maskTokens
Маскирование токенов для MLM (Masked Language Modeling).

```nim
proc maskTokens*(
  tokens: seq[int],
  tokenizer: Tokenizer,
  maskProb: float = 0.15,
  replaceMaskProb: float = 0.8,
  replaceRandomProb: float = 0.1
): seq[int]
```

**Параметры:**
- `tokens` - исходные токены
- `tokenizer` - токенизатор
- `maskProb` - вероятность маскирования (по умолчанию: 0.15)
- `replaceMaskProb` - вероятность замены на [MASK] (по умолчанию: 0.8)
- `replaceRandomProb` - вероятность замены на случайный токен (по умолчанию: 0.1)

**Применение:**
- Обучение BERT-подобных моделей
- Masked Language Modeling
- Self-supervised learning

**Пример:**
```nim
let original = tokenize("Это тестовое предложение", tokenizer)
let masked = maskTokens(original, tokenizer, maskProb = 0.15)
# masked может быть: @[1, [MASK], 3, 4, [MASK]]
```

---

### encodeWithPadding
Токенизация с автоматическим padding/truncation.

```nim
proc encodeWithPadding*(
  tokenizer: var Tokenizer,
  text: string,
  maxLength: int,
  padding: bool = true,
  truncation: bool = true
): seq[int]
```

**Пример:**
```nim
let padded = encodeWithPadding(tokenizer, "текст", maxLength = 10)
# Результат всегда длина 10
```

---

### getSubwordBreakdown
Получение разбиения текста на подслова.

```nim
proc getSubwordBreakdown*(
  text: string,
  tokenizer: var Tokenizer
): seq[string]
```

**Возвращает:** список подслов в виде строк

**Применение:**
- Визуализация токенизации
- Отладка
- Анализ качества словаря

**Пример:**
```nim
let breakdown = getSubwordBreakdown("непонятное", tokenizer)
# Результат: @["не", "##понят", "##ное"]
```

---

### estimateTokenCount
Быстрая оценка количества токенов без токенизации.

```nim
proc estimateTokenCount*(
  text: string,
  avgCharsPerToken: float = 4.0
): int
```

**Применение:**
- Предварительная оценка размера
- Валидация длины входа
- Батчинг

**Пример:**
```nim
let estimate = estimateTokenCount("Длинный текст...")
if estimate > 512:
  echo "Текст слишком длинный"
```

---

### validateTokenizer
Проверка корректности токенизатора.

```nim
proc validateTokenizer*(tokenizer: Tokenizer): seq[string]
```

**Возвращает:** список ошибок (пустой список = валиден)

**Проверяет:**
- Согласованность vocab и inverseVocab
- Наличие специальных токенов
- Корректность merges (для BPE)
- Корректность scores (для SentencePiece)

**Пример:**
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

### Byte-level encoding (для ByteLevel BPE)

```nim
proc initBytePairEncoder*(): Table[int, string]
proc initByteDecoder*(encoder: Table[int, string]): Table[string, int]
proc byteLevelEncode*(text: string): seq[int]
proc byteLevelDecode*(bytes: seq[int]): string
```

**Применение:** низкоуровневая работа с byte-level кодированием

---

## Примеры использования

### Базовый пример

```nim
import tokenization

# Обучение
let corpus = @[
  "Это первый текст для обучения",
  "А это второй текст",
  "Токенизация работает отлично"
]

var tokenizer = trainBPE(corpus, vocabSize = 500)

# Токенизация
let text = "Новый тестовый текст"
let tokens = tokenize(text, tokenizer)
echo "Tokens: ", tokens

# Декодирование
let decoded = tokenizer.decode(tokens)
echo "Decoded: ", decoded

# Метрики
let metrics = getMetrics(tokenizer, corpus)
echo "Compression ratio: ", metrics.compressionRatio
```

---

### Пакетная обработка

```nim
let texts = @[
  "Первое предложение",
  "Второе длинное предложение с большим количеством слов",
  "Третье"
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

### Работа с позициями (NER)

```nim
let text = "Иван живёт в Москве"
let offsets = tokenizeWithOffsets(text, tokenizer)

for offset in offsets:
  echo offset.token, " [", offset.startChar, ":", offset.endChar, "]"
# Результат:
# "Иван" [0:4]
# " " [4:5]
# "живёт" [5:10]
# ...
```

---

### Потоковая обработка большого файла

```nim
var totalTokens = 0
for chunk in streamTokenize("huge_file.txt", tokenizer):
  totalTokens += chunk.len
  # Обработка чанка...

echo "Total tokens processed: ", totalTokens
```

---

### Очистка и предобработка текста

```nim
let dirty = """
  <html>
    <p>Email: spam@example.com</p>
    <p>Посетите www.example.com</p>
  </html>
"""

let clean = cleanText(
  dirty,
  removeHtml = true,
  removeUrls = true,
  removeEmails = true
)

echo clean  # "Посетите"
```

---

### Сравнение токенизаторов

```nim
var bpe = trainBPE(corpus, vocabSize = 1000)
var wordpiece = trainWordPiece(corpus, vocabSize = 1000)
var sentencepiece = trainSentencePiece(corpus, vocabSize = 1000)

let results = compareTokenizers(
  "Тестовое предложение",
  @[bpe, wordpiece, sentencepiece]
)

for result in results:
  echo result.kind, ": ", result.tokens.len, " tokens"
  echo "  Decoded: ", result.decoded
```

---

### BPE Dropout для регуляризации

```nim
let text = "обучение модели"

# Создаём 5 различных токенизаций для аугментации
var augmented: seq[seq[int]]
for i in 0..4:
  augmented.add(tokenizeWithDropout(text, tokenizer, dropoutProb = 0.2))

# Все 5 будут разными!
for i, tokens in augmented:
  echo "Variant ", i, ": ", tokens
```

---

### Адаптация токенизатора к новому домену

```nim
# Исходный токенизатор обучен на общих текстах
var tokenizer = trainBPE(generalCorpus, vocabSize = 5000)

# Добавляем медицинскую терминологию
let medicalTexts = @[
  "диагностика заболеваний",
  "фармакологические препараты",
  "клиническая практика"
]

incrementalTrain(tokenizer, medicalTexts, maxNewTokens = 500)

# Теперь токенизатор знает медицинские термины
```

---

## Оптимизация производительности

### Советы по оптимизации:

1. **Используйте кэш:**
   - Кэш автоматически активен
   - Для очистки: `clearCache(tokenizer)`
   - Проверка эффективности: `tokenizer.cacheHits / (tokenizer.cacheHits + tokenizer.cacheMisses)`

2. **Пакетная обработка:**
   - Используйте `encodeBatch` вместо цикла `tokenize`
   - Значительно быстрее для множества текстов

3. **Потоковая обработка:**
   - Для файлов > 100MB используйте `streamTokenize`
   - Экономит память

4. **Выбор токенизатора:**
   - **BPE**: баланс скорости и качества
   - **WordPiece**: медленнее, лучше для BERT
   - **SentencePiece**: универсальный, хорош для любых языков
   - **ByteLevel BPE**: самый быстрый, no UNK tokens

5. **Размер словаря:**
   - Меньший словарь (1000-2000) → быстрее
   - Больший словарь (30000-50000) → лучшее качество
   - Оптимальный баланс: 8000-16000

---

## Частые вопросы (FAQ)

### Q: Как выбрать размер словаря?
**A:** 
- Маленькие корпусы (<1M слов): 1000-5000
- Средние корпусы (1M-10M слов): 8000-16000
- Большие корпусы (>10M слов): 30000-50000

### Q: Когда использовать preserveCase=true?
**A:** Когда регистр имеет значение (Named Entity Recognition, code generation). Для большинства задач NLP лучше false.

### Q: Почему появляются [UNK] токены?
**A:**
- Слишком маленький словарь
- Слишком высокий minFrequency
- Текст сильно отличается от обучающего корпуса

**Решение:**
- Увеличить vocabSize
- Уменьшить minFrequency
- Использовать ByteLevel BPE (нет UNK)
- Дообучить: `incrementalTrain`

### Q: Как ускорить токенизацию?
**A:**
1. Используйте кэш (включен по умолчанию)
2. Используйте `encodeBatch` для множества текстов
3. Компилируйте с `-d:danger --opt:speed`
4. Выбирайте ByteLevel BPE для скорости

### Q: Совместим ли ByteLevel BPE с GPT-2/3?
**A:** Да, полностью совместим. Использует те же принципы byte-pair encoding.

---

## Changelog

### Version 0.5 (2026-01-31)
- ✅ Исправлено сохранение пробелов в WordPiece
- ✅ Исправлена обработка заглавных букв (fallback на lowercase)
- ✅ Добавлен экспорт словарей в JSON
- ✅ Улучшено измерение времени в тестах
- ✅ Исправлены проверки размера словаря

### Version 0.4
- ✅ ByteLevel BPE (GPT-2/3 compatible)
- ✅ Token position tracking
- ✅ Streaming tokenization
- ✅ Vocabulary pruning
- ✅ Оптимизация производительности (20-50x)
- ✅ Кэширование
- ✅ BPE-dropout

### Version 0.3
- ✅ Расширенная очистка текста
- ✅ Маскирование токенов
- ✅ Сравнение токенизаторов

### Version 0.2
- ✅ Пакетная обработка
- ✅ Специальные токены
- ✅ Метрики

### Version 0.1
- ✅ Базовые токенизаторы: BPE, WordPiece, SentencePiece

---

## Лицензия

MIT License

---

## Контакты

**Автор:** github.com/Balans097  
**E-mail:** vasil.minsk@yahoo.com
**Версия:** 0.5  
**Дата:** 2026-01-31





