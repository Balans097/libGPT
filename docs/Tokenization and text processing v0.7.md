# Документация библиотеки токенизации

**Версия:** 0.7  
**Дата:** 2026-02-01  
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

Библиотека токенизации предоставляет полный набор инструментов для обработки и подготовки текстовых данных для задач машинного обучения и обработки естественного языка (NLP). Библиотека реализует современные алгоритмы токенизации, используемые в таких моделях как BERT, GPT-2/3, T5 и других трансформерах.

### Основные возможности

- **Множественные алгоритмы токенизации:**
  - **BPE (Byte Pair Encoding)** - классический алгоритм подсловной токенизации
  - **WordPiece** - используется в BERT, работает на уровне символов с префиксами подслов
  - **SentencePiece** - языково-независимый алгоритм на основе unigram language model
  - **Byte-Level BPE** - совместимый с GPT-2/GPT-3, работает на уровне байтов

- **Производительность и масштабируемость:**
  - Оптимизация скорости выполнения (20-50x по сравнению с наивными реализациями)
  - Кэширование токенизаций с LRU политикой
  - Параллельная пакетная обработка
  - Потоковая обработка для файлов любого размера
  - Потокобезопасная обработка для многопоточных приложений

- **Продвинутая обработка текста:**
  - Множественные стратегии нормализации текста
  - Обработка Unicode (NFKC нормализация, zero-width символы)
  - Поддержка многоязычных текстов с детекцией языка
  - Обработка чисел, эмодзи, специальных символов
  - Раскрытие сокращений и обработка контракций

- **Анализ и метрики:**
  - Комплексные метрики качества токенизации
  - Анализ словаря и статистика токенов
  - Детекция OOV (out-of-vocabulary) слов
  - Сравнение токенизаторов
  - Визуализация и отладка

- **Управление словарём:**
  - Инкрементальное обновление словаря
  - Объединение и выравнивание словарей
  - Pruning (очистка) редких токенов
  - Расширение словаря специальными токенами

- **Специализированные возможности:**
  - Subword regularization (BPE-dropout) для аугментации данных
  - Tracking позиций токенов для NER/QA задач
  - Маскирование токенов для Masked Language Modeling
  - Интеграция с популярными форматами (HuggingFace, SentencePiece, TikToken)

### Поддерживаемые сценарии использования

1. **Обучение языковых моделей** - подготовка данных для BERT, GPT, T5 и других моделей
2. **Fine-tuning** - адаптация токенизаторов под специфические домены
3. **Многоязычные приложения** - обработка текстов на разных языках
4. **Production системы** - быстрая и надёжная токенизация в реальном времени
5. **Исследования** - эксперименты с различными стратегиями токенизации

### Требования к производительности

Библиотека оптимизирована для высокой скорости работы:
- Обработка **миллионов токенов в секунду** на современных CPU
- Память: эффективное использование за счёт потоковой обработки
- Компиляция: поддержка режимов `-d:release` и `-d:danger --opt:speed` для максимальной производительности

### Совместимость

Библиотека совместима с популярными фреймворками и форматами:
- HuggingFace Tokenizers
- SentencePiece модели
- TikToken (OpenAI)
- Стандартные форматы словарей (JSON)

### Быстрый старт

```nim
import tokenization

# 1. Подготовка корпуса
let corpus = @[
  "машинное обучение это важная область",
  "глубокое обучение использует нейронные сети"
]

# 2. Обучение токенизатора
let tokenizer = trainBPE(corpus, vocabSize = 1000)

# 3. Токенизация
let tokens = tokenizer.encode("машинное обучение")

# 4. Декодирование
let text = tokenizer.decode(tokens)

# 5. Сохранение
tokenizer.save("tokenizer.json")
```

### Структура документации

Документация организована по функциональным разделам:
- **Типы данных** - описание всех типов, перечислений и структур
- **Обучение токенизаторов** - функции для создания и обучения токенизаторов
- **Токенизация и декодирование** - основные операции с текстом
- **Пакетная обработка** - эффективная обработка множества текстов
- **Работа с текстом** - нормализация, очистка и преобразование
- **Метрики и анализ** - оценка качества и анализ токенизации
- **Сохранение и загрузка** - персистентность моделей
- **Утилиты** - вспомогательные функции
- **Продвинутые функции** - специализированные возможности


---

## Типы данных

### Перечисления (Enumerations)

#### TokenizerKind
Определяет тип токенизатора.

```nim
type TokenizerKind* = enum
  tkBPE = 0              # Byte Pair Encoding
  tkWordPiece = 1        # WordPiece (BERT-style)
  tkSentencePiece = 2    # SentencePiece (unigram)
  tkByteLevelBPE = 3     # Byte-Level BPE (GPT-2/3 style)
```

**Описание значений:**
- `tkBPE` - классический Byte Pair Encoding, работает на уровне символов
- `tkWordPiece` - алгоритм WordPiece, используется в BERT
- `tkSentencePiece` - SentencePiece с unigram language model
- `tkByteLevelBPE` - BPE на уровне байтов, совместим с GPT-2/GPT-3

#### NumberNormalizationStrategy
Стратегия нормализации чисел в тексте.

```nim
type NumberNormalizationStrategy* = enum
  nsKeepOriginal      # Оставить числа как есть
  nsReplaceWithToken  # Заменить на [NUM]
  nsReplaceWithDigits # Заменить на разряды (123 -> [NUM_3DIGIT])
  nsNormalize         # Нормализовать (1,234.56 -> 1234.56)
```

**Применение:**
- `nsKeepOriginal` - сохраняет исходный формат чисел
- `nsReplaceWithToken` - заменяет все числа на специальный токен `[NUM]`
- `nsReplaceWithDigits` - заменяет числа на токены по количеству разрядов
- `nsNormalize` - приводит числа к стандартному формату без разделителей

#### EmojiStrategy
Стратегия обработки эмодзи.

```nim
type EmojiStrategy* = enum
  esKeep      # Сохранить эмодзи
  esRemove    # Удалить эмодзи
  esReplace   # Заменить на текстовое описание
  esTokenize  # Токенизировать отдельно
```

**Применение:**
- `esKeep` - оставляет эмодзи как есть
- `esRemove` - полностью удаляет эмодзи из текста
- `esReplace` - заменяет эмодзи на текстовые описания
- `esTokenize` - обрабатывает эмодзи как отдельные токены

### Структуры (Objects)

#### SpecialTokens
Специальные токены для разных задач NLP.

```nim
type SpecialTokens* = object
  padToken*: string      # Токен заполнения (padding)
  unkToken*: string      # Токен неизвестного слова (unknown)
  bosToken*: string      # Начало последовательности
  eosToken*: string      # Конец последовательности
  sepToken*: string      # Разделитель
  clsToken*: string      # Токен классификации
  maskToken*: string     # Токен маски
```

**Стандартные значения:**
- `padToken`: `"[PAD]"` - выравнивание последовательностей
- `unkToken`: `"[UNK]"` - неизвестные слова
- `bosToken`: `"[BOS]"` - маркер начала текста
- `eosToken`: `"[EOS]"` - маркер конца текста
- `sepToken`: `"[SEP]"` - разделитель сегментов (BERT)
- `clsToken`: `"[CLS]"` - для задач классификации
- `maskToken`: `"[MASK]"` - для masked language modeling

#### TokenOffset
Информация о позиции токена в тексте.

```nim
type TokenOffset* = object
  token*: string       # Сам токен
  tokenId*: int        # ID токена в словаре
  startChar*: int      # Начальная позиция в символах
  endChar*: int        # Конечная позиция в символах
  startByte*: int      # Начальная позиция в байтах
  endByte*: int        # Конечная позиция в байтах
```

**Назначение:** Используется для задач NER (Named Entity Recognition) и QA (Question Answering).

#### Tokenizer
Основной объект токенизатора.

```nim
type Tokenizer* = ref object
  kind*: TokenizerKind                    # Тип токенизатора
  vocab*: Table[string, int]              # Словарь: токен → ID
  inverseVocab*: seq[string]              # Обратный словарь: ID → токен
  merges*: seq[BPEMerge]                  # Правила слияния для BPE
  specialTokens*: SpecialTokens           # Специальные токены
  specialTokenIds*: Table[string, int]    # ID специальных токенов
  maxInputCharsPerWord*: int              # Макс. длина слова
  continuingSubwordPrefix*: string        # Префикс подслов ("##")
  scores*: Table[string, float]           # Scores для SentencePiece
  byteFallback*: bool                     # Byte fallback
  preserveCase*: bool                     # Сохранять регистр
  cache*: Table[string, seq[int]]         # Кэш токенизаций
  cacheMaxSize*: int                      # Размер кэша
  cacheHits*: int                         # Попадания в кэш
  cacheMisses*: int                       # Промахи кэша
  byteEncoder*: Table[int, string]        # Кодировщик байтов
  byteDecoder*: Table[string, int]        # Декодировщик байтов
```

#### BatchEncoding
Результат пакетной токенизации.

```nim
type BatchEncoding* = object
  inputIds*: seq[seq[int]]         # ID токенов для каждого текста
  attentionMask*: seq[seq[int]]    # Маски внимания
  tokenTypeIds*: seq[seq[int]]     # ID типа токена
  lengths*: seq[int]               # Длины последовательностей
```

#### TokenizerMetrics
Метрики производительности токенизатора.

```nim
type TokenizerMetrics* = object
  vocabSize*: int              # Размер словаря
  compressionRatio*: float     # Коэфф. сжатия (chars/tokens)
  avgTokensPerWord*: float     # Среднее токенов на слово
  vocabUtilization*: float     # Доля использованного словаря
  unkTokenRate*: float         # Доля UNK токенов
  tokensPerSecond*: float      # Скорость токенизации
```

#### VocabAnalysis
Анализ словаря токенизатора.

```nim
type VocabAnalysis* = object
  vocabSize*: int
  avgTokenLength*: float
  typeTokenRatio*: float
  coverageRate*: float
  oovRate*: float
  mostFrequent*: seq[tuple[token: string, freq: int]]
  leastFrequent*: seq[tuple[token: string, freq: int]]
  lengthDistribution*: CountTable[int]
```

### Константы

```nim
const
  MAX_INPUT_LENGTH* = 1_000_000    # Макс. длина текста (1M символов)
  MAX_VOCAB_SIZE* = 100_000        # Макс. размер словаря
  TOKENIZER_VERSION* = "1.0.0"     # Версия формата
```


---

## Обучение токенизаторов

### trainBPE
Обучает Byte Pair Encoding токенизатор.

```nim
proc trainBPE*(
  corpus: seq[string],
  vocabSize: int = 5000,
  specialTokens: SpecialTokens = defaultSpecialTokens(),
  minFrequency: int = 2,
  progressCallback: proc(current, total: int) = nil
): Tokenizer
```

**Параметры:**
- `corpus` - обучающий корпус (список текстов)
- `vocabSize` - желаемый размер словаря (по умолчанию 5000)
- `specialTokens` - специальные токены
- `minFrequency` - минимальная частота пары для слияния
- `progressCallback` - функция обратного вызова для прогресса

**Возвращает:** Обученный BPE токенизатор.

**Пример:**
```nim
let corpus = @[
  "машинное обучение это круто",
  "обучение нейронных сетей"
]
let tokenizer = trainBPE(corpus, vocabSize = 1000)
```

### trainWordPiece
Обучает WordPiece токенизатор.

```nim
proc trainWordPiece*(
  corpus: seq[string],
  vocabSize: int = 5000,
  specialTokens: SpecialTokens = defaultSpecialTokens(),
  continuingSubwordPrefix: string = "##",
  minFrequency: int = 2
): Tokenizer
```

**Параметры:**
- `corpus` - обучающий корпус
- `vocabSize` - размер словаря
- `specialTokens` - специальные токены
- `continuingSubwordPrefix` - префикс для подслов (по умолчанию `"##"`)
- `minFrequency` - минимальная частота

**Возвращает:** Обученный WordPiece токенизатор.

**Пример:**
```nim
let tokenizer = trainWordPiece(corpus, vocabSize = 1000)
let tokens = tokenizer.encode("непредсказуемый")
# Может вернуть: ["не", "##предсказ", "##уемый"]
```

### trainSentencePiece
Обучает SentencePiece токенизатор.

```nim
proc trainSentencePiece*(
  corpus: seq[string],
  vocabSize: int = 5000,
  specialTokens: SpecialTokens = defaultSpecialTokens(),
  characterCoverage: float = 0.9995,
  minFrequency: int = 2
): Tokenizer
```

**Параметры:**
- `corpus` - обучающий корпус
- `vocabSize` - размер словаря
- `characterCoverage` - доля символов для покрытия
- `minFrequency` - минимальная частота

**Возвращает:** Обученный SentencePiece токенизатор.

### trainByteLevelBPE
Обучает Byte-Level BPE токенизатор (GPT-2/GPT-3 style).

```nim
proc trainByteLevelBPE*(
  corpus: seq[string],
  vocabSize: int = 5000,
  specialTokens: SpecialTokens = defaultSpecialTokens(),
  minFrequency: int = 2
): Tokenizer
```

**Описание:** Работает на уровне байтов, гарантирует кодирование любого текста без UNK токенов.

**Пример:**
```nim
let tokenizer = trainByteLevelBPE(corpus, vocabSize = 5000)
let tokens = tokenizer.encode("Hello 🌍")  # Обработает любые символы
```

---

## Токенизация и декодирование

### encode
Кодирует текст в последовательность ID токенов.

```nim
proc encode*(
  tokenizer: Tokenizer,
  text: string,
  addSpecialTokens: bool = true,
  maxLength: int = -1,
  truncation: bool = false,
  padding: bool = false
): seq[int]
```

**Параметры:**
- `tokenizer` - токенизатор
- `text` - входной текст
- `addSpecialTokens` - добавлять ли BOS/EOS токены
- `maxLength` - максимальная длина (-1 = без ограничений)
- `truncation` - обрезать ли до maxLength
- `padding` - дополнять ли до maxLength

**Пример:**
```nim
let ids = tokenizer.encode("Привет мир", addSpecialTokens = true)
// ids = [1, 245, 678, 2]  # где 1 = [BOS], 2 = [EOS]

let paddedIds = tokenizer.encode("Короткий", maxLength = 10, padding = true)
```

### decode
Декодирует последовательность ID обратно в текст.

```nim
proc decode*(
  tokenizer: Tokenizer,
  ids: seq[int],
  skipSpecialTokens: bool = true,
  cleanUpTokenization: bool = true
): string
```

**Параметры:**
- `tokenizer` - токенизатор
- `ids` - последовательность ID токенов
- `skipSpecialTokens` - пропускать ли специальные токены
- `cleanUpTokenization` - убирать ли артефакты токенизации

**Пример:**
```nim
let text = tokenizer.decode(@[1, 245, 678, 2], skipSpecialTokens = true)
// text = "Привет мир"
```

### tokenize
Разбивает текст на токены (строки).

```nim
proc tokenize*(
  tokenizer: Tokenizer,
  text: string,
  addSpecialTokens: bool = false
): seq[string]
```

**Пример:**
```nim
let tokens = tokenizer.tokenize("машинное обучение")
// tokens = @["машин", "##ное", "обуч", "##ение"]
```

### tokenizeWithOffsets
Токенизирует текст с отслеживанием позиций.

```nim
proc tokenizeWithOffsets*(
  tokenizer: Tokenizer,
  text: string
): seq[TokenOffset]
```

**Применение:** Для задач NER, QA, где нужно знать позиции токенов.

**Пример:**
```nim
let offsets = tokenizer.tokenizeWithOffsets("Hello world")
for offset in offsets:
  echo "Token: ", offset.token
  echo "  Chars: ", offset.startChar, "..", offset.endChar
  echo "  Bytes: ", offset.startByte, "..", offset.endByte
```

---

## Пакетная обработка

### encodeBatch
Кодирует несколько текстов одновременно.

```nim
proc encodeBatch*(
  tokenizer: Tokenizer,
  texts: seq[string],
  addSpecialTokens: bool = true,
  maxLength: int = -1,
  truncation: bool = false,
  padding: bool = false
): BatchEncoding
```

**Возвращает:** `BatchEncoding` с выровненными последовательностями.

**Пример:**
```nim
let texts = @["Короткий текст", "Это более длинный текст"]
let batch = tokenizer.encodeBatch(
  texts,
  maxLength = 15,
  truncation = true,
  padding = true
)
// batch.inputIds - все последовательности длины 15
// batch.attentionMask - маски внимания
```

### encodeBatchParallel
Параллельная пакетная кодировка для больших объёмов.

```nim
proc encodeBatchParallel*(
  tokenizer: Tokenizer,
  texts: seq[string],
  addSpecialTokens: bool = true,
  maxLength: int = -1,
  numThreads: int = 0
): seq[seq[int]]
```

**Описание:** Использует многопоточность для ускорения. Может дать ускорение в 2-4 раза на многоядерных процессорах.

**Пример:**
```nim
let largeCorpus = readLargeCorpus()  # Миллион текстов
let encoded = tokenizer.encodeBatchParallel(largeCorpus, numThreads = 8)
```


---

## Работа с текстом

### cleanText
Базовая очистка текста.

```nim
proc cleanText*(
  text: string,
  removeHtml: bool = true,
  removeUrls: bool = true,
  removeEmails: bool = false,
  normalizeWhitespace: bool = true,
  toLowerCase: bool = false
): string
```

**Пример:**
```nim
let dirty = "<p>Посетите https://example.com</p>  \n\n  для деталей"
let clean = cleanText(dirty, removeHtml = true, removeUrls = true)
// clean = "Посетите для деталей"
```

### normalizeNumbers
Нормализует числа в тексте согласно стратегии.

```nim
proc normalizeNumbers*(
  text: string,
  strategy: NumberNormalizationStrategy = nsNormalize
): string
```

**Пример:**
```nim
let text1 = normalizeNumbers("У меня 5 яблок", nsReplaceWithToken)
// text1 = "У меня [NUM] яблок"

let text2 = normalizeNumbers("Цена 1234 рубля", nsReplaceWithDigits)
// text2 = "Цена [NUM_4DIGIT] рубля"
```

### handleContractions
Раскрывает сокращения в тексте.

```nim
proc handleContractions*(
  text: string,
  language: string = "en"
): string
```

**Пример:**
```nim
let en = handleContractions("I'm learning", language = "en")
// en = "I am learning"

let ru = handleContractions("это т.е. пример", language = "ru")
// ru = "это то есть пример"
```

### normalizeWhitespaceAdvanced
Продвинутая нормализация пробельных символов.

```nim
proc normalizeWhitespaceAdvanced*(
  text: string,
  preserveNewlines: bool = false
): string
```

**Описание:** Заменяет все последовательности пробельных символов на одиночные пробелы.

### handleZeroWidthChars
Удаляет zero-width символы.

```nim
proc handleZeroWidthChars*(text: string): string
```

**Описание:** Удаляет невидимые символы нулевой ширины.

### fullNormalization
Полная нормализация текста (NFKC + очистка).

```nim
proc fullNormalization*(text: string): string
```

**Описание:** Выполняет Unicode нормализацию NFKC, удаляет zero-width символы и нормализует пробелы.

---

## Метрики и анализ

### getMetrics
Вычисляет метрики производительности токенизатора.

```nim
proc getMetrics*(
  tokenizer: Tokenizer,
  corpus: seq[string]
): TokenizerMetrics
```

**Пример:**
```nim
let metrics = getMetrics(tokenizer, testCorpus)
echo "Размер словаря: ", metrics.vocabSize
echo "Коэфф. сжатия: ", metrics.compressionRatio
echo "UNK rate: ", metrics.unkTokenRate * 100, "%"
echo "Скорость: ", metrics.tokensPerSecond, " токенов/сек"
```

### analyzeVocabulary
Анализирует состав словаря токенизатора.

```nim
proc analyzeVocabulary*(
  tokenizer: Tokenizer,
  corpus: seq[string],
  topK: int = 10
): VocabAnalysis
```

**Пример:**
```nim
let analysis = analyzeVocabulary(tokenizer, corpus, topK = 20)
echo "Размер словаря: ", analysis.vocabSize
echo "Средняя длина токена: ", analysis.avgTokenLength
echo "Покрытие: ", analysis.coverageRate * 100, "%"

echo "\nСамые частые токены:"
for (token, freq) in analysis.mostFrequent:
  echo "  ", token, ": ", freq
```

### compareTokenizers
Сравнивает несколько токенизаторов на одном корпусе.

```nim
proc compareTokenizers*(
  tokenizers: seq[Tokenizer],
  corpus: seq[string],
  names: seq[string] = @[]
): Table[string, TokenizerMetrics]
```

**Пример:**
```nim
let bpe = trainBPE(corpus, 5000)
let wp = trainWordPiece(corpus, 5000)
let sp = trainSentencePiece(corpus, 5000)

let comparison = compareTokenizers(
  @[bpe, wp, sp],
  testCorpus,
  names = @["BPE", "WordPiece", "SentencePiece"]
)

for name, metrics in comparison:
  echo name, ": compression=", metrics.compressionRatio
```

### getTokenStatistics
Вычисляет статистику по токенам в корпусе.

```nim
proc getTokenStatistics*(
  tokenizer: Tokenizer,
  corpus: seq[string]
): tuple[
  totalTokens: int,
  uniqueTokens: int,
  avgLength: float,
  maxLength: int,
  minLength: int
]
```

---

## Сохранение и загрузка

### save
Сохраняет токенизатор в файл.

```nim
proc save*(tokenizer: Tokenizer, filepath: string)
```

**Формат:** JSON со всеми параметрами, словарём и правилами слияния.

**Пример:**
```nim
tokenizer.save("models/my_tokenizer.json")
```

### load
Загружает токенизатор из файла.

```nim
proc load*(filepath: string): Tokenizer
```

**Пример:**
```nim
let tokenizer = load("models/my_tokenizer.json")
```

### saveVersioned
Сохраняет токенизатор с метаданными версии.

```nim
proc saveVersioned*(
  tokenizer: Tokenizer,
  filepath: string,
  trainedOn: string = ""
)
```

**Описание:** Добавляет метаданные о версии, дате создания и обучения.

**Пример:**
```nim
tokenizer.saveVersioned(
  "models/tokenizer_v2.json",
  trainedOn = "Wikipedia RU + News 2025"
)
```

### loadVersioned
Загружает токенизатор с проверкой версии.

```nim
proc loadVersioned*(filepath: string): VersionedTokenizer
```

**Пример:**
```nim
let versioned = loadVersioned("models/tokenizer_v2.json")
echo "Версия: ", versioned.metadata.version
echo "Обучен на: ", versioned.metadata.trainedOn
let tokenizer = versioned.tokenizer
```


---

## Утилиты

### Работа со специальными токенами

#### defaultSpecialTokens
Возвращает набор специальных токенов по умолчанию.

```nim
proc defaultSpecialTokens*(): SpecialTokens
```

**Пример:**
```nim
let special = defaultSpecialTokens()
echo special.padToken   // "[PAD]"
echo special.unkToken   // "[UNK]"
```

#### addSpecialTokens
Добавляет специальные токены в токенизатор.

```nim
proc addSpecialTokens*(tokenizer: var Tokenizer, tokens: seq[string])
```

**Пример:**
```nim
var tokenizer = trainBPE(corpus, 5000)
tokenizer.addSpecialTokens(@["[QUERY]", "[ANSWER]", "[CONTEXT]"])
```

#### maskTokens
Случайно маскирует токены для Masked Language Modeling.

```nim
proc maskTokens*(
  tokenIds: seq[int],
  tokenizer: Tokenizer,
  maskProb: float = 0.15,
  randomProb: float = 0.1,
  keepProb: float = 0.1
): tuple[masked: seq[int], labels: seq[int]]
```

**Применение:** Подготовка данных для обучения BERT-подобных моделей.

**Пример:**
```nim
let tokenIds = @[101, 2003, 4521, 8765, 102]
let (masked, labels) = maskTokens(tokenIds, tokenizer, maskProb = 0.15)
// masked = @[101, 2003, [MASK_ID], 8765, 102]
// labels = @[-100, -100, 4521, -100, -100]  # -100 = игнорировать
```

### Кэширование

#### newLRUCache
Создаёт новый LRU кэш.

```nim
proc newLRUCache*(maxSize: int = 10000): LRUCache
```

#### put (LRUCache)
Добавляет значение в кэш.

```nim
proc put*(cache: var LRUCache, key: string, value: seq[int])
```

#### get (LRUCache)
Получает значение из кэша.

```nim
proc get*(cache: var LRUCache, key: string): Option[seq[int]]
```

**Пример:**
```nim
var cache = newLRUCache(maxSize = 5000)
cache.put("текст", @[1, 2, 3, 4])

let result = cache.get("текст")
if result.isSome:
  echo "Найдено в кэше: ", result.get()
```

#### getStats (LRUCache)
Получает статистику кэша.

```nim
proc getStats*(cache: LRUCache): tuple[size: int, hits: int, misses: int, hitRate: float]
```

### Валидация

#### validateInput
Проверяет корректность входного текста.

```nim
proc validateInput*(text: string): Option[string]
```

**Возвращает:** `Some(errorMessage)` если есть ошибка, `None` если текст валиден.

**Проверки:**
- Текст не пустой
- Длина не превышает MAX_INPUT_LENGTH
- Нет некорректных Unicode последовательностей

**Пример:**
```nim
let error = validateInput(userInput)
if error.isSome:
  echo "Ошибка: ", error.get()
  return
```

#### validateTokenizer
Проверяет корректность конфигурации токенизатора.

```nim
proc validateTokenizer*(tokenizer: Tokenizer): Option[string]
```

### Вспомогательные функции ID токенов

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

### Unicode операции

#### runeCount
Подсчитывает количество рун (Unicode символов).

```nim
proc runeCount*(text: string): int
```

**Пример:**
```nim
echo runeCount("Hello")    // 5
echo runeCount("Привет")   // 6
echo runeCount("🌍🌎")     // 2
```

#### truncateToRunes
Обрезает текст до заданного количества рун.

```nim
proc truncateToRunes*(text: string, maxRunes: int): string
```

**Описание:** Корректно обрезает текст с учётом многобайтовых символов.

---

## Продвинутые функции

### Продвинутые возможности BPE

#### encodeWithDropout
Токенизация с BPE-dropout для аугментации данных.

```nim
proc encodeWithDropout*(
  tokenizer: Tokenizer,
  text: string,
  dropoutRate: float = 0.1,
  seed: int = 0
): seq[int]
```

**Применение:** Субсловная регуляризация - создание множества разных токенизаций для улучшения обобщающей способности модели.

**Пример:**
```nim
let normal = tokenizer.encode("машинное обучение")
// normal = @[245, 678]

let dropout1 = tokenizer.encodeWithDropout("машинное обучение", dropoutRate = 0.3)
// dropout1 = @[24, 56, 67, 89]  # более мелкие токены

let dropout2 = tokenizer.encodeWithDropout("машинное обучение", dropoutRate = 0.3)
// dropout2 = @[245, 67, 89]  # другая сегментация
```

#### reverseBPE
Находит все возможные сегментации текста.

```nim
proc reverseBPE*(
  tokenizer: Tokenizer,
  text: string,
  maxSegmentations: int = 10
): seq[seq[string]]
```

**Применение:** Анализ токенизатора, поиск альтернативных разбиений.

### Управление словарём

#### addTokens
Добавляет новые токены в словарь.

```nim
proc addTokens*(tokenizer: var Tokenizer, tokens: seq[string])
```

**Описание:** Инкрементально расширяет словарь без переобучения.

#### findCommonTokens
Находит общие токены в нескольких токенизаторах.

```nim
proc findCommonTokens*(
  tokenizers: seq[Tokenizer],
  minCount: int = 2
): seq[string]
```

**Применение:** Анализ согласованности словарей, построение общего словаря.

#### alignVocabularies
Объединяет словари нескольких токенизаторов.

```nim
proc alignVocabularies*(
  tokenizer1: Tokenizer,
  tokenizer2: Tokenizer
): Tokenizer
```

**Применение:** Создание мультиязычных токенизаторов.

**Пример:**
```nim
let generalTokenizer = trainBPE(generalCorpus, 30000)
let domainTokenizer = trainBPE(medicalCorpus, 10000)
let combined = alignVocabularies(generalTokenizer, domainTokenizer)
```

#### pruneVocabulary
Удаляет редкие токены из словаря.

```nim
proc pruneVocabulary*(
  tokenizer: var Tokenizer,
  corpus: seq[string],
  minFrequency: int = 5,
  keepSpecial: bool = true
)
```

**Применение:** Оптимизация памяти и скорости.

**Пример:**
```nim
var tokenizer = load("large_tokenizer.json")
echo "Размер до: ", tokenizer.vocab.len
tokenizer.pruneVocabulary(corpus, minFrequency = 10)
echo "Размер после: ", tokenizer.vocab.len
```

### Многоязычность

#### detectLanguage
Определяет язык (скрипт) текста.

```nim
proc detectLanguage*(text: string): string
```

**Возвращает:** `"latin"`, `"cyrillic"`, `"cjk"`, `"arabic"`, `"mixed"` и т.д.

**Пример:**
```nim
echo detectLanguage("Hello world")        // "latin"
echo detectLanguage("Привет мир")         // "cyrillic"
echo detectLanguage("Hello Привет")       // "mixed"
```

#### detectScriptMixing
Проверяет смешивание скриптов в тексте.

```nim
proc detectScriptMixing*(
  text: string
): seq[tuple[script: string, count: int]]
```

**Применение:** Детекция code-switching, фильтрация смешанных текстов.

### Анализ OOV

#### analyzeOOVWords
Находит out-of-vocabulary слова в тексте.

```nim
proc analyzeOOVWords*(tokenizer: Tokenizer, text: string): seq[string]
```

**Применение:** Оценка покрытия словаря, поиск кандидатов для добавления.

### Извлечение границ

#### extractTokenBoundaries
Извлекает границы токенов в тексте.

```nim
proc extractTokenBoundaries*(
  tokenizer: Tokenizer,
  text: string
): seq[tuple[start: int, end: int, token: string]]
```

**Применение:** Визуализация токенизации.

### Потоковая обработка

#### tokenizeStream
Токенизирует файл в потоковом режиме.

```nim
proc tokenizeStream*(
  tokenizer: Tokenizer,
  inputPath: string,
  outputPath: string,
  batchSize: int = 1000,
  progressCallback: proc(processed: int) = nil
)
```

**Описание:** Обрабатывает большие файлы без загрузки всего содержимого в память.

**Формат вывода:** JSON Lines
```json
{"tokens": [1, 245, 678, 2], "length": 4}
{"tokens": [1, 890, 123, 456, 2], "length": 5}
```

**Пример:**
```nim
tokenizer.tokenizeStream(
  "large_corpus.txt",
  "tokenized_corpus.jsonl",
  batchSize = 5000,
  progressCallback = proc(processed: int) =
    if processed mod 10000 == 0:
      echo "Обработано строк: ", processed
)
```

### Потокобезопасность

#### newThreadSafeTokenizer
Создаёт потокобезопасную обёртку.

```nim
proc newThreadSafeTokenizer*(tokenizer: Tokenizer): ThreadSafeTokenizer
```

**Применение:** Параллельная обработка с использованием нескольких потоков.

**Пример:**
```nim
let safeTokenizer = newThreadSafeTokenizer(tokenizer)
// Теперь можно безопасно использовать из разных потоков
```

---

## Примеры использования

### Пример 1: Базовая токенизация

```nim
import tokenization

let corpus = @[
  "машинное обучение это важная область",
  "глубокое обучение использует нейронные сети"
]

let tokenizer = trainBPE(corpus, vocabSize = 1000)
let text = "машинное обучение и нейронные сети"
let tokens = tokenizer.tokenize(text)
let tokenIds = tokenizer.encode(text)

echo "Токены: ", tokens
echo "IDs: ", tokenIds

let decoded = tokenizer.decode(tokenIds)
echo "Декодировано: ", decoded

tokenizer.save("my_tokenizer.json")
```

### Пример 2: Сравнение токенизаторов

```nim
let corpus = readCorpus("data/train.txt")

let bpe = trainBPE(corpus, vocabSize = 5000)
let wordpiece = trainWordPiece(corpus, vocabSize = 5000)
let sentencepiece = trainSentencePiece(corpus, vocabSize = 5000)

let comparison = compareTokenizers(
  @[bpe, wordpiece, sentencepiece],
  testCorpus,
  names = @["BPE", "WordPiece", "SentencePiece"]
)

for name, metrics in comparison:
  echo name, ": compression=", metrics.compressionRatio
```

### Пример 3: Подготовка данных для BERT

```nim
let tokenizer = load("bert_tokenizer.json")

let sent1 = "Какая погода сегодня?"
let sent2 = "Сегодня солнечно и тепло."

let ids1 = tokenizer.encode(sent1, addSpecialTokens = false)
let ids2 = tokenizer.encode(sent2, addSpecialTokens = false)

// [CLS] sent1 [SEP] sent2 [SEP]
var inputIds = @[tokenizer.getClsTokenId()]
inputIds.add(ids1)
inputIds.add(tokenizer.getSepTokenId())
inputIds.add(ids2)
inputIds.add(tokenizer.getSepTokenId())

var tokenTypeIds: seq[int] = @[]
tokenTypeIds.add(0)  // [CLS]
for i in 0..<ids1.len: tokenTypeIds.add(0)
tokenTypeIds.add(0)  // [SEP]
for i in 0..<ids2.len: tokenTypeIds.add(1)
tokenTypeIds.add(1)  // [SEP]

let attentionMask = newSeqWith(inputIds.len, 1)
```

### Пример 4: Masked Language Modeling

```nim
let tokenizer = load("mlm_tokenizer.json")
let text = "машинное обучение использует алгоритмы"
let tokenIds = tokenizer.encode(text, addSpecialTokens = true)

let (maskedIds, labels) = maskTokens(
  tokenIds,
  tokenizer,
  maskProb = 0.15
)

echo "Оригинал: ", tokenizer.decode(tokenIds)
echo "Маскирован: ", tokenizer.decode(maskedIds)
```

### Пример 5: Потоковая обработка

```nim
let tokenizer = load("tokenizer.json")

tokenizer.tokenizeStream(
  inputPath = "large_dataset.txt",
  outputPath = "tokenized_dataset.jsonl",
  batchSize = 10000,
  progressCallback = proc(processed: int) =
    if processed mod 100000 == 0:
      echo "Обработано: ", processed
)
```

### Пример 6: Субсловная регуляризация

```nim
let tokenizer = trainBPE(corpus, vocabSize = 5000)
let text = "машинное обучение"

for i in 1..5:
  let tokens = tokenizer.encodeWithDropout(text, dropoutRate = 0.3, seed = i)
  echo "Вариант ", i, ": ", tokenizer.decode(tokens)
```

### Пример 7: Анализ словаря

```nim
let tokenizer = load("tokenizer.json")
let analysis = analyzeVocabulary(tokenizer, testCorpus, topK = 20)

echo "Размер: ", analysis.vocabSize
echo "Средняя длина: ", analysis.avgTokenLength
echo "Покрытие: ", analysis.coverageRate * 100, "%"

echo "\nТоп-20 частых:"
for (token, freq) in analysis.mostFrequent:
  echo "  '", token, "': ", freq
```

### Пример 8: Многоязычная токенизация

```nim
let tokenizerRU = trainBPE(russianCorpus, vocabSize = 30000)
let tokenizerEN = trainBPE(englishCorpus, vocabSize = 30000)

let multilingualTokenizer = alignVocabularies(tokenizerRU, tokenizerEN)

let mixedText = "Hello! Привет! How are you?"
let tokens = multilingualTokenizer.tokenize(mixedText)
echo "Токены: ", tokens
echo "Язык: ", detectLanguage(mixedText)
```

---

## Заключение

Библиотека предоставляет полный набор инструментов для токенизации текста с поддержкой современных алгоритмов (BPE, WordPiece, SentencePiece, Byte-Level BPE). Основные возможности:

- **Обучение и использование** различных типов токенизаторов
- **Сохранение и загрузка** моделей с версионированием
- **Пакетная обработка** и параллелизация
- **Потоковая обработка** для больших файлов
- **Продвинутая нормализация** текста
- **Метрики и анализ** для оценки качества
- **Многоязычная поддержка** и детекция языков
- **Субсловная регуляризация** для аугментации данных
- **Кэширование** для ускорения работы
- **Потокобезопасность** для параллельной обработки

Библиотека оптимизирована для производительности и может обрабатывать миллионы токенов в секунду на современных процессорах.

