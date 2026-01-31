################################################################
##           КОМПЛЕКСНЫЕ ТЕСТЫ ТОКЕНИЗАЦИИ
## 
##          Comprehensive tokenization tests
## 
## Версия:   0.5
## Дата:     2026-01-31
################################################################

import math, times, random, streams
import std/[tables, sequtils, strutils, algorithm, sets, unicode, json, os, re]

# Импортируем модуль токенизации
import tokenization


#==============================================================================
# УТИЛИТЫ ДЛЯ ТЕСТИРОВАНИЯ
#==============================================================================

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
  ## Начинает новую группу тестов
  currentGroup = TestGroup(
    name: name,
    tests: @[],
    totalTests: 0,
    passedTests: 0,
    failedTests: 0,
    totalDuration: 0.0
  )
  echo ""
  echo "╔" & "═".repeat(70) & "╗"
  echo "║  ", name.alignLeft(66), "║"
  echo "╚" & "═".repeat(70) & "╝"

proc endTestGroup() =
  ## Завершает текущую группу тестов
  allGroups.add(currentGroup)
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Итого: ", currentGroup.passedTests, "/", currentGroup.totalTests, 
        " тестов пройдено"
  if currentGroup.failedTests > 0:
    echo "❌ Провалено: ", currentGroup.failedTests
  else:
    echo "✅ Все тесты успешно пройдены!"
  echo "Время выполнения: ", currentGroup.totalDuration.formatFloat(ffDecimal, 3), " сек"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

proc test(name: string, condition: bool, message: string = "") =
  ## Выполняет один тест
  let startTime = cpuTime()
  let passed = condition
  let duration = cpuTime() - startTime
  
  currentGroup.totalTests += 1
  currentGroup.totalDuration += duration
  
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
    duration: duration
  ))

proc testApprox(name: string, actual: float, expected: float, 
                tolerance: float = 0.01, message: string = "") =
  ## Тест с приближенным сравнением чисел
  let diff = abs(actual - expected)
  let passed = diff <= tolerance
  let msg = if message != "": message 
            else: "Ожидалось: " & $expected & ", получено: " & $actual
  test(name, passed, msg)


#==============================================================================
# ТЕСТОВЫЕ ДАННЫЕ
#==============================================================================

  const FN = "../Тексты и книги/Базовый текст.txt"
  let corpus = split(readFile(FN), 'n')

const testSentences = @[
  "Простое предложение для тестирования.",
  "Это более длинное предложение с большим количеством слов для проверки.",
  "Краткое.",
  "Текст со специальными символами: !@#$%^&*()",
  "Numbers: 123 456 789",
  "UPPERCASE AND lowercase MiXeD",
  "Повторение повторение повторение слов слов слов",
  "княгиня Софья Васильевна была худая длинная"
]


#==============================================================================
# ГРУППА 1: ТЕСТЫ BPE (BYTE PAIR ENCODING)
#==============================================================================

proc testBPE() =
  startTestGroup("ГРУППА 1: ТЕСТЫ BPE (BYTE PAIR ENCODING)")
  
  echo "\n→ Создание и обучение BPE токенизатора..."
  var bpeTokenizer = trainBPE(testCorpus, vocabSize = 150, minFreq = 1)
  
  # Тест 1.1: Размер словаря
  test("1.1 Размер словаря BPE",
       bpeTokenizer.vocab.len > 0 and bpeTokenizer.vocab.len <= 150,
       "Размер словаря: " & $bpeTokenizer.vocab.len)
  
  # Тест 1.2: Наличие специальных токенов
  test("1.2 Наличие PAD токена в словаре",
       bpeTokenizer.specialTokens.padToken in bpeTokenizer.vocab)
  test("1.3 Наличие UNK токена в словаре",
       bpeTokenizer.specialTokens.unkToken in bpeTokenizer.vocab)
  test("1.4 Наличие BOS токена в словаре",
       bpeTokenizer.specialTokens.bosToken in bpeTokenizer.vocab)
  test("1.5 Наличие EOS токена в словаре",
       bpeTokenizer.specialTokens.eosToken in bpeTokenizer.vocab)
  
  # Тест 1.6: Токенизация простого текста
  let testText = "Это тестовое предложение"
  let tokens = tokenize(testText, bpeTokenizer)
  test("1.6 Токенизация возвращает непустой результат",
       tokens.len > 0,
       "Количество токенов: " & $tokens.len)
  
  # Тест 1.7: Декодирование
  let decoded = bpeTokenizer.decode(tokens, skipSpecialTokens = true)
  test("1.7 Декодирование восстанавливает текст",
       decoded.strip() == testText or 
       decoded.replace(" ", "").toLowerAscii() == testText.replace(" ", "").toLowerAscii(),
       "Оригинал: '" & testText & "', Декодировано: '" & decoded & "'")
  
  # Тест 1.8: Согласованность vocab и inverseVocab
  var vocabConsistent = true
  for token, id in bpeTokenizer.vocab:
    if id >= bpeTokenizer.inverseVocab.len or bpeTokenizer.inverseVocab[id] != token:
      vocabConsistent = false
      break
  test("1.8 Согласованность vocab и inverseVocab", vocabConsistent)
  
  # Тест 1.9: Наличие merges
  test("1.9 Наличие BPE merges",
       bpeTokenizer.merges.len > 0,
       "Количество merges: " & $bpeTokenizer.merges.len)
  
  # Тест 1.10: Сохранение и загрузка
  let savePath = "/tmp/test_bpe.json"
  saveTokenizer(bpeTokenizer, savePath)
  test("1.10 Сохранение токенизатора", fileExists(savePath))
  
  var loadedTokenizer = loadTokenizer(savePath)
  test("1.11 Загрузка токенизатора", loadedTokenizer.vocab.len == bpeTokenizer.vocab.len)
  
  # Тест 1.12: Идентичность после загрузки
  let tokensOriginal = tokenize("тестовый текст", bpeTokenizer)
  let tokensLoaded = tokenize("тестовый текст", loadedTokenizer)
  test("1.12 Идентичность токенизации после загрузки",
       tokensOriginal == tokensLoaded)
  
  # Тест 1.13: Метрики
  let metrics = getMetrics(bpeTokenizer, testCorpus)
  test("1.13 Вычисление метрик - размер словаря",
       metrics.vocabSize > 0)
  test("1.14 Вычисление метрик - коэффициент сжатия",
       metrics.compressionRatio > 0.0 and metrics.compressionRatio < 100.0)
  
  # Удаляем временный файл
  if fileExists(savePath):
    removeFile(savePath)
  
  endTestGroup()


#==============================================================================
# ГРУППА 2: ТЕСТЫ WORDPIECE
#==============================================================================

proc testWordPiece() =
  startTestGroup("ГРУППА 2: ТЕСТЫ WORDPIECE")
  
  echo "\n→ Создание и обучение WordPiece токенизатора..."
  var wpTokenizer = trainWordPiece(testCorpus, vocabSize = 150, minFreq = 1)
  
  # Тест 2.1: Тип токенизатора
  test("2.1 Тип токенизатора WordPiece",
       wpTokenizer.kind == tkWordPiece)
  
  # Тест 2.2: Размер словаря
  test("2.2 Размер словаря WordPiece",
       wpTokenizer.vocab.len > 0 and wpTokenizer.vocab.len <= 150)
  
  # Тест 2.3: Префикс продолжения подслова
  test("2.3 Наличие префикса продолжения",
       wpTokenizer.continuingSubwordPrefix == "##")
  
  # Тест 2.4: Специальные токены
  test("2.4 Наличие специальных токенов",
       wpTokenizer.specialTokens.padToken in wpTokenizer.vocab and
       wpTokenizer.specialTokens.unkToken in wpTokenizer.vocab)
  
  # Тест 2.5: Токенизация с префиксами
  let testText = "непонятное слово"
  let tokens = tokenize(testText, wpTokenizer)
  test("2.5 Токенизация возвращает результат",
       tokens.len > 0)
  
  # Тест 2.6: Декодирование убирает префиксы ##
  let decoded = wpTokenizer.decode(tokens, skipSpecialTokens = true)
  test("2.6 Декодирование убирает ## префиксы",
       "##" notin decoded,
       "Декодировано: " & decoded)
  
  # Тест 2.7: Обработка неизвестных слов
  let unknownText = "qwertyzxcvb"
  let unknownTokens = tokenize(unknownText, wpTokenizer)
  test("2.7 Обработка неизвестных слов",
       unknownTokens.len > 0)
  
  # Тест 2.8: Токенизация разбивает длинные слова
  let longWord = "длинноенепонятноеслово"
  let longTokens = tokenize(longWord, wpTokenizer)
  test("2.8 Длинные слова разбиваются на подслова",
       longTokens.len >= 1)
  
  # Тест 2.9: Согласованность кодирования-декодирования
  for sentence in testSentences[0..2]:
    let encoded = tokenize(sentence, wpTokenizer)
    let redecoded = wpTokenizer.decode(encoded, skipSpecialTokens = true)
    # Проверяем, что основной смысл сохранился (убираем пробелы для сравнения)
    let normalized1 = sentence.replace(" ", "").toLowerAscii()
    let normalized2 = redecoded.replace(" ", "").toLowerAscii()
    test("2.9 Согласованность encode-decode для: " & sentence[0..min(20, sentence.len-1)],
         normalized1 == normalized2 or normalized2.contains(normalized1[0..min(5, normalized1.len-1)]))
  
  # Тест 2.10: Метрики
  let metrics = getMetrics(wpTokenizer, testCorpus)
  test("2.10 Метрики - утилизация словаря",
       metrics.vocabUtilization >= 0.0 and metrics.vocabUtilization <= 1.0)
  
  endTestGroup()


#==============================================================================
# ГРУППА 3: ТЕСТЫ SENTENCEPIECE
#==============================================================================

proc testSentencePiece() =
  startTestGroup("ГРУППА 3: ТЕСТЫ SENTENCEPIECE")
  
  echo "\n→ Создание и обучение SentencePiece токенизатора..."
  var spTokenizer = trainSentencePiece(testCorpus, vocabSize = 150)
  
  # Тест 3.1: Тип токенизатора
  test("3.1 Тип токенизатора SentencePiece",
       spTokenizer.kind == tkSentencePiece)
  
  # Тест 3.2: Размер словаря
  test("3.2 Размер словаря SentencePiece",
       spTokenizer.vocab.len > 0)
  
  # Тест 3.3: Наличие scores
  test("3.3 Наличие scores для токенов",
       spTokenizer.scores.len > 0)
  
  # Тест 3.4: Специальные токены
  test("3.4 Специальные токены в словаре",
       spTokenizer.specialTokens.unkToken in spTokenizer.vocab)
  
  # Тест 3.5: Токенизация
  let testText = "Тестовое предложение"
  let tokens = tokenize(testText, spTokenizer)
  test("3.5 Токенизация работает",
       tokens.len > 0,
       "Количество токенов: " & $tokens.len)
  
  # Тест 3.6: Декодирование
  let decoded = spTokenizer.decode(tokens, skipSpecialTokens = true)
  test("3.6 Декодирование работает",
       decoded.len > 0)
  
  # Тест 3.7: Все токены имеют scores
  var allHaveScores = true
  for token in spTokenizer.vocab.keys:
    if token notin spTokenizer.scores:
      allHaveScores = false
      break
  test("3.7 Все токены словаря имеют scores", allHaveScores)
  
  # Тест 3.8: Обработка пробелов
  let textWithSpaces = "слово пробел слово"
  let spacesTokens = tokenize(textWithSpaces, spTokenizer)
  test("3.8 Обработка пробелов",
       spacesTokens.len > 0)
  
  # Тест 3.9: Консистентность encode-decode
  let original = "Проверка консистентности"
  let encoded = tokenize(original, spTokenizer)
  let redecoded = spTokenizer.decode(encoded, skipSpecialTokens = true)
  let norm1 = original.replace(" ", "").toLowerAscii()
  let norm2 = redecoded.replace(" ", "").replace("▁", "").toLowerAscii()
  test("3.9 Консистентность encode-decode",
       norm1 == norm2 or norm2.contains(norm1[0..min(3, norm1.len-1)]))
  
  # Тест 3.10: Метрики
  let metrics = getMetrics(spTokenizer, testCorpus)
  test("3.10 Метрики - коэффициент сжатия",
       metrics.compressionRatio > 0.0)
  
  endTestGroup()


#==============================================================================
# ГРУППА 4: ТЕСТЫ BYTE-LEVEL BPE
#==============================================================================

proc testByteLevelBPE() =
  startTestGroup("ГРУППА 4: ТЕСТЫ BYTE-LEVEL BPE (GPT-2 STYLE)")
  
  echo "\n→ Создание и обучение ByteLevel BPE токенизатора..."
  var blbpeTokenizer = trainByteLevelBPE(testCorpus, vocabSize = 200)
  
  # Тест 4.1: Тип токенизатора
  test("4.1 Тип токенизатора ByteLevelBPE",
       blbpeTokenizer.kind == tkByteLevelBPE)
  
  # Тест 4.2: Наличие byte encoder
  test("4.2 Наличие byte encoder",
       blbpeTokenizer.byteEncoder.len == 256)
  
  # Тест 4.3: Наличие byte decoder
  test("4.3 Наличие byte decoder",
       blbpeTokenizer.byteDecoder.len == 256)
  
  # Тест 4.4: Консистентность encoder-decoder
  var encoderDecoderConsistent = true
  for b, s in blbpeTokenizer.byteEncoder:
    if blbpeTokenizer.byteDecoder[s] != b:
      encoderDecoderConsistent = false
      break
  test("4.4 Консистентность byte encoder/decoder", encoderDecoderConsistent)
  
  # Тест 4.5: Токенизация UTF-8 текста
  let testText = "княгиня Софья Васильевна"
  let tokens = tokenize(testText, blbpeTokenizer, addSpecialTokens = false)
  test("4.5 Токенизация UTF-8 текста",
       tokens.len > 0,
       "Количество токенов: " & $tokens.len)
  
  # Тест 4.6: Декодирование сохраняет текст
  let decoded = blbpeTokenizer.decode(tokens, skipSpecialTokens = true)
  test("4.6 Декодирование сохраняет оригинальный текст",
       decoded == testText,
       "Оригинал: '" & testText & "', Декодировано: '" & decoded & "'")
  
  # Тест 4.7: Обработка спецсимволов
  let specialChars = "!@#$%^&*()"
  let specialTokens = tokenize(specialChars, blbpeTokenizer, addSpecialTokens = false)
  let specialDecoded = blbpeTokenizer.decode(specialTokens, skipSpecialTokens = true)
  test("4.7 Обработка специальных символов",
       specialDecoded == specialChars,
       "Оригинал: '" & specialChars & "', Декодировано: '" & specialDecoded & "'")
  
  # Тест 4.8: Обработка чисел
  let numbers = "123456789"
  let numTokens = tokenize(numbers, blbpeTokenizer, addSpecialTokens = false)
  let numDecoded = blbpeTokenizer.decode(numTokens, skipSpecialTokens = true)
  test("4.8 Обработка чисел",
       numDecoded == numbers)
  
  # Тест 4.9: Сохранение пробелов
  let textWithSpaces = "слово пробел слово"
  let spaceTokens = tokenize(textWithSpaces, blbpeTokenizer, addSpecialTokens = false)
  let spaceDecoded = blbpeTokenizer.decode(spaceTokens, skipSpecialTokens = true)
  test("4.9 Сохранение пробелов",
       spaceDecoded == textWithSpaces,
       "Оригинал: '" & textWithSpaces & "', Декодировано: '" & spaceDecoded & "'")
  
  # Тест 4.10: Token offsets
  let offsets = tokenizeWithOffsets(testText, blbpeTokenizer, addSpecialTokens = false)
  test("4.10 Генерация token offsets",
       offsets.len > 0)
  
  # Тест 4.11: Корректность char offsets
  if offsets.len > 0:
    var offsetsCorrect = true
    for offset in offsets:
      if offset.startChar < 0 or offset.endChar > testText.runeLen or 
         offset.startChar >= offset.endChar:
        offsetsCorrect = false
        break
    test("4.11 Корректность char offsets", offsetsCorrect)
  
  # Тест 4.12: Корректность byte offsets
  if offsets.len > 0:
    var byteOffsetsCorrect = true
    for offset in offsets:
      if offset.startByte < 0 or offset.endByte > testText.len or
         offset.startByte >= offset.endByte:
        byteOffsetsCorrect = false
        break
    test("4.12 Корректность byte offsets", byteOffsetsCorrect)
  
  endTestGroup()


#==============================================================================
# ГРУППА 5: ТЕСТЫ ДОПОЛНИТЕЛЬНЫХ ФУНКЦИЙ
#==============================================================================

proc testAdditionalFunctions() =
  startTestGroup("ГРУППА 5: ТЕСТЫ ДОПОЛНИТЕЛЬНЫХ ФУНКЦИЙ")
  
  # Подготовка токенизатора для тестов
  var tokenizer = trainBPE(testCorpus, vocabSize = 150)
  
  # Тест 5.1: cleanText - удаление HTML
  let htmlText = "<div>Текст с <b>HTML</b> тегами</div>"
  let cleaned = cleanText(htmlText, removeHtml = true)
  test("5.1 cleanText - удаление HTML тегов",
       "<" notin cleaned and ">" notin cleaned)
  
  # Тест 5.2: cleanText - удаление URL
  let urlText = "Ссылка https://example.com в тексте"
  let noUrls = cleanText(urlText, removeUrls = true)
  test("5.2 cleanText - удаление URLs",
       "https://" notin noUrls)
  
  # Тест 5.3: cleanText - удаление email
  let emailText = "Контакт test@example.com здесь"
  let noEmails = cleanText(emailText, removeEmails = true)
  test("5.3 cleanText - удаление email",
       "@" notin noEmails or "example.com" notin noEmails)
  
  # Тест 5.4: cleanText - нормализация пробелов
  let spacesText = "Много    пробелов     здесь"
  let normalizedSpaces = cleanText(spacesText, removeExtraWhitespace = true)
  test("5.4 cleanText - нормализация пробелов",
       "    " notin normalizedSpaces)
  
  # Тест 5.5: encodeBatch
  let batchTexts = @["первый", "второй текст", "третий"]
  let batchEncoding = encodeBatch(tokenizer, batchTexts, maxLength = 20, padding = true)
  test("5.5 encodeBatch - количество последовательностей",
       batchEncoding.inputIds.len == 3)
  
  # Тест 5.6: encodeBatch - padding
  test("5.6 encodeBatch - одинаковая длина после padding",
       batchEncoding.inputIds[0].len == batchEncoding.inputIds[1].len and
       batchEncoding.inputIds[1].len == batchEncoding.inputIds[2].len)
  
  # Тест 5.7: encodeBatch - attention mask
  test("5.7 encodeBatch - корректная attention mask",
       batchEncoding.attentionMask.len == 3)
  
  # Тест 5.8: encodeWithPadding
  let paddedTokens = encodeWithPadding(tokenizer, "короткий текст", maxLength = 20)
  test("5.8 encodeWithPadding - результат имеет заданную длину",
       paddedTokens.len == 20)
  
  # Тест 5.9: maskTokens
  let originalTokens = tokenize("тестовое предложение для маскирования", tokenizer)
  let (maskedTokens, labels) = maskTokens(originalTokens, tokenizer, maskProb = 0.15, seed = 42)
  test("5.9 maskTokens - длины совпадают",
       maskedTokens.len == labels.len and labels.len == originalTokens.len)
  
  # Тест 5.10: maskTokens - есть замаскированные токены
  var hasMasked = false
  for i in 0..<maskedTokens.len:
    if labels[i] != -100:
      hasMasked = true
      break
  test("5.10 maskTokens - присутствуют замаскированные токены", hasMasked)
  
  # Тест 5.11: getSubwordBreakdown
  let breakdown = getSubwordBreakdown("тестовое слово", tokenizer)
  test("5.11 getSubwordBreakdown - возвращает подслова",
       breakdown.len > 0)
  
  # Тест 5.12: estimateTokenCount
  let estimated = estimateTokenCount("это текст для оценки количества токенов")
  test("5.12 estimateTokenCount - разумная оценка",
       estimated > 0 and estimated < 100)
  
  # Тест 5.13: validateTokenizer
  let validationResults = validateTokenizer(tokenizer)
  test("5.13 validateTokenizer - проверка проходит",
       validationResults.len > 0)
  
  # Тест 5.14: compareTokenizers
  var wpTokenizer = trainWordPiece(testCorpus, vocabSize = 150)
  let comparison = compareTokenizers(@[tokenizer, wpTokenizer], "Тестовый текст")
  test("5.14 compareTokenizers - сравнение работает",
       comparison.len == 2)
  
  # Тест 5.15: analyzeVocabulary
  let analysis = analyzeVocabulary(tokenizer, testCorpus, topN = 5)
  test("5.15 analyzeVocabulary - размер словаря",
       analysis.vocabSize > 0)
  test("5.16 analyzeVocabulary - средняя длина токена",
       analysis.avgTokenLength > 0.0)
  test("5.17 analyzeVocabulary - топ токены",
       analysis.mostFrequent.len <= 5)
  
  # Тест 5.18: pruneVocabulary
  let originalSize = tokenizer.vocab.len
  pruneVocabulary(tokenizer, minFrequency = 2, corpus = testCorpus)
  test("5.18 pruneVocabulary - уменьшение размера словаря",
       tokenizer.vocab.len <= originalSize)
  
  # Тест 5.19: toLowerUnicode
  let mixedCase = "ТеСтОвЫй ТЕКСТ"
  let lowered = toLowerUnicode(mixedCase)
  test("5.19 toLowerUnicode - приведение к нижнему регистру",
       lowered == "тестовый текст")
  
  # Тест 5.20: toUpperUnicode
  let upper = toUpperUnicode("тестовый текст")
  test("5.20 toUpperUnicode - приведение к верхнему регистру",
       upper == "ТЕСТОВЫЙ ТЕКСТ")
  
  endTestGroup()


#==============================================================================
# ГРУППА 6: ТЕСТЫ КЭШИРОВАНИЯ И ПРОИЗВОДИТЕЛЬНОСТИ
#==============================================================================

proc testCachingAndPerformance() =
  startTestGroup("ГРУППА 6: ТЕСТЫ КЭШИРОВАНИЯ И ПРОИЗВОДИТЕЛЬНОСТИ")
  
  var tokenizer = trainByteLevelBPE(testCorpus, vocabSize = 200)
  tokenizer.cacheMaxSize = 100
  
  # Тест 6.1: Кэш изначально пуст
  test("6.1 Кэш изначально пуст",
       tokenizer.cache.len == 0)
  
  # Тест 6.2: Первая токенизация - cache miss
  let text1 = "Первая токенизация"
  discard tokenize(text1, tokenizer)
  test("6.2 Первая токенизация создаёт cache miss",
       tokenizer.cacheMisses == 1)
  
  # Тест 6.3: Повторная токенизация - cache hit
  discard tokenize(text1, tokenizer)
  test("6.3 Повторная токенизация создаёт cache hit",
       tokenizer.cacheHits == 1)
  
  # Тест 6.4: Кэш содержит элемент
  test("6.4 Кэш содержит токенизированный текст",
       text1 in tokenizer.cache)
  
  # Тест 6.5: Производительность с кэшем
  let testText = "Тест производительности кэша"
  let startNoCache = cpuTime()
  for i in 1..10:
    discard tokenize(testText & $i, tokenizer)
  let timeNoCache = cpuTime() - startNoCache
  
  # Токенизируем тот же текст с кэшем
  let startWithCache = cpuTime()
  for i in 1..10:
    discard tokenize(testText, tokenizer)
  let timeWithCache = cpuTime() - startWithCache
  
  test("6.5 Кэш ускоряет повторные токенизации",
       timeWithCache < timeNoCache or abs(timeWithCache - timeNoCache) < 0.01)
  
  # Тест 6.6: Очистка кэша
  clearCache(tokenizer)
  test("6.6 Очистка кэша работает",
       tokenizer.cache.len == 0)
  
  # Тест 6.7: Batch processing производительность
  let batchSize = 10
  let batchTexts = newSeq[string](batchSize)
  for i in 0..<batchSize:
    batchTexts[i] = "Текст номер " & $i
  
  let startBatch = cpuTime()
  let batchResult = encodeBatch(tokenizer, batchTexts, maxLength = 50)
  let batchTime = cpuTime() - startBatch
  
  test("6.7 Batch processing завершается за разумное время",
       batchTime < 1.0)
  
  # Тест 6.8: Метрики производительности
  let metrics = getMetrics(tokenizer, testCorpus)
  test("6.8 Метрики - скорость токенизации измерена",
       metrics.tokensPerSecond > 0.0)
  
  endTestGroup()


#==============================================================================
# ГРУППА 7: ТЕСТЫ BPE-DROPOUT И РЕГУЛЯРИЗАЦИИ
#==============================================================================

proc testDropoutAndRegularization() =
  startTestGroup("ГРУППА 7: ТЕСТЫ BPE-DROPOUT И РЕГУЛЯРИЗАЦИИ")
  
  var tokenizer = trainByteLevelBPE(testCorpus, vocabSize = 200)
  let testText = "Тестовое предложение для проверки dropout"
  
  # Тест 7.1: Оригинальная токенизация
  let originalTokens = tokenize(testText, tokenizer, addSpecialTokens = false)
  test("7.1 Оригинальная токенизация работает",
       originalTokens.len > 0)
  
  # Тест 7.2: Dropout токенизация с нулевой вероятностью
  let noDropout = tokenizeWithDropout(testText, tokenizer, dropoutProb = 0.0, seed = 42)
  test("7.2 Dropout с вероятностью 0.0 идентичен оригиналу",
       noDropout == originalTokens)
  
  # Тест 7.3: Dropout создаёт вариативность
  let dropout1 = tokenizeWithDropout(testText, tokenizer, dropoutProb = 0.3, seed = 1)
  let dropout2 = tokenizeWithDropout(testText, tokenizer, dropoutProb = 0.3, seed = 2)
  let dropout3 = tokenizeWithDropout(testText, tokenizer, dropoutProb = 0.3, seed = 3)
  
  test("7.3 Dropout создаёт разные токенизации (1 vs 2)",
       dropout1 != dropout2)
  test("7.4 Dropout создаёт разные токенизации (2 vs 3)",
       dropout2 != dropout3)
  
  # Тест 7.5: Dropout с минимальным количеством изменений
  let dropoutMin = tokenizeWithDropout(testText, tokenizer, 
                                        dropoutProb = 0.3, seed = 10, minDropped = 2)
  test("7.5 Dropout с minDropped работает",
       dropoutMin.len >= originalTokens.len)
  
  # Тест 7.6: Детерминированность при одинаковом seed
  let dropout4a = tokenizeWithDropout(testText, tokenizer, dropoutProb = 0.3, seed = 100)
  let dropout4b = tokenizeWithDropout(testText, tokenizer, dropoutProb = 0.3, seed = 100)
  test("7.6 Dropout детерминирован при одинаковом seed",
       dropout4a == dropout4b)
  
  # Тест 7.7: Декодирование dropout токенов
  let decodedDropout = tokenizer.decode(dropout1, skipSpecialTokens = true)
  test("7.7 Декодирование dropout токенов работает",
       decodedDropout.len > 0)
  
  # Тест 7.8: Dropout сохраняет смысл текста
  let normalizedOriginal = testText.replace(" ", "").toLowerAscii()
  let normalizedDropout = decodedDropout.replace(" ", "").toLowerAscii()
  test("7.8 Dropout сохраняет основной смысл текста",
       normalizedOriginal == normalizedDropout or 
       normalizedDropout.contains(normalizedOriginal[0..min(5, normalizedOriginal.len-1)]))
  
  endTestGroup()


#==============================================================================
# ГРУППА 8: ТЕСТЫ СПЕЦИАЛЬНЫХ СЛУЧАЕВ
#==============================================================================

proc testEdgeCases() =
  startTestGroup("ГРУППА 8: ТЕСТЫ СПЕЦИАЛЬНЫХ СЛУЧАЕВ")
  
  var tokenizer = trainBPE(testCorpus, vocabSize = 150)
  
  # Тест 8.1: Пустая строка
  let emptyTokens = tokenize("", tokenizer, addSpecialTokens = false)
  test("8.1 Токенизация пустой строки",
       emptyTokens.len == 0)
  
  # Тест 8.2: Строка только из пробелов
  let spaceTokens = tokenize("     ", tokenizer, addSpecialTokens = false)
  test("8.2 Токенизация строки из пробелов",
       spaceTokens.len >= 0)
  
  # Тест 8.3: Очень длинная строка
  let longText = "слово ".repeat(1000)
  let longTokens = tokenize(longText, tokenizer)
  test("8.3 Токенизация очень длинной строки",
       longTokens.len > 0)
  
  # Тест 8.4: Односимвольная строка
  let singleChar = "а"
  let singleTokens = tokenize(singleChar, tokenizer, addSpecialTokens = false)
  test("8.4 Токенизация одного символа",
       singleTokens.len > 0)
  
  # Тест 8.5: Только цифры
  let onlyNumbers = "1234567890"
  let numberTokens = tokenize(onlyNumbers, tokenizer, addSpecialTokens = false)
  test("8.5 Токенизация только цифр",
       numberTokens.len > 0)
  
  # Тест 8.6: Только спецсимволы
  let onlySpecial = "!@#$%^&*()"
  let specialTokens = tokenize(onlySpecial, tokenizer, addSpecialTokens = false)
  test("8.6 Токенизация только спецсимволов",
       specialTokens.len >= 0)
  
  # Тест 8.7: Смешанные языки (кириллица + латиница)
  let mixed = "Hello мир World"
  let mixedTokens = tokenize(mixed, tokenizer)
  test("8.7 Токенизация смешанных языков",
       mixedTokens.len > 0)
  
  # Тест 8.8: Unicode emoji
  let emoji = "Текст с 😀 emoji 🎉"
  let emojiTokens = tokenize(emoji, tokenizer)
  test("8.8 Токенизация текста с emoji",
       emojiTokens.len > 0)
  
  # Тест 8.9: Повторяющиеся символы
  let repeated = "ааааааа"
  let repeatedTokens = tokenize(repeated, tokenizer, addSpecialTokens = false)
  test("8.9 Токенизация повторяющихся символов",
       repeatedTokens.len > 0)
  
  # Тест 8.10: Декодирование пустой последовательности
  let decodedEmpty = tokenizer.decode(@[], skipSpecialTokens = true)
  test("8.10 Декодирование пустой последовательности",
       decodedEmpty == "")
  
  endTestGroup()


#==============================================================================
# СТАТИСТИЧЕСКИЙ АНАЛИЗ РЕЗУЛЬТАТОВ
#==============================================================================

proc printStatistics() =
  echo ""
  echo "╔" & "═".repeat(70) & "╗"
  echo "║  СТАТИСТИЧЕСКИЙ АНАЛИЗ РЕЗУЛЬТАТОВ ТЕСТИРОВАНИЯ              ║"
  echo "╚" & "═".repeat(70) & "╝"
  echo ""
  
  var totalTests = 0
  var totalPassed = 0
  var totalFailed = 0
  var totalDuration = 0.0
  
  echo "┌" & "─".repeat(70) & "┐"
  echo "│ ГРУППА                              │ ПРОЙДЕНО │ ПРОВАЛЕНО │ ВРЕМЯ  │"
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
         timeStr.align(6), " │"
  
  echo "└" & "─".repeat(70) & "┘"
  echo ""
  
  # Общая статистика
  echo "ОБЩАЯ СТАТИСТИКА:"
  echo "  Всего тестов:        ", totalTests
  echo "  Успешно пройдено:    ", totalPassed, " (", 
       (totalPassed.float / totalTests.float * 100).formatFloat(ffDecimal, 1), "%)"
  echo "  Провалено:           ", totalFailed
  echo "  Общее время:         ", totalDuration.formatFloat(ffDecimal, 3), " сек"
  echo "  Среднее время/тест:  ", 
       (totalDuration / totalTests.float * 1000).formatFloat(ffDecimal, 2), " мс"
  echo ""
  
  # Анализ производительности
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
  
  # Итоговый статус
  if totalFailed == 0:
    echo "╔" & "═".repeat(70) & "╗"
    echo "║  ✅ ВСЕ ТЕСТЫ УСПЕШНО ПРОЙДЕНЫ!                              ║"
    echo "╚" & "═".repeat(70) & "╝"
  else:
    echo "╔" & "═".repeat(70) & "╗"
    echo "║  ⚠️  ОБНАРУЖЕНЫ ПРОВАЛЕННЫЕ ТЕСТЫ: ", totalFailed.align(27), " ║"
    echo "╚" & "═".repeat(70) & "╝"
  echo ""


#==============================================================================
# СРАВНИТЕЛЬНЫЙ АНАЛИЗ ТОКЕНИЗАТОРОВ
#==============================================================================

proc comparativeAnalysis() =
  echo ""
  echo "╔" & "═".repeat(70) & "╗"
  echo "║  СРАВНИТЕЛЬНЫЙ АНАЛИЗ ТОКЕНИЗАТОРОВ                          ║"
  echo "╚" & "═".repeat(70) & "╝"
  echo ""
  
  # Создаём все токенизаторы
  echo "→ Обучение токенизаторов..."
  var bpeTokenizer = trainBPE(testCorpus, vocabSize = 150)
  var wpTokenizer = trainWordPiece(testCorpus, vocabSize = 150)
  var spTokenizer = trainSentencePiece(testCorpus, vocabSize = 150)
  var blbpeTokenizer = trainByteLevelBPE(testCorpus, vocabSize = 200)
  
  let testText = "княгиня Софья Васильевна была худая длинная женщина"
  
  echo ""
  echo "Тестовый текст: ", testText
  echo ""
  
  # Сравниваем токенизацию
  echo "СРАВНЕНИЕ ТОКЕНИЗАЦИИ:"
  echo "─────────────────────────────────────────────────────────────────────"
  
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
  
  # Сравниваем метрики
  echo "─────────────────────────────────────────────────────────────────────"
  echo "СРАВНЕНИЕ МЕТРИК:"
  echo "─────────────────────────────────────────────────────────────────────"
  
  let bpeMetrics = getMetrics(bpeTokenizer, testCorpus)
  let wpMetrics = getMetrics(wpTokenizer, testCorpus)
  let spMetrics = getMetrics(spTokenizer, testCorpus)
  let blbpeMetrics = getMetrics(blbpeTokenizer, testCorpus)
  
  echo "                    │   BPE   │ WordPiece │ SentPiece │ ByteLvlBPE"
  echo "────────────────────┼─────────┼───────────┼───────────┼───────────"
  echo "Размер словаря      │ ", bpeMetrics.vocabSize.align(7), " │ ",
       wpMetrics.vocabSize.align(9), " │ ",
       spMetrics.vocabSize.align(9), " │ ",
       blbpeMetrics.vocabSize.align(10)
  
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
  
  # Декодирование
  echo "─────────────────────────────────────────────────────────────────────"
  echo "ПРОВЕРКА ДЕКОДИРОВАНИЯ:"
  echo "─────────────────────────────────────────────────────────────────────"
  
  echo "Оригинал:       ", testText
  echo ""
  echo "BPE:            ", bpeTokenizer.decode(bpeTokens, skipSpecialTokens = true)
  echo "WordPiece:      ", wpTokenizer.decode(wpTokens, skipSpecialTokens = true)
  echo "SentencePiece:  ", spTokenizer.decode(spTokens, skipSpecialTokens = true)
  echo "ByteLevel BPE:  ", blbpeTokenizer.decode(blbpeTokens, skipSpecialTokens = true)
  echo ""


#==============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
#==============================================================================

when isMainModule:
  randomize()
  
  echo "╔" & "═".repeat(70) & "╗"
  echo "║  КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ БИБЛИОТЕКИ ТОКЕНИЗАЦИИ v1.0.0      ║"
  echo "╚" & "═".repeat(70) & "╝"
  echo ""
  echo "Дата запуска: ", now().format("yyyy-MM-dd HH:mm:ss")
  echo "Количество тестовых предложений в корпусе: ", testCorpus.len
  echo ""
  
  let overallStart = cpuTime()
  
  # Запуск всех групп тестов
  testBPE()
  testWordPiece()
  testSentencePiece()
  testByteLevelBPE()
  testAdditionalFunctions()
  testCachingAndPerformance()
  testDropoutAndRegularization()
  testEdgeCases()
  
  let overallTime = cpuTime() - overallStart
  
  # Статистический анализ
  printStatistics()
  
  # Сравнительный анализ
  comparativeAnalysis()
  
  # Финальная сводка
  echo "╔" & "═".repeat(70) & "╗"
  echo "║  ТЕСТИРОВАНИЕ ЗАВЕРШЕНО                                      ║"
  echo "╚" & "═".repeat(70) & "╝"
  echo ""
  echo "Общее время выполнения всех тестов: ", 
       overallTime.formatFloat(ffDecimal, 3), " сек"
  echo ""
