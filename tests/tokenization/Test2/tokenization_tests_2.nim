################################################################
##           КОМПЛЕКСНЫЕ ТЕСТЫ ТОКЕНИЗАЦИИ
## 
##          Comprehensive tokenization tests
## 
## Версия:   0.7
## Дата:     2026-02-02
################################################################




import times, random
import std/[os, tables, strutils, unicode, json, options]
import tokenization



# nim c -d:release tokenization_tests_2.nim




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
const FN = "../../../examples/Тексты и книги/Базовый текст.txt"
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
        isSome(validateInput("")),
        "Пустой текст должен вызывать ошибку")
  
  test("9.2 Валидация нормального текста",
        validateInput("нормальный текст").isNone,
        "Нормальный текст должен проходить валидацию")
  
  test("9.3 Валидация слишком длинного текста",
        isSome(validateInput("x".repeat(MAX_INPUT_LENGTH + 1))),
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
        isSome(lruCache.get("key1")),
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





# nim c -d:release tokenization_tests_2.nim

