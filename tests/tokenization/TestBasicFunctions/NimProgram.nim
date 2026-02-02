import chardet, unicode
import std/[strformat, times, strutils]


# Тестовый набор данных
type
  TestCase = object
    name: string
    text: string
    expected: Encoding
    minConfidence: float

let testCases = @[
  # UTF-8 тесты
  TestCase(
    name: "Короткий русский текст UTF-8",
    text: "Привет, мир!",
    expected: UTF8,
    minConfidence: 0.8
  ),
  TestCase(
    name: "Длинный русский текст UTF-8",
    text: """
    Однажды в студёную зимнюю пору я из лесу вышел; был сильный мороз.
    Гляжу, поднимается медленно в гору лошадка, везущая хворосту воз.
    И шествуя важно, в спокойствии чинном, лошадку ведёт под уздцы мужичок
    В больших сапогах, в полушубке овчинном, в больших рукавицах... а сам с ноготок!
    """,
    expected: UTF8,
    minConfidence: 0.85
  ),
  TestCase(
    name: "Смешанный текст (русский + английский)",
    text: "Today's temperature в Москве составляет +15°C. The weather is nice!",
    expected: UTF8,
    minConfidence: 0.7
  ),
  TestCase(
    name: "Текст с цифрами и знаками",
    text: "Цена товара: 1500 руб. Скидка: -20%. Email: test@example.com",
    expected: UTF8,
    minConfidence: 0.75
  ),
  
  # ASCII тесты
  TestCase(
    name: "Чистый ASCII",
    text: "Hello, World! This is a pure ASCII text without any special characters.",
    expected: ASCII,
    minConfidence: 1.0
  ),
  TestCase(
    name: "ASCII с цифрами",
    text: "The year 2024, temperature +25C, price $100.50",
    expected: ASCII,
    minConfidence: 1.0
  ),
  
  # Специальные случаи
  TestCase(
    name: "Только кириллица",
    text: "абвгдеёжзийклмнопрстуфхцчшщъыьэюя АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ",
    expected: UTF8,
    minConfidence: 0.8
  ),
  TestCase(
    name: "Общие русские слова",
    text: "и в не на что я с он как по но они к у ты из мы за это который",
    expected: UTF8,
    minConfidence: 0.85
  ),
  TestCase(
    name: "Текст с распространёнными биграммами",
    text: "Стоит отметить, что наш вокзал находится в центре города",
    expected: UTF8,
    minConfidence: 0.85
  ),
  
  # Пограничные случаи
  TestCase(
    name: "Очень короткий текст",
    text: "Да",
    expected: UTF8,
    minConfidence: 0.5
  ),
  TestCase(
    name: "Один символ",
    text: "A",
    expected: ASCII,
    minConfidence: 1.0
  ),
  TestCase(
    name: "Пустая строка",
    text: "",
    expected: UNKNOWN,
    minConfidence: 0.0
  )
]

# Функция запуска тестов
proc runTests() =
  echo "════════════════════════════════════════════════════════════════"
  echo "ТЕСТИРОВАНИЕ ОПРЕДЕЛИТЕЛЯ КОДИРОВКИ CHARDET"
  echo "════════════════════════════════════════════════════════════════"
  echo ""
  
  var passed = 0
  var failed = 0
  var total = testCases.len
  
  for i, test in testCases:
    echo fmt"{i+1}. {test.name}"
    echo fmt"   Текст: {test.text[0..min(50, test.text.len-1)]}..."
    
    let startTime = cpuTime()
    let (detected, confidence) = charDetDetailed(test.text)
    let elapsed = (cpuTime() - startTime) * 1000.0  # в миллисекундах
    
    echo fmt"   Ожидается: {test.expected}"
    echo fmt"   Определено: {detected}"
    echo fmt"   Уверенность: {confidence * 100:.2f}%"
    echo fmt"   Время: {elapsed:.3f} мс"
    
    if detected == test.expected and confidence >= test.minConfidence:
      echo "   Результат: PASS"
      inc passed
    else:
      echo "   Результат: FAIL"
      if detected != test.expected:
        echo fmt"     - Неверная кодировка (ожидалось {test.expected}, получено {detected})"
      if confidence < test.minConfidence:
        echo fmt"     - Низкая уверенность (ожидалось >= {test.minConfidence * 100:.0f}%, получено {confidence * 100:.2f}%)"
      inc failed
    
    echo ""
  
  # Итоговая статистика
  echo "════════════════════════════════════════════════════════════════"
  echo "ИТОГОВАЯ СТАТИСТИКА:"
  echo fmt"  Всего тестов: {total}"
  echo fmt"  Пройдено: {passed}"
  if failed > 0:
    echo fmt"  Провалено: {failed}"
  echo fmt"  Процент успеха: {(passed.float / total.float * 100.0):.2f}%"
  echo "════════════════════════════════════════════════════════════════"

# Тесты производительности
proc performanceTests() =
  echo ""
  echo "════════════════════════════════════════════════════════════════"
  echo "ТЕСТЫ ПРОИЗВОДИТЕЛЬНОСТИ"
  echo "════════════════════════════════════════════════════════════════"
  echo ""
  
  # Генерация больших текстов
  let smallText = repeat("Привет, мир! ", 10)  # ~130 символов
  let mediumText = repeat("Это тестовый русский текст для проверки производительности. ", 100)  # ~6000 символов
  let largeText = repeat("В начале было слово, и слово было у Бога, и слово было Бог. ", 1000)  # ~60000 символов
  
  proc measurePerformance(name: string, text: string, iterations: int) =
    var totalTime = 0.0
    
    for i in 1..iterations:
      let start = cpuTime()
      discard charDet(text)
      totalTime += cpuTime() - start
    
    let avgTime = (totalTime / iterations.float) * 1000.0
    let textSize = text.len.float / 1024.0  # в КБ
    let throughput = textSize / (avgTime / 1000.0)  # КБ/сек
    
    echo fmt"{name}:"
    echo fmt"  Размер текста: {text.len} символов ({textSize:.2f} КБ)"
    echo fmt"  Итераций: {iterations}"
    echo fmt"  Среднее время: {avgTime:.3f} мс"
    echo fmt"  Пропускная способность: {throughput:.2f} КБ/сек"
    echo ""
  
  measurePerformance("Маленький текст", smallText, 1000)
  measurePerformance("Средний текст", mediumText, 100)
  measurePerformance("Большой текст", largeText, 10)

# Интерактивный режим
proc interactiveMode() =
  echo ""
  echo "════════════════════════════════════════════════════════════════"
  echo "ИНТЕРАКТИВНЫЙ РЕЖИМ"
  echo "════════════════════════════════════════════════════════════════"
  echo ""
  echo "Введите текст для анализа (или команды):"
  echo "  'quit' или 'exit' - выход"
  echo "  'help' - справка"
  echo "  'stats' - статистика текущей сессии"
  echo ""
  
  var sessionStats = 0
  
  while true:
    stdout.write("> ")
    let input = stdin.readLine()
    
    if input.toLower in ["quit", "exit", "q"]:
      echo "До свидания!"
      break
    
    if input.toLower == "help":
      echo "Доступные команды:"
      echo "  quit, exit, q - выход из программы"
      echo "  help - эта справка"
      echo "  stats - статистика анализа"
      echo "  clear - очистить статистику"
      continue
    
    if input.toLower == "stats":
      echo fmt"Проанализировано текстов в этой сессии: {sessionStats}"
      continue
    
    if input.toLower == "clear":
      sessionStats = 0
      echo "Статистика очищена"
      continue
    
    if input.len == 0:
      echo "Пустая строка"
      continue
    
    inc sessionStats
    
    let start = cpuTime()
    let (encoding, confidence) = charDetDetailed(input)
    let elapsed = (cpuTime() - start) * 1000.0
    
    echo ""
    echo fmt"  Длина: {input.len} символов"
    echo fmt"  Кодировка: {encoding}"
    echo fmt"  Уверенность: {confidence * 100:.2f}%"
    echo fmt"  Время анализа: {elapsed:.3f} мс"
    
    # Дополнительная информация
    var hasCyril = false
    for rune in input.runes:
      if rune >=% Rune(0x0400) and rune <=% Rune(0x04FF):
        hasCyril = true
        break
    
    if hasCyril:
      echo "  Содержит: кириллицу"
    
    var hasLatin = false
    for c in input:
      if (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z'):
        hasLatin = true
        break
    
    if hasLatin:
      echo "  Содержит: латиницу"
    
    echo ""

# Главная функция
when isMainModule:
  runTests()
  performanceTests()
  interactiveMode()