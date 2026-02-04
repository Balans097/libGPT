################################################################
##                ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
## 
##                  Auxiliary functions
## 
## Версия:   0.2
## Дата:     2026-06-23
## Автор:    github.com/Balans097
################################################################

# 0.2 — добавлены функции определения кодировки файлов (2026-02-02)
# 0.1 — начальная реализация библиотеки (2026-01-30)




# nim c -d:release utils.nim




import std/[paths, files, tables, strutils, unicode]




const
  # Количество строк по умолчанию,
  # которые читает функция readFirstLines
  DefNumLines = 2520


type
  Encoding* = enum
    UTF8 = "UTF-8"
    UTF16LE = "UTF-16LE"
    UTF16BE = "UTF-16BE"
    CP1251 = "Windows-1251"
    KOI8R = "KOI8-R"
    KOI8U = "KOI8-U"
    ISO88595 = "ISO-8859-5"
    CP866 = "CP866"
    CP855 = "CP855"
    ASCII = "ASCII"
    UNKNOWN = "Unknown"

  EncodingScore = object
    encoding: Encoding
    score: float
    confidence: float



# Частоты букв русского языка (в процентах)
var RussianLetterFrequencies = initTable[string, float]()
RussianLetterFrequencies["о"] = 10.983
RussianLetterFrequencies["е"] = 8.483
RussianLetterFrequencies["а"] = 7.998
RussianLetterFrequencies["и"] = 7.367
RussianLetterFrequencies["н"] = 6.700
RussianLetterFrequencies["т"] = 6.318
RussianLetterFrequencies["с"] = 5.473
RussianLetterFrequencies["р"] = 4.746
RussianLetterFrequencies["в"] = 4.533
RussianLetterFrequencies["л"] = 4.343
RussianLetterFrequencies["к"] = 3.486
RussianLetterFrequencies["м"] = 3.203
RussianLetterFrequencies["д"] = 2.977
RussianLetterFrequencies["п"] = 2.804
RussianLetterFrequencies["у"] = 2.615
RussianLetterFrequencies["я"] = 2.001
RussianLetterFrequencies["ы"] = 1.898
RussianLetterFrequencies["ь"] = 1.735
RussianLetterFrequencies["г"] = 1.687
RussianLetterFrequencies["з"] = 1.641
RussianLetterFrequencies["б"] = 1.592
RussianLetterFrequencies["ч"] = 1.450
RussianLetterFrequencies["й"] = 1.208
RussianLetterFrequencies["х"] = 0.966
RussianLetterFrequencies["ж"] = 0.940
RussianLetterFrequencies["ш"] = 0.718
RussianLetterFrequencies["ю"] = 0.639
RussianLetterFrequencies["ц"] = 0.486
RussianLetterFrequencies["щ"] = 0.361
RussianLetterFrequencies["э"] = 0.331
RussianLetterFrequencies["ф"] = 0.267
RussianLetterFrequencies["ъ"] = 0.037
RussianLetterFrequencies["ё"] = 0.013

# Общие русские биграммы
const CommonRussianBigrams = [
  "ст", "но", "то", "на", "ен", "ов", "ни", "ра", "во", "ко",
  "ро", "не", "ер", "ол", "ел", "ал", "ил", "от", "ы ", "ан"
]

# Общие русские слова для проверки
const CommonRussianWords = [
  "и", "в", "не", "на", "что", "я", "с", "он", "а", "это",
  "как", "по", "но", "они", "к", "у", "ты", "из", "мы", "за"
]

# Характерные байтовые последовательности для каждой кодировки
# CP1251: часто встречаются байты 0xE0-0xFF (а-я), 0xC0-0xDF (А-Я)
# KOI8-R: часто встречаются байты 0xC0-0xDF (а-я), 0xE0-0xFF (А-Я)
proc analyzeBytePatterns(data: string): tuple[cp1251Score: float, koi8rScore: float] =
  var cp1251Pattern = 0
  var koi8rPattern = 0
  var totalBytes = 0
  
  for i in 0..<data.len:
    let b = data[i].uint8
    if b >= 0x80:
      inc totalBytes
      
      # Для CP1251: строчные буквы в диапазоне 0xE0-0xFF очень распространены
      if b >= 0xE0 and b <= 0xFF:
        inc cp1251Pattern
      
      # Для KOI8-R: строчные буквы в диапазоне 0xC0-0xDF очень распространены
      if b >= 0xC0 and b <= 0xDF:
        inc koi8rPattern
  
  if totalBytes == 0:
    return (0.0, 0.0)
  
  return (cp1251Pattern.float / totalBytes.float, koi8rPattern.float / totalBytes.float)



# Проверка валидности UTF-8
proc isValidUTF8(data: string): bool =
  var i = 0
  while i < data.len:
    let c = data[i].uint8
    
    if c <= 0x7F:
      # ASCII символ
      inc i
    elif (c and 0xE0) == 0xC0:
      # 2-байтовая последовательность
      if i + 1 >= data.len: return false
      let c1 = data[i+1].uint8
      if (c1 and 0xC0) != 0x80: return false
      # Проверка на overlong encoding (символ должен быть >= 0x80)
      let codepoint = ((c and 0x1F).int shl 6) or (c1 and 0x3F).int
      if codepoint < 0x80: return false
      inc i, 2
    elif (c and 0xF0) == 0xE0:
      # 3-байтовая последовательность
      if i + 2 >= data.len: return false
      let c1 = data[i+1].uint8
      let c2 = data[i+2].uint8
      if (c1 and 0xC0) != 0x80: return false
      if (c2 and 0xC0) != 0x80: return false
      # Проверка на overlong encoding и суррогаты
      let codepoint = ((c and 0x0F).int shl 12) or 
                      ((c1 and 0x3F).int shl 6) or 
                      (c2 and 0x3F).int
      if codepoint < 0x800: return false  # overlong
      if codepoint >= 0xD800 and codepoint <= 0xDFFF: return false  # суррогаты
      inc i, 3
    elif (c and 0xF8) == 0xF0:
      # 4-байтовая последовательность
      if i + 3 >= data.len: return false
      let c1 = data[i+1].uint8
      let c2 = data[i+2].uint8
      let c3 = data[i+3].uint8
      if (c1 and 0xC0) != 0x80: return false
      if (c2 and 0xC0) != 0x80: return false
      if (c3 and 0xC0) != 0x80: return false
      # Проверка на overlong encoding и превышение максимума
      let codepoint = ((c and 0x07).int shl 18) or 
                      ((c1 and 0x3F).int shl 12) or 
                      ((c2 and 0x3F).int shl 6) or 
                      (c3 and 0x3F).int
      if codepoint < 0x10000 or codepoint > 0x10FFFF: return false
      inc i, 4
    else:
      return false
  
  return true

# Проверка на UTF-16LE (Little Endian)
proc isUTF16LE(data: string): bool =
  if data.len < 2:
    return false
  
  # Проверка BOM для UTF-16LE (0xFF 0xFE)
  if data.len >= 2 and data[0].uint8 == 0xFF and data[1].uint8 == 0xFE:
    return true
  
  # Эвристическая проверка: в UTF-16LE каждый второй байт часто 0x00 для ASCII
  if data.len < 100:
    return false
  
  var nullCount = 0
  var oddNullCount = 0
  for i in 0..<min(100, data.len):
    if data[i].uint8 == 0:
      inc nullCount
      if i mod 2 == 1:
        inc oddNullCount
  
  # Если более 20% байтов нулевые и большинство на нечетных позициях
  return nullCount > 20 and oddNullCount > (nullCount * 7) div 10

# Проверка на UTF-16BE (Big Endian)
proc isUTF16BE(data: string): bool =
  if data.len < 2:
    return false
  
  # Проверка BOM для UTF-16BE (0xFE 0xFF)
  if data.len >= 2 and data[0].uint8 == 0xFE and data[1].uint8 == 0xFF:
    return true
  
  # Эвристическая проверка: в UTF-16BE каждый второй байт часто 0x00 для ASCII
  if data.len < 100:
    return false
  
  var nullCount = 0
  var evenNullCount = 0
  for i in 0..<min(100, data.len):
    if data[i].uint8 == 0:
      inc nullCount
      if i mod 2 == 0:
        inc evenNullCount
  
  # Если более 20% байтов нулевые и большинство на чётных позициях
  return nullCount > 20 and evenNullCount > (nullCount * 7) div 10



# Декодирование UTF-16LE
proc decodeUTF16LE(data: string): string =
  result = ""
  var i = 0
  
  # Пропускаем BOM если есть
  if data.len >= 2 and data[0].uint8 == 0xFF and data[1].uint8 == 0xFE:
    i = 2
  
  while i + 1 < data.len:
    let low = data[i].uint8
    let high = data[i+1].uint8
    let codepoint = low.int or (high.int shl 8)
    
    # Обработка суррогатных пар для символов вне BMP
    if codepoint >= 0xD800 and codepoint <= 0xDBFF:
      # Это старшая суррогатная половина, нужна младшая
      if i + 4 <= data.len:
        let low2 = data[i+2].uint8
        let high2 = data[i+3].uint8
        let codepoint2 = low2.int or (high2.int shl 8)
        
        if codepoint2 >= 0xDC00 and codepoint2 <= 0xDFFF:
          let finalCode = 0x10000 + ((codepoint - 0xD800) shl 10) + (codepoint2 - 0xDC00)
          if finalCode <= 0x10FFFF:
            result.add($Rune(finalCode))
            inc i, 4
            continue
      # Суррогатная пара неполная - добавляем replacement character
      result.add("\uFFFD")
      inc i, 2
      continue
    
    # Проверка на младшую суррогатную половину без старшей
    if codepoint >= 0xDC00 and codepoint <= 0xDFFF:
      result.add("\uFFFD")
      inc i, 2
      continue
    
    # Обычный символ
    if codepoint <= 0x10FFFF:
      result.add($Rune(codepoint))
    else:
      result.add("\uFFFD")
    inc i, 2
  
  # Обработка остаточного байта
  if i < data.len:
    result.add("\uFFFD")



# Декодирование UTF-16BE
proc decodeUTF16BE(data: string): string =
  result = ""
  var i = 0

  # Пропускаем BOM если есть
  if data.len >= 2 and data[0].uint8 == 0xFE and data[1].uint8 == 0xFF:
    i = 2
  
  while i + 1 < data.len:
    let high = data[i].uint8
    let low = data[i+1].uint8
    let codepoint = (high.int shl 8) or low.int
    
    # Обработка суррогатных пар для символов вне BMP
    if codepoint >= 0xD800 and codepoint <= 0xDBFF:
      # Это старшая суррогатная половина, нужна младшая
      if i + 4 <= data.len:
        let high2 = data[i+2].uint8
        let low2 = data[i+3].uint8
        let codepoint2 = (high2.int shl 8) or low2.int
        
        if codepoint2 >= 0xDC00 and codepoint2 <= 0xDFFF:
          let finalCode = 0x10000 + ((codepoint - 0xD800) shl 10) + (codepoint2 - 0xDC00)
          if finalCode <= 0x10FFFF:
            result.add($Rune(finalCode))
            inc i, 4
            continue
      # Суррогатная пара неполная - добавляем replacement character
      result.add("\uFFFD")
      inc i, 2
      continue
    
    # Проверка на младшую суррогатную половину без старшей
    if codepoint >= 0xDC00 and codepoint <= 0xDFFF:
      result.add("\uFFFD")
      inc i, 2
      continue
    
    # Обычный символ
    if codepoint <= 0x10FFFF:
      result.add($Rune(codepoint))
    else:
      result.add("\uFFFD")
    inc i, 2
  
  # Обработка остаточного байта
  if i < data.len:
    result.add("\uFFFD")



# Декодирование из Windows-1251
proc decodeCP1251(data: string): string =
  result = ""
  for c in data:
    let b = c.uint8
    if b >= 0xC0 and b <= 0xFF:
      # Кириллица А-Я, а-я
      result.add($Rune(0x0410 + (b - 0xC0).int))
    elif b >= 0x80 and b <= 0xBF:
      # Другие символы Windows-1251
      const cp1251Map: array[0x80..0xBF, int] = [
        0x0402, 0x0403, 0x201A, 0x0453, 0x201E, 0x2026, 0x2020, 0x2021,
        0x20AC, 0x2030, 0x0409, 0x2039, 0x040A, 0x040C, 0x040B, 0x040F,
        0x0452, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014,
        0x0098, 0x2122, 0x0459, 0x203A, 0x045A, 0x045C, 0x045B, 0x045F,
        0x00A0, 0x040E, 0x045E, 0x0408, 0x00A4, 0x0490, 0x00A6, 0x00A7,
        0x0401, 0x00A9, 0x0404, 0x00AB, 0x00AC, 0x00AD, 0x00AE, 0x0407,
        0x00B0, 0x00B1, 0x0406, 0x0456, 0x0491, 0x00B5, 0x00B6, 0x00B7,
        0x0451, 0x2116, 0x0454, 0x00BB, 0x0458, 0x0405, 0x0455, 0x0457
      ]
      result.add($Rune(cp1251Map[b]))
    else:
      result.add(c)



# Декодирование из KOI8-R - ИСПРАВЛЕННАЯ ВЕРСИЯ
proc decodeKOI8R(data: string): string =
  # Полная таблица KOI8-R для байтов 0x80-0xFF
  const koi8rTable: array[0x80..0xFF, int] = [
    # 0x80-0x8F
    0x2500, 0x2502, 0x250C, 0x2510, 0x2514, 0x2518, 0x251C, 0x2524,
    0x252C, 0x2534, 0x253C, 0x2580, 0x2584, 0x2588, 0x258C, 0x2590,
    # 0x90-0x9F
    0x2591, 0x2592, 0x2593, 0x2320, 0x25A0, 0x2219, 0x221A, 0x2248,
    0x2264, 0x2265, 0x00A0, 0x2321, 0x00B0, 0x00B2, 0x00B7, 0x00F7,
    # 0xA0-0xAF
    0x2550, 0x2551, 0x2552, 0x0451, 0x2553, 0x2554, 0x2555, 0x2556,
    0x2557, 0x2558, 0x2559, 0x255A, 0x255B, 0x255C, 0x255D, 0x255E,
    # 0xB0-0xBF
    0x255F, 0x2560, 0x2561, 0x0401, 0x2562, 0x2563, 0x2564, 0x2565,
    0x2566, 0x2567, 0x2568, 0x2569, 0x256A, 0x256B, 0x256C, 0x00A9,
    # 0xC0-0xCF (строчные: ю а б ц д е ф г х и й к л м н о)
    0x044E, 0x0430, 0x0431, 0x0446, 0x0434, 0x0435, 0x0444, 0x0433,
    0x0445, 0x0438, 0x0439, 0x043A, 0x043B, 0x043C, 0x043D, 0x043E,
    # 0xD0-0xDF (строчные: п я р с т у ж в ь ы з ш э щ ч ъ)
    0x043F, 0x044F, 0x0440, 0x0441, 0x0442, 0x0443, 0x0436, 0x0432,
    0x044C, 0x044B, 0x0437, 0x0448, 0x044D, 0x0449, 0x0447, 0x044A,
    # 0xE0-0xEF (заглавные: Ю А Б Ц Д Е Ф Г Х И Й К Л М Н О)
    0x042E, 0x0410, 0x0411, 0x0426, 0x0414, 0x0415, 0x0424, 0x0413,
    0x0425, 0x0418, 0x0419, 0x041A, 0x041B, 0x041C, 0x041D, 0x041E,
    # 0xF0-0xFF (заглавные: П Я Р С Т У Ж В Ь Ы З Ш Э Щ Ч Ъ)
    0x041F, 0x042F, 0x0420, 0x0421, 0x0422, 0x0423, 0x0416, 0x0412,
    0x042C, 0x042B, 0x0417, 0x0428, 0x042D, 0x0429, 0x0427, 0x042A
  ]
  
  result = ""
  for c in data:
    let b = c.uint8
    if b >= 0x80:
      result.add($Rune(koi8rTable[b]))
    else:
      result.add(c)



# Декодирование из KOI8-U (украинский вариант KOI8-R)
proc decodeKOI8U(data: string): string =
  # Полная таблица KOI8-U (отличается от KOI8-R в некоторых позициях)
  const koi8uTable: array[0x80..0xFF, int] = [
    # 0x80-0x8F
    0x2500, 0x2502, 0x250C, 0x2510, 0x2514, 0x2518, 0x251C, 0x2524,
    0x252C, 0x2534, 0x253C, 0x2580, 0x2584, 0x2588, 0x258C, 0x2590,
    # 0x90-0x9F
    0x2591, 0x2592, 0x2593, 0x2320, 0x25A0, 0x2219, 0x221A, 0x2248,
    0x2264, 0x2265, 0x00A0, 0x2321, 0x00B0, 0x00B2, 0x00B7, 0x00F7,
    # 0xA0-0xAF (отличия от KOI8-R: украинские буквы)
    0x2550, 0x2551, 0x2552, 0x0451, 0x0454, 0x2554, 0x0456, 0x0457,
    0x2557, 0x2558, 0x2559, 0x255A, 0x255B, 0x0491, 0x255D, 0x255E,
    # 0xB0-0xBF (отличия от KOI8-R: украинские буквы)
    0x255F, 0x2560, 0x2561, 0x0401, 0x0404, 0x2563, 0x0406, 0x0407,
    0x2566, 0x2567, 0x2568, 0x2569, 0x256A, 0x0490, 0x256C, 0x00A9,
    # 0xC0-0xCF (строчные: ю а б ц д е ф г х и й к л м н о)
    0x044E, 0x0430, 0x0431, 0x0446, 0x0434, 0x0435, 0x0444, 0x0433,
    0x0445, 0x0438, 0x0439, 0x043A, 0x043B, 0x043C, 0x043D, 0x043E,
    # 0xD0-0xDF (строчные: п я р с т у ж в ь ы з ш э щ ч ъ)
    0x043F, 0x044F, 0x0440, 0x0441, 0x0442, 0x0443, 0x0436, 0x0432,
    0x044C, 0x044B, 0x0437, 0x0448, 0x044D, 0x0449, 0x0447, 0x044A,
    # 0xE0-0xEF (заглавные: Ю А Б Ц Д Е Ф Г Х И Й К Л М Н О)
    0x042E, 0x0410, 0x0411, 0x0426, 0x0414, 0x0415, 0x0424, 0x0413,
    0x0425, 0x0418, 0x0419, 0x041A, 0x041B, 0x041C, 0x041D, 0x041E,
    # 0xF0-0xFF (заглавные: П Я Р С Т У Ж В Ь Ы З Ш Э Щ Ч Ъ)
    0x041F, 0x042F, 0x0420, 0x0421, 0x0422, 0x0423, 0x0416, 0x0412,
    0x042C, 0x042B, 0x0417, 0x0428, 0x042D, 0x0429, 0x0427, 0x042A
  ]
  
  result = ""
  for c in data:
    let b = c.uint8
    if b >= 0x80:
      result.add($Rune(koi8uTable[b]))
    else:
      result.add(c)



# Декодирование из ISO-8859-5
proc decodeISO88595(data: string): string =
  result = ""
  for c in data:
    let b = c.uint8
    if b >= 0xA0 and b <= 0xFF:
      # Кириллица в ISO-8859-5 начинается с 0xA0
      result.add($Rune(0x0400 + (b - 0xA0).int))
    else:
      result.add(c)



# Декодирование из CP866 (DOS кириллица)
proc decodeCP866(data: string): string =
  const cp866Table: array[0x80..0xFF, int] = [
    # 0x80-0x8F (А-П)
    0x0410, 0x0411, 0x0412, 0x0413, 0x0414, 0x0415, 0x0416, 0x0417,
    0x0418, 0x0419, 0x041A, 0x041B, 0x041C, 0x041D, 0x041E, 0x041F,
    # 0x90-0x9F (Р-Я, а-е)
    0x0420, 0x0421, 0x0422, 0x0423, 0x0424, 0x0425, 0x0426, 0x0427,
    0x0428, 0x0429, 0x042A, 0x042B, 0x042C, 0x042D, 0x042E, 0x042F,
    # 0xA0-0xAF (а-п)
    0x0430, 0x0431, 0x0432, 0x0433, 0x0434, 0x0435, 0x0436, 0x0437,
    0x0438, 0x0439, 0x043A, 0x043B, 0x043C, 0x043D, 0x043E, 0x043F,
    # 0xB0-0xBF (псевдографика)
    0x2591, 0x2592, 0x2593, 0x2502, 0x2524, 0x2561, 0x2562, 0x2556,
    0x2555, 0x2563, 0x2551, 0x2557, 0x255D, 0x255C, 0x255B, 0x2510,
    # 0xC0-0xCF (псевдографика)
    0x2514, 0x2534, 0x252C, 0x251C, 0x2500, 0x253C, 0x255E, 0x255F,
    0x255A, 0x2554, 0x2569, 0x2566, 0x2560, 0x2550, 0x256C, 0x2567,
    # 0xD0-0xDF (псевдографика)
    0x2568, 0x2564, 0x2565, 0x2559, 0x2558, 0x2552, 0x2553, 0x256B,
    0x256A, 0x2518, 0x250C, 0x2588, 0x2584, 0x258C, 0x2590, 0x2580,
    # 0xE0-0xEF (р-я)
    0x0440, 0x0441, 0x0442, 0x0443, 0x0444, 0x0445, 0x0446, 0x0447,
    0x0448, 0x0449, 0x044A, 0x044B, 0x044C, 0x044D, 0x044E, 0x044F,
    # 0xF0-0xFF (Ё, ё и другие символы)
    0x0401, 0x0451, 0x0404, 0x0454, 0x0407, 0x0457, 0x040E, 0x045E,
    0x00B0, 0x2219, 0x00B7, 0x221A, 0x2116, 0x00A4, 0x25A0, 0x00A0
  ]
  
  result = ""
  for c in data:
    let b = c.uint8
    if b >= 0x80:
      result.add($Rune(cp866Table[b]))
    else:
      result.add(c)



# Декодирование из CP855
proc decodeCP855(data: string): string =
  const cp855Table: array[0x80..0xFF, int] = [
    # Таблица CP855 (сербская/болгарская кириллица)
    0x0452, 0x0402, 0x0453, 0x0403, 0x0451, 0x0401, 0x0454, 0x0404,
    0x0455, 0x0405, 0x0456, 0x0406, 0x0457, 0x0407, 0x0458, 0x0408,
    0x0459, 0x0409, 0x045A, 0x040A, 0x045B, 0x040B, 0x045C, 0x040C,
    0x045E, 0x040E, 0x045F, 0x040F, 0x044E, 0x042E, 0x044A, 0x042A,
    0x0430, 0x0410, 0x0431, 0x0411, 0x0446, 0x0426, 0x0434, 0x0414,
    0x0435, 0x0415, 0x0444, 0x0424, 0x0433, 0x0413, 0x00AB, 0x00BB,
    0x2591, 0x2592, 0x2593, 0x2502, 0x2524, 0x0445, 0x0425, 0x0438,
    0x0418, 0x2563, 0x2551, 0x2557, 0x255D, 0x0439, 0x0419, 0x2510,
    0x2514, 0x2534, 0x252C, 0x251C, 0x2500, 0x253C, 0x043A, 0x041A,
    0x255A, 0x2554, 0x2569, 0x2566, 0x2560, 0x2550, 0x256C, 0x00A4,
    0x043B, 0x041B, 0x043C, 0x041C, 0x043D, 0x041D, 0x043E, 0x041E,
    0x043F, 0x2518, 0x250C, 0x2588, 0x2584, 0x041F, 0x044F, 0x2580,
    0x042F, 0x0440, 0x0420, 0x0441, 0x0421, 0x0442, 0x0442, 0x0443,
    0x0423, 0x0436, 0x0416, 0x0432, 0x0412, 0x044C, 0x042C, 0x2116,
    0x00AD, 0x044B, 0x042B, 0x0437, 0x0417, 0x0448, 0x0428, 0x044D,
    0x042D, 0x0449, 0x0429, 0x0447, 0x0427, 0x00A7, 0x25A0, 0x00A0
  ]
  
  result = ""
  for c in data:
    let b = c.uint8
    if b >= 0x80:
      result.add($Rune(cp855Table[b]))
    else:
      result.add(c)



# Проверка наличия кириллических символов в тексте
proc hasCyrillic(text: string): bool =
  for rune in text.runes:
    let runeInt = rune.int
    if (runeInt >= 0x0400 and runeInt <= 0x04FF):
      return true
  return false

# Проверка на чистый ASCII
proc isASCII(data: string): bool =
  for c in data:
    if c.uint8 > 127:
      return false
  return true



# Подсчет количества русских символов
proc countCyrillicChars(text: string): int =
  result = 0
  for rune in text.runes:
    let runeInt = rune.int
    if (runeInt >= 0x0400 and runeInt <= 0x04FF):
      inc result

# Проверка на наличие недопустимых символов (мусор после декодирования)
proc hasGarbageChars(text: string): bool =
  var invalidCount = 0
  var totalChars = 0
  
  for rune in text.runes:
    inc totalChars
    let runeInt = rune.int
    # Проверяем на непечатаемые символы (кроме пробельных)
    if runeInt < 0x20 and runeInt != 0x09 and runeInt != 0x0A and runeInt != 0x0D:
      inc invalidCount
    # Проверяем на символы частной области использования (часто появляются при ошибках)
    elif runeInt >= 0xE000 and runeInt <= 0xF8FF:
      inc invalidCount

  # Если более 5% символов - мусор, считаем кодировку неправильной
  return totalChars > 0 and (invalidCount.float / totalChars.float) > 0.05


# Анализ частоты букв
proc analyzeFrequency(text: string): float =
  ## Анализирует частоту появления русских букв и сравнивает с эталонной
  var letterCounts = initTable[string, int]()
  var totalRussianLetters = 0
  
  for rune in text.runes:
    let runeInt = rune.int
    # Только русские буквы (а-я, А-Я)
    if (runeInt >= 0x0430 and runeInt <= 0x044F) or (runeInt >= 0x0410 and runeInt <= 0x042F):
      let letter = ($rune).toLowerAscii()
      letterCounts.mgetOrPut(letter, 0).inc
      inc totalRussianLetters
  
  if totalRussianLetters < 10:
    return 0.0
  
  # Вычисляем отклонение от эталонных частот
  var score = 0.0
  for letter, expectedFreq in RussianLetterFrequencies:
    let actualCount = letterCounts.getOrDefault(letter, 0)
    let actualFreq = (actualCount.float / totalRussianLetters.float) * 100.0
    let diff = abs(actualFreq - expectedFreq)
    score += max(0.0, 1.0 - diff / 10.0)  # Чем меньше отклонение, тем выше оценка
  
  return score / RussianLetterFrequencies.len.float


# Подсчет общих биграмм
proc countCommonBigrams(text: string): int =
  ## Подсчитывает количество общих русских биграмм в тексте
  result = 0
  let lowerText = text.toLowerAscii()
  
  for bigram in CommonRussianBigrams:
    var pos = 0
    while true:
      pos = lowerText.find(bigram, pos)
      if pos == -1:
        break
      inc result
      inc pos


# Подсчет общих слов
proc countCommonWords(text: string): int =
  ## Подсчитывает количество общих русских слов в тексте
  result = 0
  let lowerText = text.toLowerAscii()
  
  for word in CommonRussianWords:
    # Ищем слово как отдельное (с пробелами или знаками препинания)
    let wordWithSpaces = " " & word & " "
    var pos = 0
    while true:
      pos = lowerText.find(wordWithSpaces, pos)
      if pos == -1:
        break
      inc result
      inc pos



# Основная функция определения кодировки
proc charDet*(text: string): Encoding =
  if text.len == 0:
    return UNKNOWN
  
  # Проверка на ASCII
  if isASCII(text):
    return ASCII
  
  # Проверка на UTF-16 (приоритет - по BOM)
  if isUTF16LE(text):
    return UTF16LE
  
  if isUTF16BE(text):
    return UTF16BE

  var scores: seq[EncodingScore]

  # Анализ байтовых паттернов исходного текста
  let (cp1251ByteScore, koi8rByteScore) = analyzeBytePatterns(text)

  # Проверка UTF-8
  if isValidUTF8(text):
    if hasCyrillic(text):
      let freq = analyzeFrequency(text)
      let bigrams = countCommonBigrams(text)
      let words = countCommonWords(text)
      let score = freq * 0.5 + (bigrams.float / 20.0) * 0.3 + (words.float / 10.0) * 0.2
      scores.add(EncodingScore(encoding: UTF8, score: score, confidence: 0.95))
    else:
      # UTF-8 без кириллицы (латиница и т.д.)
      scores.add(EncodingScore(encoding: UTF8, score: 0.8, confidence: 0.9))

  # Проверка Windows-1251
  try:
    let decoded = decodeCP1251(text)
    if hasCyrillic(decoded) and not hasGarbageChars(decoded):
      let cyrillicCount = countCyrillicChars(decoded)
      let freq = analyzeFrequency(decoded)
      let bigrams = countCommonBigrams(decoded)
      let words = countCommonWords(decoded)

      # Бонус за байтовый паттерн характерный для CP1251
      var bonus = cp1251ByteScore * 0.3
      if cyrillicCount > 50:
        bonus += 0.05

      let score = freq * 0.4 + (bigrams.float / 20.0) * 0.25 + (words.float / 10.0) * 0.15 + bonus
      scores.add(EncodingScore(encoding: CP1251, score: score, confidence: 0.85))
  except:
    discard

  # Проверка KOI8-R
  try:
    let decoded = decodeKOI8R(text)
    if hasCyrillic(decoded) and not hasGarbageChars(decoded):
      let cyrillicCount = countCyrillicChars(decoded)
      let freq = analyzeFrequency(decoded)
      let bigrams = countCommonBigrams(decoded)
      let words = countCommonWords(decoded)
      
      # Бонус за байтовый паттерн характерный для KOI8-R
      var bonus = koi8rByteScore * 0.3
      if cyrillicCount > 50:
        bonus += 0.05
      
      let score = freq * 0.4 + (bigrams.float / 20.0) * 0.25 + (words.float / 10.0) * 0.15 + bonus
      scores.add(EncodingScore(encoding: KOI8R, score: score, confidence: 0.88))
  except:
    discard

  # Проверка KOI8-U
  try:
    let decoded = decodeKOI8U(text)
    if hasCyrillic(decoded) and not hasGarbageChars(decoded):
      let cyrillicCount = countCyrillicChars(decoded)
      let freq = analyzeFrequency(decoded)
      let bigrams = countCommonBigrams(decoded)
      let words = countCommonWords(decoded)
      
      # Бонус за байтовый паттерн характерный для KOI8-U
      var bonus = koi8rByteScore * 0.3
      if cyrillicCount > 50:
        bonus += 0.05
      
      let score = freq * 0.4 + (bigrams.float / 20.0) * 0.25 + (words.float / 10.0) * 0.15 + bonus
      scores.add(EncodingScore(encoding: KOI8U, score: score, confidence: 0.85))
  except:
    discard

  # Проверка ISO-8859-5
  try:
    let decoded = decodeISO88595(text)
    if hasCyrillic(decoded) and not hasGarbageChars(decoded):
      let freq = analyzeFrequency(decoded)
      let bigrams = countCommonBigrams(decoded)
      let words = countCommonWords(decoded)
      let score = freq * 0.5 + (bigrams.float / 20.0) * 0.3 + (words.float / 10.0) * 0.2
      scores.add(EncodingScore(encoding: ISO88595, score: score, confidence: 0.75))
  except:
    discard

  # Проверка CP866
  try:
    let decoded = decodeCP866(text)
    if hasCyrillic(decoded) and not hasGarbageChars(decoded):
      let freq = analyzeFrequency(decoded)
      let bigrams = countCommonBigrams(decoded)
      let words = countCommonWords(decoded)
      let score = freq * 0.5 + (bigrams.float / 20.0) * 0.3 + (words.float / 10.0) * 0.2
      scores.add(EncodingScore(encoding: CP866, score: score, confidence: 0.70))
  except:
    discard

  # Проверка CP855
  try:
    let decoded = decodeCP855(text)
    if hasCyrillic(decoded) and not hasGarbageChars(decoded):
      let freq = analyzeFrequency(decoded)
      let bigrams = countCommonBigrams(decoded)
      let words = countCommonWords(decoded)
      let score = freq * 0.5 + (bigrams.float / 20.0) * 0.3 + (words.float / 10.0) * 0.2
      scores.add(EncodingScore(encoding: CP855, score: score, confidence: 0.68))
  except:
    discard

  # Выбор лучшей кодировки
  if len(scores) == 0:
    return UNKNOWN
  
  var best = scores[0]
  for s in scores:
    if s.score > best.score:
      best = s

  return best.encoding



# Вспомогательная функция для определения с детальной информацией
proc charDetDetailed*(text: string): tuple[encoding: Encoding, confidence: float] =
  let encoding = charDet(text)
  var confidence = 0.0
  
  case encoding
  of UTF8:
    if isValidUTF8(text):
      confidence = 0.95
  of UTF16LE:
    confidence = 0.98
  of UTF16BE:
    confidence = 0.98
  of KOI8R:
    confidence = 0.88
  of KOI8U:
    confidence = 0.85
  of CP1251:
    confidence = 0.85
  of ISO88595:
    confidence = 0.75
  of CP866:
    confidence = 0.70
  of CP855:
    confidence = 0.68
  of ASCII:
    confidence = 1.0
  of UNKNOWN:
    confidence = 0.0
  
  return (encoding, confidence)




# Публичные функции конвертации текста в UTF-8
proc toUTF8*(text: string, encoding: Encoding): string =
  ## Конвертирует текст из указанной кодировки в UTF-8
  case encoding
  of CP1251:
    return decodeCP1251(text)
  of KOI8R:
    return decodeKOI8R(text)
  of KOI8U:
    return decodeKOI8U(text)
  of CP866:
    return decodeCP866(text)
  of CP855:
    return decodeCP855(text)
  of ISO88595:
    return decodeISO88595(text)
  of UTF16LE:
    return decodeUTF16LE(text)
  of UTF16BE:
    return decodeUTF16BE(text)
  of UTF8, ASCII:
    return text
  of UNKNOWN:
    return text




proc readFirstLines*(FN: string; numLines: int = DefNumLines): seq[string] =
  ## Прочитать первые numLines строк из файла с именем FN.
  ## Если количество строк в файле меньше заданного, то вернуть столько, сколько есть.
  if not fileExists(Path(FN)): return @[]

  var f: File = nil
  if open(f, FN):
    try:
      result = newSeqOfCap[string](numLines)
      var count = 0
      for line in lines(f):
        add(result, line)
        inc count
        if count >= numLines: break
    finally:
      close(f)
  else:
    # raise newException(IOError, "Cannot open: " & FN)
    return @[]   # или raise




# Функция анализа кодировки файла
proc analyzeFile*(FN: string): Encoding =
  try:
    # Чтение заданного количества первых строк файла
    let content = join(readFirstLines(FN), "\n")
    # Чтение всего файла
    # let content = readFile(filename)
    return charDet(content)
  except:
    echo "Ошибка при чтении файла: ", getCurrentExceptionMsg()
    return UNKNOWN




proc convertFile*(inputFile: string, outputFile: string = ""): bool =
  ## Автоматически определяет кодировку входного файла и конвертирует в UTF-8
  ## Если outputFile не указан, перезаписывает исходный файл
  try:
    let content = readFile(inputFile)
    let encoding = charDet(content)

    if encoding == UNKNOWN:
      echo "Не удалось определить кодировку файла"
      return false

    let converted = toUTF8(content, encoding)
    let outFile = if outputFile == "": inputFile else: outputFile

    writeFile(outFile, converted)
    return true
  except:
    echo "Ошибка при конвертации файла: ", getCurrentExceptionMsg()
    return false
















# nim c -d:release utils.nim

