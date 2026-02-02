
################################################################
## ПРОГРАММА ПРОВЕРКИ ФУНКЦИЙ МОДУЛЯ TOKENIZATION
## 
## Версия:   0.1
## Дата:     2026-06-23
## Автор:    github.com/Balans097
################################################################

# 0.1 — начальная реализация программы



# nim c -d:release FunctionsTest.nim

import std/[encodings]
import strutils
import tokenization, utils





################################################################
###############  И С Х О Д Н Ы Е   Д А Н Н Ы Е  ################
################################################################
const
  # FN = "BadTextExampleUTF8.txt"
  # FN = "BadTextExampleUTF8BOM.txt"
  FN = "BadTextExampleKOI8-R.txt"
  # FN = "BadTextExampleANSI.txt"

let
  text = readFile(FN)





################################################################
####################  В Ы Ч И С Л Е Н И Я  #####################
################################################################

let CP = analyzeFile(FN)
echo "Кодировка:\t", CP
echo "Кодировка:\t", charDetDetailed(text)

echo "ОЧИСТКА ТЕКСТА"
echo repeat("=", 70)
echo cleanText(toUTF8(text, CP))


#[ let TT = convert(text, "UTF-8", "koi8-r")
echo cleanText(TT) ]#












# nim c -d:release FunctionsTest.nim




