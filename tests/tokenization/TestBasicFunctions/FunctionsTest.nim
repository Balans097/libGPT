
################################################################
## ПРОГРАММА ПРОВЕРКИ ФУНКЦИЙ МОДУЛЯ TOKENIZATION
## 
## Версия:   0.1
## Дата:     2026-06-23
## Автор:    github.com/Balans097
################################################################

# 0.1 — начальная реализация программы



# nim c -d:release FunctionsTest.nim



# import std/[encodings]
import strutils
# import libGPT
import tokenization, utils





################################################################
###############  И С Х О Д Н Ы Е   Д А Н Н Ы Е  ################
################################################################
const
  P2T = "Texts"
  FN = P2T & "/BadTextExampleKOI8-R.txt"
  # FN = "Texts/Воскресение (1899).fb2"
  # FN = "Texts/BadTextExampleUTF8.txt"
  # FN = "Texts/BadTextExampleANSI.txt"


let
  # Читаем заданное количество строк
  text = join(readFirstLines(FN, 2520), "\n")





################################################################
####################  В Ы Ч И С Л Е Н И Я  #####################
################################################################

let CP = analyzeFile(FN)
echo "Кодировка:\t", CP
echo "Кодировка:\t", charDetDetailed(text), '\n'

echo "ОЧИСТКА ТЕКСТА"
let CleanedText = cleanText(toUTF8(text, CP))
echo repeat("=", 64)
echo CleanedText
writeFile(P2T & "/CleanedText.txt", CleanedText)



#[ let TT = convert(text, "UTF-8", "koi8-r")
echo cleanText(TT) ]#












# nim c -d:release FunctionsTest.nim




