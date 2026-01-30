# libGPT
libGPT is a well-structured Nim library that implements a full cycle of working with GPT-like models.

libGPT — это вычислительно эффективная хорошо структурированная библиотека на Nim, реализующая полный цикл работы с GPT-подобными моделями.

Work in progress! Not for using.

## Основные характеристики библиотеки

- Чистая реализация GPT для глубокого обучения на Nim без необходимости установки сторонних библиотек и инструментария
- Поддержка обучения с нуля и генерации текста
- Модульная структура, удобная для понимания и модификации



## Testing the GPT library
![In terminal](./examples/Screenshots/Screen-01-tokenizer.png)



## Структура модулей и их назначение

| Файл              | Основная ответственность                              | Ключевые сущности / функции                              |
|-------------------|----------------------------------------------------------|------------------------------------------------------------|
| `tokenization.nim`| Токенизация текста (BPE / Byte-Pair Encoding)           | Tokenizer, encode, decode, vocab, merges                   |
| `attention.nim`   | Механизмы внимания (в первую очередь causal multi-head) | scaledDotProductAttention, MultiHeadAttention, causal mask |
| `layers.nim`      | Основные строительные блоки трансформера                | TransformerBlock / DecoderLayer, FeedForward, LayerNorm    |
| `model.nim`       | Главная модель GPT                                      | GPTConfig, GPTModel, forward, generate, load/save          |
| `generation.nim`  | Алгоритмы генерации текста                              | sample, top_k, top_p, temperature, greedy, beam search|
| `loss.nim`        | Функции потерь                                          | crossEntropyLoss, label smoothing              |
| `training.nim`    | Логика обучения модели                                  | train_step, Trainer / Loop, optimizer calls, gradient calc |
| `utils.nim`       | Вспомогательные утилиты                                 | log, random, tensor ops, save/load weights, metrics        |



## Usage
`nimble install https://github.com/Balans097/libGPT`





