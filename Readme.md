
# Telegram-бот с использованием нейросети для ответа на вопросы

Этот проект представляет собой Telegram-бота, использующего дообученную модель `ruT5-base` для генерации ответов на вопросы на основе контекста и вопроса. Мы используем модель `ruT5-base`, дообученную на датасете `SberQuAD`, чтобы научиться давать ответы на вопросы с учетом заданного контекста.

## Оглавление
- [Установка и настройка](#установка-и-настройка)
  - [Запуск бота на локальной машине](#запуск-бота-на-локальной-машине)
  - [Запуск обучения модели в Google Colab](#запуск-обучения-модели-в-google-colab)
- [Использование бота](#использование-бота)
- [Описание структуры проекта](#описание-структуры-проекта)
- [Дополнительная информация](#дополнительная-информация)

## Установка и настройка

### Запуск бота на локальной машине

1. **Клонирование репозитория**
    ```bash
    git clone https://github.com/your-repository/telegram-qa-bot.git
    cd telegram-qa-bot
    ```

2. **Создание и активация виртуального окружения**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/MacOS
    .\venv\Scripts\activate   # Windows
    ```

3. **Установка зависимостей**
    ```bash
    pip install -r requirements.txt
    ```

4. **Настройка бота**
   - Создайте бота в [BotFather](https://t.me/BotFather) и получите API токен.
   - Создайте файл `.env` в корневой директории проекта и добавьте ваш токен:
     ```
     TOKEN=YOUR_TELEGRAM_BOT_TOKEN
     ```

5. **Запуск бота**
    ```bash
    python bot.py
    ```

### Запуск обучения модели в Google Colab

1. **Создание нового Colab Notebook**:
   - Перейдите на [Google Colab](https://colab.research.google.com/).
   - Создайте новый блокнот и подключите GPU (в меню выберите `Среда выполнения` -> `Сменить среду выполнения` -> `Аппаратный ускоритель` -> `GPU`).

2. **Установка зависимостей**:
    ```python
    !pip install transformers datasets torch
    ```

3. **Загрузка и подготовка данных**:
   - Датасет `SberQuAD` можно загрузить с помощью библиотеки `datasets`:
    ```python
    from datasets import load_dataset

    # Загрузка датасета SberQuAD
    dataset = load_dataset("sberquad")
    ```

4. **Подготовка модели и токенизатора**:
    ```python
    from transformers import AutoTokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

    # Загрузка модели и токенизатора
    model_checkpoint = "cointegrated/rut5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
    ```

5. **Подготовка данных для обучения**:
   - Преобразуем данные в формат для обучения, объединяя контекст и вопрос:
    ```python
    def preprocess_data(examples):
        inputs = ['context: ' + ctx + ' question: ' + q for ctx, q in zip(examples['context'], examples['question'])]
        targets = [answer[0]['text'] for answer in examples['answers']]
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')

        labels = tokenizer(targets, max_length=150, truncation=True, padding='max_length').input_ids
        model_inputs['labels'] = labels
        return model_inputs

    tokenized_dataset = dataset.map(preprocess_data, batched=True)
    ```

6. **Настройка и запуск обучения**:
    ```python
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_strategy='epoch',
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        tokenizer=tokenizer
    )

    trainer.train()
    ```

7. **Сохранение модели**:
    ```python
    model.save_pretrained("/content/best_model")
    tokenizer.save_pretrained("/content/best_model")
    ```

8. **Скачивание модели**:
   - Архивируем и скачиваем модель на локальный диск:
    ```python
    !zip -r /content/best_model.zip /content/best_model
    from google.colab import files
    files.download("/content/best_model.zip")
    ```

## Использование бота

1. **Запуск бота**:
   - После запуска бота введите команду `/start`, чтобы начать диалог.

2. **Ввод контекста**:
   - Бот попросит ввести контекст вопроса. Введите текст, на основе которого хотите задать вопрос.

3. **Ввод вопроса**:
   - После ввода контекста, бот предложит задать вопрос. Введите свой вопрос.

4. **Получение ответа**:
   - Бот сгенерирует ответ на основе введенного контекста и вопроса.

5. **Начало с начала**:
   - Для сброса контекста и вопроса нажмите на кнопку "Начать с начала".

## Описание структуры проекта

```
telegram-qa-bot/
│
├── bot.py                      # Основной скрипт бота
├── model_training.ipynb        # Ноутбук для обучения модели в Google Colab
├── requirements.txt            # Зависимости проекта
└── README.md                   # Описание проекта
```

## Дополнительная информация

- Для обучения использовалась модель [ruT5-base](https://huggingface.co/ai-forever/ruT5-base) от SberBank AI.
- Датасет [SberQuAD](https://huggingface.co/datasets/kuznetsoffandrey/sberquad) является аналогом SQuAD для русского языка и содержит вопросы и ответы на различные тексты.
- Модель была дообучена для генерации ответа на вопрос, опираясь на контекст, предоставленный пользователем.
