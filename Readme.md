
# Telegram-бот с использованием нейросети для ответа на вопросы

#### Ссылка на бота - [здесь](https://t.me/tinek_sample_task_bot) <i>ВНИМАНИЕ! Бот работает медленно в силу слабого сервера. Ожидайте ответ, он непременно поступит, либо дайте денег на нормальный сервер)</i>

#### Этот проект представляет собой Telegram-бота, использующего дообученную модель `ruT5-base` для генерации ответов на вопросы на основе контекста и вопроса с учетом того, что ответ на вопрос уже содержится в контексте. Я использую предобученную модель `ruT5-base`, дообученную на датасете `SberQuAD`, чтобы научиться давать ответы на вопросы с учетом заданного контекста в рамках поставленной задачи.

#### В проекте используется модель вида `Seq2seq` с пайплайном ответа на вопросы. Эти модели обучены преобразовывать входную последовательность в выходную, используя архитектуру, состоящую из энкодера и декодера. 

##### Fine-tuned модель лежит вот здесь - [huggingface](https://huggingface.co/RichelieuGVG/tinek_sample_model)

### Ролик с проверкой работы бота

<a href="http://www.youtube.com/watch?feature=player_embedded&v=xe-dsLWpMxk" target="_blank"><img src="http://img.youtube.com/vi/xe-dsLWpMxk/0.jpg" 
alt="Демонстрация работы бота" width="240" height="180" border="10" /></a>

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
    git clone https://github.com/RichelieuGVG1/ml-bot-qa.git
    cd ml-bot-qa
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
    !pip install -U transformers datasets accelerate spacy bitsandbytes evaluate sentencepiece tokenizers torchinfo sacrebleu rouge_score peft jiwer wandb
    ```

3. **Авторизуйтесь с помощью токена wandb**
    ```python
    import wandb

    wandb.login()
    ```
4. **Используйте аппаратный ускоритель**
    ```python
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    ```



5. **Загрузка и подготовка данных**:
   - Датасет `SberQuAD` можно загрузить с помощью библиотеки `datasets`:
    ```python
    from datasets import load_dataset

    train_dataset = load_dataset("kuznetsoffandrey/sberquad", split="train")
    valid_dataset = load_dataset("kuznetsoffandrey/sberquad", split="validation")
    test_dataset  = load_dataset("kuznetsoffandrey/sberquad", split="test")
    ```
6. **Последовательно выполняя команды преобразования датасета приведите его к необходимому виду для дообучения**:
   - Разделите датасет на два поддатасета для валидационной и обучающей выборки
   - Удалите лишние столбцы
   - Добавьте пометку для выбранного промпта
   - Произведите токенизацию данных датасета для обучения
    ```python
    DatasetDict({
    train: Dataset({
        features: ['input_ids', 'attention_mask', 'labels'],
        num_rows: 45328
    })
    validation: Dataset({
        features: ['input_ids', 'attention_mask', 'labels'],
        num_rows: 5036
    })
})
    ```


7. **Подготовка модели и токенизатора**:
    ```python
    from transformers import AutoTokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

    # Загрузка модели и токенизатора
    model_name = 'ai-forever/ruT5-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    ```

8. **Задайте контролирующие метрики для обучения**:
    - SacreBLEU - метрика для подсчета качества перевода, которая подсчитывает общее количество слов и словосочетаний.
    - rouge - набор метрик, основанный на подсчете совпадений слов и словосочетаний.
    - chrF - метрика для подсчета совпадений символов, которые следуют друг за другом.
    ```python
    blue_metric = evaluate.load("sacrebleu")
    rouge_metric = evaluate.load("rouge")
    chrf_metric = evaluate.load("chrf")
    ```

6. **Настройка и запуск обучения**:
    ***Примечание автора: мне не хватило вычислительного времени на Google Collab, поэтому модель была обучена только на одной эпохе вместо двух с ~1416 шагами.***
    ```python
    training_args = Seq2SeqTrainingArguments(
        output_dir="./models",
        optim="adafactor",
        num_train_epochs=1, #в идеале 2 эпохи, но да ладно
        do_train=True,
        gradient_checkpointing=True,
        bf16=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=12,
        gradient_accumulation_steps=4,
        logging_dir="./logs",
        report_to="wandb",
        logging_steps=10,
        save_strategy="steps",
        save_steps=5000,
        evaluation_strategy="steps",
        eval_steps=300,
        learning_rate=3e-5,
        predict_with_generate=False,
        generation_max_length=64
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=sbersquad['train'],
        eval_dataset=sbersquad['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_fix
    )

    trainer.train()
    ```

7. **Сохранение модели**:
    ```python
    trainer.save_model(f'/wandb/best_model{MAX_INSTRUCTION_LENGTH}_t{MAX_TARGET_LENGTH}')

    import shutil
    archive_path = f'/wandb/best_model{MAX_INSTRUCTION_LENGTH}_t{MAX_TARGET_LENGTH}'
    shutil.make_archive(archive_path.replace('.zip', ''), 'zip', archive_path)
    print(f"Архив сохранен по адресу: {archive_path}")
    ```

8. **Результаты дообучения модели на датасете с метриками**
    - К сожалению, мне не хватило вычислительного времени на Google Collab, поэтому модель была обучена только на одной эпохе с ~1416 шагами.

| Шаг | Loss на валидации | Sbleu | Chr F | Rouge1 | Rouge2 | Rougel |
|-----|-------------------|-------|-------|--------|--------|--------|
| 300 | 1.025008          | 18.206400 | 62.316300 | 0.110400 | 0.035200 | 0.109800 |
| 600 | 1.007530          | 18.523100 | 62.564700 | 0.113300 | 0.036500 | 0.112800 |
| 900 | 0.959073          | 18.869000 | 63.001700 | 0.115100 | 0.035600 | 0.114600 |
| 1200| 0.944776          | 18.656300 | 62.819800 | 0.115400 | 0.035800 | 0.115000 |


## Использование бота

1. **Запуск бота**:
   - После запуска бота введите команду `/start`, чтобы начать диалог.

2. **Ввод контекста**:
   - Бот попросит ввести контекст вопроса. Введите текст, на основе которого хотите задать вопрос.

3. **Ввод вопроса**:
   - После ввода контекста, бот предложит задать вопрос. Введите свой вопрос.

4. **Получение ответа**:
   - Бот сгенерирует ответ на основе введенного контекста и вопроса.

5. **Новый вопрос**:
   - После получения ответа на вопрос бот предложит вам задать новый вопрос к старому контексту.

6. **Начало с начала**:
   - Для сброса контекста и вопроса в любое время нажмите на кнопку "Начать с начала".

## Описание структуры проекта

```
telegram-qa-bot/
│
├── bot.py                      # Основной скрипт бота
├── model_training.ipynb        # Ноутбук для обучения модели в Google Colab
├── requirements.txt            # Все зависимости проекта - ноутбук и бот
└── README.md                   # Описание проекта
```

## Дополнительная информация

- Для обучения использовалась модель [ruT5-base](https://huggingface.co/ai-forever/ruT5-base) от SberBank AI.
- Датасет [SberQuAD](https://huggingface.co/datasets/kuznetsoffandrey/sberquad) является аналогом SQuAD для русского языка и содержит вопросы и ответы на различные тексты.
- Модель была дообучена для генерации ответа на вопрос, опираясь на контекст, предоставленный пользователем. Модель не может работать в другом формате.
