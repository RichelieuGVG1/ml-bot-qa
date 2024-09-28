import os
import logging
from functools import partial
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext, ConversationHandler

from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM
import torch
from dotenv import load_dotenv

# Настройка логирования
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# Загрузка модели и токенизатора
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
'''
from safetensors import safe_open
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

model_path = "RichelieuGVG/tinek_sample_model.safetensors"
model_tensors = {}
with safe_open(model_path, framework="pt", device=0) as f:
    for k in f.keys():
        model_tensors[k] = f.get_tensor(k)
'''
# Use a pipeline as a high-level helper



saved_checkpoint = 'RichelieuGVG/tinek_sample_model'

'''
маленький кусочек кода для трансформирования .safetensors в .bin
from safetensors.torch import load_file
import torch
lora_model_path = 'best_model/model.safetensors'
bin_model_path = 'best_model/pytorch_model.bin'

torch.save(load_file(lora_model_path), bin_model_path)
'''
tokenizer = AutoTokenizer.from_pretrained(saved_checkpoint)
model = T5ForConditionalGeneration.from_pretrained(saved_checkpoint, use_safetensors=True).to(device)
'''
quantize_config = BaseQuantizeConfig(
        bits=4,
        group_size=128,
        desc_act=False
    )

model = AutoGPTQForCausalLM.from_quantized(saved_checkpoint,
        use_safetensors=True,
        model_basename='tinek_sample_model',
        device=device,
        use_triton=False,
        quantize_config=quantize_config)
'''
# Константы этапов
ASK_CONTEXT, ASK_QUESTION = range(2)

# Функция генерации текста
def generate_text(prompt, tokenizer, model, n=1, temperature=0.8, num_beams=3):
    encoded_input = tokenizer.encode_plus(prompt, return_tensors='pt')
    encoded_input = {k: v.to(model.device) for k, v in encoded_input.items()}

    resulted_tokens = model.generate(**encoded_input,
                                     eos_token_id=2,
                                     max_new_tokens=128,
                                     do_sample=True,
                                     num_beams=num_beams,
                                     num_return_sequences=n,
                                     temperature=temperature,
                                     top_p=0.9,
                                     top_k=50)
    resulted_texts = tokenizer.batch_decode(resulted_tokens, skip_special_tokens=True)
    return resulted_texts

generate_text = partial(generate_text, tokenizer=tokenizer, model=model)

def start(update: Update, context: CallbackContext) -> int:
    reply_keyboard = [['Начать с начала']]  # Кнопка для перезапуска
    update.message.reply_text(
        'Привет! Я нейросетевой бот от Василия Гурьянова для Тинькофф Sirius лагеря. Пожалуйста, введите контекст вашего вопроса.',
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
    )
    return ASK_CONTEXT

# Обработчик для перезапуска при использовании фразы "начать с начала"
def check_restart(update: Update, context: CallbackContext) -> bool:
    if 'начать с начала' in update.message.text.lower():
        context.user_data.clear()
        update.message.reply_text(
            'Все очищено! Пожалуйста, введите новый контекст вашего вопроса.'
        )
        return True
    return False

# Обработчик контекста
def get_context(update: Update, context: CallbackContext) -> int:
    # Проверяем, не запрашивает ли пользователь перезапуск
    if check_restart(update, context):
        return ASK_CONTEXT

    user = update.message.from_user
    test_context = update.message.text
    context.user_data['test_context'] = test_context
    logging.info("Контекст от пользователя %s: %s", user.first_name, test_context)

    update.message.reply_text(
        'Контекст принят. Теперь введите ваш вопрос.'
    )
    return ASK_QUESTION

# Обработчик вопроса
def get_question(update: Update, context: CallbackContext) -> int:
    # Проверяем, не запрашивает ли пользователь перезапуск
    if check_restart(update, context):
        return ASK_CONTEXT

    user = update.message.from_user
    question = update.message.text
    context.user_data['question'] = question
    logging.info("Вопрос от пользователя %s: %s", user.first_name, question)

    # Формируем запрос для генерации
    test_qa_prompt = f"Сгенерируй ответ на вопрос по тексту. Текст: '{context.user_data['test_context']}'. Вопрос: '{question}'."
    test_answers = generate_text(test_qa_prompt, n=1)[0]

    update.message.reply_text(f"<b>{test_answers}</b>", parse_mode="html")
    update.message.reply_text(
        'Можете задать еще один вопрос по тому же контексту или нажмите "Начать с начала" для рестарта.'
    )

    return ASK_QUESTION

# Обработчик команды /new_request для перезапуска
def new_request(update: Update, context: CallbackContext) -> int:
    context.user_data.clear()
    reply_keyboard = [['Начать с начала']]
    update.message.reply_text(
        'Все очищено! Пожалуйста, введите новый контекст вашего вопроса.',
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
    )
    return ASK_CONTEXT

# Обработчик текстовых сообщений, фильтрация не текстовых сообщений
def handle_text(update: Update, context: CallbackContext) -> int:
    # Проверяем, не запрашивает ли пользователь перезапуск
    if check_restart(update, context):
        return ASK_CONTEXT
    
    update.message.reply_text('Пожалуйста, отправьте текстовое сообщение.')

# Команда для завершения разговора
def cancel(update: Update, context: CallbackContext) -> int:
    update.message.reply_text('Разговор завершен. До свидания!')
    return ConversationHandler.END

def main():
    # Введите свой токен здесь
    load_dotenv()
    # Достаем токен
    TOKEN = os.getenv('TELEGRAM_TOKEN')

    updater = Updater(TOKEN)

    # Диспетчер для обработки команд и сообщений
    dp = updater.dispatcher

    # Определение последовательности команд и сообщений
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            ASK_CONTEXT: [
                MessageHandler(Filters.text & ~Filters.command, get_context),
                CommandHandler('new_request', new_request)  # Обработка команды /new_request
            ],
            ASK_QUESTION: [
                MessageHandler(Filters.text & ~Filters.command, get_question),
                CommandHandler('new_request', new_request)  # Обработка команды /new_request
            ]
        },
        fallbacks=[CommandHandler('cancel', cancel),
                   CommandHandler('new_request', new_request)]  # Обработка команды /new_request как fallback
    )

    # Добавление обработчиков
    dp.add_handler(conv_handler)
    dp.add_handler(MessageHandler(Filters.command, handle_text))
    dp.add_handler(MessageHandler(~Filters.text, handle_text))

    # Запуск бота
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
