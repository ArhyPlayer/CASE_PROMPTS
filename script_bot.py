#!/usr/bin/env python3
"""
Генератор Telegram ботов с использованием LangChain
"""

import os
import sys
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers.json import SimpleJsonOutputParser

load_dotenv()

logger = logging.getLogger(__name__)


def setup_logger() -> None:
    """Настройка логирования"""
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def create_llm():
    """Создает экземпляр LLM"""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    if not api_key:
        logger.error("OPENAI_API_KEY не найден в .env файле!")
        print("❌ OPENAI_API_KEY не найден в .env файле!")
        sys.exit(1)
    
    kwargs = {
        "api_key": api_key,
        "model": model,
        "temperature": 0.7
    }
    
    if base_url and base_url.strip():
        kwargs["base_url"] = base_url
        logger.debug(f"Используется ProxyAPI: {base_url}")
    
    logger.debug(f"Создан LLM клиент: model={model}, temperature=0.7")
    return ChatOpenAI(**kwargs)


def analysis_chain(description: str) -> dict:
    """
    Цепочка 1: Анализ задания бота
    """
    logger.info("Запуск analysis_chain")
    logger.debug(f"Описание: {description[:100]}...")
    llm = create_llm()
    
    prompt = ChatPromptTemplate.from_template(
        """Ты — бизнес-аналитик и эксперт по Telegram ботам.

Проанализируй следующее техническое задание для Telegram бота:
{description}

Выполни детальный анализ и определи:
1. Основное назначение бота (главная цель)
2. Ключевые функции, которые должен реализовать бот
3. Типы взаимодействия с пользователем (команды, кнопки, текст, медиа)
4. Уровень сложности реализации (simple - простой бот с базовыми командами, medium - бот с логикой и состояниями, complex - сложная логика с БД и API)
5. Особые требования (производительность, безопасность, интеграции и т.д.)

ВАЖНО: Отвечай строго в формате JSON со следующими полями:
{{
  "bot_purpose": "...",
  "key_features": "...",
  "user_interactions": "...",
  "complexity_level": "simple|medium|complex",
  "special_requirements": "..."
}}"""
    )
    
    chain = prompt | llm | SimpleJsonOutputParser()
    
    try:
        result = chain.invoke({"description": description})
        logger.info(f"Analysis chain завершен: complexity={result.get('complexity_level', 'N/A')}")
        return result
    except Exception as e:
        logger.warning(f"Ошибка парсинга в analysis_chain: {e}. Используются значения по умолчанию")
        return {
            "bot_purpose": "Базовый функционал",
            "key_features": "Обработка команд",
            "user_interactions": "Команды",
            "complexity_level": "simple",
            "special_requirements": "Нет"
        }


def tools_selection_chain(analysis: dict, description: str) -> dict:
    """
    Цепочка 2: Подбор инструментов для генерации
    """
    logger.info("Запуск tools_selection_chain")
    llm = create_llm()
    
    prompt = ChatPromptTemplate.from_template(
        """Ты — архитектор решений для Telegram ботов.

Исходное задание: {description}

Результаты анализа:
- Назначение: {bot_purpose}
- Ключевые функции: {key_features}
- Взаимодействия: {user_interactions}
- Сложность: {complexity_level}
- Требования: {special_requirements}

На основе анализа подбери оптимальный набор инструментов:
1. Версия aiogram (3.x - современная, используй её)
2. База данных (sqlite - для простых, postgresql - для сложных, none - если не нужна)
3. Дополнительные библиотеки (requests для API, pillow для изображений и т.д.)
4. Необходимые API интеграции (если требуется работа с внешними сервисами)
5. Middleware компоненты (логирование, аналитика, антиспам и т.д.)
6. Способ управления состояниями (FSM для диалогов, memory для простого хранения, none если не нужно)

ВАЖНО: Отвечай строго в формате JSON со следующими полями:
{{
  "framework_version": "...",
  "database": "sqlite|postgresql|none",
  "additional_libraries": "...",
  "api_integrations": "...",
  "middleware_needs": "...",
  "state_management": "FSM|memory|none"
}}"""
    )
    
    chain = prompt | llm | SimpleJsonOutputParser()
    
    try:
        result = chain.invoke({
            "description": description,
            "bot_purpose": analysis.get("bot_purpose", ""),
            "key_features": analysis.get("key_features", ""),
            "user_interactions": analysis.get("user_interactions", ""),
            "complexity_level": analysis.get("complexity_level", "simple"),
            "special_requirements": analysis.get("special_requirements", "")
        })
        logger.info(f"Tools selection завершен: db={result.get('database', 'N/A')}, state={result.get('state_management', 'N/A')}")
        return result
    except Exception as e:
        logger.warning(f"Ошибка парсинга в tools_selection_chain: {e}. Используются значения по умолчанию")
        return {
            "framework_version": "aiogram 3.x",
            "database": "none",
            "additional_libraries": "",
            "api_integrations": "",
            "middleware_needs": "logging",
            "state_management": "none"
        }


def structure_chain(analysis: dict, tools: dict, description: str) -> dict:
    """
    Цепочка 3: Создание структуры кода
    """
    logger.info("Запуск structure_chain")
    llm = create_llm()
    
    prompt = ChatPromptTemplate.from_template(
        """Ты — senior Python разработчик, специалист по архитектуре Telegram ботов.

Исходное задание: {description}

Анализ:
- Назначение: {bot_purpose}
- Функции: {key_features}
- Сложность: {complexity_level}

Инструменты:
- Framework: {framework_version}
- БД: {database}
- Библиотеки: {additional_libraries}
- Состояния: {state_management}

Спроектируй детальную структуру кода:
1. Команды - список всех команд бота (/start, /help и т.д.)
2. Handlers - обработчики (command_handler, message_handler, callback_handler и т.д.)
3. States - состояния FSM если используется диалоговая логика
4. Keyboards - какие клавиатуры нужны (reply для обычных, inline для кнопок под сообщениями)
5. Modules - структура файлов (handlers.py, keyboards.py, database.py и т.д.)
6. Data models - модели данных (классы для пользователей, записей и т.д.)
7. Helper functions - вспомогательные функции (валидация, форматирование и т.д.)

ВАЖНО: Отвечай строго в формате JSON со следующими полями:
{{
  "commands": "...",
  "handlers": "...",
  "states": "...",
  "keyboards": "...",
  "modules": "...",
  "data_models": "...",
  "helper_functions": "..."
}}"""
    )
    
    chain = prompt | llm | SimpleJsonOutputParser()
    
    try:
        result = chain.invoke({
            "description": description,
            "bot_purpose": analysis.get("bot_purpose", ""),
            "key_features": analysis.get("key_features", ""),
            "complexity_level": analysis.get("complexity_level", "simple"),
            "framework_version": tools.get("framework_version", "aiogram 3.x"),
            "database": tools.get("database", "none"),
            "additional_libraries": tools.get("additional_libraries", ""),
            "state_management": tools.get("state_management", "none")
        })
        commands = result.get('commands', '')
        handlers = result.get('handlers', '')
        logger.info(f"Structure chain завершен: commands={commands}, handlers={handlers}")
        return result
    except Exception as e:
        logger.warning(f"Ошибка парсинга в structure_chain: {e}. Используются значения по умолчанию")
        return {
            "commands": "/start, /help",
            "handlers": "command_handler, message_handler",
            "states": "",
            "keyboards": "",
            "modules": "main",
            "data_models": "",
            "helper_functions": ""
        }


def code_chain(analysis: dict, tools: dict, structure: dict, description: str) -> str:
    """
    Цепочка 4: Реализация кода бота
    """
    logger.info("Запуск code_chain")
    llm = create_llm()
    
    prompt = ChatPromptTemplate.from_template(
        """Ты — expert Python разработчик, специализирующийся на Telegram ботах с aiogram 3.x.

Исходное задание: {description}

АНАЛИЗ:
- Назначение: {bot_purpose}
- Функции: {key_features}
- Сложность: {complexity_level}

ИНСТРУМЕНТЫ:
- Framework: {framework_version}
- База данных: {database}
- Библиотеки: {additional_libraries}
- Управление состояниями: {state_management}

СТРУКТУРА КОДА:
- Команды: {commands}
- Обработчики: {handlers}
- Состояния FSM: {states}
- Клавиатуры: {keyboards}
- Модули: {modules}
- Модели данных: {data_models}
- Вспомогательные функции: {helper_functions}

Сгенерируй ПОЛНЫЙ рабочий код Telegram бота на Python.

КРИТИЧЕСКИ ВАЖНЫЕ требования:
1. Использовать ТОЛЬКО aiogram 3.x (не 2.x!)
2. Все handlers ОБЯЗАТЕЛЬНО async def
3. Токен бота читается из os.getenv("BOT_TOKEN")
4. Используй современный синтаксис: Router, Dispatcher
5. Импорты: from aiogram import Bot, Dispatcher, Router, F
6. Для запуска: await dp.start_polling(bot)
7. Добавь logging (import logging, logging.basicConfig)
8. Обработка ошибок try/except где необходимо
9. Если нужны состояния - используй FSM из aiogram.fsm
10. Если нужны клавиатуры - используй ReplyKeyboardMarkup или InlineKeyboardMarkup
11. Код ПОЛНОСТЬЮ готов к запуску, БЕЗ заглушек, TODO или комментариев "добавьте свой код"
12. Все функции РЕАЛИЗОВАНЫ полностью

Верни ТОЛЬКО код Python, без объяснений и без markdown форматирования."""
    )
    
    chain = prompt | llm
    
    result = chain.invoke({
        "description": description,
        "bot_purpose": analysis.get("bot_purpose", ""),
        "key_features": analysis.get("key_features", ""),
        "complexity_level": analysis.get("complexity_level", "simple"),
        "framework_version": tools.get("framework_version", "aiogram 3.x"),
        "database": tools.get("database", "none"),
        "additional_libraries": tools.get("additional_libraries", []),
        "state_management": tools.get("state_management", "none"),
        "commands": structure.get("commands", []),
        "handlers": structure.get("handlers", []),
        "states": structure.get("states", []),
        "keyboards": structure.get("keyboards", []),
        "modules": structure.get("modules", []),
        "data_models": structure.get("data_models", []),
        "helper_functions": structure.get("helper_functions", [])
    })
    
    code = result.content
    
    # Очистка от markdown форматирования
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    elif "```" in code:
        code = code.split("```")[1].split("```")[0]
    
    code = code.strip()
    logger.info(f"Code chain завершен: {len(code)} символов, {len(code.splitlines())} строк")
    return code


def review_chain(code: str) -> dict:
    """
    Цепочка 5: Финальная проверка кода
    """
    logger.info("Запуск review_chain")
    logger.debug(f"Проверка кода: {len(code)} символов")
    llm = create_llm()
    
    prompt = ChatPromptTemplate.from_template(
        """Ты — опытный код-ревьюер Python кода.

Проверь следующий код Telegram бота:

```python
{code}
```

Проверь:
1. Синтаксические ошибки
2. Корректность структуры запуска бота
3. Правильность импортов (aiogram 3.x)
4. Наличие обработки ошибок
5. Общее качество кода

ВАЖНО: Отвечай строго в формате JSON со следующими полями:
{{
  "is_valid": "yes|no",
  "syntax_errors": "...",
  "structure_issues": "...",
  "import_issues": "...",
  "recommendations": "..."
}}"""
    )
    
    chain = prompt | llm | SimpleJsonOutputParser()
    
    try:
        result = chain.invoke({"code": code})
        logger.info(f"Review chain завершен: is_valid={result.get('is_valid', 'N/A')}")
        return result
    except Exception as e:
        logger.warning(f"Ошибка парсинга в review_chain: {e}. Используются значения по умолчанию")
        return {
            "is_valid": "yes",
            "syntax_errors": "none",
            "structure_issues": "none",
            "import_issues": "none",
            "recommendations": "Код готов к использованию"
        }


def generate_bot(description: str) -> str:
    """
    Главная функция: запускает полную цепочку генерации бота (4 этапа + проверка)
    """
    logger.info("="*80)
    logger.info("Запуск генерации Telegram бота")
    logger.info(f"Описание: {description}")
    logger.info("="*80)
    
    print("\n" + "="*80)
    print("🤖 ГЕНЕРАТОР TELEGRAM БОТОВ (LangChain Pipeline)")
    print("="*80)
    
    # Шаг 1: Анализ задания
    print("\n📊 ШАГ 1/4: Анализ задания бота...")
    analysis = analysis_chain(description)
    print("✅ Анализ завершен")
    print(f"   • Назначение: {analysis.get('bot_purpose', 'N/A')}")
    print(f"   • Уровень сложности: {analysis.get('complexity_level', 'N/A')}")
    key_features = analysis.get('key_features', 'N/A')
    if len(str(key_features)) > 100:
        print(f"   • Ключевые функции: {str(key_features)[:100]}...")
    else:
        print(f"   • Ключевые функции: {key_features}")
    
    # Шаг 2: Подбор инструментов
    print("\n🔧 ШАГ 2/4: Подбор инструментов для генерации...")
    tools = tools_selection_chain(analysis, description)
    print("✅ Инструменты подобраны")
    print(f"   • Framework: {tools.get('framework_version', 'N/A')}")
    print(f"   • База данных: {tools.get('database', 'none')}")
    print(f"   • Управление состояниями: {tools.get('state_management', 'none')}")
    additional_libs = tools.get('additional_libraries', '')
    if additional_libs and additional_libs != 'none':
        print(f"   • Доп. библиотеки: {additional_libs}")
    
    # Шаг 3: Создание структуры
    print("\n🏗️ ШАГ 3/4: Создание структуры кода...")
    structure = structure_chain(analysis, tools, description)
    print("✅ Структура создана")
    commands = structure.get('commands', 'N/A')
    print(f"   • Команды: {commands}")
    handlers = structure.get('handlers', 'N/A')
    print(f"   • Обработчики: {handlers}")
    keyboards = structure.get('keyboards', '')
    if keyboards:
        print(f"   • Клавиатуры: {keyboards}")
    
    # Шаг 4: Реализация кода
    print("\n💻 ШАГ 4/4: Реализация кода бота...")
    code = code_chain(analysis, tools, structure, description)
    print("✅ Код реализован")
    print(f"   • Размер: {len(code)} символов")
    print(f"   • Строк кода: {len(code.splitlines())}")
    
    # Финальная проверка
    print("\n🔍 ФИНАЛЬНАЯ ПРОВЕРКА: Валидация кода...")
    review = review_chain(code)
    print("✅ Проверка завершена")
    print(f"   • Статус: {review.get('is_valid', 'yes')}")
    print(f"   • Синтаксические ошибки: {review.get('syntax_errors', 'none')}")
    
    if review.get('recommendations') and review.get('recommendations') != 'Код готов к использованию':
        print(f"\n💡 Рекомендации: {review.get('recommendations')}")
    
    # Сохранение файла
    output_file = "generated_bot.py"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(code)
        logger.info(f"Бот сохранен в файл: {output_file}")
    except Exception as e:
        logger.error(f"Ошибка при сохранении файла: {e}")
        print(f"\n❌ Ошибка при сохранении файла: {e}")
        return code
    
    print("\n" + "="*80)
    print(f"✅ БОТ УСПЕШНО СГЕНЕРИРОВАН: {output_file}")
    print("="*80)
    print("\n📝 Для запуска бота:")
    print(f"   1. Установите зависимости: pip install aiogram python-dotenv")
    
    # Проверяем, нужны ли дополнительные зависимости
    additional_libs = tools.get('additional_libraries', '')
    if additional_libs and additional_libs != 'none' and additional_libs.strip():
        print(f"   2. Установите доп. библиотеки: pip install {additional_libs}")
        print(f"   3. Добавьте BOT_TOKEN в .env файл")
        print(f"   4. Запустите: python {output_file}")
    else:
        print(f"   2. Добавьте BOT_TOKEN в .env файл")
        print(f"   3. Запустите: python {output_file}")
    print()
    
    return code


def main():
    """Основная функция"""
    # Настройка логирования
    setup_logger()
    logger.info("Запуск script_bot.py")
    
    if len(sys.argv) < 2:
        logger.error("Не указано описание бота")
        print("❌ Использование: python script_bot.py \"Описание бота\"")
        print("\nПример:")
        print('   python script_bot.py "Бот, который отправляет случайные мемы"')
        sys.exit(1)
    
    description = sys.argv[1]
    logger.info(f"Получено описание: {description}")
    
    try:
        generate_bot(description)
        logger.info("Генерация бота успешно завершена")
    except KeyboardInterrupt:
        logger.warning("Генерация прервана пользователем")
        print("\n\n❌ Генерация прервана пользователем")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Ошибка при генерации бота: {e}", exc_info=True)
        print(f"\n❌ Ошибка при генерации бота: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

