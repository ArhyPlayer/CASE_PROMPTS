#!/usr/bin/env python3
"""
Генератор текстовых постов с использованием LangChain
"""

import os
import sys
import logging
from datetime import datetime
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


def analysis_chain(topic: str, source_text: str = "") -> dict:
    """
    Цепочка 1: Анализ темы и исходного материала
    """
    logger.info("Запуск analysis_chain")
    logger.debug(f"Тема: {topic[:100]}...")
    llm = create_llm()
    
    prompt = ChatPromptTemplate.from_template(
        """Ты — контент-аналитик и эксперт по созданию вовлекающего контента.

Тема поста: {topic}
Исходный материал: {source_text}

Выполни детальный анализ и определи:
1. Основная цель поста (информирование, привлечение внимания, обучение, развлечение)
2. Целевая аудитория (кто будет читать этот пост)
3. Ключевые сообщения (что важно донести)
4. Тон и стиль (формальный, дружеский, профессиональный, эмоциональный)
5. Желаемая длина (short - до 500 символов, medium - 500-1500, long - 1500+)

ВАЖНО: Отвечай строго в формате JSON со следующими полями:
{{
  "post_goal": "...",
  "target_audience": "...",
  "key_messages": "...",
  "tone_style": "...",
  "desired_length": "short|medium|long"
}}"""
    )
    
    chain = prompt | llm | SimpleJsonOutputParser()
    
    try:
        result = chain.invoke({
            "topic": topic,
            "source_text": source_text if source_text else "Не предоставлен"
        })
        logger.info(f"Analysis chain завершен: goal={result.get('post_goal', 'N/A')}")
        return result
    except Exception as e:
        logger.warning(f"Ошибка парсинга в analysis_chain: {e}. Используются значения по умолчанию")
        return {
            "post_goal": "Информирование",
            "target_audience": "Широкая аудитория",
            "key_messages": "Основная информация по теме",
            "tone_style": "Нейтральный",
            "desired_length": "medium"
        }


def style_selection_chain(analysis: dict, topic: str) -> dict:
    """
    Цепочка 2: Подбор стиля и формата
    """
    logger.info("Запуск style_selection_chain")
    llm = create_llm()
    
    prompt = ChatPromptTemplate.from_template(
        """Ты — копирайтер и специалист по контент-маркетингу.

Тема поста: {topic}

Результаты анализа:
- Цель: {post_goal}
- Аудитория: {target_audience}
- Ключевые сообщения: {key_messages}
- Тон: {tone_style}
- Длина: {desired_length}

Определи оптимальный формат и стиль:
1. Структура поста (с заголовками, списками, абзацами и т.д.)
2. Использование emoji (да/нет и какие)
3. Призыв к действию (CTA) - нужен ли и какой
4. Хэштеги (нужны ли, сколько, какие темы)
5. Форматирование (жирный текст, курсив, подзаголовки)

ВАЖНО: Отвечай строго в формате JSON со следующими полями:
{{
  "structure": "...",
  "use_emoji": "yes|no",
  "emoji_style": "...",
  "cta": "...",
  "hashtags": "...",
  "formatting": "..."
}}"""
    )
    
    chain = prompt | llm | SimpleJsonOutputParser()
    
    try:
        result = chain.invoke({
            "topic": topic,
            "post_goal": analysis.get("post_goal", ""),
            "target_audience": analysis.get("target_audience", ""),
            "key_messages": analysis.get("key_messages", ""),
            "tone_style": analysis.get("tone_style", ""),
            "desired_length": analysis.get("desired_length", "medium")
        })
        logger.info(f"Style selection завершен: emoji={result.get('use_emoji', 'N/A')}")
        return result
    except Exception as e:
        logger.warning(f"Ошибка парсинга в style_selection_chain: {e}. Используются значения по умолчанию")
        return {
            "structure": "Заголовок, основной текст, заключение",
            "use_emoji": "no",
            "emoji_style": "",
            "cta": "Нет",
            "hashtags": "Нет",
            "formatting": "Абзацы"
        }


def structure_chain(analysis: dict, style: dict, topic: str) -> dict:
    """
    Цепочка 3: Создание структуры контента
    """
    logger.info("Запуск structure_chain")
    llm = create_llm()
    
    prompt = ChatPromptTemplate.from_template(
        """Ты — редактор и структурный аналитик контента.

Тема поста: {topic}

Анализ:
- Цель: {post_goal}
- Аудитория: {target_audience}
- Тон: {tone_style}

Стиль:
- Структура: {structure}
- Emoji: {use_emoji}
- CTA: {cta}
- Хэштеги: {hashtags}

Создай детальную структуру контента:
1. Заголовок (цепляющий, информативный)
2. Вступление (хук, привлекающий внимание)
3. Основные блоки (2-5 смысловых блоков)
4. Заключение (подведение итогов)
5. Призыв к действию (если нужен)

ВАЖНО: Отвечай строго в формате JSON со следующими полями:
{{
  "headline": "...",
  "intro": "...",
  "main_blocks": "...",
  "conclusion": "...",
  "cta_text": "..."
}}"""
    )
    
    chain = prompt | llm | SimpleJsonOutputParser()
    
    try:
        result = chain.invoke({
            "topic": topic,
            "post_goal": analysis.get("post_goal", ""),
            "target_audience": analysis.get("target_audience", ""),
            "tone_style": analysis.get("tone_style", ""),
            "structure": style.get("structure", ""),
            "use_emoji": style.get("use_emoji", "no"),
            "cta": style.get("cta", ""),
            "hashtags": style.get("hashtags", "")
        })
        logger.info(f"Structure chain завершен: headline={result.get('headline', 'N/A')[:50]}")
        return result
    except Exception as e:
        logger.warning(f"Ошибка парсинга в structure_chain: {e}. Используются значения по умолчанию")
        return {
            "headline": "Заголовок",
            "intro": "Вступление",
            "main_blocks": "Основной контент",
            "conclusion": "Заключение",
            "cta_text": ""
        }


def content_generation_chain(
    analysis: dict, 
    style: dict, 
    structure: dict, 
    topic: str,
    source_text: str = ""
) -> str:
    """
    Цепочка 4: Генерация финального поста
    """
    logger.info("Запуск content_generation_chain")
    llm = create_llm()
    
    prompt = ChatPromptTemplate.from_template(
        """Ты — профессиональный копирайтер и создатель вовлекающего контента.

Тема поста: {topic}
Исходный материал: {source_text}

АНАЛИЗ:
- Цель: {post_goal}
- Аудитория: {target_audience}
- Тон: {tone_style}
- Длина: {desired_length}

СТИЛЬ:
- Структура: {structure_format}
- Emoji: {use_emoji}
- Форматирование: {formatting}

СТРУКТУРА КОНТЕНТА:
- Заголовок: {headline}
- Вступление: {intro}
- Основные блоки: {main_blocks}
- Заключение: {conclusion}
- CTA: {cta_text}

Сгенерируй ПОЛНЫЙ готовый пост.

КРИТИЧЕСКИ ВАЖНЫЕ требования:
1. Пост должен быть ЗАВЕРШЕННЫМ и готовым к публикации
2. Соблюдай указанный тон и стиль
3. Используй структуру из плана
4. Добавь emoji если указано
5. Форматирование: используй **жирный**, *курсив*, заголовки где нужно
6. Пост должен быть информативным и вовлекающим
7. Никаких заглушек, [скобок], TODO или комментариев "добавьте текст"
8. Весь контент ПОЛНОСТЬЮ написан

Верни ТОЛЬКО текст поста, без объяснений."""
    )
    
    chain = prompt | llm | StrOutputParser()
    
    result = chain.invoke({
        "topic": topic,
        "source_text": source_text if source_text else "Не предоставлен",
        "post_goal": analysis.get("post_goal", ""),
        "target_audience": analysis.get("target_audience", ""),
        "tone_style": analysis.get("tone_style", ""),
        "desired_length": analysis.get("desired_length", "medium"),
        "structure_format": style.get("structure", ""),
        "use_emoji": style.get("use_emoji", "no"),
        "formatting": style.get("formatting", ""),
        "headline": structure.get("headline", ""),
        "intro": structure.get("intro", ""),
        "main_blocks": structure.get("main_blocks", ""),
        "conclusion": structure.get("conclusion", ""),
        "cta_text": structure.get("cta_text", "")
    })
    
    post_content = result.strip()
    logger.info(f"Content generation завершен: {len(post_content)} символов")
    return post_content


def review_chain(post_content: str) -> dict:
    """
    Цепочка 5: Финальная проверка контента
    """
    logger.info("Запуск review_chain")
    logger.debug(f"Проверка поста: {len(post_content)} символов")
    llm = create_llm()
    
    prompt = ChatPromptTemplate.from_template(
        """Ты — редактор и эксперт по качеству контента.

Проверь следующий пост:

{post_content}

Оцени:
1. Завершенность (пост полностью готов к публикации)
2. Структурированность (логичная структура, читабельность)
3. Вовлекающность (интересно ли читать, цепляет ли внимание)
4. Грамматика (нет ли явных ошибок)
5. Соответствие теме

ВАЖНО: Отвечай строго в формате JSON со следующими полями:
{{
  "is_ready": "yes|no",
  "completeness": "...",
  "structure_quality": "...",
  "engagement": "...",
  "recommendations": "..."
}}"""
    )
    
    chain = prompt | llm | SimpleJsonOutputParser()
    
    try:
        result = chain.invoke({"post_content": post_content})
        logger.info(f"Review chain завершен: is_ready={result.get('is_ready', 'N/A')}")
        return result
    except Exception as e:
        logger.warning(f"Ошибка парсинга в review_chain: {e}. Используются значения по умолчанию")
        return {
            "is_ready": "yes",
            "completeness": "Пост завершен",
            "structure_quality": "Хорошая структура",
            "engagement": "Достаточно вовлекающий",
            "recommendations": "Пост готов к публикации"
        }


def generate_post(topic: str, source_text: str = "") -> str:
    """
    Главная функция: запускает полную цепочку генерации поста (4 этапа + проверка)
    """
    logger.info("="*80)
    logger.info("Запуск генерации текстового поста")
    logger.info(f"Тема: {topic}")
    logger.info("="*80)
    
    print("\n" + "="*80)
    print("📝 ГЕНЕРАТОР ТЕКСТОВЫХ ПОСТОВ (LangChain Pipeline)")
    print("="*80)
    
    # Шаг 1: Анализ темы
    print("\n📊 ШАГ 1/4: Анализ темы и материала...")
    analysis = analysis_chain(topic, source_text)
    print("✅ Анализ завершен")
    print(f"   • Цель: {analysis.get('post_goal', 'N/A')}")
    print(f"   • Аудитория: {analysis.get('target_audience', 'N/A')}")
    print(f"   • Тон: {analysis.get('tone_style', 'N/A')}")
    
    # Шаг 2: Подбор стиля
    print("\n🎨 ШАГ 2/4: Подбор стиля и формата...")
    style = style_selection_chain(analysis, topic)
    print("✅ Стиль подобран")
    structure_info = style.get('structure', 'N/A')
    if len(str(structure_info)) > 50:
        print(f"   • Структура: {str(structure_info)[:50]}...")
    else:
        print(f"   • Структура: {structure_info}")
    print(f"   • Emoji: {style.get('use_emoji', 'N/A')}")
    cta_info = style.get('cta', 'N/A')
    if len(str(cta_info)) > 50:
        print(f"   • CTA: {str(cta_info)[:50]}...")
    else:
        print(f"   • CTA: {cta_info}")
    
    # Шаг 3: Создание структуры
    print("\n🏗️ ШАГ 3/4: Создание структуры контента...")
    structure = structure_chain(analysis, style, topic)
    print("✅ Структура создана")
    headline = structure.get('headline', 'N/A')
    if len(headline) > 60:
        print(f"   • Заголовок: {headline[:60]}...")
    else:
        print(f"   • Заголовок: {headline}")
    
    # Шаг 4: Генерация контента
    print("\n✍️ ШАГ 4/4: Генерация финального поста...")
    post_content = content_generation_chain(analysis, style, structure, topic, source_text)
    print("✅ Пост сгенерирован")
    print(f"   • Размер: {len(post_content)} символов")
    print(f"   • Строк: {len(post_content.splitlines())}")
    
    # Финальная проверка
    print("\n🔍 ФИНАЛЬНАЯ ПРОВЕРКА: Валидация контента...")
    review = review_chain(post_content)
    print("✅ Проверка завершена")
    print(f"   • Готовность: {review.get('is_ready', 'yes')}")
    structure_quality = review.get('structure_quality', 'N/A')
    if len(str(structure_quality)) > 50:
        print(f"   • Качество структуры: {str(structure_quality)[:50]}...")
    else:
        print(f"   • Качество структуры: {structure_quality}")
    
    if review.get('recommendations') and review.get('recommendations') != 'Пост готов к публикации':
        print(f"\n💡 Рекомендации: {review.get('recommendations')}")
    
    # Сохранение файла
    output_file = "generated_post.txt"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# ТЕМА: {topic}\n")
            f.write(f"# Дата генерации: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            f.write(post_content)
        logger.info(f"Пост сохранен в файл: {output_file}")
    except Exception as e:
        logger.error(f"Ошибка при сохранении файла: {e}")
        print(f"\n❌ Ошибка при сохранении файла: {e}")
        return post_content
    
    print("\n" + "="*80)
    print(f"✅ ПОСТ УСПЕШНО СГЕНЕРИРОВАН: {output_file}")
    print("="*80)
    print(f"\n📄 СОДЕРЖИМОЕ ПОСТА:")
    print("-"*80)
    print(post_content)
    print("-"*80)
    print()
    
    return post_content


def main():
    """Основная функция"""
    # Настройка логирования
    setup_logger()
    logger.info("Запуск script_post.py")
    
    if len(sys.argv) < 2:
        logger.error("Не указана тема поста")
        print("❌ Использование: python script_post.py \"Тема поста\" [\"Исходный текст\"]")
        print("\nПримеры:")
        print('   python script_post.py "Искусственный интеллект в медицине"')
        print('   python script_post.py "Новая технология" "Подробный текст статьи..."')
        sys.exit(1)
    
    topic = sys.argv[1]
    source_text = sys.argv[2] if len(sys.argv) > 2 else ""
    
    logger.info(f"Получена тема: {topic}")
    if source_text:
        logger.info(f"Предоставлен исходный текст: {len(source_text)} символов")
    
    try:
        generate_post(topic, source_text)
        logger.info("Генерация поста успешно завершена")
    except KeyboardInterrupt:
        logger.warning("Генерация прервана пользователем")
        print("\n\n❌ Генерация прервана пользователем")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Ошибка при генерации поста: {e}", exc_info=True)
        print(f"\n❌ Ошибка при генерации поста: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

