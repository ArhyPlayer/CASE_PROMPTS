#!/usr/bin/env python3
"""
CLI скрипт для работы с промптами и OpenAI API через ProxyAPI
"""

import os
import json
import sys
import io
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Настройка кодировки для корректной работы с Unicode
try:
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if sys.stdin.encoding != 'utf-8':
        sys.stdin.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, TypeError):
    # Если reconfigure недоступен, создаем обертку для stdin
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='replace')

# Загрузка переменных окружения
load_dotenv()


class PromptsManager:
    """Менеджер для работы с промптами"""
    
    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = Path(prompts_dir)
        self.prompts: List[Dict] = []
        self.load_prompts()
    
    def load_prompts(self):
        """Загружает все промпты из директории"""
        if not self.prompts_dir.exists():
            print(f"❌ Директория {self.prompts_dir} не найдена!")
            sys.exit(1)
        
        json_files = sorted(self.prompts_dir.glob("*.json"))
        
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    prompt_data = json.load(f)
                    self.prompts.append(prompt_data)
            except Exception as e:
                print(f"⚠️ Ошибка при загрузке {file_path}: {e}")
        
        if not self.prompts:
            print("❌ Не найдено ни одного промпта!")
            sys.exit(1)
        
        print(f"✅ Загружено промптов: {len(self.prompts)}")
    
    def list_prompts(self):
        """Выводит список доступных промптов"""
        print("\n" + "="*80)
        print("📋 Доступные промпты:")
        print("="*80 + "\n")
        
        for idx, prompt in enumerate(self.prompts, 1):
            print(f"{idx}. {prompt.get('name', 'Без названия')}")
            print(f"   🔖 ID: {prompt.get('prompt_id', 'N/A')}")
            print(f"   📁 Категория: {prompt.get('category', 'N/A')}")
            print(f"   📝 Описание: {prompt.get('description', 'N/A')}")
            
            # Роль - обрезаем и добавляем многоточие
            role = prompt.get('role', 'N/A')
            if len(role) > 100:
                print(f"   👤 Роль: {role[:100]}...")
            else:
                print(f"   👤 Роль: {role}")
            
            # Контекст - обрезаем и добавляем многоточие
            context = prompt.get('context', 'N/A')
            if len(context) > 100:
                print(f"   📦 Контекст: {context[:100]}...")
            else:
                print(f"   📦 Контекст: {context}")
            
            if prompt.get('test_input'):
                print(f"   ✨ Есть тестовый пример")
            
            print()
    
    def get_prompt(self, index: int) -> Optional[Dict]:
        """Возвращает промпт по индексу"""
        if 0 < index <= len(self.prompts):
            return self.prompts[index - 1]
        return None


class OpenAIClient:
    """Клиент для работы с OpenAI API через ProxyAPI"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
        self.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "2000"))
        
        if not self.api_key:
            print("❌ OPENAI_API_KEY не найден в .env файле!")
            sys.exit(1)
        
        # Инициализация клиента OpenAI с ProxyAPI
        if self.base_url and self.base_url.strip():
            print(f"🌐 Используется ProxyAPI: {self.base_url}")
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            print("🌐 Используется стандартный OpenAI API")
            self.client = OpenAI(api_key=self.api_key)
    
    def send_request(self, prompt_data: Dict, user_question: str) -> Dict:
        """
        Отправляет запрос к OpenAI API
        
        Args:
            prompt_data: Данные промпта
            user_question: Вопрос пользователя
        
        Returns:
            Dict с ответом и метаинформацией
        """
        # Формируем системное сообщение из промпта
        system_message = self._build_system_message(prompt_data)
        
        # Формируем сообщения для API
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_question}
        ]
        
        try:
            print("\n🔄 Отправляем запрос к OpenAI...")
            
            # Отправляем запрос
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Извлекаем данные из ответа
            result = {
                "answer": response.choices[0].message.content,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "finish_reason": response.choices[0].finish_reason
            }
            
            return result
            
        except Exception as e:
            print(f"\n❌ Ошибка при обращении к API: {e}")
            sys.exit(1)
    
    def _build_system_message(self, prompt_data: Dict) -> str:
        """Создает системное сообщение из данных промпта"""
        parts = []
        
        # Добавляем роль
        if "role" in prompt_data:
            parts.append(f"{prompt_data['role']}")
        
        # Добавляем контекст
        if "context" in prompt_data:
            parts.append(f"\nКОНТЕКСТ: {prompt_data['context']}")
        
        # Добавляем описание структуры в текстовом формате
        if "structure" in prompt_data:
            parts.append(f"\nФОРМАТ ОТВЕТА:")
            structure = prompt_data['structure']
            
            if 'output_format' in structure:
                parts.append(f"Формат: {structure['output_format']}")
            
            if 'components' in structure:
                parts.append("\nОтвет должен содержать следующие разделы:")
                for component in structure['components']:
                    name = component.get('name', '')
                    desc = component.get('description', '')
                    parts.append(f"- {name}: {desc}")
        
        # Добавляем требования к формату
        if "format" in prompt_data:
            format_info = prompt_data['format']
            parts.append("\nТРЕБОВАНИЯ:")
            
            if 'structure' in format_info:
                parts.append(f"- {format_info['structure']}")
            if 'length' in format_info:
                parts.append(f"- {format_info['length']}")
            if 'style' in format_info:
                parts.append(f"- {format_info['style']}")
            if 'requirements' in format_info:
                for req in format_info['requirements']:
                    parts.append(f"- {req}")
        
        # Важная инструкция о формате
        parts.append("\n⚠️ ВАЖНО: Отвечай в формате читаемого текста с использованием Markdown разметки (заголовки #, ##, списки -, **жирный текст**). НЕ используй JSON формат в ответе!")
        
        return "\n".join(parts)


def get_user_input(prompt: str, default: str = "") -> str:
    """Получает ввод пользователя с дефолтным значением"""
    if default:
        user_input = input(f"{prompt} (по умолчанию {default}): ").strip()
        return user_input if user_input else default
    else:
        return input(f"{prompt}: ").strip()


def get_multiline_input(prompt: str) -> str:
    """
    Получает многострочный ввод от пользователя
    Ввод завершается словом END на отдельной строке или EOF (Ctrl+D на Mac/Linux, Ctrl+Z на Windows)
    """
    print(f"\n{prompt}")
    print("💡 Варианты ввода:")
    print("   1. Вставьте текст, затем на новой строке напишите: END")
    print("   2. Нажмите Ctrl+D (Mac/Linux) / Ctrl+Z (Windows) для завершения")
    print("   3. Введите 'file:путь_к_файлу.txt' для чтения из файла")
    print("-" * 80)
    
    lines = []
    first_line = True
    
    try:
        while True:
            try:
                # Читаем с обработкой ошибок кодировки
                try:
                    line = input()
                except UnicodeDecodeError:
                    # Если ошибка кодировки, пытаемся прочитать с заменой некорректных символов
                    line = sys.stdin.buffer.readline().decode('utf-8', errors='replace').rstrip('\n\r')
                
                # Проверяем команду завершения
                if line.strip().upper() == 'END':
                    break
                
                # Проверяем, не команда ли это для чтения из файла
                if first_line and line.strip().startswith('file:'):
                    file_path = line.strip()[5:].strip()
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                            content = f.read()
                        print(f"✅ Прочитано из файла: {len(content)} символов")
                        print("-" * 80)
                        return content.strip()
                    except FileNotFoundError:
                        print(f"❌ Файл не найден: {file_path}")
                        print("Продолжайте ввод вручную или введите другой путь к файлу:")
                        first_line = False
                        continue
                    except Exception as e:
                        print(f"❌ Ошибка чтения файла: {e}")
                        print("Продолжайте ввод вручную:")
                        first_line = False
                        continue
                
                first_line = False
                
                # Добавляем строку (даже пустую)
                lines.append(line)
                
                # Показываем прогресс каждые 10 строк
                if len(lines) % 10 == 0:
                    print(f"  [Введено строк: {len(lines)}]", end='\r')
                    
            except EOFError:
                # Ctrl+D (Unix) или Ctrl+Z (Windows)
                break
    except KeyboardInterrupt:
        print("\n\n❌ Ввод отменен")
        sys.exit(0)
    
    # Убираем пустые строки в конце
    while lines and not lines[-1].strip():
        lines.pop()
    
    result = '\n'.join(lines)
    
    if lines:
        print(f"\n✅ Введено строк: {len(lines)}, символов: {len(result)}")
    
    print("-" * 80)
    
    # Дополнительная очистка некорректных символов
    result = result.replace('�', ' ')  # Заменяем символы замены на пробел
    
    return result.strip()


def yes_no_question(question: str, default: str = "n") -> bool:
    """Задает вопрос да/нет"""
    default_str = "y" if default.lower() == "y" else "n"
    answer = input(f"😊 {question} (y/n, по умолчанию {default_str}): ").strip()
    if not answer:
        answer = default_str
    return answer.lower() in ['y', 'yes', 'да', 'д']


def show_code_structure_submenu() -> str:
    """Показывает подменю для промпта 'Генерация структуры кода'"""
    print("\n" + "="*80)
    print("📋 ПОДМЕНЮ: Генерация структуры кода")
    print("="*80)
    print("\n1. Стандартные запросы")
    print("2. Генерация Telegram бота (LangChain)")
    print("0. Назад к выбору промпта")
    print()
    
    while True:
        choice = input("Выберите опцию (1-2, 0 для выхода): ").strip()
        
        if choice in ['1', '2', '0']:
            return choice
        else:
            print("⚠️ Некорректный выбор, попробуйте снова")


def generate_telegram_bot():
    """Запускает генератор Telegram бота"""
    print("\n" + "="*80)
    print("🤖 ГЕНЕРАТОР TELEGRAM БОТОВ")
    print("="*80)
    print("\nС помощью LangChain будет создан готовый Telegram бот на основе вашего описания.")
    print("Цепочка обработки: Анализ → Генерация кода → Проверка кода")
    print()
    
    # Получаем описание бота
    description = get_multiline_input("💬 Введите описание бота (что он должен уметь):")
    
    if not description:
        print("❌ Описание не может быть пустым!")
        return
    
    # Проверяем наличие script_bot.py
    script_path = Path("script_bot.py")
    if not script_path.exists():
        print("❌ Файл script_bot.py не найден!")
        return
    
    print("\n⏳ Запускаем генератор бота...")
    print("-" * 80)
    
    try:
        # Запускаем script_bot.py с описанием
        result = subprocess.run(
            [sys.executable, "script_bot.py", description],
            capture_output=False,
            text=True,
            encoding='utf-8'
        )
        
        if result.returncode == 0:
            print("\n" + "="*80)
            print("✅ Бот успешно сгенерирован!")
            print("="*80)
            
            # Спрашиваем, показать ли содержимое
            if yes_no_question("Показать сгенерированный код?", "n"):
                bot_file = Path("generated_bot.py")
                if bot_file.exists():
                    print("\n" + "="*80)
                    print("📄 СОДЕРЖИМОЕ: generated_bot.py")
                    print("="*80)
                    with open(bot_file, 'r', encoding='utf-8') as f:
                        print(f.read())
                    print("="*80)
        else:
            print(f"\n❌ Ошибка при генерации бота (код: {result.returncode})")
            
    except Exception as e:
        print(f"\n❌ Ошибка при запуске генератора: {e}")
    
    input("\n\nНажмите Enter для продолжения...")


def print_header():
    """Выводит заголовок программы"""
    print("\n" + "="*80)
    print("🤖 CLI Инструмент для работы с промптами и OpenAI API")
    print("="*80)


def print_request_info(model: str, temperature: float, max_tokens: int):
    """Выводит информацию о настройках запроса"""
    print("\n" + "="*80)
    print("📊 Информация о запросе:")
    print(f"  • Модель: {model}")
    print(f"  • Temperature: {temperature}")
    print(f"  • Max tokens: {max_tokens}")
    print("="*80)


def print_response_info(response_data: Dict):
    """Выводит информацию об ответе"""
    usage = response_data.get("usage", {})
    
    print("\n" + "="*80)
    print("📊 Информация о запросе:")
    print(f"  • Модель: {response_data.get('model', 'N/A')}")
    print(f"  • Использовано токенов: {usage.get('total_tokens', 0)}")
    print(f"  • Промпт токены: {usage.get('prompt_tokens', 0)}")
    print(f"  • Ответ токены: {usage.get('completion_tokens', 0)}")
    print("="*80)
    print("\n👍 Готово! До свидания!")


def configure_model(client: OpenAIClient) -> tuple:
    """Позволяет настроить параметры модели"""
    print("\n" + "="*80)
    print("⚙️ Настройки модели:")
    
    # Temperature
    temp_input = input(f"🌡️ Введите temperature (0.0-1.0, по умолчанию {client.temperature}): ").strip()
    if temp_input:
        try:
            client.temperature = float(temp_input)
            if not 0.0 <= client.temperature <= 1.0:
                print("⚠️ Temperature должна быть между 0.0 и 1.0, используем значение по умолчанию")
                client.temperature = 0.7
        except ValueError:
            print("⚠️ Некорректное значение, используем значение по умолчанию")
    
    # Max tokens
    tokens_input = input(f"📊 Введите max_tokens (по умолчанию {client.max_tokens}): ").strip()
    if tokens_input:
        try:
            client.max_tokens = int(tokens_input)
        except ValueError:
            print("⚠️ Некорректное значение, используем значение по умолчанию")
    
    # Модель
    model_input = input(f"🤖 Введите модель (по умолчанию {client.model}): ").strip()
    if model_input:
        client.model = model_input
    
    print("="*80)
    
    return client.model, client.temperature, client.max_tokens


def main():
    """Основная функция программы"""
    print_header()
    
    # Инициализация
    client = OpenAIClient()
    manager = PromptsManager()
    
    # Показываем список промптов сразу после загрузки
    manager.list_prompts()
    
    # Выбор промпта
    selected_prompt = None
    while True:
        choice = get_user_input("📋 Выберите промпт (1-3) или 'выход' для завершения", "")
        
        if choice.lower() in ['выход', 'exit', 'quit', 'q']:
            print("\n👋 До свидания!")
            sys.exit(0)
        
        try:
            prompt_index = int(choice)
            selected_prompt = manager.get_prompt(prompt_index)
            
            if selected_prompt:
                print(f"\n✅ Выбран промпт: {selected_prompt.get('name')}")
                
                # Проверяем, является ли это промптом "Генерация структуры кода"
                if selected_prompt.get('prompt_id') == 'code_structure':
                    submenu_choice = show_code_structure_submenu()
                    
                    if submenu_choice == '0':
                        # Возврат к выбору промпта
                        continue
                    elif submenu_choice == '2':
                        # Генерация Telegram бота
                        generate_telegram_bot()
                        
                        # Спрашиваем, хочет ли пользователь продолжить
                        if yes_no_question("Вернуться к меню промптов?", "y"):
                            continue
                        else:
                            print("\n👋 До свидания!")
                            sys.exit(0)
                    # submenu_choice == '1' - продолжаем стандартную обработку
                
                break
            else:
                print("⚠️ Некорректный номер промпта, попробуйте снова")
        except ValueError:
            print("⚠️ Введите число или 'выход'")
    
    # Показываем ПОЛНЫЙ тестовый вопрос
    print("\n" + "="*80)
    test_input = selected_prompt.get('test_input', '')
    if test_input:
        print(f"💡 Доступен тестовый вопрос:")
        print(f"   {test_input}")
    print("="*80)
    
    # Спрашиваем про тестовый вопрос
    if test_input:
        use_test_question = yes_no_question("Использовать тестовый вопрос?", "n")
    else:
        use_test_question = False
    
    # Настройка модели
    model, temperature, max_tokens = configure_model(client)
    
    # Сообщение о начале работы
    print(f"\n⏳ Отправляем запрос к OpenAI...")
    
    # Получаем вопрос
    if use_test_question and test_input:
        user_question = test_input
        print(f"\n✅ Используем тестовый вопрос")
    else:
        user_question = get_multiline_input("💬 Введите ваш вопрос:")
        if not user_question:
            print("❌ Вопрос не может быть пустым!")
            sys.exit(1)
    
    # Отправляем запрос
    response_data = client.send_request(selected_prompt, user_question)
    
    # Выводим ответ
    print("\n" + "="*80)
    print(f"💡 Ответ от OpenAI - {selected_prompt.get('name')}")
    print("="*80)
    print()
    print(response_data.get("answer", "Нет ответа"))
    print()
    print("="*80)
    
    # Выводим статистику
    print_response_info(response_data)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Программа прервана пользователем. До свидания!")
        sys.exit(0)

