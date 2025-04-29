"""
Скрипт для исправления проблемы с совместимостью langchain_gigachat и langchain_core
"""

import os
import subprocess
import sys

def run_command(command):
    """Запускает команду в терминале и выводит результат"""
    print(f"Выполняется: {command}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Ошибка: {e}")
        print(e.stderr)
        return False

def fix_gigachat():
    """Исправляет несовместимость versioнс"""
    print("Начало исправления зависимостей...")
    
    # Удаляем конфликтующие библиотеки
    run_command("pip uninstall -y langchain-core langchain-gigachat")
    
    # Устанавливаем совместимые версии
    run_command("pip install langchain-core==0.1.29")
    run_command("pip install langchain-gigachat==0.3.10")
    
    print("\nПроверка установленных версий:")
    run_command("pip list | grep langchain")
    
    print("\nГотово! Попробуйте запустить приложение командой:")
    print("python app.py")

if __name__ == "__main__":
    fix_gigachat() 