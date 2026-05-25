#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'MainServerv2.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()

# Найти все процессы tasklist | findstr python
# Найти netstat -ano | findstr :8000

# Запуск в консоли python manage.py runserver 0.0.0.0:8000
# .\.venv\Scripts\python.exe -m celery -A MainServerv2 worker --loglevel=info --pool=solo

# Для теста python manage.py runserver 0.0.0.0:8000 --nothreading

# Убить все процессы taskkill /F /IM python.exe
# Убить процес taskkill /PID 19004 /F