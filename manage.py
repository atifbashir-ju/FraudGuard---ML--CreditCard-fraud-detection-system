#!/usr/bin/env python
import os
import sys

def main():
    # Hardcoded to avoid environment variable issues on Render
    settings = os.environ.get('DJANGO_SETTINGS_MODULE', '').strip()
    if not settings:
        settings = 'src.dashboard.settings'
    os.environ['DJANGO_SETTINGS_MODULE'] = settings
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError("Install Django: pip install django") from exc
    execute_from_command_line(sys.argv)

if __name__ == '__main__':
 main()