#!/usr/bin/env python
"""Django command-line utility."""
import os
import sys


def main() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "fall_web.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Chưa cài Django. Hãy chạy install_dependencies.bat trước."
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
