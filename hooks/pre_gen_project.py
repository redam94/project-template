#!/usr/bin/env python
"""Pre-generation hook for cookiecutter."""

import re
import sys

MODULE_REGEX = r'^[_a-zA-Z][_a-zA-Z0-9]+$'

module_name = '{{ cookiecutter.module_name }}'

if not re.match(MODULE_REGEX, module_name):
    print(f'ERROR: {module_name} is not a valid Python module name!')
    print('Please use only letters, numbers, and underscores.')
    sys.exit(1)

print(f"Creating project: {{ cookiecutter.project_name }}")
print(f"Module name: {module_name}")
print(f"Project slug: {{ cookiecutter.project_slug }}")