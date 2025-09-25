#!/usr/bin/env python
"""Post-generation hook for cookiecutter."""

import os
import shutil
import subprocess

def remove_file(filepath):
    """Remove a file if it exists."""
    if os.path.exists(filepath):
        os.remove(filepath)

def remove_dir(dirpath):
    """Remove a directory if it exists."""
    if os.path.exists(dirpath):
        shutil.rmtree(dirpath)

def init_git():
    """Initialize git repository."""
    subprocess.call(['git', 'init'])
    subprocess.call(['git', 'add', '.'])
    subprocess.call(['git', 'commit', '-m', 'Initial commit from cookiecutter template'])

if __name__ == '__main__':
    # Remove unused license file
    if '{{ cookiecutter.open_source_license }}' == 'Not open source':
        remove_file('LICENSE')
    
    # Remove Docker files if not using Docker
    if '{{ cookiecutter.use_docker }}' != 'y':
        remove_file('Dockerfile')
        remove_file('docker-compose.yml')
        remove_file('.dockerignore')
    
    # Initialize git repository
    if '{{ cookiecutter.use_github_actions }}' == 'y':
        init_git()
    
    print("\n✨ Project successfully generated!")
    print("\n📚 Next steps:")
    print("1. cd {{ cookiecutter.project_slug }}")
    print("2. Create a virtual environment: python -m venv venv")
    print("3. Activate it: source venv/bin/activate (Unix) or venv\\Scripts\\activate (Windows)")
    print("4. Install dependencies: pip install -e '.[dev]'")
    print("5. Generate initial documentation: python scripts/generate_docs.py")
    print("\n🚀 Happy coding!")