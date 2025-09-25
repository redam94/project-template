# Quick Start Guide

This guide will get you up and running with {{cookiecutter.project_name}} in just a few minutes.
## Prerequisites

- Python {{cookiecutter.python_version}} or higher


## Installation

### Install from PyPI

```bash
pip install {{cookiecutter.project_slug}}
```

### Install from Source

```bash
git clone https://github.com/{{ cookiecutter.github_username }}/{{ cookiecutter.github_repository }}
cd {{cookiecutter.github_repository}}
pip install -e ".[dev]"
```

## Next Steps

Now that you've completed the quick start:


## Getting Help

- **Documentation**: Full documentation at [Readthedocs](https://{{ cookiecutter.readthedocs }}.readthedocs.io)
- **GitHub Issues**: Report bugs or request features
- **Community Forum**: Ask questions and share experiences

## Common Issues

### Issue: ImportError
```python
# Solution: Ensure you've installed all dependencies
pip install {{cookiecutter.project_slug}}[all]
```


Happy optimizing! 🚀