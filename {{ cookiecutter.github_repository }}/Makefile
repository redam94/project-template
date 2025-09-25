.PHONY: help clean install dev test lint format type-check docs serve-docs
.PHONY: generate-docs generate-tests auto-generate coverage quality docker
.PHONY: build publish clean-all

# Variables
PYTHON := python
PIP := pip
PROJECT := {{ cookiecutter.module_name }}
SRC_DIR := src
TEST_DIR := tests
DOCS_DIR := docs

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation
install: ## Install the package
	$(PIP) install -e .

dev: ## Install with all development dependencies
	$(PIP) install -e ".[all]"
	pre-commit install

# Testing
test: ## Run all tests
	pytest $(TEST_DIR) -v

test-unit: ## Run unit tests only
	pytest $(TEST_DIR) -v -m unit

test-integration: ## Run integration tests only
	pytest $(TEST_DIR) -v -m integration

coverage: ## Run tests with coverage report
	pytest $(TEST_DIR) --cov=$(PROJECT) --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/index.html"

tox: ## Run tests in multiple environments
	tox

# Code Quality
lint: ## Run all linting tools
	flake8 $(SRC_DIR) $(TEST_DIR)
	pylint $(SRC_DIR) || true
	mypy $(SRC_DIR)

format: ## Format code with black and isort
	isort $(SRC_DIR) $(TEST_DIR) scripts/
	black $(SRC_DIR) $(TEST_DIR) scripts/

type-check: ## Run type checking with mypy
	mypy $(SRC_DIR) --html-report mypy-report

quality: lint type-check ## Run all code quality checks
	@echo "All quality checks passed!"

pre-commit: ## Run pre-commit hooks
	pre-commit run --all-files

# Documentation
docs: ## Build documentation
	cd $(DOCS_DIR) && make clean && make html
	@echo "Documentation built in $(DOCS_DIR)/build/html/index.html"

serve-docs: ## Serve documentation locally
	cd $(DOCS_DIR) && python -m http.server --directory build/html 8000

# AI Generation
generate-docs: ## Generate documentation using AI
	$(PYTHON) scripts/generate_docs.py --source $(SRC_DIR) --enhance

generate-tests: ## Generate tests using AI
	$(PYTHON) scripts/generate_tests.py --source $(SRC_DIR) --framework pytest

auto-generate: ## Generate both docs and tests automatically
	$(PYTHON) scripts/auto_generate_all.py

analyze-quality: ## Analyze code quality and generate report
	$(PYTHON) scripts/analyze_quality.py --source $(SRC_DIR) --output quality_report.html

# Build and Release
build: clean ## Build distribution packages
	$(PYTHON) -m build

publish-test: build ## Publish to TestPyPI
	$(PYTHON) -m twine upload --repository testpypi dist/*

publish: build ## Publish to PyPI
	$(PYTHON) -m twine upload dist/*

# Docker
docker-build: ## Build Docker image
	docker build -t $(PROJECT):latest .

docker-run: ## Run Docker container
	docker run -it --rm $(PROJECT):latest

# Cleaning
clean: ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info .coverage htmlcov/ .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-docs: ## Clean documentation build
	cd $(DOCS_DIR) && make clean

clean-all: clean clean-docs ## Clean everything
	rm -rf .tox/ .mypy_cache/ mypy-report/
	rm -f coverage.xml .coverage.*

# Development workflow shortcuts
refresh: clean install test ## Clean, reinstall, and test

ci: lint test coverage ## Run CI pipeline locally

update-deps: ## Update all dependencies
	pip-compile --upgrade requirements.txt
	pip-compile --upgrade requirements-dev.txt

# Git hooks
install-hooks: ## Install git hooks
	pre-commit install
	pre-commit install --hook-type commit-msg

validate: pre-commit test ## Validate code before commit