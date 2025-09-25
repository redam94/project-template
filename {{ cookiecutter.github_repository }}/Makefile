.PHONY: help clean install dev test lint format type-check
.PHONY: docs docs-markdown docs-html serve-docs serve-markdown
.PHONY: generate-docs generate-tests auto-generate
.PHONY: coverage quality docker build publish clean-all

# Variables
PYTHON := python
PIP := pip
PROJECT := {{ cookiecutter.module_name }}
SRC_DIR := src
TEST_DIR := tests
DOCS_DIR := docs
DOCS_MD_DIR := docs_markdown

# Color output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)Available targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# === Installation ===
install: ## Install the package
	$(PIP) install -e .
	@echo "$(GREEN)✓ Package installed$(NC)"

dev: ## Install with all development dependencies
	$(PIP) install -e ".[all]"
	pre-commit install
	@echo "$(GREEN)✓ Development environment ready$(NC)"

# === Testing ===
test: ## Run all tests
	pytest $(TEST_DIR) -v

test-unit: ## Run unit tests only
	pytest $(TEST_DIR) -v -m unit

test-integration: ## Run integration tests only
	pytest $(TEST_DIR) -v -m integration

coverage: ## Run tests with coverage report
	pytest $(TEST_DIR) --cov=$(PROJECT) --cov-report=html --cov-report=term
	@echo "$(GREEN)✓ Coverage report: htmlcov/index.html$(NC)"

tox: ## Run tests in multiple environments
	tox

# === Code Quality ===
lint: ## Run all linting tools
	@echo "$(BLUE)Running linters...$(NC)"
	flake8 $(SRC_DIR) $(TEST_DIR)
	mypy $(SRC_DIR)
	@echo "$(GREEN)✓ Linting complete$(NC)"

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	isort $(SRC_DIR) $(TEST_DIR) scripts/
	black $(SRC_DIR) $(TEST_DIR) scripts/
	@echo "$(GREEN)✓ Code formatted$(NC)"

type-check: ## Run type checking with mypy
	mypy $(SRC_DIR) --html-report mypy-report
	@echo "$(GREEN)✓ Type check report: mypy-report/index.html$(NC)"

quality: lint type-check ## Run all code quality checks
	@echo "$(GREEN)✓ All quality checks passed!$(NC)"

pre-commit: ## Run pre-commit hooks
	pre-commit run --all-files

# === Documentation - Multiple Formats ===
docs: docs-html docs-markdown ## Build all documentation formats

docs-html: ## Build HTML documentation (Sphinx)
	@echo "$(BLUE)Building HTML documentation...$(NC)"
	cd $(DOCS_DIR) && make clean && make html
	@echo "$(GREEN)✓ HTML docs: $(DOCS_DIR)/build/html/index.html$(NC)"

docs-markdown: ## Build Markdown documentation
	@echo "$(BLUE)Building Markdown documentation...$(NC)"
	$(PYTHON) scripts/generate_markdown_docs.py \
		--source $(SRC_DIR) \
		--output $(DOCS_MD_DIR) \
		--style {{ cookiecutter.markdown_style|default('github') }}
	@echo "$(GREEN)✓ Markdown docs: $(DOCS_MD_DIR)/README.md$(NC)"

docs-pdf: docs-html ## Build PDF documentation
	cd $(DOCS_DIR) && make latexpdf
	@echo "$(GREEN)✓ PDF docs: $(DOCS_DIR)/build/latex/*.pdf$(NC)"

serve-docs: ## Serve HTML documentation locally
	@echo "$(BLUE)Serving HTML docs at http://localhost:8000$(NC)"
	cd $(DOCS_DIR) && python -m http.server --directory build/html 8000

serve-markdown: ## Serve Markdown documentation locally
	@echo "$(BLUE)Serving Markdown docs...$(NC)"
{% if cookiecutter.markdown_style == 'mkdocs' %}
	cd $(DOCS_MD_DIR) && mkdocs serve
{% elif cookiecutter.markdown_style == 'docusaurus' %}
	cd $(DOCS_MD_DIR) && npm run start
{% elif cookiecutter.markdown_style == 'vuepress' %}
	cd $(DOCS_MD_DIR) && vuepress dev
{% else %}
	python -m http.server --directory $(DOCS_MD_DIR) 8001
{% endif %}

# === AI Generation ===
generate-docs: ## Generate documentation using AI
	@echo "$(BLUE)Generating AI-enhanced documentation...$(NC)"
	$(PYTHON) scripts/generate_docs.py \
		--source $(SRC_DIR) \
		--style {{ cookiecutter.documentation_style }} \
		--provider {{ cookiecutter.ai_provider }} \
		--enhance
	@echo "$(GREEN)✓ Documentation generated$(NC)"

generate-docs-markdown: ## Generate markdown documentation with AI
	@echo "$(BLUE)Generating AI-enhanced markdown documentation...$(NC)"
	$(PYTHON) scripts/integrated_doc_generator.py \
		--source $(SRC_DIR) \
		--output $(DOCS_MD_DIR) \
		--ai-provider {{ cookiecutter.ai_provider }} \
		--markdown-style {{ cookiecutter.markdown_style|default('github') }} \
		--doc-style {{ cookiecutter.documentation_style }}
	@echo "$(GREEN)✓ Markdown documentation generated$(NC)"

generate-tests: ## Generate tests using AI
	@echo "$(BLUE)Generating tests with AI...$(NC)"
	$(PYTHON) scripts/generate_tests.py \
		--source $(SRC_DIR) \
		--framework {{ cookiecutter.test_framework }} \
		--provider {{ cookiecutter.ai_provider }}
	@echo "$(GREEN)✓ Tests generated$(NC)"

auto-generate: ## Generate both docs and tests automatically
	@echo "$(BLUE)Running complete AI generation pipeline...$(NC)"
	$(PYTHON) scripts/auto_generate_all.py
	@echo "$(GREEN)✓ All generation complete$(NC)"

analyze-quality: ## Analyze code quality and generate report
	@echo "$(BLUE)Analyzing code quality...$(NC)"
	$(PYTHON) scripts/analyze_quality.py \
		--source $(SRC_DIR) \
		--output quality_report.html
	@echo "$(GREEN)✓ Quality report: quality_report.html$(NC)"

# === Documentation Publishing ===
publish-docs-github: docs-markdown ## Publish docs to GitHub Pages
	@echo "$(BLUE)Publishing to GitHub Pages...$(NC)"
	ghp-import -n -p -f $(DOCS_MD_DIR)
	@echo "$(GREEN)✓ Docs published to GitHub Pages$(NC)"

publish-docs-readthedocs: docs-html ## Build for ReadTheDocs
	@echo "$(BLUE)Preparing for ReadTheDocs...$(NC)"
	cd $(DOCS_DIR) && make clean && make html
	@echo "$(GREEN)✓ Ready for ReadTheDocs$(NC)"

# === Build and Release ===
build: clean ## Build distribution packages
	@echo "$(BLUE)Building distribution packages...$(NC)"
	$(PYTHON) -m build
	@echo "$(GREEN)✓ Build complete$(NC)"

publish-test: build ## Publish to TestPyPI
	$(PYTHON) -m twine upload --repository testpypi dist/*

publish: build ## Publish to PyPI
	$(PYTHON) -m twine upload dist/*

# === Docker ===
docker-build: ## Build Docker image
	docker build -t $(PROJECT):latest .

docker-run: ## Run Docker container
	docker run -it --rm $(PROJECT):latest

# === Cleaning ===
clean: ## Clean build artifacts
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	rm -rf build/ dist/ *.egg-info .coverage htmlcov/ .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)✓ Clean complete$(NC)"

clean-docs: ## Clean all documentation
	cd $(DOCS_DIR) && make clean
	rm -rf $(DOCS_MD_DIR)
	@echo "$(GREEN)✓ Documentation cleaned$(NC)"

clean-all: clean clean-docs ## Clean everything
	rm -rf .tox/ .mypy_cache/ mypy-report/ .doc_cache/
	rm -f coverage.xml .coverage.*
	@echo "$(GREEN)✓ All clean$(NC)"

# === Development Workflows ===
refresh: clean install test ## Clean, reinstall, and test
	@echo "$(GREEN)✓ Project refreshed$(NC)"

ci: quality test coverage ## Run CI pipeline locally
	@echo "$(GREEN)✓ CI pipeline complete$(NC)"

docs-all: generate-docs-markdown docs-html ## Generate all documentation with AI
	@echo "$(GREEN)✓ All documentation generated$(NC)"

update-deps: ## Update all dependencies
	pip-compile --upgrade requirements.txt
	pip-compile --upgrade requirements-dev.txt
	@echo "$(GREEN)✓ Dependencies updated$(NC)"

# === Utility Commands ===
tree: ## Show project structure
	@tree -I '__pycache__|*.egg-info|.git|.tox|htmlcov|build|dist' --dirsfirst

stats: ## Show code statistics
	@echo "$(BLUE)Code Statistics:$(NC)"
	@echo "Lines of code:"
	@find $(SRC_DIR) -name "*.py" -type f -exec wc -l {} + | tail -1
	@echo "Number of files:"
	@find $(SRC_DIR) -name "*.py" -type f | wc -l
	@echo "Number of tests:"
	@grep -r "def test_" $(TEST_DIR) | wc -l

watch-tests: ## Watch and run tests on file changes
	watchmedo shell-command \
		--patterns="*.py" \
		--recursive \
		--command='make test' \
		$(SRC_DIR) $(TEST_DIR)