"""
Advanced AI Documentation Generator with caching, batch processing, and multiple format support.
This enhanced version includes cost optimization, parallel processing, and smart caching.
"""

import os
import ast
import json
import asyncio
import hashlib
import argparse
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocStyle(Enum):
    """Documentation style formats."""
    NUMPY = "numpy"
    GOOGLE = "google"
    SPHINX = "sphinx"
    MARKDOWN = "markdown"

@dataclass
class CodeElement:
    """Enhanced code element with additional metadata."""
    name: str
    type: str  # 'class', 'function', 'module', 'method', 'property'
    docstring: Optional[str]
    source: str
    file_path: str
    line_number: int
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    signature: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    complexity: int = 0
    hash: Optional[str] = None
    
    def __post_init__(self):
        # Calculate hash for caching
        content = f"{self.name}{self.type}{self.source}"
        self.hash = hashlib.md5(content.encode()).hexdigest()

@dataclass
class DocumentationCache:
    """Cache for generated documentation."""
    cache_dir: Path
    ttl_days: int = 7
    
    def __post_init__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "doc_cache.json"
        self.load_cache()
    
    def load_cache(self):
        """Load cache from disk."""
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                self.cache = json.load(f)
        else:
            self.cache = {}
    
    def save_cache(self):
        """Save cache to disk."""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def get(self, key: str) -> Optional[str]:
        """Get cached documentation."""
        if key in self.cache:
            entry = self.cache[key]
            # Check if cache is still valid
            cached_time = datetime.fromisoformat(entry['timestamp'])
            if datetime.now() - cached_time < timedelta(days=self.ttl_days):
                logger.debug(f"Cache hit for {key}")
                return entry['content']
        return None
    
    def set(self, key: str, content: str):
        """Cache documentation."""
        self.cache[key] = {
            'content': content,
            'timestamp': datetime.now().isoformat()
        }
        self.save_cache()

class EnhancedCodeAnalyzer:
    """Enhanced analyzer with complexity metrics and better AST parsing."""
    
    def __init__(self, source_dir: str):
        self.source_dir = Path(source_dir)
        self.elements: Dict[str, CodeElement] = {}
        self.module_tree = {}
        
    def analyze(self) -> Dict[str, CodeElement]:
        """Analyze all Python files with enhanced metrics."""
        py_files = list(self.source_dir.rglob("*.py"))
        logger.info(f"Found {len(py_files)} Python files to analyze")
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(self._analyze_file, f): f for f in py_files}
            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error analyzing {file_path}: {e}")
        
        return self.elements
    
    def _analyze_file(self, file_path: Path):
        """Analyze a single file with enhanced extraction."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            tree = ast.parse(source, filename=str(file_path))
            rel_path = file_path.relative_to(self.source_dir)
            
            # Module-level analysis
            module_key = str(rel_path)
            self.elements[module_key] = CodeElement(
                name=file_path.stem,
                type='module',
                docstring=ast.get_docstring(tree),
                source=source[:1000],
                file_path=str(rel_path),
                line_number=0,
                complexity=self._calculate_complexity(tree)
            )
            
            # Analyze all elements in the file
            self._analyze_node(tree, source, rel_path, parent=None)
            
        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")
    
    def _analyze_node(self, node, source: str, file_path: Path, parent: Optional[str] = None, depth: int = 0):
        """Recursively analyze AST nodes."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                self._process_class(child, source, file_path, parent)
                # Recursively analyze class methods
                self._analyze_node(child, source, file_path, parent=child.name, depth=depth+1)
                
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._process_function(child, source, file_path, parent)
    
    def _process_class(self, node: ast.ClassDef, source: str, file_path: Path, parent: Optional[str]):
        """Process a class definition."""
        try:
            source_segment = ast.get_source_segment(source, node)[:2000]
        except:
            source_segment = ""
        
        # Extract decorators
        decorators = [self._get_decorator_name(d) for d in node.decorator_list]
        
        # Get class signature
        bases = [self._get_name(base) for base in node.bases]
        signature = f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}"
        
        element = CodeElement(
            name=node.name,
            type='class',
            docstring=ast.get_docstring(node),
            source=source_segment,
            file_path=str(file_path),
            line_number=node.lineno,
            parent=parent,
            signature=signature,
            decorators=decorators,
            complexity=self._calculate_complexity(node)
        )
        
        key = f"{file_path}::{parent}::{node.name}" if parent else f"{file_path}::{node.name}"
        self.elements[key] = element
    
    def _process_function(self, node, source: str, file_path: Path, parent: Optional[str]):
        """Process a function/method definition."""
        try:
            source_segment = ast.get_source_segment(source, node)[:2000]
        except:
            source_segment = ""
        
        # Determine if it's a method, property, or function
        element_type = 'method' if parent else 'function'
        decorators = [self._get_decorator_name(d) for d in node.decorator_list]
        
        if 'property' in decorators:
            element_type = 'property'
        elif 'staticmethod' in decorators or 'classmethod' in decorators:
            element_type = 'method'
        
        # Get function signature
        args = []
        for arg in node.args.args:
            args.append(arg.arg)
        signature = f"{node.name}({', '.join(args)})"
        
        element = CodeElement(
            name=node.name,
            type=element_type,
            docstring=ast.get_docstring(node),
            source=source_segment,
            file_path=str(file_path),
            line_number=node.lineno,
            parent=parent,
            signature=signature,
            decorators=decorators,
            complexity=self._calculate_complexity(node)
        )
        
        key = f"{file_path}::{parent}::{node.name}" if parent else f"{file_path}::{node.name}"
        self.elements[key] = element
    
    def _calculate_complexity(self, node) -> int:
        """Calculate cyclomatic complexity of a node."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity
    
    def _get_decorator_name(self, decorator) -> str:
        """Extract decorator name."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return decorator.attr
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                return decorator.func.id
            elif isinstance(decorator.func, ast.Attribute):
                return decorator.func.attr
        return "unknown"
    
    def _get_name(self, node) -> str:
        """Get name from various node types."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return "unknown"

class BatchAIDocGenerator:
    """AI documentation generator with batch processing and caching."""
    
    def __init__(self, provider: str, cache_dir: Path, batch_size: int = 5):
        self.provider = provider
        self.batch_size = batch_size
        self.cache = DocumentationCache(cache_dir)
        self._setup_client()
        self.total_tokens = 0
        self.total_cost = 0.0
    
    def _setup_client(self):
        """Setup AI client based on provider."""
        api_key = os.getenv(f"{self.provider.upper()}_API_KEY")
        if not api_key:
            raise ValueError(f"Please set {self.provider.upper()}_API_KEY environment variable")
        
        if self.provider == "openai":
            import openai
            self.client = openai.OpenAI(api_key=api_key)
            self.model = "gpt-4-turbo-preview"
            self.cost_per_1k_input = 0.01
            self.cost_per_1k_output = 0.03
        elif self.provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model = "claude-3-sonnet-20240229"
            self.cost_per_1k_input = 0.003
            self.cost_per_1k_output = 0.015
        elif self.provider == "google":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel('gemini-pro')
            self.model = "gemini-pro"
            self.cost_per_1k_input = 0.001
            self.cost_per_1k_output = 0.002
    
    def generate_batch_documentation(self, elements: List[CodeElement], style: DocStyle) -> Dict[str, str]:
        """Generate documentation for multiple elements in batches."""
        results = {}
        
        # Filter elements that need documentation
        elements_to_process = []
        for element in elements:
            # Check cache first
            cached_doc = self.cache.get(element.hash)
            if cached_doc:
                results[element.hash] = cached_doc
            elif not element.docstring or len(element.docstring) < 50:
                elements_to_process.append(element)
        
        logger.info(f"Processing {len(elements_to_process)} elements (skipped {len(elements) - len(elements_to_process)} cached/complete)")
        
        # Process in batches
        for i in range(0, len(elements_to_process), self.batch_size):
            batch = elements_to_process[i:i + self.batch_size]
            batch_prompt = self._create_batch_prompt(batch, style)
            
            try:
                response = self._call_ai(batch_prompt)
                parsed_docs = self._parse_batch_response(response, batch)
                
                for element, doc in parsed_docs.items():
                    results[element.hash] = doc
                    self.cache.set(element.hash, doc)
                    
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                # Fall back to individual processing
                for element in batch:
                    try:
                        doc = self._generate_single_doc(element, style)
                        results[element.hash] = doc
                        self.cache.set(element.hash, doc)
                    except Exception as e2:
                        logger.error(f"Failed to generate doc for {element.name}: {e2}")
        
        logger.info(f"Total API cost: ${self.total_cost:.2f}")
        return results
    
    def _generate_single_doc(self, element: CodeElement, style: DocStyle) -> str:
        """Generate documentation for a single element."""
        prompt = self._create_single_prompt(element, style)
        response = self._call_ai(prompt)
        return response.strip()
    
    def _create_batch_prompt(self, elements: List[CodeElement], style: DocStyle) -> str:
        """Create prompt for batch documentation generation."""
        elements_desc = []
        for i, element in enumerate(elements, 1):
            elements_desc.append(f"""
Element {i}:
- Name: {element.name}
- Type: {element.type}
- Signature: {element.signature or 'N/A'}
- Complexity: {element.complexity}
- File: {element.file_path}
- Current docstring: {element.docstring or 'None'}

Source:
```python
{element.source[:500]}
```
""")
        
        return f"""Generate comprehensive {style.value} style docstrings for the following Python elements.
Return a JSON object with element names as keys and docstrings as values.

Style Requirements for {style.value}:
{self._get_style_requirements(style)}

Elements to document:
{''.join(elements_desc)}

Return ONLY a valid JSON object like:
{{
    "element_name_1": "docstring content...",
    "element_name_2": "docstring content..."
}}
"""
    
    def _create_single_prompt(self, element: CodeElement, style: DocStyle) -> str:
        """Create prompt for single documentation generation."""
        return f"""Generate a comprehensive {style.value} style docstring for this {element.type}:

Name: {element.name}
Type: {element.type}
Signature: {element.signature or 'N/A'}
File: {element.file_path}
Complexity Score: {element.complexity}

Source Code:
```python
{element.source}
```

Current docstring: {element.docstring or 'None'}

Requirements for {style.value} style:
{self._get_style_requirements(style)}

Return ONLY the docstring content without triple quotes.
"""
    
    def _get_style_requirements(self, style: DocStyle) -> str:
        """Get style-specific requirements."""
        if style == DocStyle.NUMPY:
            return """
- One-line summary
- Extended description (if needed)
- Parameters section with types
- Returns section with types
- Raises section (if applicable)
- Examples section (if helpful)
- Notes section (if relevant)
"""
        elif style == DocStyle.GOOGLE:
            return """
- One-line summary
- Extended description (if needed)
- Args: with types and descriptions
- Returns: with type and description
- Raises: exceptions that may be raised
- Example: usage examples
"""
        elif style == DocStyle.SPHINX:
            return """
- One-line summary
- Extended description
- :param name: description for each parameter
- :type name: type for each parameter
- :returns: description of return value
- :rtype: return type
- :raises: exceptions that may be raised
"""
        else:  # Markdown
            return """
- Brief description
- **Parameters** section with types
- **Returns** section
- **Examples** section
- Use markdown formatting
"""
    
    def _call_ai(self, prompt: str) -> str:
        """Call AI API and track costs."""
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=2000
                )
                # Track usage
                self.total_tokens += response.usage.total_tokens
                input_cost = (response.usage.prompt_tokens / 1000) * self.cost_per_1k_input
                output_cost = (response.usage.completion_tokens / 1000) * self.cost_per_1k_output
                self.total_cost += input_cost + output_cost
                
                return response.choices[0].message.content
                
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    temperature=0.3
                )
                return response.content[0].text
                
            elif self.provider == "google":
                response = self.client.generate_content(prompt)
                return response.text
                
        except Exception as e:
            logger.error(f"AI API call failed: {e}")
            raise
    
    def _parse_batch_response(self, response: str, batch: List[CodeElement]) -> Dict[CodeElement, str]:
        """Parse batch response from AI."""
        try:
            # Try to parse as JSON
            docs_json = json.loads(response)
            result = {}
            for element in batch:
                if element.name in docs_json:
                    result[element] = docs_json[element.name]
            return result
        except json.JSONDecodeError:
            # Fall back to splitting by element names
            result = {}
            for element in batch:
                # Try to find documentation for this element in the response
                marker = f"{element.name}:"
                if marker in response:
                    start = response.index(marker) + len(marker)
                    # Find the next element marker or end of response
                    next_markers = [f"{e.name}:" for e in batch if e != element]
                    end = len(response)
                    for next_marker in next_markers:
                        if next_marker in response[start:]:
                            end = min(end, response.index(next_marker, start))
                    
                    doc = response[start:end].strip()
                    result[element] = doc
            return result

class DocumentationReportGenerator:
    """Generate documentation coverage reports."""
    
    def __init__(self, elements: Dict[str, CodeElement]):
        self.elements = elements
        
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive documentation coverage report."""
        total_elements = len(self.elements)
        documented = sum(1 for e in self.elements.values() if e.docstring)
        undocumented = total_elements - documented
        
        by_type = {}
        for element in self.elements.values():
            if element.type not in by_type:
                by_type[element.type] = {'total': 0, 'documented': 0}
            by_type[element.type]['total'] += 1
            if element.docstring:
                by_type[element.type]['documented'] += 1
        
        # Calculate complexity metrics
        avg_complexity = sum(e.complexity for e in self.elements.values()) / total_elements if total_elements > 0 else 0
        high_complexity = [e for e in self.elements.values() if e.complexity > 10]
        
        report = {
            'summary': {
                'total_elements': total_elements,
                'documented': documented,
                'undocumented': undocumented,
                'coverage_percentage': (documented / total_elements * 100) if total_elements > 0 else 0
            },
            'by_type': by_type,
            'complexity': {
                'average': avg_complexity,
                'high_complexity_count': len(high_complexity),
                'high_complexity_elements': [
                    {'name': e.name, 'type': e.type, 'complexity': e.complexity, 'file': e.file_path}
                    for e in high_complexity[:10]  # Top 10
                ]
            },
            'undocumented_elements': [
                {'name': e.name, 'type': e.type, 'file': e.file_path}
                for e in self.elements.values() if not e.docstring
            ][:20]  # Top 20
        }
        
        return report
    
    def save_report(self, output_path: Path, format: str = 'json'):
        """Save report to file."""
        report = self.generate_report()
        
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
        elif format == 'markdown':
            self._save_markdown_report(report, output_path)
        elif format == 'html':
            self._save_html_report(report, output_path)
    
    def _save_markdown_report(self, report: Dict, output_path: Path):
        """Save report as markdown."""
        content = [
            "# Documentation Coverage Report",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n## Summary",
            f"- Total Elements: {report['summary']['total_elements']}",
            f"- Documented: {report['summary']['documented']}",
            f"- Undocumented: {report['summary']['undocumented']}",
            f"- Coverage: {report['summary']['coverage_percentage']:.1f}%",
            "\n## Coverage by Type",
        ]
        
        for type_name, stats in report['by_type'].items():
            coverage = (stats['documented'] / stats['total'] * 100) if stats['total'] > 0 else 0
            content.append(f"- {type_name}: {stats['documented']}/{stats['total']} ({coverage:.1f}%)")
        
        content.extend([
            "\n## Complexity Analysis",
            f"- Average Complexity: {report['complexity']['average']:.1f}",
            f"- High Complexity Elements: {report['complexity']['high_complexity_count']}",
            "\n### Top Complex Elements",
        ])
        
        for element in report['complexity']['high_complexity_elements']:
            content.append(f"- {element['name']} ({element['type']}): complexity={element['complexity']} in {element['file']}")
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(content))
    
    def _save_html_report(self, report: Dict, output_path: Path):
        """Save report as HTML."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Documentation Coverage Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        .metric {{ display: inline-block; margin: 20px; padding: 20px; background: #f0f0f0; border-radius: 5px; }}
        .coverage {{ font-size: 2em; font-weight: bold; color: #007bff; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #f0f0f0; }}
        .good {{ color: green; }}
        .warning {{ color: orange; }}
        .bad {{ color: red; }}
    </style>
</head>
<body>
    <h1>Documentation Coverage Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="metrics">
        <div class="metric">
            <div>Total Elements</div>
            <div class="coverage">{report['summary']['total_elements']}</div>
        </div>
        <div class="metric">
            <div>Coverage</div>
            <div class="coverage">{report['summary']['coverage_percentage']:.1f}%</div>
        </div>
        <div class="metric">
            <div>Documented</div>
            <div class="coverage good">{report['summary']['documented']}</div>
        </div>
        <div class="metric">
            <div>Undocumented</div>
            <div class="coverage bad">{report['summary']['undocumented']}</div>
        </div>
    </div>
    
    <h2>Coverage by Type</h2>
    <table>
        <tr><th>Type</th><th>Documented</th><th>Total</th><th>Coverage</th></tr>
        {"".join(f"<tr><td>{k}</td><td>{v['documented']}</td><td>{v['total']}</td><td>{v['documented']/v['total']*100:.1f}%</td></tr>" for k, v in report['by_type'].items())}
    </table>
    
    <h2>Complexity Analysis</h2>
    <p>Average Complexity: {report['complexity']['average']:.1f}</p>
    <p>High Complexity Elements: {report['complexity']['high_complexity_count']}</p>
</body>
</html>"""
        
        with open(output_path, 'w') as f:
            f.write(html)

def main():
    """Enhanced main function with more options."""
    parser = argparse.ArgumentParser(description="Advanced AI Documentation Generator")
    parser.add_argument('--source', default='src', help='Source directory to analyze')
    parser.add_argument('--docs', default='docs', help='Documentation directory')
    parser.add_argument('--cache', default='.doc_cache', help='Cache directory')
    parser.add_argument('--style', choices=['numpy', 'google', 'sphinx', 'markdown'], 
                       default='numpy', help='Documentation style')
    parser.add_argument('--provider', choices=['openai', 'anthropic', 'google'], 
                       default='openai', help='AI provider')
    parser.add_argument('--batch-size', type=int, default=5, help='Batch size for AI requests')
    parser.add_argument('--report', action='store_true', help='Generate coverage report')
    parser.add_argument('--report-format', choices=['json', 'markdown', 'html'], 
                       default='markdown', help='Report format')
    parser.add_argument('--dry-run', action='store_true', help='Analyze without generating')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Step 1: Analyze code
    logger.info(f"🔍 Analyzing code in {args.source}...")
    analyzer = EnhancedCodeAnalyzer(args.source)
    elements = analyzer.analyze()
    logger.info(f"✅ Found {len(elements)} code elements")
    
    # Generate coverage report
    if args.report or args.dry_run:
        report_gen = DocumentationReportGenerator(elements)
        report_path = Path(args.docs) / f"coverage_report.{args.report_format}"
        report_gen.save_report(report_path, args.report_format)
        logger.info(f"📊 Coverage report saved to {report_path}")
        
        if args.dry_run:
            # Just show the report
            report = report_gen.generate_report()
            print(f"\n📊 Documentation Coverage: {report['summary']['coverage_percentage']:.1f}%")
            print(f"   Total: {report['summary']['total_elements']} elements")
            print(f"   Documented: {report['summary']['documented']}")
            print(f"   Missing: {report['summary']['undocumented']}")
            return
    
    # Step 2: Generate documentation with AI
    logger.info(f"🤖 Generating documentation with {args.provider}...")
    cache_dir = Path(args.cache)
    doc_gen = BatchAIDocGenerator(args.provider, cache_dir, args.batch_size)
    
    # Convert elements to list for batch processing
    elements_list = list(elements.values())
    style = DocStyle(args.style)
    
    enhanced_docs = doc_gen.generate_batch_documentation(elements_list, style)
    logger.info(f"✅ Generated {len(enhanced_docs)} documentation entries")
    
    # Step 3: Build documentation structure
    logger.info(f"📚 Building documentation in {args.docs}...")
    # Here you would integrate with your DocumentationBuilder
    # This is a placeholder for the actual implementation
    
    logger.info("🎉 Documentation generation complete!")
    logger.info(f"💰 Total API cost: ${doc_gen.total_cost:.2f}")
    logger.info(f"📝 Total tokens used: {doc_gen.total_tokens:,}")

if __name__ == "__main__":
    main()