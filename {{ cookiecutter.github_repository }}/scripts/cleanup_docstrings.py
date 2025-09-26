#!/usr/bin/env python
"""
Automated Docstring Cleanup and Enhancement Tool
Cleans, formats, and optionally enhances docstrings in Python source files.
Supports multiple docstring styles and can use AI for enhancement.
"""

import os
import ast
import re
import argparse
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import difflib
from datetime import datetime
import json
import black

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocstringStyle(Enum):
    """Supported docstring styles."""
    NUMPY = "numpy"
    GOOGLE = "google"
    SPHINX = "sphinx"
    AUTO = "auto"  # Auto-detect

@dataclass
class DocstringComponents:
    """Components of a parsed docstring."""
    summary: str = ""
    description: str = ""
    params: Dict[str, Dict[str, str]] = field(default_factory=dict)
    returns: Dict[str, str] = field(default_factory=dict)
    raises: Dict[str, str] = field(default_factory=dict)
    examples: str = ""
    notes: str = ""
    attributes: Dict[str, Dict[str, str]] = field(default_factory=dict)
    
    def is_empty(self) -> bool:
        """Check if docstring is empty or minimal."""
        return not self.summary or len(self.summary.strip()) < 10

class DocstringParser:
    """Parse docstrings of different styles."""
    
    def __init__(self, style: DocstringStyle = DocstringStyle.AUTO):
        self.style = style
    
    def parse(self, docstring: str) -> Tuple[DocstringComponents, DocstringStyle]:
        """Parse a docstring into components."""
        if not docstring:
            return DocstringComponents(), DocstringStyle.NUMPY
        
        # Auto-detect style if needed
        detected_style = self.style
        if self.style == DocstringStyle.AUTO:
            detected_style = self._detect_style(docstring)
        
        # Parse based on style
        if detected_style == DocstringStyle.NUMPY:
            components = self._parse_numpy(docstring)
        elif detected_style == DocstringStyle.GOOGLE:
            components = self._parse_google(docstring)
        elif detected_style == DocstringStyle.SPHINX:
            components = self._parse_sphinx(docstring)
        else:
            components = self._parse_numpy(docstring)  # Default to numpy
        
        return components, detected_style
    
    def _detect_style(self, docstring: str) -> DocstringStyle:
        """Detect the docstring style."""
        if 'Parameters\n----------' in docstring or 'Returns\n-------' in docstring:
            return DocstringStyle.NUMPY
        elif 'Args:' in docstring or 'Returns:' in docstring:
            return DocstringStyle.GOOGLE
        elif ':param' in docstring or ':returns:' in docstring:
            return DocstringStyle.SPHINX
        else:
            return DocstringStyle.NUMPY  # Default
    
    def _parse_numpy(self, docstring: str) -> DocstringComponents:
        """Parse NumPy-style docstring."""
        components = DocstringComponents()
        lines = docstring.strip().split('\n')
        
        # Extract summary (first line or until first blank line)
        summary_lines = []
        idx = 0
        for idx, line in enumerate(lines):
            if line.strip() == '':
                break
            summary_lines.append(line)
        components.summary = ' '.join(summary_lines).strip()
        
        # Extract sections
        current_section = None
        section_content = []
        
        for i in range(idx + 1, len(lines)):
            line = lines[i]
            
            # Check for section headers
            if i < len(lines) - 1:
                next_line = lines[i + 1] if i + 1 < len(lines) else ''
                if set(next_line.strip()) <= {'-'} and len(next_line.strip()) >= 3:
                    # Found a section header
                    if current_section and section_content:
                        self._process_numpy_section(components, current_section, section_content)
                    current_section = line.strip()
                    section_content = []
                    i += 1  # Skip the dashes
                    continue
            
            if current_section:
                section_content.append(line)
        
        # Process last section
        if current_section and section_content:
            self._process_numpy_section(components, current_section, section_content)
        
        return components
    
    def _process_numpy_section(self, components: DocstringComponents, section: str, content: List[str]):
        """Process a NumPy docstring section."""
        section_lower = section.lower()
        
        if section_lower == 'parameters':
            components.params = self._parse_numpy_params(content)
        elif section_lower == 'returns':
            components.returns = self._parse_numpy_returns(content)
        elif section_lower == 'raises':
            components.raises = self._parse_numpy_raises(content)
        elif section_lower in ['examples', 'example']:
            components.examples = '\n'.join(content).strip()
        elif section_lower == 'notes':
            components.notes = '\n'.join(content).strip()
        elif section_lower == 'attributes':
            components.attributes = self._parse_numpy_params(content)  # Same format as params
    
    def _parse_numpy_params(self, lines: List[str]) -> Dict[str, Dict[str, str]]:
        """Parse NumPy-style parameters."""
        params = {}
        current_param = None
        current_info = {'type': '', 'description': ''}
        
        for line in lines:
            # Check if this is a parameter definition
            if line and not line[0].isspace():
                # Save previous parameter if exists
                if current_param:
                    params[current_param] = current_info
                
                # Parse new parameter
                parts = line.split(':', 1)
                if len(parts) == 2:
                    current_param = parts[0].strip()
                    current_info = {'type': parts[1].strip(), 'description': ''}
                else:
                    current_param = line.strip()
                    current_info = {'type': '', 'description': ''}
            elif line.strip() and current_param:
                # This is part of the description
                current_info['description'] += ' ' + line.strip()
        
        # Save last parameter
        if current_param:
            params[current_param] = current_info
        
        return params
    
    def _parse_numpy_returns(self, lines: List[str]) -> Dict[str, str]:
        """Parse NumPy-style returns."""
        return_info = {'type': '', 'description': ''}
        
        for i, line in enumerate(lines):
            if i == 0 and ':' in line:
                parts = line.split(':', 1)
                return_info['type'] = parts[0].strip()
                if len(parts) > 1:
                    return_info['description'] = parts[1].strip()
            elif line.strip():
                return_info['description'] += ' ' + line.strip()
        
        return return_info
    
    def _parse_numpy_raises(self, lines: List[str]) -> Dict[str, str]:
        """Parse NumPy-style raises section."""
        raises = {}
        current_exception = None
        
        for line in lines:
            if line and not line[0].isspace():
                current_exception = line.strip().rstrip(':')
                raises[current_exception] = ''
            elif line.strip() and current_exception:
                raises[current_exception] += ' ' + line.strip()
        
        return raises
    
    def _parse_google(self, docstring: str) -> DocstringComponents:
        """Parse Google-style docstring."""
        components = DocstringComponents()
        lines = docstring.strip().split('\n')
        
        # Extract summary
        summary_lines = []
        idx = 0
        for idx, line in enumerate(lines):
            if line.strip() in ['Args:', 'Returns:', 'Raises:', 'Note:', 'Example:', 'Examples:']:
                break
            if line.strip():
                summary_lines.append(line.strip())
        components.summary = ' '.join(summary_lines)
        
        # Parse sections
        current_section = None
        section_content = []
        
        for i in range(idx, len(lines)):
            line = lines[i]
            
            if line.strip() in ['Args:', 'Returns:', 'Raises:', 'Note:', 'Example:', 'Examples:', 'Attributes:']:
                if current_section:
                    self._process_google_section(components, current_section, section_content)
                current_section = line.strip().rstrip(':')
                section_content = []
            elif current_section:
                section_content.append(line)
        
        # Process last section
        if current_section:
            self._process_google_section(components, current_section, section_content)
        
        return components
    
    def _process_google_section(self, components: DocstringComponents, section: str, content: List[str]):
        """Process a Google docstring section."""
        section_lower = section.lower()
        
        if section_lower == 'args':
            components.params = self._parse_google_args(content)
        elif section_lower == 'returns':
            components.returns = self._parse_google_returns(content)
        elif section_lower == 'raises':
            components.raises = self._parse_google_raises(content)
        elif section_lower in ['example', 'examples']:
            components.examples = '\n'.join(content).strip()
        elif section_lower == 'note':
            components.notes = '\n'.join(content).strip()
        elif section_lower == 'attributes':
            components.attributes = self._parse_google_args(content)
    
    def _parse_google_args(self, lines: List[str]) -> Dict[str, Dict[str, str]]:
        """Parse Google-style arguments."""
        args = {}
        current_arg = None
        current_info = {}
        
        for line in lines:
            # Check if this is an argument definition (starts with non-whitespace)
            if line and len(line) > 0 and line[0] != ' ':
                continue  # Skip section headers
            
            stripped = line.strip()
            if stripped and ':' in stripped and not stripped[0].isspace():
                # This is a new argument
                if current_arg:
                    args[current_arg] = current_info
                
                parts = stripped.split(':', 1)
                current_arg = parts[0].strip()
                
                # Parse type and description
                if len(parts) > 1:
                    desc_part = parts[1].strip()
                    # Try to extract type in parentheses
                    type_match = re.match(r'\(([^)]+)\)\s*(.*)', desc_part)
                    if type_match:
                        current_info = {
                            'type': type_match.group(1),
                            'description': type_match.group(2)
                        }
                    else:
                        current_info = {'type': '', 'description': desc_part}
                else:
                    current_info = {'type': '', 'description': ''}
            elif stripped and current_arg:
                # Continuation of description
                current_info['description'] += ' ' + stripped
        
        # Save last argument
        if current_arg:
            args[current_arg] = current_info
        
        return args
    
    def _parse_google_returns(self, lines: List[str]) -> Dict[str, str]:
        """Parse Google-style returns."""
        content = '\n'.join(lines).strip()
        
        # Try to extract type and description
        type_match = re.match(r'^\s*([^:]+):\s*(.*)', content, re.DOTALL)
        if type_match:
            return {
                'type': type_match.group(1).strip(),
                'description': type_match.group(2).strip()
            }
        else:
            return {'type': '', 'description': content}
    
    def _parse_google_raises(self, lines: List[str]) -> Dict[str, str]:
        """Parse Google-style raises section."""
        raises = {}
        current_exception = None
        
        for line in lines:
            stripped = line.strip()
            if stripped and ':' in stripped and not line[0].isspace():
                parts = stripped.split(':', 1)
                current_exception = parts[0].strip()
                raises[current_exception] = parts[1].strip() if len(parts) > 1 else ''
            elif stripped and current_exception:
                raises[current_exception] += ' ' + stripped
        
        return raises
    
    def _parse_sphinx(self, docstring: str) -> DocstringComponents:
        """Parse Sphinx-style docstring."""
        components = DocstringComponents()
        lines = docstring.strip().split('\n')
        
        # Extract summary (lines before :param or other directives)
        summary_lines = []
        idx = 0
        for idx, line in enumerate(lines):
            if line.strip().startswith(':'):
                break
            if line.strip():
                summary_lines.append(line.strip())
        
        components.summary = ' '.join(summary_lines)
        
        # Parse directives
        for i in range(idx, len(lines)):
            line = lines[i].strip()
            
            if line.startswith(':param '):
                # Parse parameter
                match = re.match(r':param\s+(\w+):\s*(.*)', line)
                if match:
                    param_name = match.group(1)
                    param_desc = match.group(2)
                    if param_name not in components.params:
                        components.params[param_name] = {'type': '', 'description': param_desc}
                    else:
                        components.params[param_name]['description'] = param_desc
            
            elif line.startswith(':type '):
                # Parse parameter type
                match = re.match(r':type\s+(\w+):\s*(.*)', line)
                if match:
                    param_name = match.group(1)
                    param_type = match.group(2)
                    if param_name not in components.params:
                        components.params[param_name] = {'type': param_type, 'description': ''}
                    else:
                        components.params[param_name]['type'] = param_type
            
            elif line.startswith(':returns:') or line.startswith(':return:'):
                # Parse return description
                match = re.match(r':returns?:\s*(.*)', line)
                if match:
                    components.returns['description'] = match.group(1)
            
            elif line.startswith(':rtype:'):
                # Parse return type
                match = re.match(r':rtype:\s*(.*)', line)
                if match:
                    components.returns['type'] = match.group(1)
            
            elif line.startswith(':raises '):
                # Parse raises
                match = re.match(r':raises\s+(\w+):\s*(.*)', line)
                if match:
                    components.raises[match.group(1)] = match.group(2)
        
        return components

class DocstringFormatter:
    """Format docstrings according to style guidelines."""
    
    def __init__(self, style: DocstringStyle = DocstringStyle.NUMPY):
        self.style = style
        self.line_length = 79  # PEP 8 standard
    
    def format(self, components: DocstringComponents, indent: str = "    ") -> str:
        """Format components into a clean docstring."""
        if self.style == DocstringStyle.NUMPY:
            return self._format_numpy(components, indent)
        elif self.style == DocstringStyle.GOOGLE:
            return self._format_google(components, indent)
        elif self.style == DocstringStyle.SPHINX:
            return self._format_sphinx(components, indent)
        else:
            return self._format_numpy(components, indent)
    
    def _format_numpy(self, components: DocstringComponents, indent: str) -> str:
        """Format as NumPy-style docstring."""
        lines = []
        
        # Add summary
        if components.summary:
            wrapped = textwrap.fill(
                components.summary,
                width=self.line_length - len(indent),
                initial_indent='',
                subsequent_indent=''
            )
            lines.extend(wrapped.split('\n'))
        
        # Add description
        if components.description:
            lines.append('')
            wrapped = textwrap.fill(
                components.description,
                width=self.line_length - len(indent),
                initial_indent='',
                subsequent_indent=''
            )
            lines.extend(wrapped.split('\n'))
        
        # Add parameters
        if components.params:
            lines.extend(['', 'Parameters', '----------'])
            for param_name, param_info in components.params.items():
                param_type = param_info.get('type', '')
                param_desc = param_info.get('description', '')
                
                # Format parameter
                if param_type:
                    lines.append(f"{param_name} : {param_type}")
                else:
                    lines.append(f"{param_name}")
                
                # Add description with proper indentation
                if param_desc:
                    wrapped = textwrap.fill(
                        param_desc.strip(),
                        width=self.line_length - len(indent) - 4,
                        initial_indent='    ',
                        subsequent_indent='    '
                    )
                    lines.extend(wrapped.split('\n'))
        
        # Add returns
        if components.returns and (components.returns.get('type') or components.returns.get('description')):
            lines.extend(['', 'Returns', '-------'])
            
            return_type = components.returns.get('type', '')
            return_desc = components.returns.get('description', '')
            
            if return_type:
                lines.append(return_type)
            
            if return_desc:
                wrapped = textwrap.fill(
                    return_desc.strip(),
                    width=self.line_length - len(indent) - 4,
                    initial_indent='    ',
                    subsequent_indent='    '
                )
                lines.extend(wrapped.split('\n'))
        
        # Add raises
        if components.raises:
            lines.extend(['', 'Raises', '------'])
            for exception, description in components.raises.items():
                lines.append(exception)
                if description:
                    wrapped = textwrap.fill(
                        description.strip(),
                        width=self.line_length - len(indent) - 4,
                        initial_indent='    ',
                        subsequent_indent='    '
                    )
                    lines.extend(wrapped.split('\n'))
        
        # Add examples
        if components.examples:
            lines.extend(['', 'Examples', '--------'])
            lines.extend(components.examples.split('\n'))
        
        # Add notes
        if components.notes:
            lines.extend(['', 'Notes', '-----'])
            wrapped = textwrap.fill(
                components.notes,
                width=self.line_length - len(indent),
                initial_indent='',
                subsequent_indent=''
            )
            lines.extend(wrapped.split('\n'))
        
        # Join lines with proper indentation
        docstring = '\n'.join([f"{indent}{line}" if line else '' for line in lines])
        
        # Format as triple-quoted string
        return f'{indent}"""\n{docstring}\n{indent}"""'
    
    def _format_google(self, components: DocstringComponents, indent: str) -> str:
        """Format as Google-style docstring."""
        lines = []
        
        # Add summary
        if components.summary:
            wrapped = textwrap.fill(
                components.summary,
                width=self.line_length - len(indent)
            )
            lines.extend(wrapped.split('\n'))
        
        # Add description
        if components.description:
            lines.append('')
            wrapped = textwrap.fill(
                components.description,
                width=self.line_length - len(indent)
            )
            lines.extend(wrapped.split('\n'))
        
        # Add arguments
        if components.params:
            lines.extend(['', 'Args:'])
            for param_name, param_info in components.params.items():
                param_type = param_info.get('type', '')
                param_desc = param_info.get('description', '')
                
                # Format as: name (type): description
                if param_type:
                    param_line = f"    {param_name} ({param_type}): {param_desc}"
                else:
                    param_line = f"    {param_name}: {param_desc}"
                
                wrapped = textwrap.fill(
                    param_line,
                    width=self.line_length - len(indent),
                    initial_indent='',
                    subsequent_indent='        '
                )
                lines.extend(wrapped.split('\n'))
        
        # Add returns
        if components.returns and (components.returns.get('type') or components.returns.get('description')):
            lines.extend(['', 'Returns:'])
            return_type = components.returns.get('type', '')
            return_desc = components.returns.get('description', '')
            
            if return_type and return_desc:
                return_line = f"    {return_type}: {return_desc}"
            elif return_type:
                return_line = f"    {return_type}"
            else:
                return_line = f"    {return_desc}"
            
            wrapped = textwrap.fill(
                return_line,
                width=self.line_length - len(indent),
                initial_indent='',
                subsequent_indent='        '
            )
            lines.extend(wrapped.split('\n'))
        
        # Add raises
        if components.raises:
            lines.extend(['', 'Raises:'])
            for exception, description in components.raises.items():
                raise_line = f"    {exception}: {description}"
                wrapped = textwrap.fill(
                    raise_line,
                    width=self.line_length - len(indent),
                    initial_indent='',
                    subsequent_indent='        '
                )
                lines.extend(wrapped.split('\n'))
        
        # Add examples
        if components.examples:
            lines.extend(['', 'Examples:'])
            for example_line in components.examples.split('\n'):
                lines.append(f"    {example_line}")
        
        # Add notes
        if components.notes:
            lines.extend(['', 'Note:'])
            wrapped = textwrap.fill(
                components.notes,
                width=self.line_length - len(indent) - 4,
                initial_indent='    ',
                subsequent_indent='    '
            )
            lines.extend(wrapped.split('\n'))
        
        # Join lines with proper indentation
        docstring = '\n'.join([f"{indent}{line}" if line else '' for line in lines])
        
        # Format as triple-quoted string
        return f'{indent}"""\n{docstring}\n{indent}"""'
    
    def _format_sphinx(self, components: DocstringComponents, indent: str) -> str:
        """Format as Sphinx-style docstring."""
        lines = []
        
        # Add summary
        if components.summary:
            wrapped = textwrap.fill(
                components.summary,
                width=self.line_length - len(indent)
            )
            lines.extend(wrapped.split('\n'))
        
        # Add description
        if components.description:
            lines.append('')
            wrapped = textwrap.fill(
                components.description,
                width=self.line_length - len(indent)
            )
            lines.extend(wrapped.split('\n'))
        
        # Add parameters
        if components.params:
            lines.append('')
            for param_name, param_info in components.params.items():
                param_type = param_info.get('type', '')
                param_desc = param_info.get('description', '')
                
                # Add param description
                if param_desc:
                    lines.append(f":param {param_name}: {param_desc}")
                
                # Add param type
                if param_type:
                    lines.append(f":type {param_name}: {param_type}")
        
        # Add returns
        if components.returns:
            lines.append('')
            if components.returns.get('description'):
                lines.append(f":returns: {components.returns['description']}")
            if components.returns.get('type'):
                lines.append(f":rtype: {components.returns['type']}")
        
        # Add raises
        if components.raises:
            lines.append('')
            for exception, description in components.raises.items():
                lines.append(f":raises {exception}: {description}")
        
        # Join lines with proper indentation
        docstring = '\n'.join([f"{indent}{line}" if line else '' for line in lines])
        
        # Format as triple-quoted string
        return f'{indent}"""\n{docstring}\n{indent}"""'

class DocstringCleaner:
    """Clean and update docstrings in Python files."""
    
    def __init__(self, 
                 style: DocstringStyle = DocstringStyle.NUMPY,
                 fix_missing: bool = True,
                 enhance_with_ai: bool = False,
                 dry_run: bool = False):
        self.style = style
        self.fix_missing = fix_missing
        self.enhance_with_ai = enhance_with_ai
        self.dry_run = dry_run
        self.parser = DocstringParser(style)
        self.formatter = DocstringFormatter(style)
        self.changes_made = 0
        self.files_modified = []
    
    def clean_file(self, file_path: Path) -> Tuple[bool, str]:
        """Clean docstrings in a single Python file."""
        logger.info(f"Processing {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_source = f.read()
            
            # Parse the file
            tree = ast.parse(original_source, filename=str(file_path))
            
            # Process the AST and collect modifications
            modified_source = self._process_ast(tree, original_source, file_path)
            
            # Format with black if available
            try:
                modified_source = black.format_str(modified_source, mode=black.FileMode())
            except:
                pass  # Black formatting is optional
            
            # Check if changes were made
            if modified_source != original_source:
                if not self.dry_run:
                    # Write back to file
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(modified_source)
                
                self.files_modified.append(str(file_path))
                
                # Show diff if in verbose mode
                if logger.level <= logging.DEBUG:
                    diff = difflib.unified_diff(
                        original_source.splitlines(keepends=True),
                        modified_source.splitlines(keepends=True),
                        fromfile=str(file_path),
                        tofile=str(file_path)
                    )
                    logger.debug(''.join(diff))
                
                return True, modified_source
            
            return False, original_source
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return False, ""
    
    def _process_ast(self, tree: ast.AST, source: str, file_path: Path) -> str:
        """Process AST and update docstrings."""
        lines = source.splitlines()
        modifications = []  # List of (line_start, line_end, new_text)
        
        # Process all classes and functions
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                docstring = ast.get_docstring(node)
                
                # Check if docstring needs updating
                needs_update = False
                new_docstring = None
                
                if not docstring and self.fix_missing:
                    # Generate missing docstring
                    new_docstring = self._generate_docstring(node, source)
                    needs_update = True
                elif docstring:
                    # Clean existing docstring
                    cleaned = self._clean_docstring(docstring, node)
                    if cleaned != docstring:
                        new_docstring = cleaned
                        needs_update = True
                
                if needs_update and new_docstring:
                    # Calculate where to insert/replace docstring
                    docstring_node = node.body[0] if node.body and isinstance(node.body[0], ast.Expr) else None
                    
                    if docstring_node and isinstance(docstring_node.value, ast.Str):
                        # Replace existing docstring
                        start_line = docstring_node.lineno - 1
                        end_line = docstring_node.end_lineno - 1 if hasattr(docstring_node, 'end_lineno') else start_line
                        
                        # Get indentation
                        indent = self._get_indentation(lines[start_line])
                        
                        # Format new docstring
                        formatted = self.formatter.format(self._parse_to_components(new_docstring), indent)
                        
                        modifications.append((start_line, end_line, formatted))
                        self.changes_made += 1
                    elif self.fix_missing:
                        # Insert new docstring
                        insert_line = node.lineno
                        
                        # Find the line after the function/class definition
                        for i in range(node.lineno - 1, len(lines)):
                            if lines[i].rstrip().endswith(':'):
                                insert_line = i + 1
                                break
                        
                        # Get indentation for the body
                        body_indent = self._get_body_indent(node, lines)
                        
                        # Format new docstring
                        formatted = self.formatter.format(self._parse_to_components(new_docstring), body_indent)
                        
                        modifications.append((insert_line, insert_line - 1, formatted))
                        self.changes_made += 1
        
        # Apply modifications in reverse order to maintain line numbers
        modifications.sort(reverse=True, key=lambda x: x[0])
        
        for start, end, new_text in modifications:
            # Replace or insert lines
            if start > end:
                # Insert
                lines.insert(start, new_text)
            else:
                # Replace
                lines[start:end+1] = [new_text]
        
        return '\n'.join(lines)
    
    def _clean_docstring(self, docstring: str, node: ast.AST) -> str:
        """Clean and format an existing docstring."""
        # Parse the docstring
        components, detected_style = self.parser.parse(docstring)
        
        # Clean up components
        components = self._clean_components(components)
        
        # Add missing information if available from signature
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            components = self._enhance_from_signature(components, node)
        
        # Reformat
        return self._components_to_string(components)
    
    def _clean_components(self, components: DocstringComponents) -> DocstringComponents:
        """Clean up docstring components."""
        # Clean summary
        if components.summary:
            components.summary = components.summary.strip()
            # Ensure summary ends with period
            if components.summary and not components.summary.endswith('.'):
                components.summary += '.'
            # Capitalize first letter
            if components.summary:
                components.summary = components.summary[0].upper() + components.summary[1:]
        
        # Clean parameter descriptions
        for param_name, param_info in components.params.items():
            if param_info.get('description'):
                desc = param_info['description'].strip()
                # Capitalize and add period
                if desc and not desc.endswith('.'):
                    desc += '.'
                if desc:
                    desc = desc[0].upper() + desc[1:]
                param_info['description'] = desc
        
        # Clean return description
        if components.returns.get('description'):
            desc = components.returns['description'].strip()
            if desc and not desc.endswith('.'):
                desc += '.'
            if desc:
                desc = desc[0].upper() + desc[1:]
            components.returns['description'] = desc
        
        # Clean raises descriptions
        for exception, desc in components.raises.items():
            if desc:
                desc = desc.strip()
                if desc and not desc.endswith('.'):
                    desc += '.'
                if desc:
                    desc = desc[0].upper() + desc[1:]
                components.raises[exception] = desc
        
        return components
    
    def _enhance_from_signature(self, components: DocstringComponents, node: ast.AST) -> DocstringComponents:
        """Enhance docstring components from function signature."""
        # Extract parameters from signature
        if hasattr(node, 'args'):
            for arg in node.args.args:
                param_name = arg.arg
                
                # Skip 'self' and 'cls'
                if param_name in ['self', 'cls']:
                    continue
                
                # Add missing parameter
                if param_name not in components.params:
                    param_info = {'type': '', 'description': ''}
                    
                    # Try to get type from annotation
                    if arg.annotation:
                        param_info['type'] = ast.unparse(arg.annotation)
                    
                    components.params[param_name] = param_info
        
        # Extract return type from signature
        if hasattr(node, 'returns') and node.returns:
            if not components.returns.get('type'):
                components.returns['type'] = ast.unparse(node.returns)
        
        return components
    
    def _generate_docstring(self, node: ast.AST, source: str) -> str:
        """Generate a docstring for a node without one."""
        components = DocstringComponents()
        
        # Generate summary based on function/class name
        if isinstance(node, ast.ClassDef):
            components.summary = f"Class {node.name}."
        else:
            # Convert function name from snake_case to readable
            name_parts = node.name.split('_')
            readable_name = ' '.join(name_parts)
            components.summary = f"{readable_name.capitalize()}."
        
        # Add parameters for functions
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            components = self._enhance_from_signature(components, node)
        
        return self._components_to_string(components)
    
    def _components_to_string(self, components: DocstringComponents) -> str:
        """Convert components back to string (without quotes)."""
        lines = []
        
        # Summary
        if components.summary:
            lines.append(components.summary)
        
        # Description
        if components.description:
            lines.append('')
            lines.append(components.description)
        
        # Parameters
        if components.params:
            lines.extend(['', 'Parameters', '----------'])
            for param_name, param_info in components.params.items():
                if param_info.get('type'):
                    lines.append(f"{param_name} : {param_info['type']}")
                else:
                    lines.append(param_name)
                if param_info.get('description'):
                    lines.append(f"    {param_info['description']}")
        
        # Returns
        if components.returns and (components.returns.get('type') or components.returns.get('description')):
            lines.extend(['', 'Returns', '-------'])
            if components.returns.get('type'):
                lines.append(components.returns['type'])
            if components.returns.get('description'):
                lines.append(f"    {components.returns['description']}")
        
        # Raises
        if components.raises:
            lines.extend(['', 'Raises', '------'])
            for exception, description in components.raises.items():
                lines.append(exception)
                if description:
                    lines.append(f"    {description}")
        
        return '\n'.join(lines)
    
    def _parse_to_components(self, docstring: str) -> DocstringComponents:
        """Parse a docstring string to components."""
        components, _ = self.parser.parse(docstring)
        return components
    
    def _get_indentation(self, line: str) -> str:
        """Get the indentation of a line."""
        return line[:len(line) - len(line.lstrip())]
    
    def _get_body_indent(self, node: ast.AST, lines: List[str]) -> str:
        """Get the indentation for the body of a function/class."""
        # Default to 4 spaces more than the definition
        if hasattr(node, 'col_offset'):
            return ' ' * (node.col_offset + 4)
        return '    '
    
    def clean_directory(self, directory: Path, recursive: bool = True) -> Dict[str, Any]:
        """Clean all Python files in a directory."""
        pattern = '**/*.py' if recursive else '*.py'
        python_files = list(directory.glob(pattern))
        
        logger.info(f"Found {len(python_files)} Python files to process")
        
        results = {
            'total_files': len(python_files),
            'files_modified': [],
            'files_skipped': [],
            'errors': [],
            'changes_made': 0
        }
        
        for file_path in python_files:
            try:
                modified, _ = self.clean_file(file_path)
                if modified:
                    results['files_modified'].append(str(file_path))
                else:
                    results['files_skipped'].append(str(file_path))
            except Exception as e:
                results['errors'].append({
                    'file': str(file_path),
                    'error': str(e)
                })
        
        results['changes_made'] = self.changes_made
        
        return results

def main():
    """Main function for docstring cleanup tool."""
    parser = argparse.ArgumentParser(
        description="Clean and format docstrings in Python files"
    )
    
    # Input/output options
    parser.add_argument('path', help='File or directory to process')
    parser.add_argument('-r', '--recursive', action='store_true',
                       help='Process directories recursively')
    
    # Style options
    parser.add_argument('--style', choices=['numpy', 'google', 'sphinx', 'auto'],
                       default='numpy', help='Docstring style')
    
    # Processing options
    parser.add_argument('--fix-missing', action='store_true', default=True,
                       help='Add missing docstrings')
    parser.add_argument('--no-fix-missing', dest='fix_missing', action='store_false',
                       help='Do not add missing docstrings')
    
    # Other options
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be changed without modifying files')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--report', action='store_true',
                       help='Generate detailed report')
    parser.add_argument('--output', help='Output file for report (JSON)')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create cleaner
    cleaner = DocstringCleaner(
        style=DocstringStyle(args.style),
        fix_missing=args.fix_missing,
        dry_run=args.dry_run
    )
    
    # Process path
    path = Path(args.path)
    
    if path.is_file():
        # Process single file
        modified, _ = cleaner.clean_file(path)
        if modified:
            print(f"✅ Modified: {path}")
        else:
            print(f"⏭️  No changes needed: {path}")
    
    elif path.is_dir():
        # Process directory
        results = cleaner.clean_directory(path, recursive=args.recursive)
        
        # Print summary
        print("\n" + "="*50)
        print("Docstring Cleanup Summary")
        print("="*50)
        print(f"Total files processed: {results['total_files']}")
        print(f"Files modified: {len(results['files_modified'])}")
        print(f"Files skipped: {len(results['files_skipped'])}")
        print(f"Errors: {len(results['errors'])}")
        print(f"Total changes made: {results['changes_made']}")
        
        if args.dry_run:
            print("\n⚠️  DRY RUN - No files were actually modified")
        
        if results['files_modified'] and args.verbose:
            print("\nModified files:")
            for file in results['files_modified']:
                print(f"  - {file}")
        
        if results['errors']:
            print("\n❌ Errors:")
            for error in results['errors']:
                print(f"  - {error['file']}: {error['error']}")
        
        # Save report if requested
        if args.report and args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n📊 Report saved to: {args.output}")
    
    else:
        print(f"❌ Path not found: {path}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())