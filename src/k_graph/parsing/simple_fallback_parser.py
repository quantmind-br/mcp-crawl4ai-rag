"""
Simple heuristic fallback parser used when Tree-sitter is unavailable.

This parser provides minimal extraction of classes, functions, and imports
for a subset of languages (python, javascript, java, go) using regex-based
heuristics. It intentionally returns the same structure as LanguageParser.parse
expects, enabling tests to run without full Tree-sitter.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict

from ..core.interfaces import LanguageParser
from ..core.models import ParseResult


class SimpleFallbackParser(LanguageParser):
    def parse(self, file_content: str, file_path: str) -> ParseResult:
        classes: List[Dict] = []
        functions: List[Dict] = []
        imports: List[str] = []
        errors: List[str] = []

        lang = self.language_name
        text = file_content

        try:
            if lang == "python":
                # Imports
                for m in re.finditer(r"^\s*import\s+([\w\.]+)", text, re.MULTILINE):
                    imports.append(m.group(1))
                for m in re.finditer(
                    r"^\s*from\s+([\w\.]+)\s+import\s+([\w\*,\s]+)",
                    text,
                    re.MULTILINE,
                ):
                    base = m.group(1)
                    names = [n.strip() for n in m.group(2).split(",")]
                    for n in names:
                        if n:
                            imports.append(f"{base}.{n}")

                # Classes (simple python class pattern)
                for m in re.finditer(
                    r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\b", text, re.MULTILINE
                ):
                    name = m.group(1)
                    classes.append(
                        {
                            "name": name,
                            "full_name": name,
                            "methods": [],
                            "attributes": [],
                            "line_start": m.start(),
                            "line_end": m.end(),
                        }
                    )

                # Functions
                # Regular and async def
                for m in re.finditer(
                    r"^\s*(?:async\s+)?def\s+(\w+)\s*\(", text, re.MULTILINE
                ):
                    functions.append(
                        {
                            "name": m.group(1),
                            "full_name": m.group(1),
                            "params": [],
                            "return_type": "Any",
                            "line_start": m.start(),
                            "line_end": m.end(),
                        }
                    )

            elif lang in ("javascript", "typescript"):
                # Imports
                for m in re.finditer(
                    r"^\s*import\s+.*?from\s+['\"]([^'\"]+)['\"]", text, re.MULTILINE
                ):
                    imports.append(m.group(1))

                # Classes
                # Support export default class and class with extends
                class_patterns = [
                    r"^\s*export\s+default\s+class\s+([A-Za-z_][A-Za-z0-9_]*)\b",
                    r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\b",
                ]
                for pat in class_patterns:
                    for m in re.finditer(pat, text, re.MULTILINE):
                        name = m.group(1)
                        if all(c["name"] != name for c in classes):
                            classes.append(
                                {
                                    "name": name,
                                    "full_name": name,
                                    "methods": [],
                                    "attributes": [],
                                    "line_start": m.start(),
                                    "line_end": m.end(),
                                }
                            )
                # Broad fallback
                if not classes:
                    for m in re.finditer(r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)\b", text):
                        name = m.group(1)
                        if all(c["name"] != name for c in classes):
                            classes.append(
                                {
                                    "name": name,
                                    "full_name": name,
                                    "methods": [],
                                    "attributes": [],
                                    "line_start": m.start(),
                                    "line_end": m.end(),
                                }
                            )
                    classes.append(
                        {
                            "name": m.group(1),
                            "full_name": m.group(1),
                            "methods": [],
                            "attributes": [],
                            "line_start": m.start(),
                            "line_end": m.end(),
                        }
                    )

                # Functions (basic) - function declarations and generators
                for m in re.finditer(
                    r"^\s*function\*?\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
                    text,
                    re.MULTILINE,
                ):
                    functions.append(
                        {
                            "name": m.group(1),
                            "full_name": m.group(1),
                            "params": [],
                            "return_type": "Any",
                            "line_start": m.start(),
                            "line_end": m.end(),
                        }
                    )
                # Arrow functions assigned to const/let/var
                for m in re.finditer(
                    r"^\s*(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\([^\)]*\)\s*=>",
                    text,
                    re.MULTILINE,
                ):
                    name = m.group(1)
                    if all(f["name"] != name for f in functions):
                        functions.append(
                            {
                                "name": name,
                                "full_name": name,
                                "params": [],
                                "return_type": "Any",
                                "line_start": m.start(),
                                "line_end": m.end(),
                            }
                        )

            elif lang == "java":
                # Imports
                for m in re.finditer(r"^\s*import\s+([\w\.\*]+);", text, re.MULTILINE):
                    imports.append(m.group(1))

                # Classes
                for m in re.finditer(r"\bclass\s+(\w+)", text):
                    classes.append(
                        {
                            "name": m.group(1),
                            "full_name": m.group(1),
                            "methods": [],
                            "attributes": [],
                            "line_start": m.start(),
                            "line_end": m.end(),
                        }
                    )

                # Methods
                for m in re.finditer(r"\b(\w+)\s+\w+\s*\(.*?\)\s*\{", text):
                    fn = m.group(1)
                    if fn not in {
                        "if",
                        "for",
                        "while",
                        "switch",
                        "return",
                        "public",
                        "private",
                        "protected",
                    }:
                        functions.append(
                            {
                                "name": fn,
                                "full_name": fn,
                                "params": [],
                                "return_type": "Any",
                                "line_start": m.start(),
                                "line_end": m.end(),
                            }
                        )

            elif lang == "go":
                # Imports
                for m in re.finditer(
                    r"^\s*import\s+\(?\s*\"([^\"]+)\"", text, re.MULTILINE
                ):
                    imports.append(m.group(1))

                # Functions
                for m in re.finditer(r"^\s*func\s+(\w+)\s*\(", text, re.MULTILINE):
                    functions.append(
                        {
                            "name": m.group(1),
                            "full_name": m.group(1),
                            "params": [],
                            "return_type": "Any",
                            "line_start": m.start(),
                            "line_end": m.end(),
                        }
                    )

        except Exception as e:
            errors.append(str(e))

        module_name = Path(file_path).stem
        return ParseResult(
            module_name=module_name,
            file_path=file_path,
            classes=classes,
            functions=functions,
            imports=imports,
            line_count=len(text.splitlines()),
            language=lang,
            errors=errors,
        )

    def is_supported_file(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in self.supported_extensions
