#!/usr/bin/env python3
"""Generuje indeks funkcji i metod dla kodu projektu."""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "build",
    "install",
    "log",
    "out",
    "out_example",
    "niemoje",
}
VERB_HINTS = {
    "add": "Dodaje",
    "append": "Dopisuje",
    "build": "Buduje",
    "check": "Sprawdza",
    "compute": "Oblicza",
    "create": "Tworzy",
    "declare": "Deklaruje",
    "end": "Kończy",
    "ensure": "Zapewnia",
    "extract": "Wyciąga",
    "finish": "Domyka",
    "generate": "Generuje",
    "get": "Zwraca",
    "infer": "Uruchamia inferencję dla",
    "interp": "Interpoluje",
    "interpolate": "Interpoluje",
    "load": "Wczytuje",
    "main": "Stanowi punkt wejścia dla",
    "map": "Mapuje",
    "on": "Obsługuje",
    "parse": "Parsuje",
    "passes": "Weryfikuje",
    "plot": "Rysuje",
    "project": "Rzutuje",
    "request": "Wysyła żądanie dla",
    "resample": "Przeskalowuje",
    "run": "Uruchamia",
    "sanitize": "Czyści",
    "save": "Zapisuje",
    "seed": "Ustawia ziarno dla",
    "set": "Ustawia",
    "stamp": "Stempluje",
    "start": "Rozpoczyna",
    "tick": "Wykonuje cykliczny krok dla",
    "train": "Trenuje",
    "update": "Aktualizuje",
    "wait": "Czeka na",
    "wrap": "Normalizuje",
}
TOKEN_HINTS = {
    "ai": "AI",
    "bf": "bruteforce",
    "dt": "czas",
    "dyaw": "zmianę yaw",
    "err": "błąd",
    "eval": "ewaluację",
    "gt": "ground truth",
    "idx": "indeks",
    "iou": "IoU",
    "map": "mapę",
    "metadata": "metadane",
    "npz": "plik NPZ",
    "odom": "odometrię",
    "occ": "occupancy grid",
    "pgm": "plik PGM",
    "pose": "pozę",
    "qos": "QoS",
    "ref": "mapę referencyjną",
    "rgb": "RGB",
    "rmse": "RMSE",
    "robak": "Robaka",
    "rywak": "Rywaka",
    "scan": "skan",
    "slam": "SLAM",
    "sync": "synchronizację",
    "tf": "TF",
    "traj": "trajektorię",
    "trajectory": "trajektorię",
    "xy": "pozycję XY",
    "xytheta": "pozę XYTheta",
    "yaml": "plik YAML",
    "yaw": "yaw",
}


@dataclass
class FunctionEntry:
    name: str
    file: str
    line: int
    kind: str
    inputs: list[str]
    returns: str
    description: str


def rel_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def should_skip(path: Path) -> bool:
    return any(part in EXCLUDED_DIRS for part in path.parts)


def format_annotation(node: ast.AST | None) -> str:
    if node is None:
        return "brak"
    try:
        return ast.unparse(node)
    except Exception:
        return "brak"


def format_arg(arg: ast.arg, default_index: int | None = None, defaults: list[ast.AST] | None = None) -> str:
    annotation = format_annotation(arg.annotation)
    text = arg.arg if annotation == "brak" else f"{arg.arg}: {annotation}"
    if defaults is not None and default_index is not None and default_index >= 0:
        try:
            default_text = ast.unparse(defaults[default_index])
        except Exception:
            default_text = "..."
        text = f"{text} = {default_text}"
    return text


def first_sentence(text: str) -> str:
    line = " ".join(part.strip() for part in text.strip().splitlines() if part.strip())
    if not line:
        return ""
    for sep in (". ", "!\n", "?\n", "\n"):
        if sep in line:
            return line.split(sep, 1)[0].strip() + "."
    return line if line.endswith(".") else f"{line}."


def get_call_names(node: ast.AST) -> list[str]:
    names: list[str] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        name = None
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        if name and name not in names:
            names.append(name)
        if len(names) >= 3:
            break
    return names


def describe_from_name(name: str, node: ast.AST) -> str:
    clean = name.split(".")[-1].lstrip("_")
    tokens = [tok for tok in clean.split("_") if tok]
    if not tokens:
        return "Pomocnicza funkcja bez jawnego opisu."

    verb = VERB_HINTS.get(tokens[0], "Obsługuje")
    subject_tokens = tokens[1:] if len(tokens) > 1 else tokens
    subject = " ".join(TOKEN_HINTS.get(tok, tok) for tok in subject_tokens).strip()
    if not subject:
        subject = "operację pomocniczą"

    calls = get_call_names(node)
    how = ""
    if calls:
        how = f" Korzysta m.in. z: {', '.join(calls)}."
    return f"{verb} {subject}.{how}"


class FunctionCollector(ast.NodeVisitor):
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.entries: list[FunctionEntry] = []
        self.class_stack: list[str] = []
        self.function_depth = 0

    def visit_ClassDef(self, node: ast.ClassDef):
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._record_function(node, kind="method" if self.class_stack else "function")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._record_function(node, kind="method" if self.class_stack else "function")

    def _record_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, kind: str):
        is_top_level = self.function_depth == 0
        self.function_depth += 1
        if is_top_level:
            qual_name = ".".join([*self.class_stack, node.name]) if self.class_stack else node.name
            args = node.args
            inputs: list[str] = []

            pos_args = list(args.posonlyargs) + list(args.args)
            pos_defaults = [None] * (len(pos_args) - len(args.defaults)) + list(args.defaults)
            for idx, arg in enumerate(pos_args):
                default_index = idx - (len(pos_args) - len(args.defaults))
                inputs.append(format_arg(arg, default_index, list(args.defaults)))

            if args.vararg is not None:
                vararg_ann = format_annotation(args.vararg.annotation)
                inputs.append(f"*{args.vararg.arg}" if vararg_ann == "brak" else f"*{args.vararg.arg}: {vararg_ann}")

            for kw_arg, kw_default in zip(args.kwonlyargs, args.kw_defaults):
                text = format_arg(kw_arg)
                if kw_default is not None:
                    try:
                        text = f"{text} = {ast.unparse(kw_default)}"
                    except Exception:
                        text = f"{text} = ..."
                inputs.append(text)

            if args.kwarg is not None:
                kwarg_ann = format_annotation(args.kwarg.annotation)
                inputs.append(f"**{args.kwarg.arg}" if kwarg_ann == "brak" else f"**{args.kwarg.arg}: {kwarg_ann}")

            doc = first_sentence(ast.get_docstring(node) or "")
            description = doc if doc else describe_from_name(qual_name, node)
            self.entries.append(
                FunctionEntry(
                    name=qual_name,
                    file=rel_path(self.file_path),
                    line=int(node.lineno),
                    kind=kind,
                    inputs=inputs,
                    returns=format_annotation(node.returns),
                    description=description,
                )
            )

        self.generic_visit(node)
        self.function_depth -= 1


def collect_functions() -> list[FunctionEntry]:
    entries: list[FunctionEntry] = []
    for path in sorted(REPO_ROOT.rglob("*.py")):
        if should_skip(path):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        collector = FunctionCollector(path)
        collector.visit(tree)
        entries.extend(collector.entries)
    return sorted(entries, key=lambda item: (item.file, item.line, item.name))


def write_markdown(entries: list[FunctionEntry], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Indeks funkcji",
        "",
        "Zestawienie wygenerowane automatycznie na podstawie plików `.py` w repozytorium.",
        "",
    ]

    current_file = None
    for entry in entries:
        if entry.file != current_file:
            current_file = entry.file
            lines.extend([f"## {current_file}", ""])
        inputs_text = ", ".join(entry.inputs) if entry.inputs else "brak"
        lines.append(f"### {entry.name}")
        lines.append(f"- Plik: `{entry.file}:{entry.line}`")
        lines.append(f"- Typ: `{entry.kind}`")
        lines.append(f"- Wejście: `{inputs_text}`")
        lines.append(f"- Wyjście: `{entry.returns}`")
        lines.append(f"- Opis: {entry.description}")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_json(entries: list[FunctionEntry], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_repo": str(REPO_ROOT),
        "count": len(entries),
        "functions": [asdict(entry) for entry in entries],
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generuje indeks funkcji projektu.")
    parser.add_argument(
        "--output-md",
        default=str(REPO_ROOT / "docs" / "function_index.md"),
        help="Ścieżka do pliku Markdown.",
    )
    parser.add_argument(
        "--output-json",
        default=str(REPO_ROOT / "docs" / "function_index.json"),
        help="Ścieżka do pliku JSON.",
    )
    args = parser.parse_args()

    entries = collect_functions()
    write_markdown(entries, Path(args.output_md))
    write_json(entries, Path(args.output_json))
    print(f"Wygenerowano indeks funkcji: {len(entries)} pozycji")
    print(f"Markdown: {args.output_md}")
    print(f"JSON: {args.output_json}")


if __name__ == "__main__":
    main()
