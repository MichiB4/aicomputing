"""Function-signature parsing for AI Computing."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, get_args, get_origin


# Signature parsing.

_TYPE_ALIASES: Dict[str, Any] = {
    "int": int, "float": float, "str": str, "bool": bool,
    "None": type(None), "none": type(None),
    "list": list, "dict": dict, "tuple": tuple, "set": set,
    "List": list, "Dict": dict, "Tuple": tuple, "Set": set,
    "Any": Any,
}

@dataclass
class ParsedParam:
    """Parameter name, annotation, resolved type, and default source text."""

    name: str
    annotation: Optional[str] = None
    resolved_type: Any = None

    # None means absent; an explicit None default is stored as "None".
    default: Optional[str] = None


@dataclass
class ParsedSignature:
    """Function name, ordered parameters, and return-type information."""

    name: str
    params: List[ParsedParam]
    return_annotation: Optional[str] = None
    return_type: Any = None


class UnsupportedAnnotationError(ValueError):
    """A declaration uses an unsupported annotation or parameter feature."""


def validate_type_annotation(annotation: Any) -> None:
    """Reject unsupported annotations before values or model output are checked."""
    if annotation is Any or annotation is None:
        return

    origin = get_origin(annotation)
    if origin is None:
        if isinstance(annotation, type):
            # Some typing classes (e.g. TypedDict) reject isinstance checks.
            try:
                isinstance(None, annotation)
            except TypeError as exc:
                raise UnsupportedAnnotationError(
                    f"Unsupported type annotation: {annotation!r}"
                ) from exc
            return
        raise UnsupportedAnnotationError(
            f"Unsupported type annotation: {annotation!r}"
        )

    if origin not in (list, dict, tuple, set):
        raise UnsupportedAnnotationError(
            f"Unsupported type annotation: {annotation!r}"
        )

    args = get_args(annotation)
    if not args:
        # Bare typing aliases are supported; explicit empty generic arguments
        # (including tuple[()]) are not currently checked by the validator.
        if annotation not in (List, Dict, Tuple, Set):
            raise UnsupportedAnnotationError(
                f"Unsupported empty generic annotation: {annotation!r}"
            )
        return

    if origin in (list, set) and len(args) != 1:
        raise UnsupportedAnnotationError(
            f"Expected one type argument in {annotation!r}"
        )
    if origin is dict and len(args) != 2:
        raise UnsupportedAnnotationError(
            f"Expected two type arguments in {annotation!r}"
        )
    if origin is tuple and len(args) == 2 and args[1] is Ellipsis:
        validate_type_annotation(args[0])
        return

    for arg in args:
        validate_type_annotation(arg)


def _resolve_annotation(node: ast.expr) -> Any:
    """Resolve only known names and container subscriptions, without evaluation."""
    if isinstance(node, ast.Name) and node.id in _TYPE_ALIASES:
        return _TYPE_ALIASES[node.id]
    if isinstance(node, ast.Constant) and node.value is None:
        return type(None)
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        origin = _TYPE_ALIASES.get(node.value.id)
        if origin in (list, dict, tuple, set):
            arguments = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            if not arguments:
                raise UnsupportedAnnotationError("Empty generic annotations are unsupported")
            resolved = []
            for index, argument in enumerate(arguments):
                # Only tuple[T, ...] permits an ellipsis as a type argument.
                if (
                    origin is tuple
                    and len(arguments) == 2
                    and index == 1
                    and isinstance(argument, ast.Constant)
                    and argument.value is Ellipsis
                ):
                    resolved.append(Ellipsis)
                else:
                    resolved.append(_resolve_annotation(argument))
            annotation = origin[resolved[0] if len(resolved) == 1 else tuple(resolved)]
            validate_type_annotation(annotation)
            return annotation
    raise UnsupportedAnnotationError(f"Unsupported type annotation: {ast.unparse(node)!r}")


def resolve_type_str(s: str) -> Any:
    """Parse a supported annotation expression without executing it."""
    if not isinstance(s, str):
        raise TypeError("Type annotation must be a string")
    try:
        expression = ast.parse(s.strip(), mode="eval")
    except SyntaxError as exc:
        raise UnsupportedAnnotationError(f"Invalid type annotation: {s!r}") from exc
    return _resolve_annotation(expression.body)


def parse_signature(sig: str) -> ParsedSignature:
    """Read one declaration, retaining annotations/defaults as source expressions.

    Compilation checks declaration rules such as duplicate names. The resulting
    code object is discarded; annotations and default expressions never run.
    """
    if not isinstance(sig, str):
        raise TypeError("Function signature must be a string")
    source = "def " + sig.strip() + ":\n    pass\n"
    try:
        tree = ast.parse(source, filename="<signature>")
        if (
            len(tree.body) != 1
            or not isinstance(tree.body[0], ast.FunctionDef)
            or len(tree.body[0].body) != 1
            or not isinstance(tree.body[0].body[0], ast.Pass)
        ):
            raise ValueError("Expected a function signature without a body or additional statements")
        declaration = tree.body[0]
        compile(tree, "<signature>", "exec", dont_inherit=True)
    except SyntaxError as exc:
        raise ValueError(f"Invalid function signature {sig!r}: {exc.msg}") from None

    if getattr(declaration, "type_params", []):
        raise UnsupportedAnnotationError("Generic function type parameters are unsupported")
    arguments = declaration.args
    if arguments.vararg is not None or arguments.kwarg is not None:
        raise UnsupportedAnnotationError("Variadic parameters are unsupported")

    positional = arguments.posonlyargs + arguments.args
    parameters = positional + arguments.kwonlyargs
    required_positional_count = len(positional) - len(arguments.defaults)
    positional_defaults = [None] * required_positional_count + arguments.defaults
    defaults = positional_defaults + arguments.kw_defaults
    params = []
    for parameter, default in zip(parameters, defaults):
        annotation = (
            ast.get_source_segment(source, parameter.annotation)
            if parameter.annotation is not None else None
        )
        params.append(ParsedParam(
            name=parameter.arg,
            annotation=annotation,
            resolved_type=_resolve_annotation(parameter.annotation) if annotation is not None else Any,
            default=ast.get_source_segment(source, default) if default is not None else None,
        ))

    return ParsedSignature(
        name=declaration.name,
        params=params,
        return_annotation=(
            ast.get_source_segment(source, declaration.returns)
            if declaration.returns is not None else None
        ),
        return_type=_resolve_annotation(declaration.returns) if declaration.returns is not None else Any,
    )
