"""Python declaration syntax, fixed annotation rules, and unevaluated defaults."""
from pathlib import Path
import tempfile
from typing import Any
import unittest

from aicomputing import parse_signature
from aicomputing.types import UnsupportedAnnotationError, resolve_type_str


class SignatureParsingTests(unittest.TestCase):
    def test_nested_defaults_and_quoted_delimiters_remain_source_expressions(self):
        signature = parse_signature(
            "build(pair: tuple[int, int] = (1, 2), *, "
            "options: dict[str, list[int]] = {'x': [3, 4]}, label: str = 'na\u00efve(,)') -> tuple[int, int]"
        )
        self.assertEqual([parameter.name for parameter in signature.params], ["pair", "options", "label"])
        self.assertEqual([parameter.default for parameter in signature.params],
                         ["(1, 2)", "{'x': [3, 4]}", "'na\u00efve(,)'"])
        self.assertEqual(signature.params[1].resolved_type, dict[str, list[int]])
        self.assertEqual(signature.return_type, tuple[int, int])

    def test_positional_and_keyword_defaults_distinguish_required_from_explicit_none(self):
        signature = parse_signature("f(first: int = 1, /, second: int = 2, *, required: str, optional: Any = None)")
        self.assertEqual([parameter.name for parameter in signature.params], ["first", "second", "required", "optional"])
        self.assertEqual([parameter.default for parameter in signature.params], ["1", "2", None, "None"])
        self.assertIsNone(signature.return_annotation)
        self.assertIs(signature.return_type, Any)

    def test_invalid_declarations_are_rejected_including_compiler_only_errors(self):
        for signature in ("1f(x)", "class(x)", "f(class)", "f(x, x)", "f(x, /, *, x)",
                          "__debug__()", "f(__debug__)", "f(x=1, y)", "f(x, /, /)",
                          "f(x=)", "f(x:)", "f(,x)"):
            with self.subTest(signature=signature), self.assertRaises(ValueError):
                parse_signature(signature)

    def test_function_bodies_and_additional_declarations_are_rejected(self):
        for signature in ("f():\n    return 1", "f():\n    pass\noutside = 1\n#",
                          "f():\n    pass\n\ndef another()"):
            with self.subTest(signature=signature), self.assertRaises(ValueError):
                parse_signature(signature)

    def test_defaults_are_never_executed(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "must-not-exist.txt"
            expression = f"__import__('pathlib').Path({str(target)!r}).write_text('executed')"
            signature = parse_signature(f"f(value={expression}) -> int")
            self.assertEqual(signature.params[0].default, expression)
            self.assertFalse(target.exists())

    def test_unsupported_annotation_expressions_are_never_executed(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "must-not-exist.txt"
            expression = f"__import__('pathlib').Path({str(target)!r}).touch()"
            with self.assertRaises(UnsupportedAnnotationError):
                parse_signature(f"f(value: {expression})")
            self.assertFalse(target.exists())

    def test_variadic_parameters_remain_unsupported(self):
        for signature in ("f(*args)", "f(x, *args: int)", "f(**kwargs)", "f(x, **kwargs: Any)"):
            with self.subTest(signature=signature), self.assertRaisesRegex(UnsupportedAnnotationError, "Variadic"):
                parse_signature(signature)

    def test_generic_function_type_parameters_remain_unsupported(self):
        with self.assertRaises(ValueError):
            parse_signature("f[T](value: int) -> int")

    def test_non_string_signatures_are_caller_errors(self):
        for signature in (None, False, 7, b"f(x)"):
            with self.subTest(signature=signature), self.assertRaises(TypeError):
                parse_signature(signature)

    def test_existing_annotation_types_and_aliases_are_preserved(self):
        for text, expected in (("int", int), ("float", float), ("str", str), ("bool", bool),
                               ("None", type(None)), ("none", type(None)), ("Any", Any),
                               ("List[int]", list[int]), ("Dict[str, int]", dict[str, int]),
                               ("Tuple[int, ...]", tuple[int, ...]), ("Set[str]", set[str]),
                               ("dict[str, list[tuple[int, ...]]]", dict[str, list[tuple[int, ...]]]),
                               ("list [ int ]", list[int])):
            with self.subTest(annotation=text):
                self.assertEqual(resolve_type_str(text), expected)

    def test_annotations_outside_the_supported_subset_are_rejected(self):
        for annotation in ("Unknown", "Optional[int]", "int | None", "typing.List[int]", "'int'",
                           "list[int, str]", "dict[str]", "dict[str, int, bool]", "list[123]",
                           "list[...]", "tuple[int, int, ...]", "tuple[()]", "True", ""):
            with self.subTest(annotation=annotation), self.assertRaises(UnsupportedAnnotationError):
                resolve_type_str(annotation)


if __name__ == "__main__":
    unittest.main()
