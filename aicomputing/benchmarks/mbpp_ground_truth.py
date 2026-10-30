"""Typed JSON values for frozen benchmark inputs and targets."""

import json
import math


def encode_value(value):
    """Encode supported built-in value trees, preserving types and dict order."""
    active = set()

    def encode(item):
        kind = type(item)
        if item is None:
            return ["none"]
        if kind is bool:
            return ["bool", item]
        if kind is int:
            return ["int", hex(item)]
        if kind is float:
            if not math.isfinite(item):
                raise ValueError("Nonfinite MBPP values cannot be graded")
            return ["float", item.hex()]
        if kind is str:
            return ["str", item]
        if kind is bytes:
            return ["bytes", item.hex()]
        if kind not in (list, tuple, dict, set, frozenset):
            raise TypeError(f"Unsupported MBPP value type: {kind.__name__}")

        identity = id(item)
        if identity in active:
            raise ValueError("Cyclic MBPP values cannot be frozen")
        active.add(identity)
        try:
            if kind is dict:
                return ["dict", [[encode(key), encode(val)]
                                 for key, val in item.items()]]
            entries = [encode(entry) for entry in item]
            if kind in (set, frozenset):
                entries.sort(key=json.dumps)
            return [kind.__name__, entries]
        finally:
            active.remove(identity)

    return encode(value)


def decode_value(node):
    """Decode typed JSON data without evaluating source or loading objects."""
    if type(node) is not list or not node or type(node[0]) is not str:
        raise ValueError("Invalid MBPP typed value")
    tag = node[0]
    if tag == "none" and len(node) == 1:
        return None
    if len(node) != 2:
        raise ValueError("Invalid MBPP typed value")
    payload = node[1]
    if tag == "bool" and type(payload) is bool:
        return payload
    if tag == "str" and type(payload) is str:
        return payload
    if tag in ("int", "float", "bytes") and type(payload) is str:
        try:
            if tag == "int":
                value = int(payload, 16)
            elif tag == "float":
                value = float.fromhex(payload)
            else:
                value = bytes.fromhex(payload)
            if encode_value(value) != node:
                raise ValueError("Noncanonical MBPP typed value")
            return value
        except (ValueError, OverflowError) as exc:
            raise ValueError("Invalid MBPP typed value") from exc
    if type(payload) is not list:
        raise ValueError("Invalid MBPP typed value")
    if tag == "dict":
        result = {}
        for pair in payload:
            if type(pair) is not list or len(pair) != 2:
                raise ValueError("Invalid MBPP dictionary entry")
            key, value = decode_value(pair[0]), decode_value(pair[1])
            try:
                if key in result:
                    raise ValueError("Duplicate MBPP dictionary key")
                result[key] = value
            except TypeError as exc:
                raise ValueError("Unhashable MBPP dictionary key") from exc
        return result
    if tag in ("list", "tuple", "set", "frozenset"):
        entries = [decode_value(entry) for entry in payload]
        if tag == "list":
            return entries
        if tag == "tuple":
            return tuple(entries)
        try:
            result = set(entries) if tag == "set" else frozenset(entries)
        except TypeError as exc:
            raise ValueError("Unhashable MBPP set member") from exc
        if len(result) != len(entries):
            raise ValueError("Duplicate MBPP set member")
        return result
    raise ValueError("Invalid MBPP typed value")
