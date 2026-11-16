"""Native usage and visible-output evidence must survive transport adaptation."""
import unittest
from evaluation.responses_transport import normalize_data
from evaluation.common import classify_exception


class ResponsesTransportTests(unittest.TestCase):
    def test_usage_counts_reasoning_once_and_discards_hidden_payload(self):
        raw, evidence = normalize_data({"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
            "output": [{"type": "reasoning", "encrypted_content": "hidden"},
                       {"type": "message", "content": [{"type": "output_text", "text": "visible"}]},
                       {"type": "function_call", "arguments": "hidden"}],
            "usage": {"input_tokens": 10, "output_tokens": 50, "total_tokens": 60,
                      "output_tokens_details": {"reasoning_tokens": 40}}})
        self.assertEqual(raw["usage"]["completion_tokens"], 50)
        self.assertEqual(raw["choices"][0]["finish_reason"], "length")
        self.assertEqual(raw["tool_calls"], ["function_call"])
        self.assertEqual(evidence["unsupported_output_types"], ["function_call"])
        self.assertNotIn("hidden", str(evidence))

    def test_error_allowed_values_do_not_misclassify_unsupported_none(self):
        error = RuntimeError()
        error.status_code = 400
        error.body = {"param": "reasoning_effort", "message": "Unsupported value: 'max'. Supported values: 'none', 'high'."}
        self.assertIs(classify_exception(error)[1], False)
        error.body["message"] = "Unsupported value: 'none'. Supported values: 'high'."
        self.assertIs(classify_exception(error)[1], True)
