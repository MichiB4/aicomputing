"""Provider identity, credential separation and token-cost accounting."""
import os
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation import common
from evaluation.model_profiles import get_model_profile
from evaluation.config import configure


class ModelProfileTests(unittest.TestCase):
    def test_profiles_keep_provider_specific_rates_and_connections(self):
        luna = get_model_profile("gpt-6-luna")
        terra = get_model_profile("gpt-6.1-terra")
        qwen = get_model_profile("qwen-3.8-27b")
        local = get_model_profile("qwen3:0.6b")
        self.assertEqual(terra.input_rate, Decimal("1") / 1_000_000)
        self.assertEqual(terra.output_rate, Decimal("5") / 1_000_000)
        self.assertNotEqual(luna.input_rate, terra.input_rate)
        self.assertEqual(qwen.api_base, "https://api.cerebras.ai/v1")
        self.assertEqual(qwen.key_env, "CEREBRAS_API_KEY")
        self.assertIsNone(qwen.service_tier)
        self.assertEqual(qwen.output_reservation, 32768)
        self.assertEqual(local.input_rate + local.output_rate, 0)
        self.assertIsNone(local.key_env)
        self.assertEqual(get_model_profile("gpt-6-luna", reasoning_effort="max").api_format, "responses")
        with self.assertRaises(ValueError):
            get_model_profile("typo-model")

    def test_credentials_are_never_borrowed_from_another_provider(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY":"openai-test", "CEREBRAS_API_KEY":"cerebras-test"}):
            self.assertEqual(common.read_key(None,get_model_profile("gpt-6-luna").key_env), "openai-test")
            self.assertEqual(common.read_key(None,get_model_profile("qwen-3.8-27b").key_env), "cerebras-test")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"keys.txt"
            path.write_text('OPENAI_API_KEY="one"\nCEREBRAS_API_KEY="two"\n')
            self.assertEqual(common.read_key(path,get_model_profile("qwen-3.8-27b").key_env),"two")
            with self.assertRaises(ValueError):common.read_key(path,get_model_profile("qwen3:0.6b").key_env)

    def test_reported_cache_usage_is_charged_at_terra_rates(self):
        events=[]
        ledger=common.Ledger(events.append,Decimal("1"),get_model_profile("gpt-6.1-terra"),max_attempts=1)
        record,reservation,started=ledger.before([{"role":"user","content":"x"*3000}])
        record.update(sdk_dispatched=True,sdk_wire_checked=True)
        ledger.after(record,reservation,started,sdk_response={
            "returned_model":"gpt-6.1-terra","input_tokens":1000,"output_tokens":100,
            "cached_input_tokens_reported":400,"cache_write_tokens_reported":200,
            "service_tier":"default","finish_reason":"stop"})
        self.assertEqual(ledger.charged,Decimal("0.001170"))
        self.assertEqual(record["charge_basis"],"reported_usage")
        self.assertIsNone(ledger.stop_reason)
        self.assertEqual([event["event"] for event in events],["request_start","request_complete"])

    def test_missing_usage_keeps_the_full_reservation(self):
        ledger=common.Ledger([].append,Decimal("1"),get_model_profile("gpt-6-luna"),max_attempts=1)
        record,reservation,started=ledger.before([{"role":"user","content":"hello"}])
        record.update(sdk_dispatched=True,sdk_wire_checked=True)
        ledger.after(record,reservation,started,exception=RuntimeError())
        self.assertEqual(ledger.charged,reservation)
        self.assertFalse(record["usage_complete"])

    def test_custom_models_require_explicit_price_and_capacity(self):
        with self.assertRaises(ValueError):configure({"model":"another-model","provider":"openai"})
        settings,profile=configure({"model":"another-model","provider":"openai","input_price":1.5,
                                    "output_price":3,"max_output_tokens":1000})
        self.assertEqual(profile.model,"another-model")
        self.assertEqual(profile.output_reservation,1000)
        self.assertEqual(profile.input_rate,Decimal("1.5")/1_000_000)


if __name__ == "__main__": unittest.main()
