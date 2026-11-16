"""Backend contracts using in-memory HTTP and local inference stubs."""
import contextlib
import json
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx
from openai import OpenAI
from ollama import Client as OllamaClient

from aicomputing import ExecuteRequest, VirtualRuntime
from aicomputing.answer_protocol import frame_answer
from aicomputing.backends.base import Backend, BackendCommunicationError, BackendTimeoutError, GenerateResponse, ToolUseDetected
from aicomputing.backends.hf import HFBackend
from aicomputing.backends.ollama import OllamaBackend
from aicomputing.backends.openai_compat import OpenAICompatBackend


def completion(*, usage=True, choices=True, content=None):
    response = dict(id="offline", object="chat.completion", created=0, model="offline",
                    choices=[dict(index=0, message=dict(role="assistant", content=(
                        frame_answer('{"result": 7}') if content is None else content)),
                                  finish_reason="stop")] if choices else [])
    if usage:
        response["usage"] = dict(prompt_tokens=5, completion_tokens=7, total_tokens=12)
    return response


class PromptOnlyBackend(Backend):
    name = "prompt-only"
    default_model = "offline"

    def __init__(self, text):
        self.calls = []
        self.response = GenerateResponse(text=text, model="offline")

    def generate(self, prompt, **settings):
        self.calls.append((prompt, settings))
        return self.response


class BackendSettingTests(unittest.TestCase):
    def test_invalid_connection_and_device_settings_fail_before_setup(self):
        settings = ((OpenAICompatBackend, "base_url", "_get_client"),
                    (OpenAICompatBackend, "api_key", "_get_client"),
                    (OllamaBackend, "host", "_get_client"), (HFBackend, "device", "_load"))
        cases = [(value, TypeError) for value in (False, True, 0, 1.5, [], {}, (), b"setting")]
        cases += [(value, ValueError) for value in ("", "  ", "\t\n", "\u00a0\u2003")]
        for backend_type, name, setup in settings:
            with patch.object(backend_type, setup) as initialize:
                for value, error in cases:
                    with self.subTest(backend=backend_type.__name__, setting=name, value=value):
                        with self.assertRaisesRegex(error, name):
                            backend_type(model="offline", **{name: value})
                backend_type(model="offline", **{name: None}).close()
                initialize.assert_not_called()


class DefaultChatAdapterTests(unittest.TestCase):
    def test_single_user_content_and_generation_settings_are_forwarded_unchanged(self):
        settings = dict(model="custom-model", temperature=0.7, top_p=0.8, max_new_tokens=10, stop=["END"])
        for content in ("", " \tQuestion\n\nMore detail.\r\n "):
            with self.subTest(content=content):
                backend = PromptOnlyBackend("reply")
                messages = [{"role": "user", "content": content}]
                response = backend.generate_chat(messages, **settings)
                self.assertIs(response, backend.response)
                self.assertEqual(backend.calls, [(content, settings)])
                self.assertEqual(messages, [{"role": "user", "content": content}])

    def test_conversations_and_non_user_roles_are_rejected_before_generation(self):
        cases = ([], [{"role": "system", "content": "Instruction"}],
                 [{"role": "assistant", "content": "Previous answer"}],
                 [{"role": "developer", "content": "Instruction"}],
                 [{"role": "system", "content": "Instruction"}, {"role": "user", "content": "Question"}],
                 [{"role": "user", "content": "Earlier question"}, {"role": "user", "content": "Question"}],
                 [{"content": "Question"}])
        for messages in cases:
            with self.subTest(messages=messages):
                backend = PromptOnlyBackend("reply")
                with self.assertRaisesRegex(ValueError, "only one user message.*implement generate_chat"):
                    backend.generate_chat(messages)
                self.assertFalse(backend.calls)

    def test_malformed_messages_and_nontext_content_are_rejected_before_generation(self):
        cases = (None, "Question", {"role": "user", "content": "Question"},
                 ({"role": "user", "content": "Question"},), [None], ["Question"],
                 [{"role": "user"}], [{"role": "user", "content": None}],
                 [{"role": "user", "content": 7}], [{"role": "user", "content": ["Question"]}],
                 [{"role": "user", "content": b"Question"}])
        for messages in cases:
            with self.subTest(messages=messages):
                backend = PromptOnlyBackend("reply")
                with self.assertRaises(TypeError):
                    backend.generate_chat(messages)
                self.assertFalse(backend.calls)

    def test_additional_message_fields_require_a_chat_implementation(self):
        for extra in ({"name": "alice"}, {"images": ["image"]}, {"tool_calls": []}):
            with self.subTest(extra=extra):
                backend = PromptOnlyBackend("reply")
                with self.assertRaisesRegex(ValueError, "only role and content fields"):
                    backend.generate_chat([{"role": "user", "content": "Question", **extra}])
                self.assertFalse(backend.calls)

    def test_code_execution_works_with_a_prompt_only_custom_backend(self):
        backend = PromptOnlyBackend("def f(): return 7")
        with VirtualRuntime(backend=backend, strategy="code_execute") as runtime:
            result = runtime.execute(ExecuteRequest(signature="f() -> int", docstring="Return seven.", inputs={}))
        self.assertEqual(result.result, 7)
        self.assertEqual(result.execution_details["status"], "ok")
        self.assertEqual(result.model_calls, 1)
        self.assertEqual(len(backend.calls), 1)
        self.assertIn("Return seven.", backend.calls[0][0])


class OpenAIBackendTests(unittest.TestCase):
    def client(self, handler, **options):
        transport = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(transport.close)
        factory = patch("openai.OpenAI", side_effect=lambda **kwargs: OpenAI(http_client=transport, **kwargs))
        self.addCleanup(factory.stop)
        constructor = factory.start()
        backend = OpenAICompatBackend(model="offline", api_key="offline-key", base_url="https://offline.invalid/v1", **options)
        self.addCleanup(backend.close)
        return backend, constructor

    def test_default_sdk_retries_are_zero_and_a_server_failure_is_sent_once(self):
        calls = []
        def handler(request):
            calls.append(request)
            return httpx.Response(500, json={"error": {"message": "offline failure"}})
        backend, constructor = self.client(handler, request_diagnostics=True)
        constructor.assert_not_called()
        with self.assertRaises(BackendCommunicationError) as caught:
            backend.generate("prompt")
        self.assertEqual(len(calls), 1)
        self.assertEqual(constructor.call_args.kwargs["max_retries"], 0)
        self.assertEqual(caught.exception.diagnostics["sdk_max_retries"], 0)
        self.assertEqual(caught.exception.diagnostics["http_status"], 500)

    def test_connection_overrides_are_forwarded_literally_and_only_none_is_omitted(self):
        for base_url in (None, "https://literal.invalid/Custom/v1/"):
            for api_key in (None, "  offline-key  "):
                with self.subTest(base_url=base_url, explicit_key=api_key is not None):
                    with patch("openai.OpenAI") as constructor:
                        backend = OpenAICompatBackend(model="offline", base_url=base_url, api_key=api_key)
                        constructor.assert_not_called()
                        backend._get_client()
                        expected = {"max_retries": 0}
                        if base_url is not None:
                            expected["base_url"] = base_url
                        if api_key is not None:
                            expected["api_key"] = api_key
                        constructor.assert_called_once_with(**expected)
                        backend.close()

    def test_none_connection_settings_use_the_sdk_environment_defaults(self):
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(200, json=completion())
        with httpx.Client(transport=httpx.MockTransport(handler)) as transport:
            with patch.dict("os.environ", {"OPENAI_BASE_URL": "https://environment.invalid/v1",
                                           "OPENAI_API_KEY": "offline-environment-key"}):
                with patch("openai.OpenAI", side_effect=lambda **options: OpenAI(http_client=transport, **options)):
                    with OpenAICompatBackend(model="offline", base_url=None, api_key=None) as backend:
                        backend.generate("prompt")
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].url.host, "environment.invalid")
        self.assertEqual(requests[0].headers["authorization"], "Bearer offline-environment-key")

    def test_explicit_retry_timeout_and_generation_controls_are_forwarded(self):
        bodies = []
        def handler(request):
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json=completion())
        backend, constructor = self.client(handler, sdk_max_retries=3, timeout_seconds=180, reasoning_effort="low")
        response = backend.generate("prompt", max_new_tokens=23, temperature=0.4, top_p=0.9)
        self.assertEqual(constructor.call_args.kwargs["max_retries"], 3)
        self.assertEqual(constructor.call_args.kwargs["timeout"], 180)
        self.assertEqual((bodies[0]["max_tokens"], bodies[0]["temperature"], bodies[0]["top_p"]), (23, 0.4, 0.9))
        self.assertEqual(bodies[0]["reasoning_effort"], "low")
        self.assertEqual((response.input_tokens, response.output_tokens), (5, 7))

    def test_omitted_limits_and_native_reasoning_are_not_invented(self):
        bodies = []
        def handler(request):
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json=completion())
        backend, constructor = self.client(handler)
        backend.generate("first")
        backend.generate("second")
        self.assertEqual(constructor.call_count, 1)
        self.assertTrue(all("max_tokens" not in body and "reasoning_effort" not in body for body in bodies))

    def test_only_none_inherits_the_configured_model(self):
        models = []
        def handler(request):
            models.append(json.loads(request.content)["model"])
            return httpx.Response(200, json=completion())
        backend, _ = self.client(handler)
        inherited = backend.generate("prompt", model=None)
        overridden = backend.generate("prompt", model="provider/Custom:Tag")
        self.assertEqual(models, ["offline", "provider/Custom:Tag"])
        self.assertEqual((inherited.model, overridden.model), ("offline", "provider/Custom:Tag"))

    def test_read_timeout_is_normalized_without_extra_requests(self):
        calls = []
        def handler(request):
            calls.append(request)
            raise httpx.ReadTimeout("offline timeout", request=request)
        backend, _ = self.client(handler, request_diagnostics=True)
        with self.assertRaises(BackendTimeoutError) as caught:
            backend.generate("prompt")
        self.assertEqual(len(calls), 1)
        self.assertEqual(caught.exception.diagnostics["httpx_timeout_phases"], ["read"])

    def test_missing_usage_stays_unknown_in_package_results(self):
        backend, _ = self.client(lambda request: httpx.Response(200, json=completion(usage=False)))
        with VirtualRuntime(backend=backend) as runtime:
            result = runtime.execute(ExecuteRequest(signature="f() -> int", docstring="Return seven.", inputs={}))
        self.assertEqual(result.result, 7)
        self.assertIsNone(result.input_tokens)
        self.assertIsNone(result.output_tokens)
        self.assertIsNone(result.total_tokens)

    def test_original_text_is_retained_and_unfinished_thinking_is_not_parsed(self):
        for closed in (True, False):
            with self.subTest(closed=closed):
                text = '  <think>private draft' + ('</think>\n' if closed else '\n')
                text += frame_answer('{"result": 7}') + '  \n'
                backend, _ = self.client(lambda request, text=text: httpx.Response(200, json=completion(content=text)))
                with VirtualRuntime(backend=backend) as runtime:
                    result = runtime.execute(ExecuteRequest(signature="f() -> int", docstring="Return seven.",
                                                            inputs={}, max_retries=0))
                self.assertEqual(result.raw_text, text)
                self.assertEqual(result.result, 7 if closed else None)
                self.assertEqual(result.parse_status, "parsed" if closed else "failed")

    def test_validation_retries_use_cleaned_text_without_changing_raw_evidence(self):
        for strategy, calls_per_attempt in (("direct", 1), ("contract", 2), ("consistency", 5)):
            with self.subTest(strategy=strategy):
                bodies = []
                def handler(request, bodies=bodies, strategy=strategy, calls_per_attempt=calls_per_attempt):
                    bodies.append(json.loads(request.content))
                    value = "wrong type" if len(bodies) <= calls_per_attempt else 7
                    payload = {"result": value}
                    if strategy == "contract" and len(bodies) % 2 == 0:
                        payload["verdict"] = "accept"
                    text = '  <think>private draft</think>\n' + frame_answer(json.dumps(payload)) + '  \n'
                    return httpx.Response(200, json=completion(content=text))
                backend, _ = self.client(handler)
                with VirtualRuntime(backend=backend, strategy=strategy) as runtime:
                    result = runtime.execute(ExecuteRequest(signature="f() -> int", docstring="Return seven.",
                                                            inputs={}, max_retries=1))
                self.assertEqual((result.result, result.retries, result.model_calls), (7, 1, 2 * calls_per_attempt))
                retry_prompt = bodies[calls_per_attempt]["messages"][0]["content"]
                self.assertIn("Previous output:", retry_prompt)
                self.assertIn("wrong type", retry_prompt)
                self.assertNotIn("private draft", retry_prompt)
                self.assertIn("  <think>private draft</think>", result.raw_text)

    def test_invalid_provider_response_is_a_backend_error(self):
        backend, _ = self.client(lambda request: httpx.Response(200, json=completion(choices=False)))
        with self.assertRaisesRegex(BackendCommunicationError, "no completion choices"):
            backend.generate("prompt")

    def test_tool_call_messages_are_rejected_even_with_a_framed_answer(self):
        fields = (
            {"tool_calls": [dict(id="call-offline", type="function", function=dict(name="calculate", arguments="{}"))]},
            {"function_call": dict(name="calculate", arguments="{}")},
        )
        for field in fields:
            with self.subTest(field=next(iter(field))):
                payload = completion()
                payload["choices"][0]["message"].update(field)
                backend, _ = self.client(lambda request, payload=payload: httpx.Response(200, json=payload))
                with self.assertRaisesRegex(ToolUseDetected, "message\\."):
                    backend.generate("prompt")

    def test_tool_finish_reasons_and_populated_provider_fields_are_rejected(self):
        payloads = []
        for reason in ("tool_calls", "function_call", "tool_use"):
            payload = completion()
            payload["choices"][0]["finish_reason"] = reason
            payloads.append(payload)
        payload = completion()
        payload["provider_details"] = {"code_execution_result": {"value": 7}}
        payloads.append(payload)
        for payload in payloads:
            with self.subTest(payload=payload):
                backend, _ = self.client(lambda request, payload=payload: httpx.Response(200, json=payload))
                with self.assertRaises(ToolUseDetected):
                    backend.generate("prompt")

    def test_unused_tool_fields_and_text_discussing_tools_are_accepted(self):
        text = 'The fields tool_calls and function_call describe tools. {"tool_calls": ["example"]}\n'
        text += frame_answer('{"result": 7}')
        for unused in (None, []):
            with self.subTest(unused=unused):
                payload = completion(content=text)
                payload["choices"][0]["message"].update(tool_calls=unused, function_call=None)
                payload["provider_details"] = {"tool_outputs": unused}
                backend, _ = self.client(lambda request, payload=payload: httpx.Response(200, json=payload))
                with VirtualRuntime(backend=backend) as runtime:
                    result = runtime.execute(ExecuteRequest(signature="f() -> int", docstring="Return seven.", inputs={}))
                self.assertEqual(result.result, 7)
                self.assertEqual(result.raw_text, text)

    def test_tool_call_reply_stops_before_validation_retries_or_local_execution(self):
        payload = completion(content="```python\ndef f(): return 7\n```")
        payload["choices"][0]["finish_reason"] = "tool_calls"
        for strategy in ("direct", "code_execute"):
            for mode in ("strict", "warn", "none"):
                with self.subTest(strategy=strategy, mode=mode):
                    calls = []
                    def handler(request, calls=calls):
                        calls.append(request)
                        return httpx.Response(200, json=payload)
                    backend, _ = self.client(handler)
                    with patch("aicomputing.strategies.code_execute.execute_code_with_retries") as worker:
                        with VirtualRuntime(backend=backend, strategy=strategy, validation_mode=mode) as runtime:
                            with self.assertRaises(ToolUseDetected):
                                runtime.execute(ExecuteRequest(signature="f() -> int", docstring="Return seven.",
                                                               inputs={}, max_retries=3))
                        worker.assert_not_called()
                    self.assertEqual(len(calls), 1)

    def test_codegen_executes_cleaned_source_and_retains_the_original_reply(self):
        text = '  <think>```python\ndef f(): return 999\n```</think>\n'
        text += '```python\ndef f(): return 7\n```  \n'
        backend, _ = self.client(lambda request: httpx.Response(200, json=completion(content=text)))
        execution = dict(status="ok", value=7, error_type=None, elapsed_seconds=0.01)
        with patch("aicomputing.strategies.code_execute.execute_code_with_retries", return_value=execution) as worker:
            with VirtualRuntime(backend=backend, strategy="code_execute") as runtime:
                result = runtime.execute(ExecuteRequest(signature="f() -> int", docstring="Return seven.", inputs={}))
        self.assertEqual(result.result, 7)
        self.assertEqual(result.raw_text, text)
        self.assertEqual(result.generated_code, "def f(): return 7")
        self.assertEqual(worker.call_args.args[0], result.generated_code)

    def test_close_is_lazy_and_retains_a_client_when_cleanup_fails(self):
        backend = OpenAICompatBackend(model="offline")
        backend.close()
        failed = SimpleNamespace(close=Mock(side_effect=RuntimeError("cleanup")))
        backend._client = failed
        with self.assertRaises(RuntimeError):
            backend.close()
        self.assertIs(backend._client, failed)
        failed.close.side_effect = None
        backend.close()
        self.assertIsNone(backend._client)

    def test_invalid_retry_settings_fail_before_client_creation(self):
        for retries in (-1, True, None, 1.5):
            with self.subTest(retries=retries), self.assertRaises(ValueError):
                OpenAICompatBackend(sdk_max_retries=retries)


class OllamaBackendTests(unittest.TestCase):
    def client(self, handler, **options):
        clients = []
        def create(**kwargs):
            client = OllamaClient(transport=httpx.MockTransport(handler), **kwargs)
            self.addCleanup(client._client.close)
            clients.append(client)
            return client
        factory = patch("ollama.Client", side_effect=create)
        self.addCleanup(factory.stop)
        constructor = factory.start()
        backend = OllamaBackend(model="offline", **options)
        self.addCleanup(backend.close)
        return backend, constructor, clients

    def test_clients_are_lazy_isolated_and_recreated_after_close(self):
        response = {"message": {"role": "assistant", "content": frame_answer('{"result": 7}')}}
        with patch.dict("os.environ", {"OLLAMA_HOST": "http://offline.invalid", "OLLAMA_API_KEY": ""}):
            first, constructor, clients = self.client(lambda request: httpx.Response(200, json=response))
            second = OllamaBackend(model="offline")
            self.addCleanup(second.close)
            first.close()
            constructor.assert_not_called()
            first.generate("first")
            first.generate("reuse")
            second.generate("second")
            self.assertEqual(constructor.call_count, 2)
            self.assertIsNot(first._client, second._client)
            first.close()
            self.assertTrue(clients[0]._client.is_closed)
            self.assertFalse(clients[1]._client.is_closed)
            second.generate("still open")
            first.generate("recreated")
            self.assertEqual(constructor.call_count, 3)
            self.assertIs(first._client, clients[2])

    def test_host_resolution_follows_the_sdk_without_eager_client_creation(self):
        cases = [(None, "", "http://127.0.0.1:11434"),
                 (None, "offline.invalid:12345/proxy", "http://offline.invalid:12345/proxy"),
                 (None, "https://offline.invalid/proxy/", "https://offline.invalid/proxy"),
                 (None, "[::1]:12345/proxy", "http://[::1]:12345/proxy"),
                 ("explicit.invalid:12346/proxy", "ignored.invalid", "http://explicit.invalid:12346/proxy")]
        for host, environment_host, expected in cases:
            with self.subTest(host=host, environment_host=environment_host):
                with patch.dict("os.environ", {"OLLAMA_HOST": environment_host, "OLLAMA_API_KEY": ""}):
                    backend, constructor, clients = self.client(lambda request: httpx.Response(200, json={
                        "message": {"role": "assistant", "content": frame_answer('{"result": 7}')}}), host=host)
                    constructor.assert_not_called()
                    backend.generate("prompt")
                    self.assertEqual(str(clients[0]._client.base_url).rstrip("/"), expected)

    def test_host_values_are_passed_to_the_sdk_without_rewriting(self):
        for host in (None, "https://literal.invalid/Custom/v1/", " localhost:12345/path "):
            with self.subTest(host=host), patch("ollama.Client") as constructor:
                backend = OllamaBackend(model="offline", host=host)
                constructor.assert_not_called()
                backend._get_client()
                constructor.assert_called_once_with(host=host)
                backend.close()

    def test_generation_uses_the_owned_endpoint_and_authentication(self):
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={"message": {"role": "assistant", "content": frame_answer('{"result": 7}')}})
        with patch.dict("os.environ", {"OLLAMA_HOST": "https://offline.invalid/proxy", "OLLAMA_API_KEY": "offline-token"}):
            backend, constructor, clients = self.client(handler, think=False)
            response = backend.generate("prompt")
        self.assertEqual(constructor.call_count, 1)
        self.assertEqual(response.requested_reasoning, {"think": False})
        self.assertEqual([request.url.path for request in requests], ["/proxy/api/chat"])
        self.assertTrue(all(request.headers["authorization"] == "Bearer offline-token" for request in requests))
        self.assertFalse(clients[0]._client.is_closed)

    def test_tool_calls_are_rejected_with_or_without_an_answer(self):
        for content in ("", frame_answer('{"result": 7}')):
            with self.subTest(content=content):
                payload = {"message": {"role": "assistant", "content": content,
                                       "tool_calls": [{"function": {"name": "calculate", "arguments": {}}}]},
                           "done_reason": "stop"}
                backend, _, _ = self.client(lambda request, payload=payload: httpx.Response(200, json=payload))
                with self.assertRaisesRegex(ToolUseDetected, "message\\.tool_calls"):
                    backend.generate("prompt")

    def test_tool_call_termination_reasons_are_rejected_without_tool_fields(self):
        for reason in ("tool_calls", "function_call", "tool-use"):
            with self.subTest(reason=reason):
                payload = {"message": {"role": "assistant", "content": frame_answer('{"result": 7}')},
                           "done_reason": reason}
                backend, _, _ = self.client(lambda request, payload=payload: httpx.Response(200, json=payload))
                with self.assertRaisesRegex(ToolUseDetected, "finish_reason"):
                    backend.generate("prompt")

    def test_dictionary_responses_use_the_same_tool_indicator_check(self):
        payloads = [
            {"message": {"content": frame_answer('{"result": 7}'),
                         "tool_calls": [{"function": {"name": "calculate", "arguments": {}}}]}},
            {"message": {"content": frame_answer('{"result": 7}')}, "done_reason": "tool_calls"},
            {"message": {"content": frame_answer('{"result": 7}')}, "provider_details": {"tool_result": {"value": 7}}},
        ]
        for payload in payloads:
            with self.subTest(payload=payload):
                backend = OllamaBackend(model="offline")
                client = SimpleNamespace(chat=Mock(return_value=payload))
                with patch.object(backend, "_get_client", return_value=client):
                    with self.assertRaises(ToolUseDetected):
                        backend.generate("prompt")
                self.assertEqual(client.chat.call_count, 1)

    def test_unused_tool_fields_and_text_discussing_tools_are_accepted(self):
        text = 'The fields tool_calls and code_execution describe tools. {"tool_calls": ["example"]}\n'
        text += frame_answer('{"result": 7}')
        for unused in (None, []):
            with self.subTest(unused=unused):
                payload = {"message": {"role": "assistant", "content": text, "tool_calls": unused,
                                       "thinking": "Consider the words function_call and code_execution."},
                           "done_reason": "stop"}
                backend, _, _ = self.client(lambda request, payload=payload: httpx.Response(200, json=payload))
                with VirtualRuntime(backend=backend) as runtime:
                    result = runtime.execute(ExecuteRequest(signature="f() -> int", docstring="Return seven.", inputs={}))
                self.assertEqual(result.result, 7)
                self.assertEqual(result.raw_text, text)

    def test_tool_call_reply_stops_before_validation_retries_or_local_execution(self):
        payload = {"message": {"role": "assistant", "content": "def f(): return 7",
                               "tool_calls": [{"function": {"name": "calculate", "arguments": {}}}]},
                   "done_reason": "stop"}
        for strategy in ("direct", "code_execute"):
            for mode in ("strict", "warn", "none"):
                with self.subTest(strategy=strategy, mode=mode):
                    calls = []
                    def handler(request, calls=calls):
                        calls.append(request)
                        return httpx.Response(200, json=payload)
                    backend, _, _ = self.client(handler)
                    with patch("aicomputing.strategies.code_execute.execute_code_with_retries") as worker:
                        with VirtualRuntime(backend=backend, strategy=strategy, validation_mode=mode) as runtime:
                            with self.assertRaises(ToolUseDetected):
                                runtime.execute(ExecuteRequest(signature="f() -> int", docstring="Return seven.",
                                                               inputs={}, max_retries=3))
                        worker.assert_not_called()
                    self.assertEqual(len(calls), 1)

    def test_raw_content_is_preserved_while_thinking_is_excluded_from_parsing(self):
        text = '  <think>private draft</think>\n' + frame_answer('{"result": 7}') + '  \n'
        client = SimpleNamespace(chat=Mock(return_value={"message": {"content": text}}))
        backend = OllamaBackend(model="offline")
        with patch.object(backend, "_get_client", return_value=client):
            with VirtualRuntime(backend=backend) as runtime:
                result = runtime.execute(ExecuteRequest(signature="f() -> int", docstring="Return seven.", inputs={}))
        self.assertEqual(result.raw_text, text)
        self.assertEqual(result.result, 7)

    def test_generation_controls_native_reasoning_and_usage_are_forwarded(self):
        client = SimpleNamespace(chat=Mock(return_value={"message": {"content": frame_answer('{"result": 7}')},
                                                       "prompt_eval_count": 5, "eval_count": 7, "done_reason": "stop"}))
        backend = OllamaBackend(model="offline", think=False)
        with patch.object(backend, "_get_client", return_value=client):
            response = backend.generate("prompt", max_new_tokens=23, temperature=0.4, top_p=0.9)
            backend.generate("prompt", max_new_tokens=None)
        first, second = client.chat.call_args_list
        self.assertEqual(first.kwargs["options"], {"temperature": 0.4, "top_p": 0.9, "num_predict": 23})
        self.assertIs(first.kwargs["think"], False)
        self.assertEqual(second.kwargs["options"]["num_predict"], -1)
        self.assertEqual((response.input_tokens, response.output_tokens), (5, 7))

    def test_timeout_and_missing_usage_have_consistent_meanings(self):
        backend = OllamaBackend(model="offline")
        client = SimpleNamespace(chat=Mock(side_effect=httpx.ReadTimeout("offline")))
        with patch.object(backend, "_get_client", return_value=client):
            with self.assertRaises(BackendTimeoutError):
                backend.generate("prompt")
            client.chat.side_effect = None
            client.chat.return_value = {"message": {"content": frame_answer('{"result": 7}')}}
            response = backend.generate("prompt")
        self.assertIsNone(response.input_tokens)
        self.assertIsNone(response.output_tokens)

    def test_only_none_inherits_the_configured_model(self):
        client = SimpleNamespace(chat=Mock(return_value={"message": {"content": frame_answer('{"result": 7}')}}))
        backend = OllamaBackend(model="configured-model")
        with patch.object(backend, "_get_client", return_value=client):
            inherited = backend.generate("prompt", model=None)
            overridden = backend.generate("prompt", model="provider/Custom:Tag")
        self.assertEqual([call.kwargs["model"] for call in client.chat.call_args_list],
                         ["configured-model", "provider/Custom:Tag"])
        self.assertEqual((inherited.model, overridden.model), ("configured-model", "provider/Custom:Tag"))

    def test_owned_transport_cleanup_is_repeatable(self):
        transport = SimpleNamespace(close=Mock())
        backend = OllamaBackend(model="offline", host="http://offline.invalid")
        backend._client = SimpleNamespace(_client=transport)
        backend.close()
        backend.close()
        transport.close.assert_called_once()
        self.assertIsNone(backend._client)

    def test_failed_cleanup_retains_the_client_for_another_attempt(self):
        transport = SimpleNamespace(close=Mock(side_effect=RuntimeError("cleanup")))
        backend = OllamaBackend(model="offline")
        client = SimpleNamespace(_client=transport)
        backend._client = client
        with self.assertRaisesRegex(RuntimeError, "cleanup"):
            backend.close()
        self.assertIs(backend._client, client)
        transport.close.side_effect = None
        backend.close()
        self.assertIsNone(backend._client)


class Token(int):
    def item(self):
        return int(self)


class Batch(dict):
    @property
    def input_ids(self):
        return self["input_ids"]

    def to(self, device):
        return self


class Tokenizer:
    chat_template = "offline template"
    model_max_length = 16
    pad_token_id = 0
    eos_token_id = 2

    def __init__(self):
        self.apply_chat_template = Mock(return_value="rendered prompt")
        self.encoding = []

    def __call__(self, text, **options):
        self.encoding.append((text, options))
        return Batch(input_ids=[[Token(1), Token(3), Token(4)]])

    def decode(self, tokens, **options):
        return frame_answer('{"result": 7}')


class HFBackendTests(unittest.TestCase):
    def test_device_defaults_and_explicit_strings_are_resolved_without_rewriting(self):
        for device, cuda, expected in ((None, False, "cpu"), (None, True, "cuda"),
                                       ("cpu", True, "cpu"), ("cuda:2", False, "cuda:2"), (" mps ", True, " mps ")):
            with self.subTest(device=device, cuda=cuda):
                is_available = Mock(return_value=cuda)
                model = SimpleNamespace(to=Mock())
                tokenizer = SimpleNamespace(pad_token="<pad>", eos_token="<eos>")
                modules = {
                    "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=is_available)),
                    "transformers": SimpleNamespace(
                        AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)),
                        AutoModelForCausalLM=SimpleNamespace(from_pretrained=Mock(return_value=model))),
                }
                with patch.dict(sys.modules, modules):
                    backend = HFBackend(model="offline", device=device)
                    backend._load()
                model.to.assert_called_once_with(expected)
                self.assertEqual(is_available.call_count, 1 if device is None else 0)
                self.assertEqual(backend.cache_state()["device"], device)
                backend.close()
                self.assertEqual(backend._device, device)

    def backend(self, *, tokens=(7, 2), context=16, thinking=None):
        backend = HFBackend(model="offline", enable_thinking=thinking)
        backend._device = "cpu"
        backend._tokenizer = Tokenizer()
        generated = [[Token(v) for v in (1, 3, 4, *tokens)]]
        backend._model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=context),
                                        generation_config=SimpleNamespace(eos_token_id=2),
                                        generate=Mock(return_value=generated))
        torch = patch.dict(sys.modules, {"torch": SimpleNamespace(inference_mode=contextlib.nullcontext)})
        torch.start()
        self.addCleanup(torch.stop)
        return backend

    def test_explicit_limits_templates_sampling_and_usage(self):
        backend = self.backend(thinking=False)
        response = backend.generate("prompt", max_new_tokens=4, temperature=0.7, top_p=0.9)
        options = backend._model.generate.call_args.kwargs
        self.assertEqual((options["max_new_tokens"], options["temperature"], options["top_p"]), (4, 0.7, 0.9))
        self.assertTrue(options["do_sample"])
        self.assertFalse(backend._tokenizer.encoding[0][1]["add_special_tokens"])
        self.assertIs(backend._tokenizer.apply_chat_template.call_args.kwargs["enable_thinking"], False)
        self.assertEqual((response.input_tokens, response.output_tokens, response.finish_reason), (3, 2, "stop"))

    def test_chat_preserves_conversation_roles_and_contents(self):
        backend = self.backend()
        messages = [{"role": "system", "content": "Follow the specification."},
                    {"role": "user", "content": "Earlier question"},
                    {"role": "assistant", "content": "Earlier answer"},
                    {"role": "user", "content": "  Current question\n"}]
        backend.generate_chat(messages, max_new_tokens=4)
        backend._tokenizer.apply_chat_template.assert_called_once_with(
            messages, tokenize=False, add_generation_prompt=True)

    def test_fixed_model_backend_accepts_none_or_its_configured_name(self):
        backend = self.backend()
        for model in (None, "offline"):
            self.assertEqual(backend.generate("prompt", model=model, max_new_tokens=4).model, "offline")
        with self.assertRaisesRegex(ValueError, "configured for"):
            backend.generate("prompt", model="another-model", max_new_tokens=4)
        self.assertEqual(backend._model.generate.call_count, 2)
        self.assertEqual(backend._tokenizer.encoding[0][0], ["rendered prompt"])

    def test_single_prompt_and_single_user_chat_have_identical_generation(self):
        backend = self.backend(thinking=False)
        prompt = "  Return seven.\n"
        first = backend.generate(prompt, max_new_tokens=4)
        options = backend._model.generate.call_args.kwargs
        second = backend.generate_chat([{"role": "user", "content": prompt}], max_new_tokens=4)
        self.assertEqual(first, second)
        self.assertEqual(backend._model.generate.call_args.kwargs, options)
        self.assertEqual(backend._tokenizer.apply_chat_template.call_args_list[0],
                         backend._tokenizer.apply_chat_template.call_args_list[1])

    def test_without_a_template_a_single_user_prompt_is_used_unchanged(self):
        backend = self.backend()
        backend._tokenizer.chat_template = None
        prompt = "  Return seven.\n"
        backend.generate_chat([{"role": "user", "content": prompt}], max_new_tokens=4)
        self.assertEqual(backend._tokenizer.encoding, [([prompt], dict(return_tensors="pt", add_special_tokens=True))])
        backend._tokenizer.apply_chat_template.assert_not_called()

    def test_thinking_override_requires_a_chat_template_before_generation(self):
        for thinking in (False, True):
            with self.subTest(thinking=thinking):
                backend = self.backend(thinking=thinking)
                backend._tokenizer.chat_template = None
                with self.assertRaisesRegex(ValueError, "enable_thinking requires a tokenizer chat template"):
                    backend.generate("prompt", max_new_tokens=4)
                backend._model.generate.assert_not_called()

    def test_without_a_template_unsupported_conversations_are_rejected_before_generation(self):
        for messages in ([], [{"role": "system", "content": "Instruction"}],
                         [{"role": "assistant", "content": "Previous answer"}],
                         [{"role": "user", "content": "Earlier question"},
                          {"role": "user", "content": "Current question"}]):
            with self.subTest(messages=messages):
                backend = self.backend()
                backend._tokenizer.chat_template = None
                with self.assertRaisesRegex(ValueError, "at least one message|only one user message"):
                    backend.generate_chat(messages, max_new_tokens=4)
                backend._model.generate.assert_not_called()
                self.assertEqual(backend._tokenizer.encoding, [])

    def test_a_template_error_is_not_replaced_by_flattened_generation(self):
        backend = self.backend()
        backend._tokenizer.apply_chat_template.side_effect = ValueError("Unsupported conversation roles")
        with self.assertRaisesRegex(ValueError, "Unsupported conversation roles"):
            backend.generate_chat([{"role": "system", "content": "Instruction"},
                                   {"role": "user", "content": "Question"}], max_new_tokens=4)
        backend._model.generate.assert_not_called()
        self.assertEqual(backend._tokenizer.encoding, [])

    def test_template_prefilled_thinking_is_parsed_without_altering_generated_text(self):
        for closed in (True, False):
            with self.subTest(closed=closed):
                backend = self.backend()
                backend._tokenizer.apply_chat_template.return_value = "rendered prompt<think>\n"
                text = '  private draft' + ('</think>\n' if closed else '\n')
                text += frame_answer('{"result": 7}') + '  \n'
                backend._tokenizer.decode = Mock(return_value=text)
                with VirtualRuntime(backend=backend) as runtime:
                    result = runtime.execute(ExecuteRequest(signature="f() -> int", docstring="Return seven.",
                                                            inputs={}, max_new_tokens=4, max_retries=0))
                self.assertEqual(result.raw_text, text)
                self.assertEqual(result.result, 7 if closed else None)
                self.assertEqual(result.parse_status, "parsed" if closed else "failed")

    def test_omitted_limit_uses_remaining_context(self):
        backend = self.backend()
        backend.generate("prompt", max_new_tokens=None)
        self.assertEqual(backend._model.generate.call_args.kwargs["max_new_tokens"], 13)

    def test_unknown_context_requires_an_explicit_limit(self):
        backend = self.backend(context=None)
        backend._tokenizer.model_max_length = 10**30
        with self.assertRaisesRegex(ValueError, "finite model context"):
            backend.generate("prompt")
        backend._model.generate.assert_not_called()
        backend.generate("prompt", max_new_tokens=100)
        self.assertEqual(backend._model.generate.call_args.kwargs["max_new_tokens"], 100)

    def test_explicit_budget_exceeding_context_is_rejected_before_inference(self):
        for context, limit, available in ((16, 14, 13), (3, 1, 0), (2, 1, 0)):
            with self.subTest(context=context, limit=limit):
                backend = self.backend(context=context)
                with self.assertRaises(ValueError) as caught:
                    backend.generate("prompt", max_new_tokens=limit)
                error = str(caught.exception)
                self.assertIn("input uses 3 tokens", error)
                self.assertIn(f"requests {limit} output tokens", error)
                self.assertIn(f"{context}-token context", error)
                self.assertIn(f"{available} output tokens available", error)
                backend._model.generate.assert_not_called()

    def test_explicit_budget_fitting_context_is_forwarded_unchanged(self):
        backend = self.backend()
        backend.generate("prompt", max_new_tokens=13)
        self.assertEqual(backend._model.generate.call_args.kwargs["max_new_tokens"], 13)

    def test_context_guard_uses_text_config_then_model_config_then_tokenizer(self):
        for source in ("text_config", "model_config", "tokenizer"):
            with self.subTest(source=source):
                backend = self.backend(context=16 if source == "model_config" else None)
                if source == "text_config":
                    backend._model.config = SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=16),
                                                            max_position_embeddings=64)
                if source != "tokenizer":
                    backend._tokenizer.model_max_length = 64
                with self.assertRaisesRegex(ValueError, "16-token context"):
                    backend.generate("prompt", max_new_tokens=14)
                backend._model.generate.assert_not_called()

    def test_reaching_a_limit_without_eos_is_truncation(self):
        backend = self.backend(tokens=(7, 8, 9, 10))
        self.assertEqual(backend.generate("prompt", max_new_tokens=4).finish_reason, "length")

    def test_close_drops_model_state_without_loading_it(self):
        backend = self.backend()
        backend.close()
        self.assertIsNone(backend._model)
        self.assertIsNone(backend._tokenizer)
        self.assertIsNone(backend._device)


if __name__ == "__main__":
    unittest.main()
