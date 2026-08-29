from __future__ import annotations

from types import SimpleNamespace

from agent.llm import OpenAIStructuredClient
from agent.propose import LLMHypothesis


def test_responses_structured_output_and_usage_are_accounted() -> None:
    parsed = LLMHypothesis.model_validate(
        {
            "id": "h_sdk",
            "target_node": "model",
            "rationale": "bounded test",
            "method_source": "unit test",
            "expected_delta": 0.003,
            "expected_cost_minutes": 1,
            "patch": {
                "op": "replace_params",
                "node": "model",
                "params": [{"name": "seed", "value": 1}],
            },
        }
    )

    class Responses:
        def __init__(self) -> None:
            self.kwargs = None

        def parse(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                output_parsed=parsed,
                usage=SimpleNamespace(input_tokens=123, output_tokens=17),
            )

    responses = Responses()
    result = OpenAIStructuredClient(
        model="gpt-5.6-terra",
        reasoning_effort="low",
        client=SimpleNamespace(responses=responses),
    ).parse(LLMHypothesis, instructions="system", input_text="context")

    assert result.value == parsed
    assert result.usage.to_log() == {"in": 123, "out": 17}
    assert responses.kwargs["text_format"] is LLMHypothesis
    assert responses.kwargs["reasoning"] == {"effort": "low"}


def test_live_hypothesis_schema_contains_no_open_ended_object_maps() -> None:
    from openai.lib._pydantic import to_strict_json_schema

    schema = to_strict_json_schema(LLMHypothesis)

    def walk(value):
        if isinstance(value, dict):
            assert value.get("additionalProperties") is not True
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(schema)
    runtime = LLMHypothesis.model_validate(
        {
            "id": "h_wire",
            "target_node": "model",
            "rationale": "strict wire conversion",
            "method_source": "unit test",
            "expected_delta": 0.002,
            "expected_cost_minutes": 1,
            "patch": {
                "op": "replace_params",
                "node": "model",
                "params": [{"name": "seed", "value": 29}],
            },
        }
    ).to_runtime()
    assert runtime.patch.params == {"seed": 29}
