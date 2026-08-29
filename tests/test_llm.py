from __future__ import annotations

from types import SimpleNamespace

from agent.llm import OpenAIStructuredClient
from agent.propose import Hypothesis


def test_responses_structured_output_and_usage_are_accounted() -> None:
    parsed = Hypothesis.model_validate(
        {
            "id": "h_sdk",
            "target_node": "model",
            "rationale": "bounded test",
            "method_source": "unit test",
            "expected_delta": 0.003,
            "expected_cost_minutes": 1,
            "patch": {"op": "replace_params", "node": "model", "params": {"seed": 1}},
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
    ).parse(Hypothesis, instructions="system", input_text="context")

    assert result.value == parsed
    assert result.usage.to_log() == {"in": 123, "out": 17}
    assert responses.kwargs["text_format"] is Hypothesis
    assert responses.kwargs["reasoning"] == {"effort": "low"}

