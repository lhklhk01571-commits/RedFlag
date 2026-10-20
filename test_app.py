"""
실제 API를 호출하지 않고 가짜 응답으로 토론 흐름을 검증하는 테스트.

실행: python -m unittest test_app -v
"""

import os
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

for _name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
    os.environ.setdefault(_name, "test-key")

import app  # noqa: E402

BRAND_WORDS = ["OpenAI", "Anthropic", "Google", "GPT", "Claude", "Gemini"]


class FakeBackends:
    """세 회사 SDK 클라이언트를 흉내 내며 모든 호출 인자를 기록."""

    def __init__(self, fail=None, empty_gpt=False):
        self.calls = []
        self.fail = fail or (lambda provider, call_no: False)
        self.empty_gpt = empty_gpt
        self._lock = threading.Lock()
        self._counter = 0
        self.judge_hook = None

        self.openai = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self._openai)))
        self.anthropic = SimpleNamespace(messages=SimpleNamespace(create=self._anthropic))
        self.google = SimpleNamespace(models=SimpleNamespace(generate_content=self._google))

    def _record(self, provider, system, user, kwargs):
        with self._lock:
            self._counter += 1
            call_no = len([c for c in self.calls if c["provider"] == provider]) + 1
            self.calls.append({"provider": provider, "system": system, "user": user, "kwargs": kwargs})
            text = f"발언 #{self._counter}"
        if self.fail(provider, call_no):
            raise RuntimeError(f"{provider} quota exceeded (call {call_no})")
        return text

    def _openai(self, **kwargs):
        messages = kwargs["messages"]
        system = messages[0]["content"] if messages[0]["role"] == "system" else ""
        text = self._record("gpt", system, messages[-1]["content"], kwargs)
        content = None if self.empty_gpt else text
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="length")],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
        )

    def _anthropic(self, **kwargs):
        is_judge = "system" not in kwargs
        if is_judge and self.judge_hook:
            self.judge_hook()
        text = self._record("claude", kwargs.get("system", ""), kwargs["messages"][-1]["content"], kwargs)
        if is_judge:
            text = "최종 통합 결론 본문"
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking="내부 사고 과정"),
                SimpleNamespace(type="text", text=text),
            ],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=200, output_tokens=80),
        )

    def _google(self, model, contents, config):
        text = self._record("gemini", config.system_instruction or "", contents, {"model": model, "config": config})
        return SimpleNamespace(
            text=text,
            candidates=[SimpleNamespace(finish_reason="STOP")],
            usage_metadata=SimpleNamespace(prompt_token_count=300, candidates_token_count=40, thoughts_token_count=60),
        )

    def debater_calls(self):
        return [c for c in self.calls if c["system"]]

    def judge_call(self):
        judges = [c for c in self.calls if not c["system"]]
        assert len(judges) == 1, judges
        return judges[0]


class DebateTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fake = FakeBackends()
        patches = [
            mock.patch.object(app, "LOG_DIR", self.tmp.name),
            mock.patch.object(app, "REASONING_EFFORT", "medium"),
            mock.patch.object(app, "JUDGE_KEY", "claude"),
            mock.patch.object(app, "JUDGE_MODEL_ID", app.ANTHROPIC_MODEL),
            mock.patch.object(app, "openai_client", self.fake.openai),
            mock.patch.object(app, "anthropic_client", self.fake.anthropic),
            mock.patch.object(app, "google_client", self.fake.google),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.client = app.app.test_client()

    def run_debate(self, rounds, topic="테스트 주제", reference=""):
        res = self.client.post("/api/debate", json={"topic": topic, "rounds": rounds, "reference": reference})
        self.assertEqual(res.status_code, 200, res.get_json())
        return res.get_json()

    def test_round1_debaters_answer_independently(self):
        self.run_debate(1)
        calls = self.fake.debater_calls()
        self.assertEqual(sorted(c["provider"] for c in calls), ["claude", "gemini", "gpt"])
        for c in calls:
            self.assertIn("초기 입장", c["system"])
            self.assertIn("(아직 발언 없음", c["user"])

    def test_later_rounds_see_only_previous_rounds(self):
        self.run_debate(3)
        calls = self.fake.debater_calls()
        self.assertEqual(len(calls), 9)
        for round_no in (2, 3):
            for c in calls[(round_no - 1) * 3: round_no * 3]:
                self.assertIn("1) 동의", c["system"])
                self.assertIn(f"[{round_no - 1}라운드", c["user"])
                self.assertNotIn(f"[{round_no}라운드", c["user"])

    def test_no_model_or_brand_names_reach_models(self):
        self.run_debate(3)
        texts = [c["system"] + c["user"] for c in self.fake.calls]
        names = BRAND_WORDS + [app.OPENAI_MODEL, app.ANTHROPIC_MODEL, app.GOOGLE_MODEL]
        for text in texts:
            for name in names:
                self.assertNotIn(name, text)

    def test_each_debater_knows_its_own_label(self):
        data = self.run_debate(1)
        provider_by_label = {v: k for k, v in data["model_labels"].items()}
        for c in self.fake.debater_calls():
            label = c["system"].split("토론자 ", 1)[1][0]
            self.assertEqual(provider_by_label[label], c["provider"])

    def test_red_team_rotates_from_round_two(self):
        data = self.run_debate(4)
        red = {}
        for e in data["history"]:
            if e["red_team"]:
                self.assertNotIn(e["round"], red)
                red[e["round"]] = e["model"]
        self.assertEqual(sorted(red), [2, 3, 4])
        self.assertEqual(sorted(red.values()), ["claude", "gemini", "gpt"])
        red_prompts = [c for c in self.fake.debater_calls() if "레드팀입니다" in c["system"]]
        self.assertEqual(len(red_prompts), 3)

    def test_failed_debater_is_flagged_and_excluded_from_context(self):
        self.fake.fail = lambda provider, call_no: provider == "gemini" and call_no == 1
        data = self.run_debate(2)

        self.assertEqual(len(data["failures"]), 1)
        failure = data["failures"][0]
        self.assertEqual(failure["round"], 1)
        self.assertIn("Google", failure["name"])
        self.assertIn("quota exceeded", failure["error"])

        judge_prompt = self.fake.judge_call()["user"]
        self.assertIn("(응답 실패 - 발언 없음)", judge_prompt)
        self.assertNotIn("quota exceeded", judge_prompt)
        for c in self.fake.debater_calls()[3:]:
            self.assertIn("(응답 실패 - 발언 없음)", c["user"])
            self.assertNotIn("quota exceeded", c["user"])

    def test_empty_response_is_treated_as_failure(self):
        self.fake.empty_gpt = True
        data = self.run_debate(1)
        gpt_entry = next(e for e in data["history"] if e["model"] == "gpt")
        self.assertIn("finish_reason=length", gpt_entry["error"])
        self.assertEqual(len(data["failures"]), 1)

    def test_reasoning_models_get_matching_parameters(self):
        self.run_debate(1)
        by_provider = {c["provider"]: c["kwargs"] for c in self.fake.debater_calls()}

        gpt = by_provider["gpt"]
        self.assertEqual(gpt["max_completion_tokens"], 16000)
        self.assertEqual(gpt["reasoning_effort"], "medium")
        self.assertNotIn("max_tokens", gpt)
        self.assertNotIn("temperature", gpt)

        claude = by_provider["claude"]
        self.assertEqual(claude["max_tokens"], 16000)
        self.assertEqual(claude["output_config"], {"effort": "medium"})

        config = by_provider["gemini"]["config"]
        self.assertEqual(config.max_output_tokens, 32768)
        self.assertEqual(config.thinking_config.thinking_level.value, "MEDIUM")
        self.assertIn("토론자", config.system_instruction)

    def test_empty_reasoning_effort_sends_no_reasoning_parameters(self):
        with mock.patch.object(app, "REASONING_EFFORT", ""):
            self.run_debate(1)
        by_provider = {c["provider"]: c["kwargs"] for c in self.fake.debater_calls()}
        self.assertNotIn("reasoning_effort", by_provider["gpt"])
        self.assertNotIn("output_config", by_provider["claude"])
        self.assertIsNone(by_provider["gemini"]["config"].thinking_config)

    def test_claude_thinking_block_is_not_shown(self):
        data = self.run_debate(1)
        claude_entry = next(e for e in data["history"] if e["model"] == "claude")
        self.assertTrue(claude_entry["content"].startswith("발언 #"))
        self.assertEqual(data["verdict"], "최종 통합 결론 본문")

    def test_reference_reaches_debaters_and_judge(self):
        reference = "SMP 단가 120원/kWh" + "가" * 6000
        self.run_debate(1, reference=reference)
        for c in self.fake.calls:
            self.assertIn("[사용자 제공 전제 자료]", c["user"])
            self.assertIn("SMP 단가 120원/kWh", c["user"])
            self.assertNotIn("가" * (app.MAX_REFERENCE_CHARS + 1), c["user"])

    def test_log_is_saved_before_judge_and_after(self):
        seen = {}

        def check_log_before_judge():
            files = os.listdir(self.tmp.name)
            seen["files"] = files
            with open(os.path.join(self.tmp.name, files[0]), encoding="utf-8") as f:
                seen["content"] = f.read()

        self.fake.judge_hook = check_log_before_judge
        data = self.run_debate(2, topic="1MW 태양광, 지금 시작해도 될까?")

        self.assertEqual(len(seen["files"]), 1)
        self.assertIn("(토론 진행 중", seen["content"])
        self.assertNotIn("?", data["log_file"])

        with open(os.path.join(self.tmp.name, data["log_file"]), encoding="utf-8") as f:
            log = f.read()
        self.assertIn("최종 통합 결론 본문", log)
        for model_id in (app.OPENAI_MODEL, app.ANTHROPIC_MODEL, app.GOOGLE_MODEL):
            self.assertIn(model_id, log)
        self.assertIn("익명 매핑", log)
        self.assertIn("추론 강도: medium", log)
        self.assertIn("전체 토큰", log)

    def test_usage_totals_include_thinking_and_judge(self):
        data = self.run_debate(2)
        # 라운드당 gpt 100/50, claude 200/80, gemini 300/(40+60) + 통합(claude) 200/80
        self.assertEqual(data["usage_total"], {"input": 2 * 600 + 200, "output": 2 * 230 + 80})

    def test_judge_failure_is_reported_separately(self):
        self.fake.fail = lambda provider, call_no: provider == "claude" and call_no == 2
        data = self.run_debate(1)
        self.assertEqual(data["verdict"], "")
        self.assertIn("quota exceeded", data["judge_error"])

    def test_request_validation(self):
        res = self.client.post("/api/debate", json={"topic": "  ", "rounds": 3})
        self.assertEqual(res.status_code, 400)
        res = self.client.post("/api/debate", data="not json", content_type="application/json")
        self.assertEqual(res.status_code, 400)
        self.assertEqual(self.run_debate("99")["max_rounds"], 5)
        self.assertEqual(self.run_debate("abc")["max_rounds"], 3)
        self.assertEqual(self.run_debate(0)["max_rounds"], 1)


if __name__ == "__main__":
    unittest.main()
