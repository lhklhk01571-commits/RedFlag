"""
AI 토론 경쟁 모델 (AI Debate Arena)
====================================
OpenAI, Anthropic(Claude), Google(Gemini) 세 AI가 익명(토론자 A/B/C)으로 주어진 주제에 대해
N라운드에 걸쳐 서로의 의견을 수용·반영하며 제안을 발전시키고,
마지막에 통합 AI가 수렴된 내용을 종합해 최종 통합 결론을 도출하는 단일 파일 웹앱.

실행 방법:
    1) .env.example을 복사해 .env를 만들고 OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY 설정
    2) pip install -r requirements.txt
    3) python app.py
    4) 브라우저에서 http://localhost:5000 접속
"""

import os
import random
import operator
import traceback
from datetime import datetime
from typing import Annotated, Dict, List, Optional, Tuple, TypedDict

from flask import Flask, render_template_string, request, jsonify
from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------------------
# AI 클라이언트 초기화
# ------------------------------------------------------------------
import openai
import anthropic
from google import genai
from google.genai import types as genai_types

_missing_keys = [k for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY") if not os.getenv(k)]
if _missing_keys:
    raise SystemExit(f".env에 다음 API 키가 없습니다: {', '.join(_missing_keys)} (.env.example 참고)")

# 3사 모델을 같은 가격대로 맞춰야 토론 결과가 모델 성능차가 아닌 논리/근거의 질로 갈립니다.
# (2026-10 기준 세 모델 모두 100만 토큰당 입력 $2 / 출력 $10. 단, Gemini 4 Argon은 12/1부터 $4 / $20)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-6.1-sol")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-4-argon")

# 세 모델의 추론(thinking) 강도를 같은 수준으로 맞춥니다. 세 회사가 공통으로 지원하는 값만 허용.
# 추론 기능이 없는 구형 모델(gpt-4o 등)을 쓸 때는 .env에서 REASONING_EFFORT= 로 비워두세요.
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "medium").strip().lower()
if REASONING_EFFORT not in ("", "low", "medium", "high"):
    raise SystemExit("REASONING_EFFORT는 low / medium / high 중 하나이거나 빈 값이어야 합니다.")

# 통합 담당 AI ("gpt" / "claude" / "gemini"). JUDGE_MODEL을 지정하면 토론자와 다른 모델로 통합합니다.
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "claude")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "")

# 출력 한도에는 사고 토큰도 포함되므로 실제 답변 길이보다 훨씬 넉넉해야 응답이 잘리지 않습니다.
# Gemini 4 Argon은 32K 미만 한도에서 사고만 하다 빈 응답을 낼 수 있어 Google 권장값을 따릅니다.
OUTPUT_TOKEN_LIMIT = {"gpt": 16000, "claude": 16000, "gemini": 32768}
# Gemini 4 Argon 가이드 권장값 (호출당 최소 20분)
REQUEST_TIMEOUT_SECONDS = 1200
MAX_REFERENCE_CHARS = 5000

openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=REQUEST_TIMEOUT_SECONDS)
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), timeout=REQUEST_TIMEOUT_SECONDS)
google_client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY"),
    http_options=genai_types.HttpOptions(
        timeout=REQUEST_TIMEOUT_SECONDS * 1000,
        retry_options=genai_types.HttpRetryOptions(attempts=3),
    ),
)

# 토론 로그 자동 저장 폴더 (API 호출 없이 로컬 파일로만 저장 - 토큰 소모 없음)
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

from langgraph.graph import StateGraph, START, END

# ------------------------------------------------------------------
# 상태(State) 정의
# ------------------------------------------------------------------
class DebateState(TypedDict):
    topic: str
    reference: str
    started_at: str
    today: str
    max_rounds: int
    current_round: int
    # 한 라운드의 세 발언이 병렬로 추가되므로 리스트를 이어붙이는 reducer가 필요
    history: Annotated[List[Dict], operator.add]
    label_map: Dict[str, str]
    red_team_order: List[str]
    log_file: str
    verdict: str
    judge_error: str
    judge_usage: Dict[str, int]


MODEL_KEYS = ["gpt", "claude", "gemini"]

DEBATER_MODELS = {"gpt": OPENAI_MODEL, "claude": ANTHROPIC_MODEL, "gemini": GOOGLE_MODEL}

ROLE_NAMES = {
    "gpt": f"OpenAI · {OPENAI_MODEL}",
    "claude": f"Anthropic · {ANTHROPIC_MODEL}",
    "gemini": f"Google · {GOOGLE_MODEL}",
}

JUDGE_KEY = JUDGE_PROVIDER if JUDGE_PROVIDER in MODEL_KEYS else "claude"
JUDGE_MODEL_ID = JUDGE_MODEL or DEBATER_MODELS[JUDGE_KEY]

DEBATER_COMMON = (
    '당신은 AI 토론 패널의 토론자 {label}입니다. 토론 주제: "{topic}"\n'
    "오늘 날짜: {today}\n"
    "토론자는 A, B, C로만 지칭하고, 자신이나 다른 토론자가 어떤 AI 모델·회사인지 언급하지 마세요.\n"
    "가격·금리·정책·시장 규모처럼 학습 시점 이후 바뀌었을 수 있는 수치에는 '(확인 필요)'를 붙이고, "
    "사용자가 제공한 전제 자료가 있으면 그 수치를 우선 사용하세요.\n"
)

# 첫 발언: 다른 토론자의 발언을 보지 않은 상태에서 초기 입장/제안을 제시
DEBATER_SYSTEM_TEMPLATE_FIRST = DEBATER_COMMON + (
    "이 주제에 대한 당신의 초기 입장과 제안을, 핵심 근거와 함께 제시하세요.\n"
    "다른 토론자들도 지금 동시에 각자의 초기 입장을 작성하고 있으며, 이후 라운드에서 서로의 의견을 "
    "검토하며 이 제안을 발전시켜 나갈 것이므로, 지금은 명확하고 검증 가능한 출발점이 되는 제안을 작성하세요.\n"
    "응답은 한국어로, 400자 이내로 작성하세요."
)

# 2라운드 이후: 동의/보완 -> 갱신된 제안 -> 유지하는 이견 순서로 의견을 수렴·고도화
DEBATER_SYSTEM_TEMPLATE_NEXT = DEBATER_COMMON + (
    "이 토론의 목표는 승패를 가르는 것이 아니라, 서로의 의견 중 합리적인 부분을 수용하며 "
    "해결책을 점점 더 정교하고 실행 가능한 형태로 발전시키는 것입니다.\n"
    "아래 [지금까지의 토론 내용]을 검토한 뒤, 다음 4가지를 순서대로 작성하세요.\n"
    "1) 동의: 다른 토론자의 의견 중 합리적이라고 판단되는 부분과 그 이유\n"
    "2) 보완/반박: 동의하기 어렵거나 보완이 필요한 부분과 그 이유\n"
    "3) 갱신된 제안: 1)과 2)를 반영하여, 이번 라운드 기준 당신의 제안/결론을 한층 더 구체화하고 "
    "발전시켜 다시 제시 (이전 라운드 내용을 그대로 반복하지 말 것)\n"
    "4) 유지하는 이견: 여전히 동의하지 않는 부분 (없으면 '없음')\n"
    "새로운 근거나 논리 없이 다수 의견에 맞추려고 입장을 바꾸지 마세요. 입장을 바꿨다면 "
    "어느 토론자의 어떤 논거 때문인지 밝히세요.\n"
    "응답은 한국어로, 500자 이내로 작성하세요."
)

RED_TEAM_INSTRUCTION = (
    "\n\n[이번 라운드 특별 역할: 레드팀]\n"
    "이번 라운드에서 당신은 레드팀입니다. 형식은 그대로 지키되, 2)에서 지금 형성되고 있는 공통 결론의 "
    "가장 약한 고리, 숨은 전제, 실패 시나리오를 최소 2가지 집중적으로 공격하세요. "
    "1)은 꼭 필요한 부분만 짧게 쓰세요."
)

JUDGE_SYSTEM_TEMPLATE = (
    '당신은 AI 토론 결과를 종합하는 통합(Synthesis) 담당자입니다. 토론 주제: "{topic}"\n'
    "오늘 날짜: {today}\n\n"
    "{reference_block}"
    "아래는 토론자 A, B, C 세 명이 {rounds}라운드에 걸쳐, 매 라운드 서로의 의견 중 합리적인 "
    "부분을 수용하며 자신의 제안을 갱신해온 토론 전체 내용입니다. 마지막 라운드의 제안일수록 "
    "더 발전·수렴된 상태입니다. '(레드팀)' 표시가 붙은 발언은 공통 결론의 약점을 집중 공격하는 "
    "역할을 맡은 발언이고, '응답 실패'로 표시된 발언은 오류로 누락된 것입니다.\n"
    "(어떤 AI 모델인지는 의도적으로 가렸습니다. 특정 모델/브랜드에 대한 선호나 "
    "편견 없이, 오직 내용의 논리·근거만으로 종합하세요.)\n\n"
    "{full_history}\n\n"
    "당신의 역할은 '승자 1명을 뽑는 것'이 아니라, 토론을 통해 수렴된 내용을 종합하여 "
    "가장 합리적이고 실행 가능한 하나의 통합안을 제시하는 것입니다. "
    "다음 형식의 한국어 보고서를 작성하세요.\n"
    "1. 라운드별 수렴 과정 요약 (각 토론자가 어떤 의견을 수용/반영하며 입장을 어떻게 "
    "발전시켰는지 핵심만 간결하게)\n"
    "2. 최종 통합 결론 (3자의 마지막 라운드 제안을 종합한 하나의 최적안). 핵심 주장마다 "
    "[3/3 합의], [2/3 합의], [이견] 중 하나로 합의 수준을 표시하고, 소수 의견은 지우지 말고 "
    "근거와 함께 남기세요. 레드팀 지적 중 해소되지 않은 것은 남은 리스크로 명시하고, 새 근거 없이 "
    "서로 맞춰주느라 수렴한 것으로 보이는 부분이 있으면 지적하세요.\n"
    "3. 검증이 필요한 수치·전제 (토론에서 출처 없이 쓰인 수치, 학습 시점 이후 바뀌었을 수 있는 "
    "가격·금리·정책 등을 목록으로)\n"
    "4. 실행 가능한 결론 (Action Item, 2~3개)\n"
    "5. 사용자가 현실적으로 스스로 점검하고 깊이 고민해 보아야 할 핵심 질문 3가지\n"
    "   (토론 주제와 직접 관련된, 의사결정 전 반드시 자문해야 할 날카로운 질문으로 작성. "
    "일반론적 질문 금지, 사용자의 구체적 상황을 가정한 질문으로 작성)\n\n"
    "[분량 지침] 전체 보고서는 공백 포함 약 2000자 분량으로, 각 항목을 충분히 "
    "상세하고 구체적으로 서술하세요. 너무 짧게 요약하지 말고, 근거와 예시를 곁들여 "
    "심도 있게 작성하세요."
)


def red_team_for(state: DebateState, round_no: int) -> Optional[str]:
    """2라운드부터 매 라운드 한 명씩 돌아가며 레드팀을 맡김 (순서는 토론마다 무작위)."""
    if round_no < 2:
        return None
    order = state["red_team_order"]
    return order[(round_no - 2) % len(order)]


def sorted_history(state: DebateState) -> List[Dict]:
    return sorted(state["history"], key=lambda e: (e["round"], state["label_map"][e["model"]]))


def format_history(state: DebateState, entries: List[Dict]) -> str:
    """토론 기록을 익명 라벨(A/B/C)로만 표기 - 토론자와 통합 AI 모두 실제 모델명을 보지 않음."""
    label_map = state["label_map"]
    lines = []
    for entry in sorted(entries, key=lambda e: (e["round"], label_map[e["model"]])):
        role = " (레드팀)" if entry.get("red_team") else ""
        body = "(응답 실패 - 발언 없음)" if entry.get("error") else entry["content"]
        lines.append(f"[{entry['round']}라운드 - 토론자 {label_map[entry['model']]}{role}]\n{body}")
    return "\n\n".join(lines)


def reference_block(state: DebateState) -> str:
    reference = state.get("reference", "").strip()
    return f"[사용자 제공 전제 자료]\n{reference}\n\n" if reference else ""


def build_debater_prompt(state: DebateState, model_key: str) -> str:
    round_no = state["current_round"]
    template = DEBATER_SYSTEM_TEMPLATE_FIRST if round_no == 1 else DEBATER_SYSTEM_TEMPLATE_NEXT
    prompt = template.format(label=state["label_map"][model_key], topic=state["topic"], today=state["today"])
    if red_team_for(state, round_no) == model_key:
        prompt += RED_TEAM_INSTRUCTION
    return prompt


def build_debater_message(state: DebateState) -> str:
    """같은 라운드의 다른 발언은 빼고, 직전 라운드까지의 기록만 전달 (발언 순서 편향 방지)."""
    previous = [e for e in state["history"] if e["round"] < state["current_round"]]
    context = format_history(state, previous) if previous else "(아직 발언 없음 - 첫 라운드입니다)"
    return f"{reference_block(state)}[지금까지의 토론 내용]\n{context}"


def build_judge_prompt(state: DebateState) -> str:
    return JUDGE_SYSTEM_TEMPLATE.format(
        topic=state["topic"],
        today=state["today"],
        reference_block=reference_block(state),
        rounds=state["max_rounds"],
        full_history=format_history(state, state["history"]),
    )


# ------------------------------------------------------------------
# 회사별 API 호출 (응답 텍스트, 토큰 사용량) 반환. 빈 응답은 실패로 처리.
# ------------------------------------------------------------------
def call_gpt(model: str, system_prompt: str, user_prompt: str) -> Tuple[str, Dict[str, int]]:
    messages = [{"role": "user", "content": user_prompt}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    extra = {"reasoning_effort": REASONING_EFFORT} if REASONING_EFFORT else {}
    resp = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=OUTPUT_TOKEN_LIMIT["gpt"],
        **extra,
    )
    choice = resp.choices[0]
    text = (choice.message.content or "").strip()
    if not text:
        raise RuntimeError(f"빈 응답 (finish_reason={choice.finish_reason})")
    usage = {"input": resp.usage.prompt_tokens, "output": resp.usage.completion_tokens} if resp.usage else {}
    return text, usage


def call_claude(model: str, system_prompt: str, user_prompt: str) -> Tuple[str, Dict[str, int]]:
    extra = {}
    if system_prompt:
        extra["system"] = system_prompt
    if REASONING_EFFORT:
        extra["output_config"] = {"effort": REASONING_EFFORT}
    resp = anthropic_client.messages.create(
        model=model,
        max_tokens=OUTPUT_TOKEN_LIMIT["claude"],
        messages=[{"role": "user", "content": user_prompt}],
        **extra,
    )
    # 사고(thinking) 블록이 text 블록보다 먼저 올 수 있으므로 text 블록만 모음
    text = "".join(block.text for block in resp.content if block.type == "text").strip()
    if not text:
        raise RuntimeError(f"빈 응답 (stop_reason={resp.stop_reason})")
    return text, {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens}


def call_gemini(model: str, system_prompt: str, user_prompt: str) -> Tuple[str, Dict[str, int]]:
    config = genai_types.GenerateContentConfig(
        system_instruction=system_prompt or None,
        max_output_tokens=OUTPUT_TOKEN_LIMIT["gemini"],
        thinking_config=genai_types.ThinkingConfig(thinking_level=REASONING_EFFORT) if REASONING_EFFORT else None,
    )
    resp = google_client.models.generate_content(model=model, contents=user_prompt, config=config)
    text = (resp.text or "").strip()
    if not text:
        reason = resp.candidates[0].finish_reason if resp.candidates else "후보 응답 없음"
        raise RuntimeError(f"빈 응답 (finish_reason={reason})")
    meta = resp.usage_metadata
    usage = {}
    if meta:
        usage = {
            "input": meta.prompt_token_count or 0,
            "output": (meta.candidates_token_count or 0) + (meta.thoughts_token_count or 0),
        }
    return text, usage


CALLERS = {"gpt": call_gpt, "claude": call_claude, "gemini": call_gemini}


# ------------------------------------------------------------------
# LangGraph 노드
# ------------------------------------------------------------------
def start_round(state: DebateState) -> Dict:
    return {"current_round": state["current_round"] + 1}


def make_debater_node(model_key: str):
    def debater_node(state: DebateState) -> Dict:
        round_no = state["current_round"]
        entry = {
            "round": round_no,
            "model": model_key,
            "content": "",
            "error": None,
            "usage": {},
            "red_team": red_team_for(state, round_no) == model_key,
        }
        try:
            entry["content"], entry["usage"] = CALLERS[model_key](
                DEBATER_MODELS[model_key], build_debater_prompt(state, model_key), build_debater_message(state)
            )
        except Exception as e:  # noqa: BLE001
            entry["error"] = str(e)
        return {"history": [entry]}

    return debater_node


def end_round(state: DebateState) -> Dict:
    # 라운드마다 중간 저장 - 도중에 오류가 나도 그때까지의 토론 기록은 남음
    save_debate_log_safely(state)
    return {}


def should_continue(state: DebateState) -> str:
    if state["current_round"] < state["max_rounds"]:
        return "continue"
    return "judge"


def judge_node(state: DebateState) -> Dict:
    update = {"verdict": "", "judge_error": "", "judge_usage": {}}
    try:
        update["verdict"], update["judge_usage"] = CALLERS[JUDGE_KEY](JUDGE_MODEL_ID, "", build_judge_prompt(state))
    except Exception as e:  # noqa: BLE001
        update["judge_error"] = str(e)
    save_debate_log_safely({**state, **update})
    return update


# ------------------------------------------------------------------
# 집계 / 토론 로그 자동 저장 (로컬 파일 I/O — API/토큰 추가 소모 없음)
# ------------------------------------------------------------------
def collect_failures(state: DebateState) -> List[Dict]:
    return [
        {
            "round": e["round"],
            "name": ROLE_NAMES[e["model"]],
            "label": state["label_map"][e["model"]],
            "error": e["error"],
        }
        for e in sorted_history(state)
        if e.get("error")
    ]


def usage_totals(state: DebateState) -> Dict[str, int]:
    usages = [e.get("usage") or {} for e in state["history"]] + [state.get("judge_usage") or {}]
    return {
        "input": sum(u.get("input", 0) for u in usages),
        "output": sum(u.get("output", 0) for u in usages),
    }


def make_log_filename(topic: str, started: datetime) -> str:
    safe_topic = "".join(c for c in topic if c.isalnum() or c in (" ", "_", "-")).strip()[:30]
    return f"{started.strftime('%Y%m%d_%H%M%S')}_{safe_topic or 'debate'}.md"


def save_debate_log(state: DebateState) -> None:
    """토론 상태를 logs/ 폴더의 마크다운 파일로 저장 (같은 파일을 라운드마다 덮어씀)."""
    label_map = state["label_map"]
    mapping = " · ".join(f"토론자 {label_map[k]} = {ROLE_NAMES[k]}" for k in sorted(MODEL_KEYS, key=label_map.get))
    lines = [
        f"# 토론 주제: {state['topic']}",
        "",
        f"- 일시: {state['started_at']}",
        f"- 라운드 수: {state['max_rounds']}",
        f"- 추론 강도: {REASONING_EFFORT or '(지정 안 함)'}",
        f"- 통합 담당: {JUDGE_MODEL_ID}",
        f"- 익명 매핑: {mapping}",
        "",
    ]
    if state.get("reference"):
        lines += ["## 사용자 제공 전제 자료", "", state["reference"], ""]
    lines += ["---", ""]

    for entry in sorted_history(state):
        role = " · 레드팀" if entry.get("red_team") else ""
        lines.append(
            f"## {entry['round']}라운드 - 토론자 {label_map[entry['model']]} ({ROLE_NAMES[entry['model']]}){role}"
        )
        lines.append("")
        lines.append(f"> 응답 실패: {entry['error']}" if entry.get("error") else entry["content"])
        if entry.get("usage"):
            lines.append("")
            lines.append(f"_토큰: 입력 {entry['usage']['input']:,} · 출력 {entry['usage']['output']:,}_")
        lines.append("")

    lines += ["---", "", "## 최종 통합 결론", ""]
    if state.get("verdict"):
        lines.append(state["verdict"])
    elif state.get("judge_error"):
        lines.append(f"> 통합 실패: {state['judge_error']}")
    else:
        lines.append("(토론 진행 중 - 아직 결론 없음)")

    totals = usage_totals(state)
    lines += ["", "---", "", f"- 전체 토큰: 입력 {totals['input']:,} · 출력 {totals['output']:,} (출력에는 사고 토큰 포함)", ""]

    with open(os.path.join(LOG_DIR, state["log_file"]), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_debate_log_safely(state: DebateState) -> None:
    """로그 저장에 실패해도 토론 진행에는 영향이 없도록 처리."""
    try:
        save_debate_log(state)
    except Exception:  # noqa: BLE001
        traceback.print_exc()


# ------------------------------------------------------------------
# LangGraph 그래프 구성
#   start_round -> (gpt | claude | gemini 동시 발언) -> end_round -> 다음 라운드 또는 judge
# ------------------------------------------------------------------
workflow = StateGraph(DebateState)
workflow.add_node("start_round", start_round)
for _key in MODEL_KEYS:
    workflow.add_node(_key, make_debater_node(_key))
workflow.add_node("end_round", end_round)
workflow.add_node("judge", judge_node)

workflow.add_edge(START, "start_round")
for _key in MODEL_KEYS:
    workflow.add_edge("start_round", _key)
workflow.add_edge(MODEL_KEYS, "end_round")
workflow.add_conditional_edges("end_round", should_continue, {"continue": "start_round", "judge": "judge"})
workflow.add_edge("judge", END)

debate_graph = workflow.compile()

# ------------------------------------------------------------------
# Flask 웹 앱
# ------------------------------------------------------------------
app = Flask(__name__)

INDEX_HTML = """
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI 토론 경쟁 모델</title>
<style>
  :root {
    --bg: #0f1115; --panel: #181b22; --border: #2a2f3a;
    --text: #e6e8ec; --muted: #9aa3b2; --accent: #6ea8fe;
    --gpt: #74c69d; --claude: #f0a868; --gemini: #8b9bf4; --judge: #ff6b81;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: -apple-system, "Segoe UI", "Malgun Gothic", sans-serif;
    background: var(--bg); color: var(--text); padding: 24px;
  }
  .wrap { max-width: 860px; margin: 0 auto; }
  h1 { font-size: 22px; margin-bottom: 4px; }
  .sub { color: var(--muted); margin-bottom: 24px; font-size: 14px; }
  .card {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 12px; padding: 20px; margin-bottom: 16px;
  }
  label { display: block; margin-bottom: 6px; font-size: 13px; color: var(--muted); }
  input[type=text], select, textarea {
    width: 100%; padding: 10px 12px; border-radius: 8px;
    border: 1px solid var(--border); background: #11141a; color: var(--text);
    font-size: 14px; margin-bottom: 14px; font-family: inherit;
  }
  textarea { min-height: 84px; resize: vertical; line-height: 1.5; }
  .hint { color: var(--muted); font-size: 12px; margin: -8px 0 14px; }
  button {
    background: var(--accent); color: #0f1115; font-weight: 600;
    border: none; padding: 12px 20px; border-radius: 8px; cursor: pointer; font-size: 14px;
  }
  button:disabled { opacity: 0.6; cursor: not-allowed; }
  .copy-btn {
    background: transparent; color: var(--accent); border: 1px solid var(--accent);
    padding: 4px 10px; font-size: 12px; font-weight: 600; border-radius: 6px;
    float: right; margin-top: -2px;
  }
  .verdict-head { display: flex; justify-content: space-between; align-items: center; }
  .log-note { color: var(--muted); font-size: 12px; margin-top: 10px; }
  .warn {
    background: #33260f; border: 1px solid #a8741a; color: #ffd28a;
    border-radius: 8px; padding: 10px 12px; font-size: 13px; line-height: 1.6; margin: 8px 0 12px;
  }
  .badge {
    display: inline-block; padding: 2px 8px; border-radius: 6px;
    font-size: 12px; font-weight: 700; margin-bottom: 8px;
  }
  .badge.gpt { background: var(--gpt); color: #06281c; }
  .badge.claude { background: var(--claude); color: #2b1700; }
  .badge.gemini { background: var(--gemini); color: #161b40; }
  .badge.judge { background: var(--judge); color: #2b0009; }
  .badge.red { background: #ff6b6b; color: #2b0000; }
  .badge.fail { background: #5c1d1d; color: #ffb3b3; }
  .entry { padding: 12px 0; border-bottom: 1px solid var(--border); }
  .entry:last-child { border-bottom: none; }
  .entry-content { white-space: pre-wrap; line-height: 1.6; font-size: 14px; }
  .entry-content.failed { color: var(--muted); font-style: italic; }
  .verdict-box { white-space: pre-wrap; line-height: 1.7; font-size: 15px; }
  details summary { cursor: pointer; color: var(--accent); font-size: 14px; padding: 6px 0; }
  #loading { display: none; color: var(--muted); font-size: 14px; }
  .round-tag { color: var(--muted); font-size: 12px; margin-left: 6px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>🤖 AI 토론 경쟁 모델</h1>
  <div class="sub">OpenAI · Anthropic · Google 세 AI가 익명(토론자 A/B/C)으로 동시에 발언하며 의견을 발전시키고, 최종 통합 결론을 도출합니다.</div>

  <div class="card">
    <label for="topic">토론 주제</label>
    <input type="text" id="topic" placeholder="예: 1MW 태양광 발전 사업, 지금 시작하는 것이 합리적인가?">

    <label for="reference">전제 자료 (선택)</label>
    <textarea id="reference" maxlength="5000" placeholder="예: 부지 임대료 연 2,000만 원, 최근 SMP 단가, 자기자본 3억 원, 대출 금리 5% ..."></textarea>
    <div class="hint">내 상황의 수치나 최신 가격·정책을 적어두면 토론자와 통합 AI가 모두 이 자료를 우선 근거로 사용합니다.</div>

    <label for="rounds">토론 라운드 수</label>
    <select id="rounds">
      <option value="1">1라운드</option>
      <option value="2">2라운드</option>
      <option value="3" selected>3라운드 (기본)</option>
      <option value="4">4라운드</option>
      <option value="5">5라운드</option>
    </select>

    <button id="runBtn" onclick="runDebate()">토론 시작</button>
    <span id="loading">⏳ AI들이 토론 중입니다... (라운드 수에 따라 수 분 소요)</span>
  </div>

  <div id="result"></div>
</div>

<script>
async function runDebate() {
  const topic = document.getElementById('topic').value.trim();
  const reference = document.getElementById('reference').value.trim();
  const rounds = document.getElementById('rounds').value;
  if (!topic) { alert('토론 주제를 입력하세요.'); return; }

  const btn = document.getElementById('runBtn');
  const loading = document.getElementById('loading');
  const result = document.getElementById('result');
  btn.disabled = true; loading.style.display = 'inline'; result.innerHTML = '';

  try {
    const res = await fetch('/api/debate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({topic, reference, rounds})
    });
    const data = await res.json();
    if (data.error) {
      result.innerHTML = `<div class="card">오류: ${escapeHtml(data.error)}</div>`;
      return;
    }

    const names = data.role_names || {};
    const labels = data.model_labels || {};

    let logHtml = '';
    data.history.forEach(e => {
      const redTag = e.red_team ? ' <span class="badge red">레드팀</span>' : '';
      const failTag = e.error ? ' <span class="badge fail">응답 실패</span>' : '';
      const body = e.error
        ? `<div class="entry-content failed">${escapeHtml(e.error)}</div>`
        : `<div class="entry-content">${escapeHtml(e.content)}</div>`;
      logHtml += `<div class="entry">
        <span class="badge ${e.model}">${escapeHtml(names[e.model] || e.model)}</span>${redTag}${failTag}
        <span class="round-tag">${e.round}라운드 · 토론자 ${escapeHtml(labels[e.model] || '?')}</span>
        ${body}
      </div>`;
    });

    let warnHtml = '';
    if (data.failures && data.failures.length) {
      const items = data.failures.map(f => `${f.round}라운드 ${escapeHtml(f.name)}`).join(', ');
      warnHtml = `<div class="warn">⚠️ 일부 발언이 오류로 빠졌습니다: ${items}<br>
        이 결론은 빠진 발언 없이 작성됐으므로, 참여한 AI가 3개보다 적었다는 점을 감안해서 보세요.
        오류 내용은 아래 상세 로그에서 확인할 수 있습니다.</div>`;
    }

    const verdictHtml = data.judge_error
      ? `<div class="warn">❌ 최종 통합 결론 작성에 실패했습니다: ${escapeHtml(data.judge_error)}</div>`
      : `<div class="verdict-box" id="verdictText">${escapeHtml(data.verdict)}</div>`;
    const copyBtn = data.judge_error
      ? ''
      : '<button class="copy-btn" id="copyBtn" onclick="copyVerdict()">📋 복사</button>';

    const logNote = data.log_file
      ? `<div class="log-note">📁 토론 로그 저장됨: logs/${escapeHtml(data.log_file)}</div>`
      : '';

    const u = data.usage_total || {input: 0, output: 0};
    const usageNote = `<div class="log-note">🔢 토큰 사용량: 입력 ${u.input.toLocaleString()} · 출력 ${u.output.toLocaleString()} (출력에는 사고 토큰 포함)</div>`;

    let legendNote = '';
    if (data.label_map) {
      const parts = Object.entries(data.label_map).sort().map(([k, v]) => `${k} = ${escapeHtml(v)}`);
      legendNote = `<div class="log-note">🏷️ 익명 토론자 매핑 (토론자와 통합 AI에게는 비공개로 진행됨): ${parts.join(' · ')}</div>`;
    }

    result.innerHTML = `
      <div class="card">
        <div class="verdict-head">
          <span class="badge judge">최종 통합 결론</span>
          ${copyBtn}
        </div>
        ${warnHtml}
        ${verdictHtml}
        ${logNote}
        ${usageNote}
        ${legendNote}
      </div>
      <div class="card">
        <details><summary>토론 상세 로그 보기 (${data.history.length}개 발언)</summary>
          <div style="margin-top:12px">${logHtml}</div>
        </details>
      </div>
    `;
  } catch (err) {
    result.innerHTML = `<div class="card">요청 중 오류가 발생했습니다: ${escapeHtml(String(err))}</div>`;
  } finally {
    btn.disabled = false; loading.style.display = 'none';
  }
}

function copyVerdict() {
  const el = document.getElementById('verdictText');
  const text = el.innerText;
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.getElementById('copyBtn');
    const original = btn.textContent;
    btn.textContent = '✅ 복사됨';
    setTimeout(() => { btn.textContent = original; }, 1500);
  }).catch(() => {
    alert('클립보드 복사에 실패했습니다. 직접 선택해서 복사해주세요.');
  });
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/api/debate", methods=["POST"])
def api_debate():
    data = request.get_json(force=True, silent=True)
    if not isinstance(data, dict):
        data = {}
    topic = (data.get("topic") or "").strip()
    reference = (data.get("reference") or "").strip()[:MAX_REFERENCE_CHARS]
    try:
        rounds = int(data.get("rounds", 3))
    except (TypeError, ValueError):
        rounds = 3
    rounds = max(1, min(rounds, 5))

    if not topic:
        return jsonify({"error": "토론 주제를 입력해주세요."}), 400

    # 브랜드/자기선호 편향 방지: 토론 시작 시점부터 gpt/claude/gemini -> A/B/C 무작위 배정
    labels = ["A", "B", "C"]
    random.shuffle(labels)
    red_team_order = MODEL_KEYS[:]
    random.shuffle(red_team_order)
    started = datetime.now()

    initial_state: DebateState = {
        "topic": topic,
        "reference": reference,
        "started_at": started.strftime("%Y-%m-%d %H:%M:%S"),
        "today": started.strftime("%Y-%m-%d"),
        "max_rounds": rounds,
        "current_round": 0,
        "history": [],
        "label_map": dict(zip(MODEL_KEYS, labels)),
        "red_team_order": red_team_order,
        "log_file": make_log_filename(topic, started),
        "verdict": "",
        "judge_error": "",
        "judge_usage": {},
    }

    try:
        result = debate_graph.invoke(initial_state)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    label_map = result["label_map"]
    log_saved = os.path.exists(os.path.join(LOG_DIR, result["log_file"]))
    return jsonify(
        {
            "topic": result["topic"],
            "max_rounds": result["max_rounds"],
            "verdict": result["verdict"],
            "judge_error": result["judge_error"],
            "history": sorted_history(result),
            "failures": collect_failures(result),
            "usage_total": usage_totals(result),
            "role_names": ROLE_NAMES,
            "model_labels": label_map,
            # 익명 라벨(A/B/C) -> 실제 AI 매핑 (토론자·통합 AI에게는 비공개, 사용자 화면에는 공개)
            "label_map": {label_map[k]: ROLE_NAMES[k] for k in MODEL_KEYS},
            "log_file": result["log_file"] if log_saved else None,
        }
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    print("=" * 50)
    print("AI 토론 경쟁 모델 서버 시작")
    for _key in MODEL_KEYS:
        print(f"  토론자: {ROLE_NAMES[_key]}")
    print(f"  통합 담당: {JUDGE_MODEL_ID} / 추론 강도: {REASONING_EFFORT or '(지정 안 함)'}")
    print(f"브라우저에서 http://localhost:{port} 접속")
    print("=" * 50)
    # 디버그 모드는 오류 화면에서 서버 코드를 실행할 수 있는 콘솔을 열어주므로 기본은 끔
    app.run(host="127.0.0.1", port=port, debug=os.getenv("FLASK_DEBUG") == "1")
