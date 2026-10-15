"""
AI 토론 경쟁 모델 (AI Debate Arena)
====================================
GPT-4(OpenAI), Claude(Anthropic), Gemini(Google) 세 AI가 주어진 주제에 대해
N라운드에 걸쳐 서로의 의견을 수용·반영하며 제안을 발전시키고,
마지막에 통합 AI가 수렴된 내용을 종합해 최종 통합 결론을 도출하는 단일 파일 웹앱.

실행 방법:
    1) .env 파일에 OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY 설정
    2) pip install -r requirements.txt
    3) python app.py
    4) 브라우저에서 http://localhost:5000 접속
"""

import os
import json
import random
import traceback
from datetime import datetime
from typing import TypedDict, List, Dict

from flask import Flask, render_template_string, request, jsonify
from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------------------
# AI 클라이언트 초기화
# ------------------------------------------------------------------
import openai
import anthropic
from google import genai

# 3사 모델을 모두 '플래그십' 등급으로 맞춰야 토론 결과가 모델 성능차가 아닌
# 논리/근거의 질로 갈립니다. (gemini-2.5-flash는 경량 모델이라 gpt-4o/claude-sonnet
# 대비 체급이 낮아 항상 밀리는 경향이 있었음 -> pro로 상향)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.5-pro")

# 심판을 맡을 모델 (기본: Claude). "gpt" / "claude" / "gemini" 중 선택 가능
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "claude")

openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
google_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY", ""))

# 토론 로그 자동 저장 폴더 (API 호출 없이 로컬 파일로만 저장 - 토큰 소모 없음)
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

from langgraph.graph import StateGraph, END

# ------------------------------------------------------------------
# 상태(State) 정의
# ------------------------------------------------------------------
class DebateState(TypedDict):
    topic: str
    max_rounds: int
    current_round: int
    history: List[Dict]
    verdict: str
    label_map: Dict[str, str]


ROLE_NAMES = {
    "gpt": "GPT-4 (OpenAI)",
    "claude": "Claude (Anthropic)",
    "gemini": "Gemini (Google)",
}

# 첫 발언: 토론 주제에 대한 초기 입장/제안을 제시
DEBATER_SYSTEM_TEMPLATE_FIRST = (
    '당신은 AI 토론 패널의 일원입니다. 토론 주제: "{topic}"\n'
    "이 주제에 대한 당신의 초기 입장과 제안을, 핵심 근거와 함께 제시하세요.\n"
    "이후 라운드에서 다른 토론자들의 의견을 검토하며 이 제안을 발전시켜 나갈 것이므로, "
    "지금은 명확하고 검증 가능한 출발점이 되는 제안을 작성하세요.\n"
    "응답은 한국어로, 400자 이내로 작성하세요."
)

# 2라운드 이후: 동의/보완 -> 갱신된 제안 순서로 의견을 수렴·고도화
DEBATER_SYSTEM_TEMPLATE_NEXT = (
    '당신은 AI 토론 패널의 일원입니다. 토론 주제: "{topic}"\n'
    "이 토론의 목표는 승패를 가르는 것이 아니라, 서로의 의견 중 합리적인 부분을 수용하며 "
    "해결책을 점점 더 정교하고 실행 가능한 형태로 발전시키는 것입니다.\n"
    "아래 [지금까지의 토론 내용]을 검토한 뒤, 다음 3가지를 순서대로 작성하세요.\n"
    "1) 동의: 다른 토론자의 의견 중 합리적이라고 판단되는 부분과 그 이유\n"
    "2) 보완/반박: 동의하기 어렵거나 보완이 필요한 부분과 그 이유\n"
    "3) 갱신된 제안: 1)과 2)를 반영하여, 이번 라운드 기준 당신의 제안/결론을 한층 더 구체화하고 "
    "발전시켜 다시 제시 (이전 라운드 내용을 그대로 반복하지 말 것)\n"
    "응답은 한국어로, 500자 이내로 작성하세요."
)


def pick_debater_prompt(state: DebateState) -> str:
    """첫 발언인지 여부에 따라 적절한 디베이터 시스템 프롬프트를 선택."""
    template = DEBATER_SYSTEM_TEMPLATE_FIRST if not state["history"] else DEBATER_SYSTEM_TEMPLATE_NEXT
    return template.format(topic=state["topic"])


JUDGE_SYSTEM_TEMPLATE = (
    '당신은 AI 토론 결과를 종합하는 통합(Synthesis) 담당자입니다. 토론 주제: "{topic}"\n\n'
    "아래는 토론자 A, B, C 세 명이 {rounds}라운드에 걸쳐, 매 라운드 서로의 의견 중 합리적인 "
    "부분을 수용하며 자신의 제안을 갱신해온 토론 전체 내용입니다. 마지막 라운드의 제안일수록 "
    "더 발전·수렴된 상태입니다.\n"
    "(어떤 AI 모델인지는 의도적으로 가렸습니다. 특정 모델/브랜드에 대한 선호나 "
    "편견 없이, 오직 내용의 논리·근거만으로 종합하세요.)\n\n"
    "{full_history}\n\n"
    "당신의 역할은 '승자 1명을 뽑는 것'이 아니라, 토론을 통해 수렴된 내용을 종합하여 "
    "가장 합리적이고 실행 가능한 하나의 통합안을 제시하는 것입니다. "
    "다음 형식의 한국어 보고서를 작성하세요.\n"
    "1. 라운드별 수렴 과정 요약 (각 토론자가 어떤 의견을 수용/반영하며 입장을 어떻게 "
    "발전시켰는지 핵심만 간결하게)\n"
    "2. 최종 통합 결론 (3자의 마지막 라운드 제안을 종합한 하나의 최적안. 3자 모두 "
    "공통적으로 합의에 도달한 부분과, 여전히 트레이드오프/이견이 남아있는 부분을 구분해서 명시)\n"
    "3. 실행 가능한 결론 (Action Item, 2~3개)\n"
    "4. 사용자가 현실적으로 스스로 점검하고 깊이 고민해 보아야 할 핵심 질문 3가지\n"
    "   (토론 주제와 직접 관련된, 의사결정 전 반드시 자문해야 할 날카로운 질문으로 작성. "
    "일반론적 질문 금지, 사용자의 구체적 상황을 가정한 질문으로 작성)\n\n"
    "[분량 지침] 전체 보고서는 공백 포함 약 2000자 분량으로, 각 항목을 충분히 "
    "상세하고 구체적으로 서술하세요. 너무 짧게 요약하지 말고, 근거와 예시를 곁들여 "
    "심도 있게 작성하세요."
)


def build_context(state: DebateState) -> str:
    """지금까지의 토론 기록을 텍스트로 변환."""
    if not state["history"]:
        return "(아직 발언 없음 - 첫 라운드입니다)"
    lines = []
    for entry in state["history"]:
        lines.append(f"[{entry['round']}라운드 - {ROLE_NAMES[entry['model']]}]\n{entry['content']}")
    return "\n\n".join(lines)


def build_judge_context(state: DebateState, label_map: Dict[str, str]) -> str:
    """심판 전달용 토론 기록 - 모델명을 익명 라벨(A/B/C)로 가려 브랜드/자기선호 편향을 방지."""
    lines = []
    for entry in state["history"]:
        label = label_map[entry["model"]]
        lines.append(f"[{entry['round']}라운드 - 토론자 {label}]\n{entry['content']}")
    return "\n\n".join(lines)


def _append(state: DebateState, model_key: str, content: str) -> None:
    state["history"].append(
        {"round": state["current_round"] + 1, "model": model_key, "content": content}
    )


# ------------------------------------------------------------------
# 각 AI 노드 (LangGraph 노드 함수)
# ------------------------------------------------------------------
def gpt_node(state: DebateState) -> DebateState:
    context = build_context(state)
    system_prompt = pick_debater_prompt(state)
    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"[지금까지의 토론 내용]\n{context}"},
            ],
            max_tokens=750,
        )
        content = resp.choices[0].message.content.strip()
    except Exception as e:  # noqa: BLE001
        content = f"[오류: GPT-4 응답 실패 - {e}]"
    _append(state, "gpt", content)
    return state


def claude_node(state: DebateState) -> DebateState:
    context = build_context(state)
    system_prompt = pick_debater_prompt(state)
    try:
        resp = anthropic_client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=750,
            system=system_prompt,
            messages=[{"role": "user", "content": f"[지금까지의 토론 내용]\n{context}"}],
        )
        content = resp.content[0].text.strip()
    except Exception as e:  # noqa: BLE001
        content = f"[오류: Claude 응답 실패 - {e}]"
    _append(state, "claude", content)
    return state


def gemini_node(state: DebateState) -> DebateState:
    context = build_context(state)
    system_prompt = pick_debater_prompt(state)
    try:
        resp = google_client.models.generate_content(
            model=GOOGLE_MODEL,
            contents=f"{system_prompt}\n\n[지금까지의 토론 내용]\n{context}",
        )
        content = resp.text.strip()
    except Exception as e:  # noqa: BLE001
        content = f"[오류: Gemini 응답 실패 - {e}]"
    _append(state, "gemini", content)
    # 한 라운드(3개 AI 모두 발언)가 끝났으므로 라운드 카운트 증가
    state["current_round"] += 1
    return state


def judge_node(state: DebateState) -> DebateState:
    # 브랜드/자기선호 편향 방지: gpt/claude/gemini -> A/B/C 익명 라벨을 매 토론마다 무작위 배정
    keys = ["gpt", "claude", "gemini"]
    labels = ["A", "B", "C"]
    random.shuffle(labels)
    label_map = dict(zip(keys, labels))

    full_history = build_judge_context(state, label_map)
    prompt = JUDGE_SYSTEM_TEMPLATE.format(
        topic=state["topic"], rounds=state["max_rounds"], full_history=full_history
    )

    def _judge_with_claude():
        resp = anthropic_client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=5000,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()

    def _judge_with_gpt():
        resp = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5000,
        )
        return resp.choices[0].message.content.strip()

    def _judge_with_gemini():
        resp = google_client.models.generate_content(model=GOOGLE_MODEL, contents=prompt)
        return resp.text.strip()

    judges = {"claude": _judge_with_claude, "gpt": _judge_with_gpt, "gemini": _judge_with_gemini}
    try:
        verdict = judges.get(JUDGE_PROVIDER, _judge_with_claude)()
    except Exception as e:  # noqa: BLE001
        verdict = f"[오류: 심판 응답 실패 - {e}]"

    state["verdict"] = verdict
    # 익명 라벨(A/B/C) -> 실제 AI 매핑 (심판에게는 비공개, 사용자 화면에는 공개)
    state["label_map"] = {label_map[k]: ROLE_NAMES[k] for k in keys}
    return state


def should_continue(state: DebateState) -> str:
    if state["current_round"] < state["max_rounds"]:
        return "continue"
    return "judge"


# ------------------------------------------------------------------
# 토론 로그 자동 저장 (로컬 파일 I/O — API/토큰 추가 소모 없음)
# ------------------------------------------------------------------
def save_debate_log(result: DebateState) -> str:
    """토론 결과를 logs/ 폴더에 마크다운 파일로 저장하고, 파일명을 반환."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = "".join(
        c for c in result["topic"] if c.isalnum() or c in (" ", "_", "-")
    ).strip()[:30]
    filename = f"{timestamp}_{safe_topic or 'debate'}.md"
    filepath = os.path.join(LOG_DIR, filename)

    lines = [
        f"# 토론 주제: {result['topic']}",
        "",
        f"- 일시: {timestamp}",
        f"- 라운드 수: {result['max_rounds']}",
        "",
        "---",
        "",
    ]
    for entry in result["history"]:
        lines.append(f"## {entry['round']}라운드 - {ROLE_NAMES[entry['model']]}")
        lines.append("")
        lines.append(entry["content"])
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 최종 통합 결론")
    lines.append("")
    lines.append(result["verdict"])
    lines.append("")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return filename


# ------------------------------------------------------------------
# LangGraph 그래프 구성
# ------------------------------------------------------------------
workflow = StateGraph(DebateState)
workflow.add_node("gpt", gpt_node)
workflow.add_node("claude", claude_node)
workflow.add_node("gemini", gemini_node)
workflow.add_node("judge", judge_node)

workflow.set_entry_point("gpt")
workflow.add_edge("gpt", "claude")
workflow.add_edge("claude", "gemini")
workflow.add_conditional_edges("gemini", should_continue, {"continue": "gpt", "judge": "judge"})
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
  input[type=text], select {
    width: 100%; padding: 10px 12px; border-radius: 8px;
    border: 1px solid var(--border); background: #11141a; color: var(--text);
    font-size: 14px; margin-bottom: 14px;
  }
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
  .badge {
    display: inline-block; padding: 2px 8px; border-radius: 6px;
    font-size: 12px; font-weight: 700; margin-bottom: 8px;
  }
  .badge.gpt { background: var(--gpt); color: #06281c; }
  .badge.claude { background: var(--claude); color: #2b1700; }
  .badge.gemini { background: var(--gemini); color: #161b40; }
  .badge.judge { background: var(--judge); color: #2b0009; }
  .entry { padding: 12px 0; border-bottom: 1px solid var(--border); }
  .entry:last-child { border-bottom: none; }
  .entry-content { white-space: pre-wrap; line-height: 1.6; font-size: 14px; }
  .verdict-box { white-space: pre-wrap; line-height: 1.7; font-size: 15px; }
  details summary { cursor: pointer; color: var(--accent); font-size: 14px; padding: 6px 0; }
  #loading { display: none; color: var(--muted); font-size: 14px; }
  .round-tag { color: var(--muted); font-size: 12px; margin-left: 6px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>🤖 AI 토론 경쟁 모델</h1>
  <div class="sub">GPT-4 · Claude · Gemini 가 의견을 수렴하며 발전시키고, 최종 통합 결론을 도출합니다.</div>

  <div class="card">
    <label for="topic">토론 주제</label>
    <input type="text" id="topic" placeholder="예: 1MW 태양광 발전 사업, 지금 시작하는 것이 합리적인가?">

    <label for="rounds">토론 라운드 수</label>
    <select id="rounds">
      <option value="1">1라운드</option>
      <option value="2">2라운드</option>
      <option value="3" selected>3라운드 (기본)</option>
      <option value="4">4라운드</option>
      <option value="5">5라운드</option>
    </select>

    <button id="runBtn" onclick="runDebate()">토론 시작</button>
    <span id="loading">⏳ AI들이 토론 중입니다... (라운드 수에 따라 30초~수분 소요)</span>
  </div>

  <div id="result"></div>
</div>

<script>
const ROLE_LABELS = {gpt: "GPT-4", claude: "Claude", gemini: "Gemini"};

async function runDebate() {
  const topic = document.getElementById('topic').value.trim();
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
      body: JSON.stringify({topic, rounds})
    });
    const data = await res.json();
    if (data.error) {
      result.innerHTML = `<div class="card">오류: ${data.error}</div>`;
      return;
    }

    let logHtml = '';
    data.history.forEach(e => {
      logHtml += `<div class="entry">
        <span class="badge ${e.model}">${ROLE_LABELS[e.model]}</span>
        <span class="round-tag">${e.round}라운드</span>
        <div class="entry-content">${escapeHtml(e.content)}</div>
      </div>`;
    });

    const logNote = data.log_file
      ? `<div class="log-note">📁 토론 로그 저장됨: logs/${escapeHtml(data.log_file)}</div>`
      : '';

    let legendNote = '';
    if (data.label_map) {
      const parts = Object.entries(data.label_map).map(([k, v]) => `${k} = ${v}`);
      legendNote = `<div class="log-note">🏷️ 익명 토론자 매핑 (통합 AI에게는 비공개로 진행됨): ${parts.join(' · ')}</div>`;
    }

    result.innerHTML = `
      <div class="card">
        <div class="verdict-head">
          <span class="badge judge">최종 통합 결론</span>
          <button class="copy-btn" id="copyBtn" onclick="copyVerdict()">📋 복사</button>
        </div>
        <div class="verdict-box" id="verdictText">${escapeHtml(data.verdict)}</div>
        ${logNote}
        ${legendNote}
      </div>
      <div class="card">
        <details><summary>토론 상세 로그 보기 (${data.history.length}개 발언)</summary>
          <div style="margin-top:12px">${logHtml}</div>
        </details>
      </div>
    `;
  } catch (err) {
    result.innerHTML = `<div class="card">요청 중 오류가 발생했습니다: ${err}</div>`;
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
    data = request.get_json(force=True)
    topic = (data.get("topic") or "").strip()
    try:
        rounds = int(data.get("rounds", 3))
    except (TypeError, ValueError):
        rounds = 3
    rounds = max(1, min(rounds, 5))

    if not topic:
        return jsonify({"error": "토론 주제를 입력해주세요."}), 400

    initial_state: DebateState = {
        "topic": topic,
        "max_rounds": rounds,
        "current_round": 0,
        "history": [],
        "verdict": "",
        "label_map": {},
    }

    try:
        result = debate_graph.invoke(initial_state)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    # 토론 로그 자동 저장 (실패해도 응답에는 영향 없도록 처리)
    try:
        result["log_file"] = save_debate_log(result)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        result["log_file"] = None

    return jsonify(result)


if __name__ == "__main__":
    print("=" * 50)
    print("AI 토론 경쟁 모델 서버 시작")
    print("브라우저에서 http://localhost:5000 접속")
    print("=" * 50)
    app.run(debug=True, port=5000)
