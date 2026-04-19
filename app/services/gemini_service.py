import re
from typing import Any, List, Tuple, Union

import google.generativeai as genai

from app.core.config import GEMINI_API_KEY, logger
from app.schemas.chat_schema import Message
from app.services.translator_service import detect_language
from app.utils.heuristics import get_fallback

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

_HISTORY_LIMIT = 6
_MODEL_NAME = "models/gemini-1.5-flash"

_BANNED_PHRASES = (
    "tell me more",
    "i'm here for you",
    "i’m here for you",
    "im here for you",
    "i understand",
)

_SYSTEM_RULES = """You are an emotional-support assistant.
Hard rules (do not break):
- Read the full transcript; your answer MUST reference specific words, situations, or feelings the user actually mentioned.
- Never use these phrases (even in different casing): "tell me more", "I'm here for you", "I understand" (do not use "I understand" as empty empathy).
- Never reply with generic one-line comfort that could apply to anyone.
- Give at least two concrete suggestions, reflections, or questions that tie directly to the user's last message and prior turns.
- Match the requested output language exactly.
- Be concise but substantive (roughly 80–220 words unless the user asked for brevity)."""


def _msg_role_text(msg: Union[Message, dict, Any]) -> Tuple[str, str]:
    if isinstance(msg, dict):
        return str(msg.get("role", "user") or "user"), str(msg.get("text", "") or "")
    if isinstance(msg, Message):
        return str(msg.role), str(msg.text)
    if hasattr(msg, "model_dump"):
        d = msg.model_dump()
        return str(d.get("role", "user") or "user"), str(d.get("text", "") or "")
    role = getattr(msg, "role", None)
    text = getattr(msg, "text", None)
    if role is not None and text is not None:
        return str(role), str(text)
    return "user", ""


def _normalize_role(role: str) -> str:
    r = (role or "").strip().lower()
    if r in ("ai", "assistant", "bot", "model", "system"):
        return "assistant"
    return "user"


def _build_transcript(history: List[Union[Message, dict]], text: str) -> str:
    window = list(history[-_HISTORY_LIMIT:]) if history else []
    lines: List[str] = []
    for msg in window:
        role_raw, content = _msg_role_text(msg)
        content = (content or "").strip()
        if not content:
            continue
        label = "Assistant" if _normalize_role(role_raw) == "assistant" else "User"
        lines.append(f"{label}: {content}")
    if not lines and (text or "").strip():
        lines.append(f"User: {text.strip()}")
    return "\n".join(lines).strip()


def _contains_banned(text: str) -> bool:
    low = (text or "").lower()
    return any(p in low for p in _BANNED_PHRASES)


def _strip_prefix_labels(text: str) -> str:
    return re.sub(r"^\[?reply\]?\s*:?\s*", "", text or "", flags=re.IGNORECASE).strip()


def _extract_response_text(response: Any) -> str:
    if not response:
        return ""
    try:
        t = getattr(response, "text", None)
        if t:
            return str(t).strip()
    except Exception:
        pass
    chunks: List[str] = []
    for cand in getattr(response, "candidates", None) or []:
        content = getattr(cand, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", None) or []:
            pt = getattr(part, "text", None)
            if pt:
                chunks.append(str(pt))
    return "".join(chunks).strip()


def _word_count(s: str) -> int:
    t = s or ""
    if re.search(r"[\u0600-\u06FF]", t):
        return len([p for p in re.split(r"\s+", t.strip()) if p])
    return len(re.findall(r"\b\w+\b", t))


def _is_low_quality(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return True
    if _word_count(s) < 8:
        return True
    if len(s) < 40:
        return True
    low = s.lower()
    if low.count("happy to help") or low.count("feel free to reach out"):
        return True
    return False


def _snippet(text: str, max_len: int = 140) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    if len(t) <= max_len:
        return t
    return t[: max_len - 3].rstrip() + "..."


def _keyword_plan(low: str, lang: str) -> Tuple[str, str, str]:
    if lang.startswith("ar"):
        if any(k in low for k in ("عمل", "شغل", "مدير", "زميل")):
            return (
                "حدد مهمة واحدة صغيرة تنجزها اليوم وقلل التشتت عن بريدك ساعة.",
                "اكتب ما يزعجك في جملتين ثم قرر هل هو واقعي أم افتراض.",
                "خذ استراحة ١٠ دقائق بعيدًا عن الشاشة قبل الرد على رسائل العمل.",
            )
        if any(k in low for k in ("تعب", "نوم", "أرق", "إرهاق")):
            return (
                "ثبت وقت نوم واستيقاظ بنفس الفترة لثلاثة أيام متتالية.",
                "قلل الكافيين بعد الظهر واستبدله بماء أو شاي خفيف.",
                "جرب ٥ دقائق تمطيط خفيف قبل النوم.",
            )
        if any(k in low for k in ("قلق", "خوف", "توتر")):
            return (
                "سمِّ المخاوف التي تتحكم بك الآن في جملة واحدة لكل منها.",
                "تنفس ٤–٧–٨ لأربع دورات مع التركيز على الزفير.",
                "اقترح على نفسك خطوة دقيقة خلال الساعة القادمة فقط.",
            )
        return (
            "اكتب ما الذي يزيد الضغط الآن في ثلاث نقاط قصيرة.",
            "اختر خطوة واحدة صغيرة يمكن تنفيذها خلال ٢٠ دقيقة.",
            "حدد شخصًا أو مصدرًا يمكن أن يدعمك عمليًا في هذه النقطة.",
        )
    if any(k in low for k in ("work", "boss", "job", "office", "coworker", "deadline")):
        return (
            "Pick one deliverable for today and silence notifications for a 45-minute focus block.",
            "Write the worry in two sentences, then mark what is inside your control versus outside it.",
            "Schedule a 10-minute walk before replying to tense work messages.",
        )
    if any(k in low for k in ("tired", "sleep", "insomnia", "exhausted", "fatigue")):
        return (
            "Anchor a fixed wake time for three days, even if bedtime shifts slightly.",
            "Cut caffeine after 2 p.m. and replace late scrolling with a paper book or calm audio.",
            "Add five minutes of light stretching before bed to signal wind-down.",
        )
    if any(k in low for k in ("anxious", "anxiety", "panic", "worried", "fear", "nervous")):
        return (
            "Name the fear in one sentence each so it is visible instead of vague.",
            "Run four cycles of 4-7-8 breathing with a longer exhale than inhale.",
            "Commit to one tiny action in the next hour instead of solving everything.",
        )
    if any(k in low for k in ("friend", "partner", "family", "relationship", "mom", "dad")):
        return (
            "State what you need from the other person in one clear, non-accusatory sentence.",
            "Ask for a specific time to talk rather than debating over text.",
            "Note one boundary you want to keep and one concession you can offer.",
        )
    return (
        "Write three bullets: trigger, feeling, and one lever you can move today.",
        "Shrink the goal to a 20-minute version so momentum is realistic.",
        "Identify one person, tool, or habit that would reduce friction if used once today.",
    )


def _emotion_anchor(emotion: str, lang: str) -> str:
    e = (emotion or "").lower()
    if lang.startswith("ar"):
        mapping = {
            "sadness": "ما تشاركه يوضح أن الحمل ثقيل الآن، وهذا طبيعي.",
            "fear": "القلق الذي وصفته منطقي أمام غموض الموقف.",
            "anger": "الانزعاج الذي ظهر في كلامك يستحق أن يُسمّى ويُعالَج خطوة بخطوة.",
            "joy": "من الجيد أن تلاحظ لحظة إيجابية؛ ابنِ عليها بخطوة صغيرة تُثبتها.",
            "surprise": "الموقف المفاجئ يحتاج ترتيبًا للحقائق قبل القرار.",
        }
        return mapping.get(e, "لخص ما يزيد الضغط في جملتين ثم اختر خطوة واحدة للساعة القادمة.")
    mapping = {
        "sadness": "What you described sounds genuinely heavy, and it makes sense it would weigh on you.",
        "fear": "The worry you surfaced fits an uncertain situation; narrowing facts from assumptions will help.",
        "anger": "The frustration in your message is signal, not noise—channel it into one boundary or request.",
        "joy": "It is worth anchoring this positive moment with a small follow-up action so it lasts.",
        "surprise": "A sudden shift is easier to navigate once the facts are listed in order.",
    }
    return mapping.get(
        e,
        "Ground the situation in two short sentences, then pick one lever you can move within the hour.",
    )


def _contextual_override(text: str, emotion: str, lang: str) -> str:
    low = (text or "").lower()
    s = _snippet(text, 160)
    a, b, c = _keyword_plan(low, lang)
    anchor = _emotion_anchor(emotion, lang)
    if lang.startswith("ar"):
        return (
            f"{anchor} اعتمادًا على ما قلت (\"{s}\")، جرّب ما يلي بشكل عملي:\n"
            f"1) {a}\n2) {b}\n3) {c}"
        )
    return (
        f"{anchor} Based on what you shared about \"{s}\", try these concrete moves:\n"
        f"1) {a}\n2) {b}\n3) {c}"
    )


def _build_prompt(transcript: str, lang: str, strict_extra: str = "") -> str:
    lang_name = "Arabic" if lang.startswith("ar") else "English"
    extra = f"\nAdditional constraint: {strict_extra}\n" if strict_extra else ""
    return (
        f"{_SYSTEM_RULES}\n\n"
        f"Output language: {lang_name} ({lang}).\n"
        f"{extra}"
        "## Conversation (most recent last; answer the final User turn)\n"
        f"{transcript}\n\n"
        "Write the assistant reply now. Follow every hard rule."
    )


def _call_gemini(model: genai.GenerativeModel, prompt: str) -> Tuple[Any, str]:
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.55,
            "top_p": 0.9,
            "max_output_tokens": 512,
        },
    )
    raw = _extract_response_text(response)
    print("[Gemini DEBUG] final prompt:\n", prompt, flush=True)
    print("[Gemini DEBUG] raw response:\n", raw, flush=True)
    logger.debug("Gemini final prompt:\n%s", prompt)
    logger.debug("Gemini raw response:\n%s", raw)
    return response, raw


def generate_gemini_response(
    text: str,
    emotion: str,
    lang: str,
    history: List[Union[Message, dict]],
) -> str:
    if not GEMINI_API_KEY:
        return _contextual_override(text, emotion, lang)

    transcript = _build_transcript(history, text)
    prompt = _build_prompt(transcript, lang)

    try:
        model = genai.GenerativeModel(_MODEL_NAME)
        _, raw = _call_gemini(model, prompt)
        reply = _strip_prefix_labels(raw)

        if not reply or _contains_banned(reply) or _is_low_quality(reply):
            retry_prompt = _build_prompt(
                transcript,
                lang,
                strict_extra=(
                    "Your last draft was too generic or used forbidden filler. "
                    "Rewrite with zero banned phrases, at least eight meaningful words, "
                    "and explicit references to details in the transcript."
                ),
            )
            _, raw2 = _call_gemini(model, retry_prompt)
            reply = _strip_prefix_labels(raw2)

        if _contains_banned(reply):
            logger.warning("Gemini reply contained banned phrases; using contextual override.")
            reply = _contextual_override(text, emotion, lang)
        elif _is_low_quality(reply):
            logger.warning("Gemini reply low quality; using contextual override.")
            reply = _contextual_override(text, emotion, lang)

        detected = detect_language(reply)
        if detected != lang and len((text or "").strip()) > 0:
            logger.warning(
                "Gemini language mismatch (expected %s, got %s); using contextual override.",
                lang,
                detected,
            )
            reply = _contextual_override(text, emotion, lang)

        return reply

    except Exception as e:
        logger.error("Gemini generation failed: %s", e)
        try:
            return _contextual_override(text, emotion, lang)
        except Exception:
            return get_fallback(lang, emotion)
