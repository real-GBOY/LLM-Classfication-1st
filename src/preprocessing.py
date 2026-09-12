"""Text-shaping utilities shared by the TF-IDF baseline and the DeBERTa pipeline."""
from src.data import parse_turns

TURN_SEP = " "
NULL_TURN_PLACEHOLDER = "[NO RESPONSE]"


def render_turns(turns: list) -> str:
    """Joins a list of conversation turns into one string, substituting a
    placeholder for content-filtered (null) turns instead of crashing or
    silently dropping them (dropping would misrepresent how many turns the
    conversation had).
    """
    parts = [t if isinstance(t, str) else NULL_TURN_PLACEHOLDER for t in turns]
    return TURN_SEP.join(parts)


def render_prompt_cell(raw: str) -> str:
    return render_turns(parse_turns(raw))


def build_compare_text(prompt_raw: str, response_a_raw: str, response_b_raw: str) -> str:
    """Builds the '[PROMPT] ... [RESPONSE A] ... [RESPONSE B] ...' text used
    by the TF-IDF baseline. No length budget is applied here -- TF-IDF has no
    hard sequence-length limit, unlike DeBERTa.
    """
    prompt = render_prompt_cell(prompt_raw)
    resp_a = render_prompt_cell(response_a_raw)
    resp_b = render_prompt_cell(response_b_raw)
    return f"[PROMPT] {prompt} [RESPONSE A] {resp_a} [RESPONSE B] {resp_b}"


TRUNCATED_MARKER = " ... [TRUNCATED] ... "


def _truncate_words_head_tail(text: str, word_budget: int, head_frac: float = 0.7) -> str:
    """Keeps the first `head_frac` of the budget from the start of `text` and
    the rest from the end, joined by an explicit truncation marker.

    Head+tail (rather than plain right-truncation) is used because response
    text tends to carry its direct answer near the start and any
    summary/conclusion near the end -- a technique or code walkthrough that
    gets cut off after the first paragraph loses exactly the part a human
    rater would use to judge quality.
    """
    words = text.split()
    if len(words) <= word_budget:
        return text
    head_n = max(1, int(word_budget * head_frac))
    tail_n = max(0, word_budget - head_n)
    if tail_n == 0:
        return " ".join(words[:head_n])
    return " ".join(words[:head_n]) + TRUNCATED_MARKER + " ".join(words[-tail_n:])


def _truncate_prompt_turns(turns: list, word_budget: int) -> str:
    """Fills the prompt's word budget starting from the MOST RECENT turn
    backward, since in multi-turn conversations the final turn is almost
    always the actual question being judged; earlier turns are useful
    context but expendable first when the budget is tight.
    """
    rendered = [t if isinstance(t, str) else NULL_TURN_PLACEHOLDER for t in turns]
    kept = []
    remaining = word_budget
    for turn in reversed(rendered):
        turn_words = turn.split()
        if len(turn_words) <= remaining:
            kept.append(turn)
            remaining -= len(turn_words)
        else:
            if remaining > 0:
                kept.append(_truncate_words_head_tail(turn, remaining, head_frac=1.0))
            break
    kept.reverse()
    return TURN_SEP.join(kept)


def build_truncated_compare_text(
    prompt_raw: str,
    response_a_raw: str,
    response_b_raw: str,
    max_words: int = 340,
    prompt_frac: float = 0.2,
    response_frac: float = 0.4,
) -> str:
    """Builds the DeBERTa input text with a word-count budget applied BEFORE
    tokenization, so that a long response_a can't push response_b out of the
    tokenizer's hard sequence_length window entirely.

    `max_words` should be set conservatively below the model's true token
    limit (roughly 0.75 tokens/word for English SentencePiece), leaving
    slack for [CLS]/[SEP] and the literal [PROMPT]/[RESPONSE A]/[RESPONSE B]
    markers. The tokenizer's own sequence_length truncation is still the
    exact backstop -- this is a coarse pre-truncation, not a precise one.
    """
    prompt_budget = int(max_words * prompt_frac)
    response_budget = int(max_words * response_frac)

    prompt = _truncate_prompt_turns(parse_turns(prompt_raw), prompt_budget)
    resp_a = _truncate_words_head_tail(render_prompt_cell(response_a_raw), response_budget)
    resp_b = _truncate_words_head_tail(render_prompt_cell(response_b_raw), response_budget)

    return f"[PROMPT] {prompt} [RESPONSE A] {resp_a} [RESPONSE B] {resp_b}"
