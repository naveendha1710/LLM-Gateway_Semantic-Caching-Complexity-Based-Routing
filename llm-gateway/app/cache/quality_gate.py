"""Quality gate — decide whether a response is worth caching.

A response is cached only when it passes the gate. The gate checks:

1. The response has at least one choice with non-empty content.
2. ``finish_reason`` is exactly ``"stop"`` (``"length"`` means truncated).
3. Token usage is present and non-negative.
4. The response is not flagged as ``degraded`` (degraded responses come
   from a fallback provider and may be lower quality).
5. Content is not mid-sentence truncated (must end with terminal punctuation).

The gate is intentionally conservative: it is cheaper to skip caching a
borderline response than to serve a bad cached answer to future requests.
"""

from __future__ import annotations

import logging
import re

from app.api.schemas.responses import ChatCompletionResponse

logger = logging.getLogger(__name__)

_GOOD_FINISH_REASONS = {"stop"}

# Terminal punctuation that indicates a complete sentence/thought
_TERMINAL_PUNCTUATION = re.compile(r'[.!?]["\']?\s*$')


def passes_quality_gate(response: ChatCompletionResponse) -> bool:
    """Return ``True`` if ``response`` is suitable for caching.

    The gate is strict: any response with ``finish_reason != "stop"`` is
    rejected. This includes ``"length"`` (truncated due to token limit),
    ``"content_filter"``, ``"tool_calls"``, etc. We also reject responses
    that appear mid-sentence (no terminal punctuation).
    """
    if response.degraded:
        logger.debug("quality_gate_rejected_degraded")
        return False

    if not response.choices:
        logger.debug("quality_gate_rejected_no_choices")
        return False

    choice = response.choices[0]
    if not choice.message.content or not choice.message.content.strip():
        logger.debug("quality_gate_rejected_empty_content")
        return False

    # Reject any finish_reason other than "stop" — "length" means truncated
    if choice.finish_reason not in _GOOD_FINISH_REASONS:
        logger.debug(
            "quality_gate_rejected_finish_reason",
            extra={"finish_reason": choice.finish_reason},
        )
        return False

    # Reject mid-sentence truncation (no terminal punctuation)
    content = choice.message.content.strip()
    if not _TERMINAL_PUNCTUATION.search(content):
        logger.debug(
            "quality_gate_rejected_mid_sentence",
            extra={"content_preview": content[:100]},
        )
        return False

    usage = response.usage
    if usage is None:
        logger.debug("quality_gate_rejected_no_usage")
        return False
    if usage.total_tokens < 0 or usage.prompt_tokens < 0 or usage.completion_tokens < 0:
        logger.debug("quality_gate_rejected_negative_usage")
        return False

    return True
