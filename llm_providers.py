"""
LLM provider abstraction. order_agent.py only knows about this interface --
it never imports the anthropic or google-genai SDKs directly. Each provider
owns its own conversation history in whatever native format that API expects,
since Claude and Gemini shape messages differently.

Tool schemas passed in are always the generic format used in order_agent.py:
{"name": ..., "description": ..., "input_schema": <JSON schema dict>}
Each provider converts that to its own native tool format internally.
"""

import json
import os
import time
from typing import Callable


class LLMProvider:
    """Common interface every provider implements."""

    def send(self, user_text: str, system_prompt: str, tools: list[dict],
              tool_executor: Callable[[str, dict], dict]) -> str:
        """Send one user turn, running the full tool-call loop if needed,
        and return the model's final text reply."""
        raise NotImplementedError


API_TIMEOUT_SECONDS = 15.0  # fail fast with the "having trouble" fallback rather
# than blocking a live phone call in silence if the LLM API hangs/retries internally.

MAX_ATTEMPTS = 2  # one retry -- a single timeout/5xx is often transient (especially
# on free-tier API access, which runs on shared, lower-priority infrastructure and
# is more prone to occasional server-side slowness) and frequently succeeds a moment
# later. Worst case this doubles the wait before the "having trouble" fallback; best
# case the customer never notices anything happened.
RETRY_DELAY_SECONDS = 0.5


def _call_with_retry(fn: Callable[[], object], label: str):
    """Call fn() (a zero-arg callable making one API request), retrying once on
    any failure before giving up. Re-raises the last error if every attempt fails,
    for the caller's existing except-and-fallback handling."""
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return fn()
        except Exception as e:
            last_error = e
            print(f"[{label}] attempt {attempt}/{MAX_ATTEMPTS} FAILED: {type(e).__name__}: {e}", flush=True)
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_DELAY_SECONDS)
    raise last_error


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str | None = None):
        import anthropic

        self.client = anthropic.Anthropic(timeout=API_TIMEOUT_SECONDS)
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-opus-5")
        self.history: list[dict] = []

    def send(self, user_text, system_prompt, tools, tool_executor) -> str:
        import anthropic

        self.history.append({"role": "user", "content": user_text})
        anthropic_tools = [
            {"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]}
            for t in tools
        ]

        while True:
            try:
                response = _call_with_retry(
                    lambda: self.client.messages.create(
                        model=self.model,
                        max_tokens=1024,
                        system=system_prompt,
                        tools=anthropic_tools,
                        messages=self.history,
                    ),
                    "AnthropicProvider",
                )
            except Exception:
                # Every attempt failed -- degrade gracefully (the customer hears a
                # retry prompt) rather than crash the whole call. Don't append
                # anything further -- history stays valid for the next turn.
                return (
                    "Sorry, I'm having trouble connecting right now. "
                    "Could you say that again in a moment?"
                )
            self.history.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                return "".join(block.text for block in response.content if block.type == "text")

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result = tool_executor(block.name, block.input)
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)}
                )
            self.history.append({"role": "user", "content": tool_results})


class GeminiProvider(LLMProvider):
    def __init__(self, model: str | None = None):
        from google import genai
        from google.genai import types

        # timeout is in milliseconds for this SDK's HttpOptions -- without it, a
        # hung/slow request has no bound and can block a live call in silence
        # far longer than API_TIMEOUT_SECONDS would suggest.
        self.client = genai.Client(http_options=types.HttpOptions(timeout=int(API_TIMEOUT_SECONDS * 1000)))
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
        self.history: list = []  # list[types.Content]

    def send(self, user_text, system_prompt, tools, tool_executor) -> str:
        from google.genai import types

        gemini_tool = types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=t["name"],
                    description=t["description"],
                    parameters_json_schema=t["input_schema"],
                )
                for t in tools
            ]
        )
        config = types.GenerateContentConfig(system_instruction=system_prompt, tools=[gemini_tool])

        self.history.append(types.Content(role="user", parts=[types.Part(text=user_text)]))

        while True:
            try:
                response = _call_with_retry(
                    lambda: self.client.models.generate_content(
                        model=self.model, contents=self.history, config=config
                    ),
                    "GeminiProvider",
                )
            except Exception:
                # Every attempt failed -- degrade gracefully (the customer hears a
                # retry prompt) rather than crash the whole call. Don't append
                # anything further -- history stays valid for the next turn.
                return (
                    "Sorry, I'm having trouble connecting right now. "
                    "Could you say that again in a moment?"
                )
            self.history.append(response.candidates[0].content)

            function_calls = response.function_calls
            if not function_calls:
                return response.text or ""

            response_parts = []
            for fc in function_calls:
                result = tool_executor(fc.name, fc.args or {})
                response_parts.append(types.Part.from_function_response(name=fc.name, response=result))
            self.history.append(types.Content(role="user", parts=response_parts))


def build_provider(name: str | None) -> LLMProvider:
    name = (name or "anthropic").strip().lower()
    if name == "anthropic":
        return AnthropicProvider()
    if name == "gemini":
        return GeminiProvider()
    raise ValueError(f"Unknown LLM_PROVIDER '{name}'. Use 'anthropic' or 'gemini'.")
