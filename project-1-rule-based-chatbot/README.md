# Project 1 — Rule-Based AI Chatbot 🤖

**Goal:** Create a simple rule-based chatbot that responds to predefined user inputs.

This is the foundation phase: no machine learning yet — pure control flow, decision-making
logic, and deterministic "white box" AI (traceable Input → Logic → Output).

## How it meets the spec

| Requirement | Implementation |
|---|---|
| Input loop | Continuous `while True` cycle |
| Sanitization | `raw_input.lower().strip()` handles case & whitespace |
| Knowledge base | Dictionary with 11 intents (spec asks for 5+) |
| Fallback | `responses.get(user_input, fallback)` — lookup + fallback in one atomic operation |
| Exit strategy | Clean `break` on `bye` / `exit` / `quit` / `goodbye` |

The knowledge base is a **dictionary (hash map)** rather than an if/elif ladder:
lookup is O(1) constant time regardless of how many rules are added, while an
if/elif ladder degrades linearly O(n) and accumulates technical debt.

## Run it

```bash
python3 chatbot.py
```

Example session:

```
DecoBot: Welcome to DecodeLabs! Type 'help' for options or 'bye' to exit.
You: hello
DecoBot: Hi there! I'm DecoBot, your DecodeLabs assistant. Ask me anything!
You: what is ai
DecoBot: AI is the simulation of human intelligence by machines. I'm the deterministic kind...
You: bye
DecoBot: Goodbye! Thanks for chatting with DecodeLabs.
```

## Key skills demonstrated

Control flow, decision-making logic, input sanitization, hash-map data structures,
and basic AI concepts (deterministic rule engines as the guardrail layer used in
modern LLM systems).
