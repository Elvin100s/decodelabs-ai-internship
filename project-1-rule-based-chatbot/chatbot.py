"""
Project 1 — Rule-Based AI Chatbot (DecodeLabs Industrial Training Kit)

A deterministic, dictionary-driven chatbot following the IPO model:
  INPUT   -> sanitization & normalization (.lower().strip())
  PROCESS -> intent matching via dictionary lookup, O(1)
  OUTPUT  -> response generation with a safe fallback

Runs in a continuous loop until the user issues an exit command.
"""

BOT_NAME = "DecoBot"

# Knowledge base: dictionary with 5+ intents (the "logic skeleton").
# Dictionary lookup is O(1) — preferred over an if/elif ladder, which is O(n).
RESPONSES = {
    "hello": f"Hi there! I'm {BOT_NAME}, your DecodeLabs assistant. Ask me anything!",
    "hi": f"Hello! {BOT_NAME} at your service.",
    "how are you": "Running at 100% uptime — thanks for asking! How are you?",
    "what is your name": f"I'm {BOT_NAME}, a rule-based chatbot built with pure if-else logic and dictionaries.",
    "what can you do": "I can greet you, tell you about myself, share the time rules I live by, and say goodbye. Try 'help'.",
    "who made you": "I was built by a DecodeLabs AI intern as Project 1: the foundation phase.",
    "what is ai": "AI is the simulation of human intelligence by machines. I'm the deterministic kind: 100% hard-coded, zero hallucination risk!",
    "tell me a joke": "Why did the chatbot cross the road? To .get() to the other side!",
    "help": "Try: hello | how are you | what is your name | what can you do | who made you | what is ai | tell me a joke | bye",
    "thanks": "You're welcome!",
    "thank you": "Anytime! That's what I'm here for.",
}

EXIT_COMMANDS = {"bye", "exit", "quit", "goodbye"}

FALLBACK = "I do not understand that yet. Type 'help' to see what I know."


def get_response(raw_input: str) -> str:
    """PROCESS phase: sanitize the input and match it against the knowledge base."""
    clean_input = raw_input.lower().strip()
    return RESPONSES.get(clean_input, FALLBACK)


def main() -> None:
    print(f"{BOT_NAME}: Welcome to DecodeLabs! Type 'help' for options or 'bye' to exit.")

    # The heartbeat: infinite loop until the kill command.
    while True:
        raw_input_text = input("You: ")
        clean_input = raw_input_text.lower().strip()

        if not clean_input:
            print(f"{BOT_NAME}: You didn't type anything — try 'help'.")
            continue

        # Exit strategy: clean break command.
        if clean_input in EXIT_COMMANDS:
            print(f"{BOT_NAME}: Goodbye! Thanks for chatting with DecodeLabs.")
            break

        print(f"{BOT_NAME}: {get_response(clean_input)}")


if __name__ == "__main__":
    main()
