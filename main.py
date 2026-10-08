"""Command-line loop for testing the order-taking agent as a 'customer'."""

from order_agent import OrderAgent, load_menu


def print_welcome() -> None:
    restaurant_name = load_menu()["restaurant_name"]
    print("=" * 60)
    print(f"{restaurant_name} -- Order Line (Phase 1: text-based demo)")
    print("=" * 60)
    print("Type your order like a customer would. Type 'quit' to leave the call.\n")


def run_call() -> bool:
    """Runs one customer call. Returns True if the user wants to start another."""
    agent = OrderAgent()
    print(f"Agent: {agent.greeting()}")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAgent: Call ended.")
            return False

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Agent: No problem, have a great day!")
            return False

        reply = agent.process_message(user_input)
        print(f"Agent: {reply}")

        if agent.order_finalized:
            again = input("\n[Order saved to orders.db. Start a new order? (y/n)] ").strip().lower()
            return again.startswith("y")


def main() -> None:
    print_welcome()
    keep_going = True
    while keep_going:
        keep_going = run_call()


if __name__ == "__main__":
    main()
