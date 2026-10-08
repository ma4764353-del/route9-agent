"""Print every logged call: the full conversation, then whether it ended in a saved order."""

import json
import sqlite3

from order_agent import DB_PATH


def main() -> None:
    conn = sqlite3.connect(DB_PATH)

    call_ids = [
        row[0]
        for row in conn.execute(
            "SELECT DISTINCT call_id FROM conversation_log ORDER BY MIN(timestamp) OVER (PARTITION BY call_id)"
        )
    ]
    # Some sqlite builds don't support window functions in this shorthand; fall back if needed.
    if not call_ids:
        call_ids = [row[0] for row in conn.execute("SELECT DISTINCT call_id FROM conversation_log")]

    if not call_ids:
        print("No logged calls yet. Run main.py and place an order first.")
        return

    for call_id in call_ids:
        messages = conn.execute(
            "SELECT timestamp, role, message FROM conversation_log WHERE call_id = ? ORDER BY id",
            (call_id,),
        ).fetchall()

        print("=" * 60)
        print(f"Call {call_id}  (started {messages[0][0]})")
        print("=" * 60)
        for _, role, message in messages:
            label = "You " if role == "customer" else "Agent"
            print(f"{label}: {message}")

        order = conn.execute(
            "SELECT id, total, items_json FROM orders WHERE call_id = ?", (call_id,)
        ).fetchone()
        print("-" * 60)
        if order:
            order_id, total, items_json = order
            print(f"-> Order #{order_id} saved. Total: ${total:.2f}")
            for item in json.loads(items_json):
                mods = f" ({', '.join(item['modifiers'])})" if item["modifiers"] else ""
                print(f"     {item['quantity']}x {item['name']}{mods}")
        else:
            print("-> No order saved for this call (cancelled or left incomplete).")
        print()

    conn.close()


if __name__ == "__main__":
    main()
