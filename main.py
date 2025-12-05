from app.large_language_model import ChatAssistant

if __name__ == "__main__":
    assistant = ChatAssistant()
    while True:
        q = input("你: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            break
        print("助手:", assistant.stream_chat(q))