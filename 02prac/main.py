import asyncio
from utils.node import app


input={"query":"Docling이 다른 라이브러리와 다른 점이 무엇인지 설명해줘"}
config={"configurable": {"thread_id": 0}}
async def main():
    async for event in app.astream_events(
        input=input, stream_mode="events", version="v2",config=config
        ):

        kind = event["event"]

        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"].content
            print(chunk, end="", flush=True)

        elif kind == "on_retriever_end ":
            print(event)

        elif kind == "on_chat_model_end":
            print("\n\n")
            

if __name__ == "__main__":
    asyncio.run(main())