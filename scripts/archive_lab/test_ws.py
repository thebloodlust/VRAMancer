import asyncio
import websockets
async def hello():
    try:
        async with websockets.connect("ws://127.0.0.1:5060/?device=smartphone&vram=4&battery=50") as ws:
            print("Connected!")
            await asyncio.sleep(1)
    except Exception as e:
        print("Error:", e)
asyncio.run(hello())
