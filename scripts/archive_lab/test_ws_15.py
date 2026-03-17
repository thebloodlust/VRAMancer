import asyncio
import websockets

async def handler(websocket):
    print("NOUVELLE CONNEXION", websocket.request.path)
    try:
        await websocket.send("Hello")
        print("Envoyé. Attente message...")
        async for message in websocket:
            print("Reçu:", message)
    except Exception as e:
        print("Erreur:", e)
    finally:
        print("Fermé code:", websocket.close_code)

async def main():
    print("Démarrage...")
    # Serve on random port
    async with websockets.serve(handler, "127.0.0.1", 5061):
        await asyncio.Future()  # run forever

asyncio.run(main())
