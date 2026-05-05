import asyncio
import websockets

async def listen():
    uri = "ws://localhost:8000/ws/predictions"
    try:
        async with websockets.connect(uri) as websocket:
            print("🟢 Connected to WebSocket! Waiting for predictions...")
            while True:
                message = await websocket.recv()
                print(f"🚨 New Alert: {message}")
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(listen())