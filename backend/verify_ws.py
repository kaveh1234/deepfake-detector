import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://localhost:8000/ws"
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to WebSocket Server")
            
            # Send start session command
            start_msg = {
                "type": "control",
                "data": {
                    "action": "start_session"
                }
            }
            await websocket.send(json.dumps(start_msg))
            print("Sent start_session")
            
            # Wait for response
            response = await websocket.recv()
            print(f"Received: {response}")
            
            data = json.loads(response)
            if data['type'] == 'session_update':
                print("Session verification SUCCESS")
            else:
                print("Session verification FAILED")
                
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(test_websocket())
