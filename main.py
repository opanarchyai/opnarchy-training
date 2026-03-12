import asyncio
import json
import threading
import os
import numpy as np
from aiohttp import web
import websockets
import pybullet as p
import pybullet_data

state = {
    "episode": 0,
    "current_reward": 0,
    "best_reward": -999,
    "pickup_success": False,
    "progress_percent": 0,
    "total_steps": 0,
}

connected_clients = set()

def training_loop():
    try:
        p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        total_episodes = 1000

        for episode in range(total_episodes):
            p.resetSimulation()
            p.setGravity(0, 0, -9.81)
            p.loadURDF("plane.urdf")

            robot = p.loadURDF(
                "kuka_iiwa/model.urdf",
                [0, 0, 0],
                useFixedBase=True
            )

            ball_pos = [
                0.5 + np.random.uniform(-0.1, 0.1),
                np.random.uniform(-0.1, 0.1),
                0.1
            ]
            ball = p.loadURDF("sphere_small.urdf", ball_pos)

            episode_reward = 0
            pickup = False
            prev_distance = None

            for step in range(200):
                joint_angles = [
                    np.random.uniform(-0.5, 0.5)
                    for _ in range(7)
                ]

                for i in range(7):
                    p.setJointMotorControl2(
                        robot, i,
                        p.POSITION_CONTROL,
                        targetPosition=joint_angles[i]
                    )

                p.stepSimulation()

                ball_now, _ = p.getBasePositionAndOrientation(ball)
                end_effector = p.getLinkState(robot, 6)[0]

                distance = np.linalg.norm(
                    np.array(end_effector) - np.array(ball_now)
                )

                if prev_distance is not None:
                    if distance < prev_distance:
                        episode_reward += 1
                    else:
                        episode_reward -= 1

                prev_distance = distance

                if distance < 0.12:
                    episode_reward += 100
                    pickup = True
                    break

                state["total_steps"] += 1

            ep = episode + 1
            r = round(episode_reward, 2)

            state["episode"] = ep
            state["current_reward"] = r
            state["pickup_success"] = pickup
            state["progress_percent"] = round(
                ep / total_episodes * 100, 1
            )

            if r > state["best_reward"]:
                state["best_reward"] = r

            print(f"Episode {ep} | Reward: {r} | Pickup: {pickup}")

    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()

async def broadcast_loop():
    last_episode = -1
    while True:
        await asyncio.sleep(1)
        if state["episode"] != last_episode:
            last_episode = state["episode"]
            if connected_clients:
                message = json.dumps({
                    "type": "episode_update",
                    "episode": state["episode"],
                    "current_reward": state["current_reward"],
                    "best_reward": state["best_reward"],
                    "pickup_success": state["pickup_success"],
                    "progress_percent": state["progress_percent"],
                    "total_steps": state["total_steps"]
                })
                await asyncio.gather(
                    *[client.send(message)
                      for client in connected_clients],
                    return_exceptions=True
                )

async def ws_handler(websocket, path=None):
    connected_clients.add(websocket)
    print(f"Client connected. Total: {len(connected_clients)}")
    try:
        await websocket.send(json.dumps({
            "type": "welcome",
            "message": "Connected to Opnarchy Training",
            "current_state": state
        }))
        await websocket.wait_closed()
    finally:
        connected_clients.discard(websocket)

async def http_handler(request):
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Opnarchy Training</title>
        <meta http-equiv="refresh" content="2">
        <style>
            body {{
                background: #0a0a0a;
                color: #00d4ff;
                font-family: monospace;
                padding: 40px;
            }}
            h1 {{ color: #7b2fff; }}
            .stat {{
                font-size: 1.4em;
                margin: 10px 0;
            }}
            .success {{ color: #00ff88; }}
        </style>
    </head>
    <body>
        <h1>Opnarchy Robot Training</h1>
        <div class="stat">Episode: {state["episode"]}</div>
        <div class="stat">
            Current Reward: {state["current_reward"]}
        </div>
        <div class="stat">
            Best Reward: {state["best_reward"]}
        </div>
        <div class="stat success">
            Pickup: {state["pickup_success"]}
        </div>
        <div class="stat">
            Progress: {state["progress_percent"]}%
        </div>
        <div class="stat">
            Steps: {state["total_steps"]}
        </div>
        <div class="stat">
            WebSocket: wss://web-production-91a926.up.railway.app/ws
        </div>
    </body>
    </html>
    """
    return web.Response(text=html, content_type="text/html")

async def main():
    t = threading.Thread(target=training_loop, daemon=True)
    t.start()
    print("Training started")

    asyncio.create_task(broadcast_loop())

    port = int(os.getenv("PORT", 8000))

    app = web.Application()
    app.router.add_get("/", http_handler)
    app.router.add_get("/ws", handle_ws_upgrade)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    print(f"Server on port {port}")
    await asyncio.Future()

async def handle_ws_upgrade(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    connected_clients.add(ws)
    print(f"WS Client connected. Total: {len(connected_clients)}")
    try:
        await ws.send_str(json.dumps({
            "type": "welcome",
            "message": "Connected to Opnarchy Training",
            "current_state": state
        }))
        async for msg in ws:
            pass
    finally:
        connected_clients.discard(ws)
    return ws

if __name__ == "__main__":
    asyncio.run(main())
```

---

This runs everything on one single port that Railway assigns. WebSocket connects via:
```
wss://web-production-91a926.up.railway.app/ws
