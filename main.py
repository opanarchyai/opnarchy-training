import asyncio
import json
import threading
import os
import numpy as np
from aiohttp import web

state = {
    "episode": 0,
    "current_reward": 0,
    "best_reward": -999,
    "pickup_success": False,
    "progress_percent": 0,
    "total_steps": 0,
    "joint_angles": [0] * 17,
    "robot_position": [0, 0, 0],
    "is_alive": True
}

connected_clients = set()

def training_loop():
    try:
        import gymnasium as gym
        env = gym.make("Humanoid-v4")
        obs, _ = env.reset()
        total_episodes = 999999
        episode_reward = 0
        episode_count = 0
        best_reward = -999

        while episode_count < total_episodes:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            state["total_steps"] += 1

            joint_angles = obs[:17].tolist()
            state["joint_angles"] = [
                round(float(a), 4) for a in joint_angles
            ]
            state["current_reward"] = round(episode_reward, 2)
            state["is_alive"] = not terminated

            if terminated or truncated:
                episode_count += 1
                r = round(episode_reward, 2)

                if r > best_reward:
                    best_reward = r

                state["episode"] = episode_count
                state["best_reward"] = round(best_reward, 2)
                state["progress_percent"] = round(
                    episode_count / total_episodes * 100, 4
                )

                print(
                    f"Episode {episode_count} | "
                    f"Reward: {r} | "
                    f"Best: {best_reward}"
                )

                obs, _ = env.reset()
                episode_reward = 0

    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()

async def broadcast_loop():
    last_episode = -1
    last_step = -1
    while True:
        await asyncio.sleep(0.1)
        current_step = state["total_steps"]
        if (state["episode"] != last_episode or
                current_step != last_step):
            last_episode = state["episode"]
            last_step = current_step
            if connected_clients:
                message = json.dumps({
                    "type": "training_update",
                    "episode": state["episode"],
                    "current_reward": state["current_reward"],
                    "best_reward": state["best_reward"],
                    "progress_percent": state["progress_percent"],
                    "total_steps": state["total_steps"],
                    "joint_angles": state["joint_angles"],
                    "is_alive": state["is_alive"]
                })
                await asyncio.gather(
                    *[client.send_str(message)
                      for client in connected_clients],
                    return_exceptions=True
                )

async def http_handler(request):
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Opnarchy Humanoid Training</title>
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
            .alive {{ color: #00ff88; }}
            .dead {{ color: #ff4444; }}
        </style>
    </head>
    <body>
        <h1>Opnarchy Humanoid Training</h1>
        <div class="stat">
            Episode: {state["episode"]}
        </div>
        <div class="stat">
            Reward: {state["current_reward"]}
        </div>
        <div class="stat">
            Best: {state["best_reward"]}
        </div>
        <div class="stat">
            Steps: {state["total_steps"]}
        </div>
        <div class="stat">
            Progress: {state["progress_percent"]}%
        </div>
        <div class="stat {'alive' if state['is_alive'] else 'dead'}">
            Status: {'Walking' if state['is_alive'] else 'Fell Over'}
        </div>
        <div class="stat">
            Joints: {len(state["joint_angles"])} active
        </div>
    </body>
    </html>
    """
    return web.Response(text=html, content_type="text/html")

async def handle_ws_upgrade(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    connected_clients.add(ws)
    print(f"Client connected. Total: {len(connected_clients)}")
    try:
        await ws.send_str(json.dumps({
            "type": "welcome",
            "message": "Opnarchy Humanoid Training",
            "current_state": state
        }))
        async for msg in ws:
            pass
    finally:
        connected_clients.discard(ws)
    return ws

async def main():
    t = threading.Thread(target=training_loop, daemon=True)
    t.start()
    print("Humanoid training started")

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

if __name__ == "__main__":
    asyncio.run(main())
```

---

Also update **requirements.txt** to this:
```
gymnasium[mujoco]==0.29.1
numpy==1.23.5
websockets==11.0.3
aiohttp==3.8.5
setuptools==68.0.0
