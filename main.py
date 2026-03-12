import asyncio
import json
import threading
import time
import numpy as np
from aiohttp import web
import websockets

# Training state shared between threads
state = {
    "episode": 0,
    "current_reward": 0,
    "best_reward": -999,
    "pickup_success": False,
    "progress_percent": 0,
    "total_steps": 0,
    "running": True
}

connected_clients = set()

def training_loop():
    try:
        import pybullet as p
        import pybullet_data
        from stable_baselines3 import SAC
        import gymnasium as gym
        from gymnasium import spaces

        class RobotArmEnv(gym.Env):
            def __init__(self):
                super().__init__()
                self.action_space = spaces.Box(
                    low=-1, high=1, shape=(7,), dtype=np.float32
                )
                self.observation_space = spaces.Box(
                    low=-10, high=10, shape=(13,), dtype=np.float32
                )
                self.physics_client = p.connect(p.DIRECT)
                p.setAdditionalSearchPath(pybullet_data.getDataPath())
                self.robot = None
                self.ball = None
                self.step_count = 0
                self.reset()

            def reset(self, seed=None):
                p.resetSimulation()
                p.setGravity(0, 0, -9.81)
                p.loadURDF("plane.urdf")
                self.robot = p.loadURDF(
                    "kuka_iiwa/model.urdf",
                    [0, 0, 0],
                    useFixedBase=True
                )
                ball_start = [
                    0.5 + np.random.uniform(-0.1, 0.1),
                    0.0 + np.random.uniform(-0.1, 0.1),
                    0.1
                ]
                self.ball = p.loadURDF(
                    "sphere_small.urdf",
                    ball_start
                )
                self.step_count = 0
                return self._get_obs(), {}

            def _get_obs(self):
                joint_states = p.getJointStates(
                    self.robot, range(7)
                )
                joint_angles = [s[0] for s in joint_states]
                ball_pos, _ = p.getBasePositionAndOrientation(
                    self.ball
                )
                end_effector = p.getLinkState(
                    self.robot, 6
                )[0]
                return np.array(
                    joint_angles + list(ball_pos) +
                    list(end_effector),
                    dtype=np.float32
                )

            def step(self, action):
                for i in range(7):
                    p.setJointMotorControl2(
                        self.robot, i,
                        p.POSITION_CONTROL,
                        targetPosition=action[i] * 2
                    )
                p.stepSimulation()
                self.step_count += 1

                ball_pos, _ = p.getBasePositionAndOrientation(
                    self.ball
                )
                end_effector = p.getLinkState(
                    self.robot, 6
                )[0]

                distance = np.linalg.norm(
                    np.array(end_effector) -
                    np.array(ball_pos)
                )

                pickup = distance < 0.12
                reward = -distance + (100 if pickup else 0)
                done = pickup or self.step_count >= 200

                state["total_steps"] += 1
                state["progress_percent"] = round(
                    state["total_steps"] / 1000000 * 100, 1
                )

                return self._get_obs(), reward, done, False, {}

        env = RobotArmEnv()
        model = SAC("MlpPolicy", env, verbose=0)

        class EpisodeCallback:
            def __init__(self):
                self.episode_rewards = []
                self.current_reward = 0

            def on_step(self, locals_dict):
                reward = locals_dict.get("rewards", [0])[0]
                done = locals_dict.get("dones", [False])[0]
                self.current_reward += reward

                if done:
                    ep = state["episode"] + 1
                    r = round(self.current_reward, 2)
                    pickup = self.current_reward > 50

                    state["episode"] = ep
                    state["current_reward"] = r
                    state["pickup_success"] = pickup
                    if r > state["best_reward"]:
                        state["best_reward"] = r

                    self.current_reward = 0
                    print(f"Episode {ep} | "
                          f"Reward: {r} | "
                          f"Pickup: {pickup}")
                return True

        from stable_baselines3.common.callbacks import BaseCallback

        class SB3Callback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.ep_callback = EpisodeCallback()
                self.current_reward = 0

            def _on_step(self):
                reward = self.locals["rewards"][0]
                done = self.locals["dones"][0]
                self.current_reward += reward

                if done:
                    ep = state["episode"] + 1
                    r = round(self.current_reward, 2)
                    pickup = self.current_reward > 50

                    state["episode"] = ep
                    state["current_reward"] = r
                    state["pickup_success"] = pickup
                    if r > state["best_reward"]:
                        state["best_reward"] = r

                    self.current_reward = 0
                    print(f"Episode {ep} | "
                          f"Reward: {r} | "
                          f"Pickup: {pickup}")
                return True

        callback = SB3Callback()
        model.learn(
            total_timesteps=1000000,
            callback=callback
        )

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

async def ws_handler(websocket):
    connected_clients.add(websocket)
    print(f"Client connected. Total: {len(connected_clients)}")
    try:
        await websocket.send(json.dumps({
            "type": "welcome",
            "message": "Connected to Opnarchy Training Server",
            "current_state": state
        }))
        await websocket.wait_closed()
    finally:
        connected_clients.discard(websocket)
        print(f"Client disconnected. "
              f"Total: {len(connected_clients)}")

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
        <div class="stat">
            Episode: {state["episode"]}
        </div>
        <div class="stat">
            Current Reward: {state["current_reward"]}
        </div>
        <div class="stat">
            Best Reward: {state["best_reward"]}
        </div>
        <div class="stat success">
            Pickup Success: {state["pickup_success"]}
        </div>
        <div class="stat">
            Progress: {state["progress_percent"]}%
        </div>
        <div class="stat">
            Total Steps: {state["total_steps"]}
        </div>
    </body>
    </html>
    """
    return web.Response(text=html, content_type="text/html")

async def main():
    training_thread = threading.Thread(
        target=training_loop,
        daemon=True
    )
    training_thread.start()
    print("Training thread started")

    asyncio.create_task(broadcast_loop())

    app = web.Application()
    app.router.add_get("/", http_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 8000)
    await site.start()
    print("HTTP server started on port 8000")

    async with websockets.serve(
        ws_handler, "0.0.0.0", 8765
    ):
        print("WebSocket server started on port 8765")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
