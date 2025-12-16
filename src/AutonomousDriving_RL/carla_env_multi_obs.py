# carla_env/carla_env_multi_obs.py
import carla
import numpy as np
import random
import time
from gymnasium import Env, spaces


class CarlaEnvMultiObs(Env):
    def __init__(self, keep_alive_after_exit=False):
        super(CarlaEnvMultiObs, self).__init__()
        self.client = None
        self.world = None
        self.vehicle = None
        self.actor_list = []
        self.frame_count = 0
        self.max_frames = 1000
        self.prev_x = 0.0
        self.spectator = None
        self.keep_alive = keep_alive_after_exit

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if self.client is None:
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(20.0)
            print("🔄 连接 CARLA...")
            self.world = self.client.get_world()
            # 🔥 清除所有现有 actor，避免冲突
            self._clear_existing_actors()
        else:
            if not self.keep_alive:
                self._destroy_actors()

        self.spawn_vehicle()

        # 🔥 等待几帧让物理稳定 + 强制刷新视角
        for _ in range(5):
            self.world.tick()
            time.sleep(0.05)
        self._update_spectator_view()

        self.frame_count = 0
        obs = self.get_observation()
        self.prev_x = obs[0]
        return obs, {}

    def _clear_existing_actors(self):
        """清除地图上所有车辆、行人、传感器"""
        actors = self.world.get_actors()
        vehicles = actors.filter('vehicle.*')
        walkers = actors.filter('walker.*')
        sensors = actors.filter('sensor.*')
        destroy_list = list(vehicles) + list(walkers) + list(sensors)
        if destroy_list:
            self.client.apply_batch_sync([
                carla.command.DestroyActor(x.id) for x in destroy_list
            ])
            print(f"🧹 清除了 {len(destroy_list)} 个现有 actor")

    def spawn_vehicle(self):
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        if not vehicle_bp:
            vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))

        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("❌ 地图中没有可用的 spawn points！")

        self.vehicle = None
        for i, transform in enumerate(spawn_points):
            # 强制 z 高度为 0.3，避免穿地或悬空
            safe_location = carla.Location(
                x=transform.location.x,
                y=transform.location.y,
                z=max(transform.location.z, 0.1) + 0.2
            )
            safe_transform = carla.Transform(safe_location, transform.rotation)

            try:
                self.vehicle = self.world.try_spawn_actor(vehicle_bp, safe_transform)
                if self.vehicle is not None:
                    self.actor_list.append(self.vehicle)
                    loc = safe_transform.location
                    print(
                        f"✅ 车辆生成成功: {self.vehicle.type_id} | 位置: ({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f}) | 使用 spawn 点 #{i}")
                    break
            except Exception:
                continue

        if self.vehicle is None:
            raise RuntimeError("❌ 所有 spawn 点均被占用，无法生成车辆！")

        self.spectator = self.world.get_spectator()
        print("🎥 第三人称视角已激活")

    def _update_spectator_view(self):
        if self.vehicle and self.spectator:
            vehicle_transform = self.vehicle.get_transform()
            offset = carla.Location(x=-6.0, z=2.5)
            spectator_location = vehicle_transform.transform(offset)
            spectator_rotation = carla.Rotation(
                pitch=-10,
                yaw=vehicle_transform.rotation.yaw + 180,
                roll=vehicle_transform.rotation.roll
            )
            self.spectator.set_transform(
                carla.Transform(spectator_location, spectator_rotation)
            )

    def _destroy_actors(self):
        if not self.keep_alive:
            for actor in self.actor_list:
                if actor and actor.is_alive:
                    actor.destroy()
            self.actor_list.clear()
            for _ in range(3):
                self.world.tick()
                time.sleep(0.1)

    def get_observation(self):
        if not self.vehicle or not self.vehicle.is_alive:
            return np.zeros(4, dtype=np.float32)
        loc = self.vehicle.get_location()
        vel = self.vehicle.get_velocity()
        return np.array([loc.x, loc.y, vel.x, vel.y], dtype=np.float32)

    def step(self, action):
        throttle, steer, brake = action
        control = carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake)
        )
        self.vehicle.apply_control(control)
        self.world.tick()
        self.frame_count += 1
        self._update_spectator_view()

        if not self.vehicle or not self.vehicle.is_alive:
            return np.zeros(4, dtype=np.float32), -10.0, True, False, {}

        obs = self.get_observation()
        x, y, vx, vy = obs
        speed = np.linalg.norm([vx, vy])
        forward_reward = 0.1 * (x - self.prev_x)
        speed_reward = 0.5 * speed
        penalty = -1.0 if (speed > 1.0 and abs(x - self.prev_x) < 0.1) else 0.0
        reward = forward_reward + speed_reward + penalty
        self.prev_x = x

        terminated = False
        truncated = self.frame_count >= self.max_frames
        return obs, reward, terminated, truncated, {}

    def close(self):
        if not self.keep_alive:
            self._destroy_actors()
        else:
            print("ℹ️ 车辆已保留，可在 CARLA 中继续观察！")
