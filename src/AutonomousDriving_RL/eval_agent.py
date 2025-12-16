# eval_agent.py
import os
import sys
import numpy as np  # 🔥 已添加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from carla_env.carla_env_multi_obs import CarlaEnvMultiObs


def main():
    model_path = "final_model.zip"
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return

    print("🔄 加载模型...")
    model = PPO.load(model_path)
    print("✅ 模型加载成功！")

    # 🔥 keep_alive_after_exit=True → 车辆不销毁
    env = CarlaEnvMultiObs(keep_alive_after_exit=True)

    try:
        obs, _ = env.reset()
        print("▶️ 开始驾驶演示（运行 200 步）...")

        for step in range(200):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            if step % 50 == 0:
                x, y, vx, vy = obs
                speed = np.linalg.norm([vx, vy])
                print(f"  Step {step}: 位置=({x:.1f}, {y:.1f}), 速度={speed:.2f} m/s")

        print("✅ 演示完成！车辆保留在 CARLA 中。")
    except KeyboardInterrupt:
        print("\n🛑 用户中断")
    finally:
        env.close()


if __name__ == "__main__":
    main()