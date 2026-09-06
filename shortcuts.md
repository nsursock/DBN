# Bench FPS

cd ~/Desktop/Work/dirty-bot-natured && source .venv/bin/activate

clear; 

.venv/bin/python utils/bench/scale.py normal --env cartpole --algo ppo \
  --envs 1024 2048 4096 8192 16384 32768 65536
.venv/bin/python utils/bench/scale.py normal --env pendulum --algo sac \
  --envs 1024 2048 4096 8192 16384 32768 65536
.venv/bin/python utils/bench/scale.py normal --env pendulum --algo td3 \
  --envs 1024 2048 4096 8192 16384 32768 65536



cd ~/Desktop/Work/dirty-bot-natured && source .venv/bin/activate

clear; 
.venv/bin/python utils/bench/scale.py normal --env cartpole --algo ppo --max-envs 1_000
.venv/bin/python utils/bench/scale.py normal --env pendulum --algo sac --max-envs 1_000
.venv/bin/python utils/bench/scale.py normal --env pendulum --algo td3 --max-envs 1_000



# PPO on CartPole
.venv/bin/python utils/bench/solve.py --target cartpole_ppo --envs 32 64 128 256 512

# SAC on Pendulum
.venv/bin/python utils/bench/solve.py --target pendulum_sac --envs 32 64 128 256 512 --max-timesteps 1_000_000

# TD3 on Pendulum
.venv/bin/python utils/bench/solve.py --target pendulum_td3 --envs 32 64 128 256 512 --max-timesteps 1_000_000