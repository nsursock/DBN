# Bench FPS

cd ~/Desktop/dirty-bot-natured && source .venv/bin/activate

clear; 

.venv/bin/python utils/bench/scale.py normal --env cartpole --algo ppo --max-envs 100_000
.venv/bin/python utils/bench/scale.py normal --env pendulum --algo sac --max-envs 100_000
.venv/bin/python utils/bench/scale.py normal --env pendulum --algo td3 --max-envs 100_000

