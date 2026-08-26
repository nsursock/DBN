# Bench FPS

cd ~/Desktop/dirty-bot-natured && source .venv/bin/activate

clear; 

PYTHONPATH=scripts:tests:utils .venv/bin/python utils/bench/scale.py normal --env cartpole --algo ppo --max-envs 100_000
PYTHONPATH=scripts:tests:utils .venv/bin/python utils/bench/scale.py normal --env pendulum --algo sac --max-envs 100_000
PYTHONPATH=scripts:tests:utils .venv/bin/python utils/bench/scale.py normal --env pendulum --algo td3 --max-envs 100_000

