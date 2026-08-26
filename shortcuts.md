# Bench FPS

cd ~/Desktop/dirty-bot-natured && source .venv/bin/activate

clear; PYTHONPATH=scripts:tests:utils .venv/bin/python utils/bench_fps.py normal --env cartpole --algo ppo --max-envs 100_000 --plateau-pct -100
clear; PYTHONPATH=scripts:tests:utils .venv/bin/python utils/bench_fps.py normal --env pendulum --algo sac --max-envs 100_000 --plateau-pct -100

