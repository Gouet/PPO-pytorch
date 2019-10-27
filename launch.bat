call C:\Users\Victor\Anaconda3\Scripts\activate.bat
call conda activate GYM_ENV_RL

# --scenario=BipedalWalkerHardcore-v2
# --scenario=BipedalWalkerHardcore-v2 --scenario=Breakout-v0

python train.py --scenario=Assault-v0
pause 