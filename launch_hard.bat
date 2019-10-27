call C:\Users\Victor\Anaconda3\Scripts\activate.bat
call conda activate GYM_ENV_RL

python train.py --scenario=Enduro-v0
python train.py --scenario=Atlantis-v0
python train.py --scenario=BattleZone-v0
python train.py --scenario=Alien-v0

pause