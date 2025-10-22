srun -p Pixel -N1 --gres=gpu:8 --cpus-per-task=128 python train/main.py --scale_lr False --logdir train/logs/step2_train_TDM/ --base train/config/step2_train_TDM.yaml -t True --gpus 0,1,2,3,4,5,6,7
## Or run the following command.
## python train/main.py --scale_lr False --logdir train/logs/step2_train_TDM/ --base train/config/step2_train_TDM.yaml -t True --gpus 0,