srun -p Pixel -N1 --gres=gpu:8 --cpus-per-task=128 python train/main.py --scale_lr False --logdir train/logs/step1_train_IDM/ --base train/config/step1_train_IDM.yaml -t True --gpus 0,1,2,3,4,5,6,7
## Or run the following command.
## python train/main.py --scale_lr False --logdir train/logs/step1_train_IDM/ --base train/config/step1_train_IDM.yaml -t True --gpus 0,