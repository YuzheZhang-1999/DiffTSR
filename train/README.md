## Run the following four steps to train the entire DiffTSR model

## cd to the root path
```bash
cd DiffTSR/
```

## Stpe 0
```bash
bash train/0_step0_train_IDM_VAE.sh
```

## Step 1
```bash
bash train/1_step1_train_IDM.sh
```

## Step 2
```bash
bash train/2_step2_train_TDM.sh
```

## Step 3
```bash
bash train/3_step3_train_DiffTSR.sh
```