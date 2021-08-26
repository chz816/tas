# Topic-Guided Abstractive Text Summarization: a Joint Learning Approach

## Local Setup
Tested with Python 3.7 via virtual environment. Clone the repo, go to the repo folder, setup the virtual environment, and install the required packages:
```bash
$ python3.7 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

### Install ```apex```
Based on the recommendation from HuggingFace, both finetuning and eval are 30% faster with ```--fp16```. For that you need to install ```apex```.
```bash
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Data
Create a directory for data used in this work named ```data```:
```bash
$ mkdir data
```

### CNN/DM
```bash
$ wget https://cdn-datasets.huggingface.co/summarization/cnn_dm_v2.tgz
$ tar -xzvf cnn_dm_v2.tgz
$ mv cnn_cln data/cnndm
```

### XSUM
```bash
$ wget https://cdn-datasets.huggingface.co/summarization/xsum.tar.gz
$ tar -xzvf xsum.tar.gz
$ mv xsum data/xsum
```

#### Preprocessing for XSUM
```bash
$ python preprocess_xsum.py
```

## Training
### CNN/DM
Our model is warmed up using ```sshleifer/distilbart-cnn-12-6```:
```bash
$ DATA_DIR=data/cnndm
$ OUTPUT_DIR=log/cnndm/alpha-0-beta-1-3e-5

$ python -m torch.distributed.launch --nproc_per_node=3  taas_finetune_trainer.py \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --learning_rate=3e-5 \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 64 \
  --do_train \
  --evaluation_strategy steps \
  --freeze_embeds \
  --freeze_encoder \
  --save_total_limit 5 \
  --save_steps 500 \
  --logging_steps 500 \
  --num_train_epochs 5 \
  --model_name_or_path sshleifer/distilbart-cnn-12-6 \
  --fp16 \
  --loss_alpha 0 \
  --loss_beta 1
```

### XSUM
Our model is warmed up using ```sshleifer/distilbart-xsum-12-6```:
```bash
$ DATA_DIR=data/xsum
$ OUTPUT_DIR=log/xsum/alpha-0.1-beta-1-4e-5-lda

$ python -m torch.distributed.launch --nproc_per_node=3  taas_finetune_trainer.py \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --learning_rate=4e-5 \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 64 \
  --do_train \
  --evaluation_strategy steps \
  --freeze_embeds \
  --freeze_encoder \
  --save_total_limit 5 \
  --save_steps 500 \
  --logging_steps 500 \
  --num_train_epochs 20 \
  --model_name_or_path sshleifer/distilbart-xsum-12-6 \
  --fp16 \
  --loss_alpha 0.1 \
  --loss_beta 1 \
  --topic_model_type 'LDA'
```

## Evaluation
We release the pre-trained checkpoints:
- [CNN/DM](https://drive.google.com/file/d/1zbJfkSSHDGdfeoC3XuaecAeOar1lQLw6/view?usp=sharing)
- [XSUM](https://drive.google.com/file/d/1O2gsvBmRX068NLphhJH2jMUheIYFMUnJ/view?usp=sharing)
### CNN/DM
CNN/DM requires an extra postprocessing step.
```bash
$ export DATA=cnndm
$ export DATA_DIR=data/$DATA
$ export CHECKPOINT_DIR=log/cnndm/alpha-0-beta-1-3e-5-load-topic-3e-4-val_avg_loss-1624.6068-step-56000/val_avg_loss-0.9381-step-3000
$ export OUTPUT_DIR=output/$DATA/alpha-0-beta-1-3e-5-load-topic-3e-4-val_avg_loss-1624.6068-step-56000-val_avg_loss-0.9381-step-3000

$ python -m torch.distributed.launch --nproc_per_node=3  taas_eval.py \
    --model_name sshleifer/distilbart-cnn-12-6  \
    --save_dir $OUTPUT_DIR \
    --data_dir $DATA_DIR \
    --bs 64 \
    --fp16 \
    --use_checkpoint \
    --checkpoint_path $CHECKPOINT_DIR
    
$ python postprocess_cnndm.py \
    --src_file $OUTPUT_DIR/test_generations.txt \
    --tgt_file $DATA_DIR/test.target
```

### XSUM
```bash
$ export DATA=xsum
$ export DATA_DIR=data/$DATA
$ export CHECKPOINT_DIR=log-dimhead/xsum/alpha-0.1-beta-1-3e-5-topic1024/val_avg_loss-100.3857-step-2500
$ export OUTPUT_DIR=output/$DATA/alpha-0.1-beta-1-3e-5-topic1024-val_avg_loss-100.3857-step-2500

$ python -m torch.distributed.launch --nproc_per_node=3  taas_eval.py \
    --model_name sshleifer/distilbart-xsum-12-6  \
    --save_dir $OUTPUT_DIR \
    --data_dir $DATA_DIR \
    --bs 64 \
    --fp16 \
    --use_checkpoint \
    --checkpoint_path $CHECKPOINT_DIR
```