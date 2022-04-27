# KLUE_DST_as_Prompting
This repository is Korean version implementation of "[Dialogue State Tracking with a Language Model using Schema-Driven Prompting](https://aclanthology.org/2021.emnlp-main.404/)" on [KLUE-DST (a.k.a WoS)](https://github.com/KLUE-benchmark/KLUE) dataset.

<p align="center">
  <img src="./sdp-dst.png" width="80%" height="80%">
</p>

## Leaderboard
**All our scores are calculated with the validation set** because the test set is not open to public. I submitted the codes
to the [KLUE-Benchmark](https://klue-benchmark.com/tasks/73/data/download) and I am waiting for the evaluation results
for the test set.
|Model|Joint Goal Accuracy|Slot F1-Score|
|-----|-------------------|-------------|
|KLUE-RoBERTa-small|46.62 %|91.44 %|
|KLUE-BERT-base|46.64 %|91.61 %|
|KLUE-RoBERTa-base|47.49 %|91.64 %|
|KLUE-RoBERTa-large|50.22 %|92.23 %|
|**KE-T5-base (Ours)**|**71.19 %**|**99.25 %**|

The pre-trained LM used in this repository is [KE-T5-base](https://github.com/AIRC-KETI/ke-t5).

## Installation
This repository is available in Ubuntu 20.04 LTS, and it is not tested in other OS.
```
conda create -n dst_prompt python=3.7.10
conda activate dst_prompt
cd KLUE_DST_as_Prompting
pip install -r requirements.txt
```

## Download KLUE-DST Dataset
You can download the dataset from [KLUE-Benchmark](https://klue-benchmark.com/tasks/73/data/download) or the
following commands.

```
cd kluewos11
wget https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000073/data/wos-v1.1.tar.gz
tar -xvf wos-v1.1.tar.gz
cd wos-v1.1/
mv ontology.json wos-v1.1_dev.json wos-v1.1_dev_sample_10.json wos-v1.1_train.json ..
cd ..
rm wos-v1.1.tar.gz
rm -r wos-v1.1
```
Additionally, I put `schema.json` which is made by myself in the `kluewos11` folder for prompt-based fine-tuning.

## Preprocess Data
You can get `dev.idx`, `dev.json`, `train.idx`, and `train.json` files after pre-processing in the `kluewos11` folder.
```
cd ..
sh preprocess.sh
```

## Prompt-based Fine-tuning
Please set the arguments `CUDA_VISIBLE_DEVICES`, `gradient_accumulation_steps`, `per_device_train_batch_size`, and `per_device_eval_batch_size` in `train.sh` to suit your learning environment first, and then
```
sh train.sh
```
Fine-tuning takes approximately 24 hours on 2 NVIDIA Titan RTX. 

## Evaluation 
Please make prediction `.json` file before evaluation. `--checkpoint_model_path` should be changed by yourself.
```
sh postprocess.sh
```
You can get the evaluation scores on your terminal with the prediction `.json` file.
```
sh get_metrics.sh
```

## Reference
This repository is based on the following paper:
```bib
@inproceedings{lee2021dialogue,
  title={Dialogue State Tracking with a Language Model using Schema-Driven Prompting},
  author={Lee, Chia-Hsuan and Cheng, Hao and Ostendorf, Mari},
  booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  pages={4937--4949},
  year={2021}
}
```
