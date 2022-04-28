# KLUE-DST as Prompting
This repository is Korean version implementation of "[Dialogue State Tracking with a Language Model using Schema-Driven Prompting](https://aclanthology.org/2021.emnlp-main.404/)" on [KLUE-DST (a.k.a WoS)](https://github.com/KLUE-benchmark/KLUE) dataset.

<p align="center">
  <img src="./sdp-dst.png" width="70%" height="70%">
</p>

## Leaderboard
**All our scores are calculated with the validation set because the test set is not open to public**. I submitted the codes
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

## Examples
```python
import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("KETI-AIR/ke-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("KETI-AIR/ke-t5-base")

# please change the checkpoint path after fine-tuning yourself
ckpt_params = torch.load('./out/ke_t5b_kluewos11/checkpoint-124014/pytorch_model.bin', map_location='cpu')
model.resize_token_embeddings(len(tokenizer))
model.load_state_dict(ckpt_params)
model.eval()
```
```python
# example 1
dialog_history = "[user] 명동 쇼핑 거리에 대해 물어볼게 있는데 영업시간이랑 입장료, 주소를 알려주세요. \
                  [domain] 관광 가볼 만한 장소 또는 공간을 찾으세요 [slot] 이름 관광지의 이름이 무엇인지"
input_ids = tokenizer(dialog_history, return_tensors='pt').input_ids
value = model.generate(input_ids)
print(tokenizer.decode(value[0], skip_special_tokens=True))
>>> 명동 쇼핑 거리
```
```python
# example 2
dialog_history = "[user] 서울 북쪽에 경관이 좋은 공원을 찾고 있습니다. [domain] 관광 가볼 만한 장소 또는 공간을 찾으세요 \
                  [slot] 경치 좋은 관광지의 경치가 만족스러운지 [candidates] none, dontcare, yes, no"
input_ids = tokenizer(dialog_history, return_tensors='pt').input_ids
value = model.generate(input_ids)
print(tokenizer.decode(value[0], skip_special_tokens=True))
>>> yes
```

## Installation
This repository is available in Ubuntu 20.04 LTS, and it is not tested in other OS.
```
conda create -n klue_dst python=3.7.10
conda activate klue_dst
cd KLUE_DST_as_Prompting
pip install -r requirements.txt
```

## Download KLUE-DST Dataset
You can download the dataset with the following commands.
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
Additionally, I put `schema.json` which is made by myself in the `./kluewos11` for prompt-based fine-tuning.

## Preprocess Data
You can get `dev.idx`, `dev.json`, `train.idx`, and `train.json` in the `./kluewos11` after preprocessing.
```
cd ..
sh preprocess.sh
```

## Prompt-based Fine-tuning
Please set the training arguments `CUDA_VISIBLE_DEVICES`, `--gradient_accumulation_steps`, `--per_device_train_batch_size`, and `--per_device_eval_batch_size` in `train.sh` properly to suit your learning environment first, and then
```
sh train.sh
```
Fine-tuning takes approximately 24 hours on 2 NVIDIA Titan RTX for 3 epochs, also it can be different for each learning environment.

## Evaluation 
Please make `pred.json` before evaluation. `--checkpoint_model_path` should be changed by yourself.
```
sh postprocess.sh
```
You can get the evaluation scores on your terminal from the `pred.json`.
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
