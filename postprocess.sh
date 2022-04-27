python postprocess.py \
    --dev_dir ./kluewos11/dev.json \
    --pretrained_model_name_or_path KETI-AIR/ke-t5-base \
    --checkpoint_model_path ./out/ke_t5b_kluewos11/checkpoint-124014/pytorch_model.bin \
    --encoder_max_seq_len 512 \
    --decoder_max_seq_len 32 \
    --pred_dir ./kluewos11/pred.json