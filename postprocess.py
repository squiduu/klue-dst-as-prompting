import json
import torch
import argparse
from tqdm import tqdm

from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForSeq2SeqLM


def main(args):
    # get dialogs in dev dataset
    dev_dialogs = []
    dev_file = open(file=args.dev_file, mode="r", encoding="utf-8")
    for line in dev_file.readlines():
        dev_dialog = json.loads(line)
        dev_dialogs.append(dev_dialog["dialogue"])

    # set pre-trained tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model_name_or_path)
    # load fine-tuned checkpoint
    ckpt_params = torch.load(args.checkpoint_model_path, map_location="cuda")
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(ckpt_params)
    model.eval()

    preds = []
    for input_dialog in tqdm(dev_dialogs, total=len(dev_dialogs)):
        tokens = tokenizer.tokenize(input_dialog)
        if len(tokens) > args.encoder_max_seq_len:
            tokens = tokens[-args.encoder_max_seq_len :]
            modified_input_dialog = tokenizer.convert_tokens_to_string(tokens)
            enc_input = tokenizer(
                text=modified_input_dialog,
                padding=True,
                truncation=True,
                max_length=args.encoder_max_seq_len,
                return_tensors="pt",
            )
            output = model.generate(input_ids=enc_input["input_ids"], max_length=args.decoder_max_seq_len)
            pred = tokenizer.decode(output[0], skip_special_tokens=True)
        else:
            enc_input = tokenizer(
                text=input_dialog,
                padding=True,
                truncation=True,
                max_length=args.encoder_max_seq_len,
                return_tensors="pt",
            )
            output = model.generate(input_ids=enc_input["input_ids"], max_length=args.decoder_max_seq_len)
            pred = tokenizer.decode(output[0], skip_special_tokens=True)

        preds.append(pred)

    with open(file=args.pred_dir, mode="w", encoding="utf-8") as pred_file:
        json.dump(obj=preds, fp=pred_file, ensure_ascii=False)

    print("Post-processing is finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dev_dir", default="./kluewos11/dev.json", type=str)
    parser.add_argument("--pretrained_model_name_or_path", default="KETI-AIR/ke-t5-base", type=str)
    parser.add_argument(
        "--checkpoint_model_path", default="./out/ke_t5b_kluewos11/checkpoint-124014/pytorch_model.bin", type=str
    )
    parser.add_argument("--encoder_max_seq_len", default=512, type=int)
    parser.add_argument("--decoder_max_seq_len", default=32, type=int)
    parser.add_argument("--pred_dir", default="./kluewos11/pred.json", type=str)

    args = parser.parse_args()

    main(args)
