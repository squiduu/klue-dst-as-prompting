import json
import argparse


def main(args):
    # get slot accuracy
    gold_values = []
    gold_file = open(file=args.gold_dir, mode="r", encoding="utf-8")
    for line in gold_file.readlines():
        gold_value = json.loads(line)
        gold_values.append(gold_value["state"])
    gold_file.close()

    pred_values = []
    with open(file=args.pred_dir, mode="r", encoding="utf-8") as f:
        pred_values = json.load(f)

    slot_acc = []
    for i in range(len(gold_values)):
        if pred_values[i] == gold_values[i]:
            slot_acc.append(1.0)
        else:
            slot_acc.append(0.0)

    print(f"Slot Accuracy: {sum(slot_acc) / len(slot_acc) * 100:.2f} %")

    # get joint goal accuracy
    jga = []
    for i in range(0, len(gold_values), 45):
        match = []
        for j in range(i, i + 44):
            if pred_values[j] == gold_values[j]:
                match.append(1.0)
            else:
                match.append(0.0)

        if match.count(1.0) == len(match):
            jga.append(1.0)
        else:
            jga.append(0.0)

    print(f"Joint Goal Accuracy: {sum(jga) / len(jga) * 100:.2f} %")

    # get F1 score
    #################################
    #                gold    gold   #
    #              | value | none | #
    # pred | value | TP    | FP   | #
    # pred | none  | FN    | TN   | #
    #################################
    tp, fp, fn, tn = 0.0, 0.0, 0.0, 0.0
    eps = 1e-20
    for i in range(len(gold_values)):
        if pred_values[i] != "none" and gold_values[i] != "none" and pred_values[i] == gold_values[i]:
            tp += 1.0
        if pred_values[i] != "none" and gold_values[i] == "none":
            fp += 1.0
        if pred_values[i] == "none" and gold_values[i] != "none":
            fn += 1.0
        if pred_values[i] == "none" and gold_values[i] == "none":
            tn += 1.0

    recall = tp / (tp + fn + eps)
    precision = tp / (tp + fp + eps)
    f1_score = 2 * precision * recall / (precision + recall)

    print(f"Slot F1-Score: {f1_score * 100:.2f} %")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_dir", default="./kleuwos11/dev.json", type=str)
    parser.add_argument("--pred_dir", default="./kleuwos11/pred.json", type=str)
    args = parser.parse_args()

    main(args)
