import argparse
from io import TextIOWrapper
import os
import json
from tqdm import tqdm


def preprocess_kluewos11(
    dialog_file_dir: str,
    schema: list,
    output: TextIOWrapper,
    idx_output: TextIOWrapper,
    excluded_domains: list,
    domain_idx: dict,
    args,
):
    """Preprocess KLUE-DST (a.k.a WoS) v1.1 data.
    input = {
        `dialogue`: '[user] U_usr [system] U_sys [domain] dom dom_desc [slot] slot slot_desc [candidates] pv',
        `state`: 'active slot value'
        }

    Args:
        dialog_file_dir (str): raw dialogue .json files directory.
        schema (list): a schema .json file directory.
        output (TextIOWrapper): a file to save preprocessed dialogue data.
        idx_output (TextIOWrapper): a file to save preprocessed dialogue indices data.
        excluded_domains (list): domains that are not in test set.
        domain_idx (dict): indices for all domains.
    """
    dialog_filename = dialog_file_dir.split("/")[-1]

    with open(file=dialog_file_dir, mode="r", encoding="utf-8") as dialog_file_dir:
        # dialog_data: list(dict)
        dialog_data = json.load(dialog_file_dir)

    for dialog_idx in tqdm(range(len(dialog_data)), desc="Preprocess dialogues", total=len(dialog_data)):
        # initialize current dialog (str)
        current_dialog = ""

        # dialog: dict
        dialog = dialog_data[dialog_idx]
        # single_dialog (list(dict)): a single dialogue for each `dialogue_id`
        for single_dialog in dialog["dialogue"]:
            # speaker (str): a speaker for each turn
            speaker = " [" + str.lower(single_dialog["role"]) + "] "
            # uttr (str): an utterance for each turn
            uttr = single_dialog["text"]
            # add the speaker and utterance to the current dialogue
            current_dialog += speaker
            current_dialog += uttr

            if single_dialog["role"] == "user":
                # initialize active slot values
                active_slot_values = {}
                for slotvalue in single_dialog["state"]:
                    # value (str): a value for each user turn
                    slot = str.split(slotvalue, sep="-")[:-1]
                    slot = slot[0] + "-" + slot[1]
                    value = str.split(slotvalue, sep="-")[-1]
                    # active_slot_values (dict(str)): slot-value pairs for each user turn
                    active_slot_values[slot] = value

                # iterate thourgh each domain-slot pair in each user turn
                for domain in schema:
                    # skip domains that are not in the test set
                    if domain["domain"] in excluded_domains:
                        continue

                    # slots (list(dict)): containing slotnames, slot descriptions, and categorical states
                    slots = domain["slots"]
                    for slot in slots:
                        domain_name, slot_name = str.split(slot["name"], sep="-")

                        # generate schema prompt with or without natural langauge descriptions
                        schema_prompt = ""
                        # add domain prompt
                        schema_prompt += (
                            " [domain] " + domain_name + " " + domain["description"]
                            if args.use_domain_desc
                            else domain_name
                        )
                        # add slot prompt
                        schema_prompt += (
                            " [slot] " + slot_name + " " + slot["description"] if args.use_slot_desc else slot_name
                        )
                        if args.use_possible_cat_value:
                            # only append possible values if the slot is categorical
                            if slot["is_categorical"]:
                                candidate_values = ", ".join(slot["possible_values"])
                                # add candidate values prompt
                                schema_prompt += " [candidates] " + candidate_values

                        if slot["name"] in active_slot_values.keys():
                            # target_value (str): a target value for each turn
                            target_value = active_slot_values[slot["name"]]
                        else:
                            # special token for non-active slots
                            target_value = "none"

                        # make a form of input sequences
                        dialog_line = {"dialogue": current_dialog + schema_prompt, "state": target_value}
                        output.write(json.dumps(dialog_line, ensure_ascii=False))
                        output.write("\n")

                        # write idx file for postprocessing decoding
                        idx_for_dec = [
                            dialog_filename,
                            str(dialog_idx),
                            str(domain_idx[domain_name]),
                            domain_name,
                            slot_name,
                        ]
                        idx_output.write("|||".join(idx_for_dec))
                        idx_output.write("\n")


def main():
    parser = argparse.ArgumentParser()

    # set arguments
    parser.add_argument("--data_name", default=None, type=str, required=True, help="Data files name to preprocess.")
    parser.add_argument("--data_dir", default=None, type=str, required=True, help="Data files directory to preprocess.")
    parser.add_argument(
        "--schema_dir", default=None, type=str, required=True, help="Schema files directory to preprocess."
    )
    parser.add_argument(
        "--use_domain_desc",
        default=False,
        type=bool,
        required=True,
        help="Whether or not to use domain descriptions to input sequences.",
    )
    parser.add_argument(
        "--use_slot_desc",
        default=False,
        type=bool,
        required=True,
        help="Whether or not to use slot descriptions to input sequences.",
    )
    parser.add_argument(
        "--use_possible_cat_value",
        default=False,
        type=bool,
        required=True,
        help="Whether or not to use possible values as suffix for categorical slots.",
    )

    args = parser.parse_args()

    print(vars(args))

    # read schema file
    with open(file=args.schema_dir, mode="r", encoding="utf-8") as f:
        schema = json.load(f)

    # set domain indicies
    domain_idx = {
        "관광": 0,
        "숙소": 1,
        "식당": 2,
        "지하철": 3,
        "택시": 4,
    }

    # skip domains that are not in the test set
    excluded_domains = []

    for run_type in ["train", "dev"]:
        print(f"-------- Start to preprocess {args.data_name} {run_type} set --------")

        # set output, in advance
        output = open(os.path.join(args.data_dir, f"{run_type}.json"), mode="w")
        idx_output = open(os.path.join(args.data_dir, f"{run_type}.idx"), mode="w")
        # dialog_files (list(str)): dialogue data .json files
        dialog_file_dir = os.path.join(args.data_dir, f"wos-v1.1_{run_type}.json")

        # preprocess
        if dialog_file_dir.split("/")[-1] != "schema.json":
            preprocess_kluewos11(
                dialog_file_dir=dialog_file_dir,
                schema=schema,
                output=output,
                idx_output=idx_output,
                excluded_domains=excluded_domains,
                domain_idx=domain_idx,
                args=args,
            )

        idx_output.close()
        output.close()

    # notify finishing
    print(f"-------- Finish preprocessing of {args.data_name} --------")


if __name__ == "__main__":
    main()
