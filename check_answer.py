import json


base = {}

with open("baseline_results.jsonl") as f:
    for line in f:
        x=json.loads(line)
        base[x["idx"]] = x


with open("finetuned_results.jsonl") as f:
    for line in f:
        x=json.loads(line)

        if base[x["idx"]]["correct"] and not x["correct"]:

            print("题号:",x["idx"])

            print("\nBASE:")
            print(base[x["idx"]]["generated_text"])

            print("\nLORA:")
            print(x["generated_text"])

            break