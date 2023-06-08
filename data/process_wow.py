import json
from nlgeval import NLGEval
import re
import os

EOS = "<|endoftext|>"
scorer = NLGEval(
    no_skipthoughts=True,
    no_glove=True,
    metrics_to_omit=[
        "Bleu_1",
        "Bleu_2",
        "Bleu_3",
        "Bleu_4",
        # 'ROUGE_L',
        "CIDEr",
        "METEOR",
    ],
)


def get_pseudo_knowledge(knowledge, candidates, comment):
    if len(candidates) == 0:
        return knowledge
    scores = [
        scorer.compute_individual_metrics([s], comment)["ROUGE_L"] for s in candidates
    ]
    max_score = max(scores)
    index = scores.index(max_score)
    return candidates[index]


def get_checked_passage(checked_sentence, passages, checked_passage):
    # if checked_passage is empty, search the knowledge in passages
    # if not found too, infer from name chosen_xxx_0
    if len(checked_passage):
        return list(checked_passage.values())[0].strip()
    knowledge = list(checked_sentence.values())[0].strip()
    for passage_ in passages:
        for key, candidates in passage_.items():
            if knowledge in candidates:
                return key
    name = re.search(r"chosen_(.*)_\d", list(checked_sentence.keys())[0])
    if name is None:
        name = re.search(r"partner_(.*)_\d", list(checked_sentence.keys())[0])
    if name is None:
        name = re.search(r"self_(.*)_\d", list(checked_sentence.keys())[0])
    if name is None:
        import pdb

        pdb.set_trace()
    return name.group(1).replace("_", " ")


def convert(path, save_path):
    with open(path, "r") as f:
        data = json.load(f)

    converted_data = []
    acc = 0
    for dialog in data:
        dialog_history = []
        for turn in dialog["dialog"]:
            if (
                "checked_sentence" in turn
                and isinstance(turn["checked_sentence"], dict)
                and len(turn["checked_sentence"]) > 0
            ):
                assert len(turn["checked_sentence"]) == 1
                knowledge = list(turn["checked_sentence"].values())[0].strip()
                if knowledge != "no_passages_used":
                    retrieved_passages = turn["retrieved_passages"]
                    candidates = []
                    for topic_docs in retrieved_passages:
                        docs = list(topic_docs.values())[0]
                        candidates += docs
                    candidates = [c.strip() for c in candidates]
                    pseudo_knowledge = get_pseudo_knowledge(
                        knowledge, candidates, turn["text"].strip()
                    )
                    acc += pseudo_knowledge == knowledge
                    if len(dialog_history) == 0:
                        context = "".join(dialog_history)
                    else:
                        context = "".join(dialog_history[:-1])
                    checked_passage = get_checked_passage(
                        turn["checked_sentence"],
                        retrieved_passages,
                        turn["checked_passage"],
                    )
                    assert len(checked_passage) > 0
                    converted_data.append(
                        {
                            "dialog_history": context,
                            "knowledge": knowledge,
                            "pseudo_knowledge": pseudo_knowledge,
                            "text": turn["text"].strip(),
                            "chosen_topic": dialog["chosen_topic"],
                            "checked_passage": checked_passage,
                        }
                    )
            dialog_history.append(turn["text"].strip())
            dialog_history.append(EOS)
    print("saving", len(converted_data))
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_path, "w") as f:
        for item in converted_data:
            f.write(json.dumps(item) + "\n")


convert("wizard_of_wiki/test_random_split.json", "wow/test_random_split.txt")
convert("wizard_of_wiki/train.json", "wow/train.txt")
convert("wizard_of_wiki/test_topic_split.json", "wow/test_topic_split.txt")
convert("wizard_of_wiki/valid_random_split.json", "wow/valid_random_split.txt")
convert("wizard_of_wiki/valid_topic_split.json", "wow/valid_topic_split.txt")
