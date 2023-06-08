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


def get_knowledge(turn, max_knowledge_length):
    docs = turn["context"]["contents"]
    select = turn["context"]["selected_contents"][1:]
    selected_doc_index = set()
    selected_sent_index = set()
    checked_doc = None
    sent = []
    assert len(docs) == len(select)
    for i in range(len(docs)):
        if len(docs[i]["content"]) != len(select[i]):
            import pdb

            pdb.set_trace()
        assert len(docs[i]["content"]) == len(select[i])
        for j in range(len(docs[i]["content"])):
            if select[i][j]:
                sent.append(docs[i]["content"][j])
                selected_doc_index.add(i)
                selected_sent_index.add((i, j))
                if checked_doc is None:
                    checked_doc = docs[i]["title"]
    sent = " ".join(sent)
    if sent == "" or len(sent.split()) > max_knowledge_length:
        sent = "no_passages_used"
    return sent, checked_doc, len(selected_doc_index) > 1, len(selected_sent_index) > 1


def get_retrieved_sentences(turn):
    candidates = []
    for doc in turn["context"]["contents"]:
        for sent in doc["content"]:
            candidates.append(sent)
    return candidates


def convert(path, save_path, max_knowledge_length=10000):
    # if the knowledge is larger than max_knowledge_length,
    # I assume the knowledge sentence is not properly splitted,
    # and hence discard the knowledge
    converted_data = []
    acc = 0
    multi_doc_cnt = 0
    multi_sent_cnt = 0

    with open(path, "r") as f:
        for i, line in enumerate(f):
            dialog = json.loads(line)
            if len(dialog) > 1:
                import pdb

                pdb.set_trace()
            assert len(dialog) == 1
            dialog = list(dialog.values())[0]
            dialog_history = []
            for turn in dialog["dialog_history"]:
                if (
                    turn["action"] != "Wizard => Apprentice"
                    and "SearchAgent" in turn["action"]
                ):
                    continue
                if turn["action"] == "Apprentice => Wizard":
                    dialog_history.append(turn["text"].strip())
                    dialog_history.append(EOS)
                    continue
                knowledge, checked_passage, multi_doc, multi_sent = get_knowledge(
                    turn, max_knowledge_length
                )
                if knowledge != "no_passages_used":
                    multi_doc_cnt += multi_doc
                    multi_sent_cnt += multi_sent
                    candidates = get_retrieved_sentences(turn)
                    pseudo_knowledge = get_pseudo_knowledge(
                        knowledge, candidates, turn["text"].strip()
                    )
                    acc += pseudo_knowledge == knowledge
                    if len(dialog_history) == 0:
                        context = "".join(dialog_history)
                    else:
                        context = "".join(dialog_history[:-1])
                    assert len(checked_passage) > 0
                    converted_data.append(
                        {
                            "dialog_history": context,
                            "knowledge": knowledge,
                            "pseudo_knowledge": pseudo_knowledge,
                            "text": turn["text"].strip(),
                            "apprentice_persona": dialog["apprentice_persona"],
                            "checked_passage": checked_passage,
                        }
                    )
                dialog_history.append(turn["text"].strip())
                dialog_history.append(EOS)
    print("saving", len(converted_data))
    print("Multi-doc selection rate", multi_doc_cnt / len(converted_data))
    print("Multi-sentence selection rate", multi_sent_cnt / len(converted_data))
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_path, "w") as f:
        for item in converted_data:
            f.write(json.dumps(item) + "\n")


convert("wizard_of_internet/valid.jsonl", "wizint/valid-short-knowledge.txt", 50)
convert("wizard_of_internet/train.jsonl", "wizint/train-short-knowledge.txt", 50)
convert("wizard_of_internet/test.jsonl", "wizint/test-short-knowledge.txt", 50)
