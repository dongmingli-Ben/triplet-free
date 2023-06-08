import argparse
import os
import numpy as np

import torch
from module.optimizer import build_optimizer
from module.datasets.dataset_doc import build_dataloader, build_dataloader_from_file
from module.trainers.trainer_sql import EncoderDecoderTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration


def parse_args():
    parser = argparse.ArgumentParser()

    # data related
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/mnt/lidongming/weibo_post_commenting/small",
        help="Directory containing data",
    )
    parser.add_argument("--bs", type=int, default=32, help="batch size")
    parser.add_argument(
        "--workers", type=int, default=16, help="workers for dataloader"
    )

    # model related
    parser.add_argument(
        "--backbone", type=str, default="dialogpt-medium", help="backbone of the model"
    )
    parser.add_argument(
        "--reward-model",
        type=str,
        default="dialogpt-medium",
        help="model used for reward",
    )
    # image related

    # sql settings
    parser.add_argument(
        "--sql-implementation",
        type=str,
        default="v2_v2r_v3_v3r",
        help="loss implementation",
    )
    parser.add_argument(
        "--update",
        type=str,
        default="polyak",
        choices=["polyak", "copy"],
        help="How to update the target model (slow copy of current model)",
    )
    parser.add_argument(
        "--polyak",
        type=float,
        default=1e-3,
        help="coefficient in updating target model",
    )
    parser.add_argument(
        "--train-mode",
        type=str,
        default="sql-mix",
        choices=["mle", "sql-mix", "sql-onpolicy", "sql-offpolicy"],
        help="How to perform training."
        "mle: MLE objective on training data; "
        "sql-mix: both sql-onpolicy and sql-offpolicy; "
        "sql-onpolicy: training samples are on-policy samples; "
        "sql-offpolicy: training samples are from training set",
    )
    parser.add_argument(
        "--warmup-mode",
        type=str,
        default="sql-offpolicy",
        choices=["mle", "sql-mix", "sql-onpolicy", "sql-offpolicy"],
        help="The same as --train-mode",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5000,
        help="Gradient update steps for warm up",
    )
    parser.add_argument(
        "--nll-coefficient",
        type=float,
        default=10,
        help="coefficient for nll (consistency) reward",
    )
    parser.add_argument(
        "--doc-coefficient",
        type=float,
        default=10,
        help="coefficient for doc (knowledge) reward",
    )
    parser.add_argument(
        "--length-coefficient",
        type=float,
        default=1,
        help="coefficient for length (consistency) reward",
    )
    parser.add_argument(
        "--reward-shaping-min", type=float, default=-50, help="min for reshaping reward"
    )
    parser.add_argument(
        "--reward-shaping-max", type=float, default=50, help="max for reshaping reward"
    )
    parser.add_argument(
        "--reward-min", type=float, default=-10, help="min for raw reward"
    )
    parser.add_argument(
        "--reward-max", type=float, default=0, help="max for raw reward"
    )
    parser.add_argument(
        "--bleu-margin", type=float, default=0.3, help="max margin for bleu-n reward"
    )
    parser.add_argument(
        "--bleu-reward", type=str, default="2", help="n for bleu-n reward"
    )
    parser.add_argument(
        "--length-margin", type=float, default=0, help="max margin for length reward"
    )
    parser.add_argument(
        "--nll-mode",
        type=str,
        default="token",
        choices=["token", "sentence"],
        help="token-level or sentence-level nll",
    )
    parser.add_argument(
        "--nll-condition", type=str, default="post_doc", choices=["post", "post_doc"]
    )

    # training settings
    parser.add_argument("--device", type=str, default="cuda", help="device to train")
    parser.add_argument("--optimizer", type=str, default="Adam", help="optimizer")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument(
        "--max-epoch", type=int, default=10, help="maximum epochs for training"
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=5,
        help="num of rounds not improved and stop",
    )
    parser.add_argument(
        "--early-stopping-metric", type=str, default="ppl", help="metric to early-stop"
    )
    parser.add_argument(
        "--save-dir", type=str, default="save", help="directory to save checkpoints"
    )
    parser.add_argument(
        "--log-period", type=int, default=1000, help="logging in training"
    )
    parser.add_argument(
        "--max-checkpoint-num",
        type=int,
        default=10,
        help="maximum number of checkpoints (for auto-removal)",
    )
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=1000,
        help="number of batches per evaluation",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="", help="checkpoint path to resume training"
    )
    parser.add_argument(
        "--load-history",
        dest="load_history",
        action="store_true",
        help="whether to load history when loading checkpoint",
    )
    parser.set_defaults(load_history=False)
    parser.add_argument(
        "--load-target-model",
        dest="load_target_model",
        action="store_true",
        help="whether to load the target model from checkpoint",
    )
    parser.set_defaults(load_target_model=False)
    parser.add_argument(
        "--from-distributed",
        dest="from_distributed",
        action="store_true",
        help="whether the checkpoint is distributed",
    )
    parser.set_defaults(from_distributed=False)
    parser.add_argument(
        "--last-utterence",
        dest="last_utterence_only",
        action="store_true",
        help="whether the checkpoint is distributed",
    )
    parser.set_defaults(last_utterence_only=False)
    parser.add_argument(
        "--evaluate-only",
        dest="evaluate_only",
        action="store_true",
        help="whether to evaluate",
    )
    parser.set_defaults(evaluate_only=False)

    # decoding related
    parser.add_argument(
        "--decode",
        type=str,
        default="top-k",
        choices=["top-k", "greedy", "random", "top-p"],
        help="methods for decoding",
    )
    parser.add_argument("--top-k", type=int, default=5, help="k in top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.5, help="p in top-p sampling")
    parser.add_argument(
        "--max-len", type=int, default=64, help="max length used in decoding"
    )

    # others
    parser.add_argument(
        "--fp16",
        dest="fp16",
        action="store_true",
        help="whether to use mix-precision for training",
    )
    parser.set_defaults(fp16=False)
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="cache",
        help="directory to store cached index file",
    )
    parser.add_argument(
        "--tensorboard",
        dest="tensorboard",
        action="store_true",
        help="whether to use tensorboard",
    )
    parser.set_defaults(tensorboard=False)
    parser.add_argument(
        "--mlflow", dest="mlflow", action="store_true", help="whether to use mlflow"
    )
    parser.set_defaults(mlflow=False)
    parser.add_argument(
        "--exp-name",
        type=str,
        default="sql",
        help="name of the experiment (for mlflow)",
    )
    parser.add_argument("--seed", default=23473, type=int, help="seed")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    print("Building tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    r_tokenizer = AutoTokenizer.from_pretrained(args.reward_model)
    print("Building model")
    model = T5ForConditionalGeneration.from_pretrained(args.backbone).to(
        torch.device(args.device)
    )
    optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), args.lr)

    print(f"Building dataset from {args.data_dir}")
    train_dataloader = build_dataloader(args, tokenizer, r_tokenizer, "train")
    valid_dataloader = build_dataloader(args, tokenizer, r_tokenizer, "dev")
    test_dataloader = build_dataloader(args, tokenizer, r_tokenizer, "test")

    if args.evaluate_only:
        testset_paths = [
            "data/wow/test_random_split.txt",
            "data/wow/test_topic_split.txt",
            "data/wizint/test-short-knowledge.txt",
        ]
        testset_names = ["wow_seen", "wow_unseen", "wizint_short"]
        testset_loaders = []
        for path, name in zip(testset_paths, testset_names):
            testset_loaders.append(
                build_dataloader_from_file(args, path, tokenizer, r_tokenizer, False)
            )
    else:
        testset_loaders = []
        testset_names = []

    print(f"Start training with maximum {args.max_epoch} epochs")
    print(f"Model backbone: {args.backbone}")
    print(f"Optimizer: {args.optimizer}, lr: {args.lr}")
    print(f"Early stopping {args.early_stopping} on {args.early_stopping_metric}")
    print(f"Using bleu-{args.bleu_reward} reward")
    print(f"Using nll reward with condition on {args.nll_condition}")
    print(
        f"Reward coefficients: nll {args.nll_coefficient}, doc {args.doc_coefficient}, "
        f"length {args.length_coefficient}"
    )
    trainer = EncoderDecoderTrainer(
        model,
        "copy",
        optimizer,
        tokenizer,
        r_tokenizer,
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        args,
    )
    trainer.train_tests(args.eval_batches, testset_loaders, testset_names)


if __name__ == "__main__":
    main()
