config=tripletfree
checkpoint=checkpoint/release.pt


echo ${config}

python train_doc_eng.py \
--data-dir data/wow \
--cache-dir cache/wow \
--save-dir save/${config} \
--evaluate-only \
--checkpoint ${checkpoint} \
--backbone t5-base \
--reward-model dialogpt-medium \
--last-utterence \
--exp-name sql-release \
--max-len 40 \
--bleu-margin 0.3 \
--bleu-reward 2 \
--length-margin 30 \
--nll-coefficient 0.5 \
--nll-mode sentence \
--nll-condition post \
--length-coefficient 0.5 \
--reward-min -90 \
--reward-max 0 \
--reward-shaping-min -50 \
--reward-shaping-max 50 \
--doc-coefficient 100 \
--sql-implementation v2_v2r_v3_v3r \
--update polyak \
--polyak 1e-3 \
--train-mode sql-offpolicy \
--warmup-mode sql-offpolicy \
--warmup-steps 20000 \
--tensorboard \
--mlflow \
--bs 4 \
--workers 4 \
--optimizer Adam \
--lr 1e-5 \
--max-epoch 5 \
--early-stopping 6 \
--early-stopping-metric val/rewards/reward/mean \
--log-period 100 \
--decode top-p \
--top-p 0.4 \
--eval-batches 5000 \
--seed 2834