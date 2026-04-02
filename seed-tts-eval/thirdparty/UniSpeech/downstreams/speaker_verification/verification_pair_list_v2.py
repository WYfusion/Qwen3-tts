'''
python verification_tsv.py $tsv1 $tsv2 --model_name wavlm_large --checkpoint wavlm_large_finetune.pth --scores $score_file --wav1_start_sr 0 --wav2_start_sr 0 --wav1_end_sr -1 --wav2_end_sr -1
'''
import tqdm
import argparse
from verification import init_model, verification
import os
import math

parser = argparse.ArgumentParser()
parser.add_argument('pair')
parser.add_argument('--model_name')
parser.add_argument('--checkpoint')
parser.add_argument('--scores')
parser.add_argument('--wav1_start_sr', type=int)
parser.add_argument('--wav2_start_sr', type=int)
parser.add_argument('--wav1_end_sr', type=int)
parser.add_argument('--wav2_end_sr', type=int)
parser.add_argument('--wav2_cut_wav1', type=bool, default=False)
parser.add_argument('--device', default="cuda:0")
parser.add_argument('--upstream_ckpt', default=None)
args = parser.parse_args()

f = open(args.pair)
lines = f.readlines()
f.close()

tsv1 = []
tsv2 = []
for line in lines:
    e = line.strip().split('|')
    if len(e) == 4:
        part1, _, _, part2 = line.strip().split('|')
    else:
        part1, part2 = line.strip().split('|')[:2]
    tsv1.append(part1)
    tsv2.append(part2)

scores_w = open(args.scores, 'w')
assert len(tsv1) == len(tsv2)

model = None
score_list = []
num_missing = 0
num_exception = 0
num_none = 0
print(
    f"[verification_pair_list_v2] start: pairs={len(tsv1)} model_name={args.model_name} "
    f"checkpoint={args.checkpoint} upstream_ckpt={args.upstream_ckpt} device={args.device}",
    flush=True,
)
print("[verification_pair_list_v2] initializing model before iterating over pairs", flush=True)
try:
    model = init_model(args.model_name, args.checkpoint, upstream_ckpt=args.upstream_ckpt)
    print("[verification_pair_list_v2] model initialization finished", flush=True)
except Exception as e:
    print(f"[verification_pair_list_v2] fatal: model initialization failed: {e}", flush=True)
    scores_w.write(
        f"# fatal\tmodel_init\tmodel_name={args.model_name}\tcheckpoint={args.checkpoint}\t"
        f"upstream_ckpt={args.upstream_ckpt}\t{e}\n"
    )
    scores_w.flush()
    scores_w.close()
    raise SystemExit(1)

started_inference = False
for t1, t2 in tqdm.tqdm(zip(tsv1, tsv2), total=len(tsv1)):
    t1_path = t1.strip()
    t2_path = t2.strip()
    if not os.path.exists(t1_path) or not os.path.exists(t2_path):
        num_missing += 1
        scores_w.write(f"# missing\t{t1_path}\t{t2_path}\n")
        scores_w.flush()
        continue
    try:
        if not started_inference:
            print(
                f"[verification_pair_list_v2] first valid pair entering inference: {t1_path} | {t2_path}",
                flush=True,
            )
            started_inference = True
        sim, model = verification(args.model_name, t1_path, t2_path, use_gpu=True, checkpoint=args.checkpoint, wav1_start_sr=args.wav1_start_sr, wav2_start_sr=args.wav2_start_sr, wav1_end_sr=args.wav1_end_sr, wav2_end_sr=args.wav2_end_sr, model=model, wav2_cut_wav1=args.wav2_cut_wav1, device=args.device, upstream_ckpt=args.upstream_ckpt)
    except Exception as e:
        num_exception += 1
        print(f"[verification_pair_list_v2] failed: {t1_path} | {t2_path} | {e}", flush=True)
        scores_w.write(f"# exception\t{t1_path}\t{t2_path}\t{e}\n")
        scores_w.flush()
        continue

    if sim is None:
        num_none += 1
        scores_w.write(f"# none\t{t1_path}\t{t2_path}\n")
        scores_w.flush()
        continue
    scores_w.write(f'{t1_path}_{args.wav1_start_sr}_{args.wav1_end_sr}|{t2_path}_{args.wav2_start_sr}_{args.wav2_end_sr}\t{sim.cpu().item()}\n')
    # print(f'{t1_path}_{args.wav1_start_sr}_{args.wav1_end_sr}|{t2_path}_{args.wav2_start_sr}_{args.wav2_end_sr}\t{sim.cpu().item()}')
    score_list.append(sim.cpu().item())
    scores_w.flush()
num_pairs = len(tsv1)
num_success = len(score_list)
avg_score = float("nan") if not score_list else (sum(score_list) / len(score_list))
scores_w.write(f"# stats\ttotal={num_pairs}\tsuccess={num_success}\tmissing={num_missing}\tnone={num_none}\texception={num_exception}\n")
if math.isnan(avg_score):
    scores_w.write('avg score: nan\n')
    print(
        f"[verification_pair_list_v2] no valid scores produced. "
        f"total={num_pairs}, missing={num_missing}, none={num_none}, exception={num_exception}",
        flush=True,
    )
else:
    scores_w.write(f'avg score: {avg_score}\n')
    print(
        f"[verification_pair_list_v2] avg score: {avg_score:.6f} | "
        f"total={num_pairs}, success={num_success}, missing={num_missing}, none={num_none}, exception={num_exception}",
        flush=True,
    )
scores_w.flush()
scores_w.close()
# print(f'avg score: {round(sum(score_list)/len(score_list), 3)}')
