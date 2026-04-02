import sys
import numpy as np

infile=sys.argv[1]
outfile=sys.argv[2]

fout = open(outfile, "w")
scores = []
for line in open(infile, "r").readlines():
    if line.startswith("#"):
        fout.write(line)
        continue
    if line.startswith("avg score:"):
        fout.write(line)
        continue
    if not line.strip():
        continue
    item, score = line.strip().split("\t")
    scores.append(float(score))
    fout.write(line)

if not scores:
    fout.write("ASV: nan\n")
    fout.write("ASV-var: nan\n")
    print("ASV: nan")
    print("ASV-var: nan")
    fout.close()
    raise SystemExit(0)

res = round(np.mean(np.array(scores)), 3)
res_var = round(np.var(np.array(scores)), 3)
fout.write(f"ASV: {res}\n")
fout.write(f"ASV-var: {res_var}\n")
print(f"ASV: {res}")
print(f"ASV-var: {res_var}")
fout.close()
