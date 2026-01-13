import os
import subprocess

base_cmd = [
    "python", "train_u.py",
    "-m", "/vast/cw4287/gaussian-model/train_WD_scales_3_log2sigma2",
    "-s", "/home/cw4287/gaussian-dataset/tandt/train",
    "--eval",
    "--data_device", "cuda",
    "--switch_to_wd", "True",
    "--iterations", "50000",
    #"--test_iterations", "100", "200", "300", "500", "1000" ,"1500", "2000" ,"2500", "3000","3500", "4000","4500", "5000", #"8000", "10000" ,"15000" ,"20000", "25000", "30000",
]

render_cmd = [
    "python", "render.py",
    "-m", "/vast/cw4287/gaussian-model/train_WD_scales_3_log2sigma2",
    "-s", "/home/cw4287/gaussian-dataset/tandt/train",
    "--eval",
    "--data_device", "cuda",
]

# factors = [ 1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01]
# scales  = [ 4, 2,   3,   4,   2,    3,    4]

factors = [0.01]
scales  = [3]
log2sigma = [2]


os.makedirs("logs_erank", exist_ok=True)

i=1
import itertools
for f,s,l in itertools.product(factors,scales,log2sigma):

    log_path = f"logs_erank/run_{i}_factor{f}_scale{s}_log2signma{l}.log"
    log_path1 = f"logs_erank/render.log"

    cmd = base_cmd + ["--factor", str(f), "--scale", str(s), "--log2sigma", str(l)]
    cmd2=render_cmd

    print(f"\n==============================")
    print(f"Run {i}: factor={f}, scale={s}")
    print("Command:", " ".join(cmd))
    print(f"Logging to: {log_path}")
    print(f"==============================\n")

    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in proc.stdout:
            print(line, end="")
            log_f.write(line)

        ret = proc.wait()
    i=i+1

    if ret != 0:
        print(f"Run {i} FAILED with code {ret}. Continuing to next run.\n")
        continue

    with open(log_path1, "w") as log_f1:
        proc = subprocess.Popen(
            cmd2,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in proc.stdout:
            print(line, end="")
            log_f1.write(line)

        ret = proc.wait()

    if ret != 0:
        print("render complete")
