import os
import subprocess

base_cmd = [
    "python", "train_u.py",
    "-m", "/home/cw4287/gaussian-model/train1",
    "-s", "/home/cw4287/gaussian-dataset/tandt/train",
    "--eval",
    "--data_device", "cuda",
    "--switch_to_wd", "True",
    "--iterations", "2000",
    "--test_iterations", "100" , "200" , "300", "400", "500", "600", "700" ,"800" ,"900", "1000", "1500", "2000",
]

# factors = [ 1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01]
# scales  = [ 4, 2,   3,   4,   2,    3,    4]

factors = [ 0.001, 0.001, 0.001, 0.0001, 0.0001, 0.0001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001 ]
scales  = [ 2,   3,   4,   2,    3,    4, 5, 2, 3, 4, 5 ]

os.makedirs("logs", exist_ok=True)

for i, (f, s) in enumerate(zip(factors, scales), start=1):

    log_path = f"logs/run_{i}_factor{f}_scale{s}.log"

    cmd = base_cmd + ["--factor", str(f), "--scale", str(s)]

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

    if ret != 0:
        print(f"Run {i} FAILED with code {ret}. Continuing to next run.\n")
        continue
