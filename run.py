import subprocess
import sys
import time
import shutil
import os

PYTHON = sys.executable

ALGORITHMS = ["fedavg", "fedavgm", "fedprox", "fedadam", "fedadagrad", "fedyogi"]


def clean_predictions():
    shutil.rmtree("predictions", ignore_errors=True)
    os.makedirs("predictions", exist_ok=True)


def start_process(script, args=""):
    command = f"{PYTHON} {script} {args}"
    return subprocess.Popen(command, shell=False, creationflags=subprocess.CREATE_NEW_CONSOLE)


def run_experiment(algo):
    print(f"\n🚀 Running {algo.upper()}...\n")

    clean_predictions()

    server = start_process("server.py", f"--algorithm {algo}")

    time.sleep(5)

    clients = []
    for c in ["client1.py", "client2.py", "client3.py", "client4.py"]:
        p = start_process(c)
        clients.append(p)
        time.sleep(1)

    print("[LAUNCHER] All processes started")

    server.wait()

    print(f"\n✅ {algo.upper()} completed\n")

    for c in clients:
        if c.poll() is None:
            c.terminate()

    time.sleep(2)

    subprocess.run([PYTHON, "final_generate_plot.py", "--algo", algo])


def main():
    shutil.rmtree("results", ignore_errors=True)

    for algo in ALGORITHMS:
        run_experiment(algo)

    print("\n📊 Generating comparison plots...\n")
    subprocess.run([PYTHON, "compare_algorithms.py"])

    print("\n🎯 ALL DONE!")


if __name__ == "__main__":
    main()