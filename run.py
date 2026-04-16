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

def start_in_new_terminal(script, args=[]):
    return subprocess.Popen(
        [PYTHON, script] + args, 
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )

def run_experiment(algo):
    print(f"\n🚀 Running {algo.upper()}...\n")
    clean_predictions()

    server = start_in_new_terminal("server.py", ["--algorithm", algo])
    time.sleep(5)

    clients = [start_in_new_terminal(c) for c in ["client1.py", "client2.py", "client3.py", "client4.py"]]
    
    print("[LAUNCHER] All processes started in separate windows.")
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