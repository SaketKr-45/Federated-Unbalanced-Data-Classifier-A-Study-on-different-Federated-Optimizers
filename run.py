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
    """
    Start process directly using the current Python executable.
    This prevents zombie processes on Windows by allowing graceful termination.
    """
    command = f"{PYTHON} {script} {args}"
    # creationflags=subprocess.CREATE_NEW_CONSOLE opens separate windows. 
    # Remove it if you want everything in one terminal.
    return subprocess.Popen(command, shell=False, creationflags=subprocess.CREATE_NEW_CONSOLE)


def run_experiment(algo):
    print(f"\n🚀 Running {algo.upper()}...\n")
    clean_predictions()

    # =========================
    # SERVER
    # =========================
    server = start_process("server.py", f"--algorithm {algo}")

    time.sleep(5)

    # =========================
    # CLIENTS
    # =========================
    clients = []
    for c in ["client1.py", "client2.py", "client3.py", "client4.py"]:
        p = start_process(c)
        clients.append(p)
        time.sleep(1)

    print("[LAUNCHER] All processes started")

    # Wait for server to finish
    server.wait()
    print(f"\n✅ {algo.upper()} completed\n")

    for c in clients:
        if c.poll() is None:
            c.terminate()

    time.sleep(2)

    # Generate plots
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