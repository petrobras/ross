import os
import pathlib
import subprocess
import ross as rs

Path = pathlib.Path

current_path = Path.cwd()
bench_dir = Path(os.path.dirname(rs.__file__)) / "benchmark"
snakeviz_dir = Path(bench_dir) / "snakeviz_inputs"

if not snakeviz_dir.is_dir():
    Path.mkdir(Path.joinpath(bench_dir, "snakeviz_inputs"))

os.chdir(bench_dir / "snakeviz_inputs")

bench_type = input(
    f"\nRunning ross_benchmarks in version: {rs.__version__}\n\n"
    f"What kind of Benchmarks do you want to run?"
    f"       \n 1 - Campbell Diagram"
    f"       \n 2 - Convergence Analysis"
    f"       \n 3 - Frequency Response"
    f"       \n 4 - Modal Analysis"
    f"       \n 5 - Static Analysis\n"
)


inputs = os.listdir(bench_dir / "snakeviz_inputs")
saving_path = f"version_{rs.__version__}"
if not os.path.isdir(saving_path):
    os.mkdir(saving_path)

os.chdir(bench_dir / "snakeviz_inputs")
os.chdir(os.getcwd() + f"/{saving_path}")

if bench_type == "1":
    bashCommand = f"python -m cProfile -o campbell.prof {bench_dir}/cProfile/benchmark_campbell_diagram.py"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    bashCommand = f"snakeviz campbell.prof"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
elif bench_type == "2":
    bashCommand = f"python -m cProfile -o convergence.prof {bench_dir}/cProfile/benchmark_convergence.py"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    bashCommand = f"snakeviz convergence.prof"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
elif bench_type == "3":
    bashCommand = f"python -m cProfile -o freq_response.prof {bench_dir}/cProfile/benchmark_freq_response.py"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    bashCommand = f"snakeviz freq_response.prof"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
elif bench_type == "4":
    bashCommand = (
        f"python -m cProfile -o modal.prof {bench_dir}/cProfile/benchmark_modal.py"
    )
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    bashCommand = f"snakeviz modal.prof"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
elif bench_type == "5":
    bashCommand = f"python -m cProfile -o static.prof {bench_dir}/cProfile/benchmark_static_analysis.py"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    bashCommand = f"snakeviz static.prof"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
