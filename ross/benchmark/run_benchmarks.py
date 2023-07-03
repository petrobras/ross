import subprocess
import ross as rs
import cProfile
from pathlib import Path

ross_path = Path(rs.__file__).parent
benchmark_path = ross_path / "benchmark"
snakeviz_path = benchmark_path / "snakeviz_inputs"

if not snakeviz_path.is_dir():
    Path.mkdir(snakeviz_path)

bench_type = input(
    f"\nRunning ross_benchmarks in version: {rs.__version__}\n\n"
    f"What kind of Benchmarks do you want to run?"
    f"       \n 1 - Campbell Diagram"
    f"       \n 2 - Convergence Analysis"
    f"       \n 3 - Frequency Response"
    f"       \n 4 - Modal Analysis"
    f"       \n 5 - Static Analysis\n"
)


saving_path = snakeviz_path / f"version_{rs.__version__}"
if not saving_path.is_dir():
    Path.mkdir(saving_path)

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
    cProfile.run(
        "rotor = rs.rotor_example(); rotor.run_modal(speed=0)",
        filename=saving_path / "modal.prof",
    )
    bashCommand = f"snakeviz {saving_path}/modal.prof"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
elif bench_type == "5":
    bashCommand = f"python -m cProfile -o static.prof {bench_dir}/cProfile/benchmark_static_analysis.py"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    bashCommand = f"snakeviz static.prof"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
