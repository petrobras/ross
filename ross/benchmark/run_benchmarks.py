import subprocess
import ross as rs
import cProfile
import pytest
from pathlib import Path

pytest.skip("Skipping benchmarks", allow_module_level=True)

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
    cProfile.run(
        "rotor = rs.rotor_example(); rotor.run_campbell(speed_range=[0, 1, 2, 3, 4])",
        filename=saving_path / "campbell.prof",
    )
    bashCommand = f"snakeviz {saving_path}/campbell.prof"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
elif bench_type == "2":
    cProfile.run(
        "rotor = rs.rotor_example(); rotor.convergence()",
        filename=saving_path / "convergence.prof",
    )
    bashCommand = f"snakeviz {saving_path}/convergence.prof"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
elif bench_type == "3":
    cProfile.run(
        "rotor = rs.rotor_example(); rotor.run_freq_response()",
        filename=saving_path / "freq_response.prof",
    )
    bashCommand = f"snakeviz {saving_path}/freq_response.prof"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
elif bench_type == "4":
    cProfile.run(
        "rotor = rs.rotor_example(); rotor.run_modal(speed=0)",
        filename=saving_path / "modal.prof",
    )
    bashCommand = f"snakeviz {saving_path}/modal.prof"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
elif bench_type == "5":
    cProfile.run(
        "rotor = rs.rotor_example(); rotor.run_static()",
        filename=saving_path / "static.prof",
    )
    bashCommand = f"snakeviz {saving_path}/static.prof"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
