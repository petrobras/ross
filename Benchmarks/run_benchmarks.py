import os
import pathlib
import subprocess
import ross as rs

Path = pathlib.Path

current_path = os.getcwd()
bench_dir = Path(os.path.dirname(os.path.dirname(rs.__file__)))/'Benchmarks'

if not os.path.isdir(bench_dir/'Snakeviz_inputs'):
    os.mkdir(bench_dir/'Snakeviz_inputs')

os.chdir(bench_dir/'Snakeviz_inputs')

bench_type = input(f"\nRunning ross_benchmarks in version: {rs.__version__}\n\n"
                   f"What kind of Benchmarks do you want to run?"
                   f"       \n 1 - Campbell Diagram"
                   f"       \n 2 - Convergence Analysis"
                   f"       \n 3 - Frequency Response"
                   f"       \n 4 - Modal Analysis"
                   f"       \n 5 - Static Analysis\n"
                   )


inputs = os.listdir(bench_dir/"Snakeviz_inputs")
saving_path = f"version_{rs.__version__}"
if not os.path.isdir(saving_path):
    os.mkdir(saving_path)

os.chdir(bench_dir/'Snakeviz_inputs')
os.chdir(os.getcwd()+f"/{saving_path}")

if bench_type == '2':
    bashCommand = f"python -m cProfile -o Convergence.prof {bench_dir}/cProfile/Benchmark_Convergence.py"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    bashCommand = f"snakeviz Convergence.prof"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
elif bench_type == '1':
    bashCommand = f"python -m cProfile -o Campbell.prof {bench_dir}/cProfile/Benchmark_Campbell_diagram.py"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    bashCommand = f"snakeviz Campbell.prof"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
elif bench_type == '3':
    bashCommand = f"python -m cProfile -o Freq_response.prof {bench_dir}/cProfile/Benchmark_Freq_response.py"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    bashCommand = f"snakeviz Freq_response.prof"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
elif bench_type == '4':
    bashCommand = f"python -m cProfile -o Modal.prof {bench_dir}/cProfile/Benchmark_Modal.py"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    bashCommand = f"snakeviz Modal.prof"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
elif bench_type == '5':
    bashCommand = f"python -m cProfile -o Static.prof {bench_dir}/cProfile/Benchmark_Static_analysis.py"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    bashCommand = f"snakeviz Static.prof"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)



