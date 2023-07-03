import cProfile
import ross as rs
import numpy as np

rotor = rs.Rotor.load("Benchmarks")
cProfile.run(rotor.run_campbell())
