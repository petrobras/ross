import cProfile
import ross as rs

rotor = rs.Rotor.load('Benchmarks')
cProfile.run('rotor.run_modal(speed=125.0)')
