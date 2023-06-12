import cProfile
import ross as rs


# Estrutrar class
'''
- Criar classe benchmark
- benchmark default (colocar dentro da pasta "benchmark" (?) )
- Criar documentação

'''
rotor = rs.Rotor.load('Benchmarks')
cProfile.run(rotor.run_modal())
