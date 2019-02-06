# Todo: translate comments, names of variables and functions.
# Todo: check PEP 8

from ross.fluid_flow import fluid_flow as flow
import numpy as np

# These are the data related to the fluid flow problem

# GRID:
# Número de pontos ao longo da direção Z (direção do escoamento):
#NZ=150
#NZ =100
NZ = 10

# Número de pontos ao longo da direção TETA:
#OBS: NTETA deve ser ímpar!
#NTETA=37
#NTETA =17
NTETA = 20

#Número de pontos ao longo da direção r:
NRAIO =11

# Número de intervalo em Z:
NintervZ=NZ-1

#Número de intervalo em TETA:
NintervTETA=NTETA-1

# Número de intervalo em r:
NintervRAIO=NRAIO-1

# Comprimento na direção Z (m):
Lb=1.

# Comprimento na direção TETA (rad):
Lteta=2.*np.pi

# Tamanho do intervalo na direção Z:
DZ=Lb/NintervZ

# Tamanho do intervalo na direção TETA:
DTETA=Lteta/NintervTETA

# Número de nós na malha:
NTOTAL=NZ*NTETA

###########################################################################
#CONDIÇÕES DE OPERAÇÃO

# Rotação do rotor (rad/s):
omega=100.*2*np.pi/60
#omega=5*np.pi/3

# Pressão de Entrada (Pa):
Pent=392266.
#Pent=192244

# Pressão de Saída (Pa):
Ps=100000.

###########################################################################
#DADOS GEOMÉTRICOS DO PROBLEMA

# Raio menor do rotor (m):
raioVale = 0.034
#raioVale=0.036

# Raio maior do rotor (m):
raioCrista=0.039
#raioCrista=0.037

# Raio do estator (m):
Ro=0.04

# Passo do rotor (m) (comprimento de onda senoidal):
lOnda=0.18
#lOnda=0.059995

# Excentricidade (m) (distância entre os centros do rotor e do estator):
Xe = 0.
#Xe = 0.001;
Ye = 0.
#%Ye = 0.001

###########################################################################
# CARACTERÍSTICAS DO FLUIDO:

# Viscosidade (Pa.s):
visc=0.001 # Água
#visc=0.042 # Purolub 46
#visc=0.433 # Purolub 150

# Densidade do fluido(Kg/m^3):
Rho=1000. # Água
#Rho=868. # Purolub 46
# Rho=885. # Purolub 150


if __name__ == "__main__":
    my_pressure_matrix = flow.PressureMatrix(NZ, NTETA, NRAIO, NintervZ, NintervTETA, NintervRAIO, Lb, Lteta, DZ,
                                             DTETA, NTOTAL, omega, Pent, Ps, raioVale, raioCrista, Ro, lOnda, Xe,
                                             Ye, visc, Rho)
    P = my_pressure_matrix.calculate_pressure_matrix()
    print(P)
