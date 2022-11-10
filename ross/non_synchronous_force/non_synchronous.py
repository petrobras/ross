# Class to aplicate the non synchronous force in the rotor

#import ross as rs
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots

def non_synchronous_force(F,m,a,k,omega,s,time_non_sync_params):

    # Variables:

    mx = 14.29

    mz = 14.29

    ax = 2.871

    az = -2.871

    kx = 1.195*(10**6)

    kz = 1.195*(10**6)

    # Equations:

    time_assync = np.arange(time_non_sync_params[0],time_non_sync_params[1]+time_non_sync_params[2],time_non_sync_params[2])

    Amplitude_x = []

    Amplitude_z = []

    qx = []

    qz = []

    if type(omega) == int:

        count = 1

    else:

        count = len(omega)

    for w in range(count):

        if type(omega) == int:

            aux_w = omega
        
        else:

            aux_w = omega[w]

        Amplitude_x.append((F*(mz*s**2*aux_w**2 + ax*s*aux_w**2 - kz))/(ax*az*s**2*aux_w**4 - kx*kz + kx*mz*s**2*aux_w**2 + kz*mx*s**2*aux_w**2 - mx*mz*s**4*aux_w**4))

        Amplitude_z.append((F*(mx*s**2*aux_w**2 + az*s*aux_w**2 - kx))/(ax*az*s**2*aux_w**4 - kx*kz + kx*mz*s**2*aux_w**2 + kz*mx*s**2*aux_w**2 - mx*mz*s**4*aux_w**4))

        aux_x = []

        aux_z = []

        for ii in time_assync:

            aux_x.append(Amplitude_x[-1]*np.sin(s*aux_w*ii))

            aux_z.append(Amplitude_z[-1]*np.cos(s*aux_w*ii))
        
        qx.append(aux_x)

        qz.append(aux_z)

    # Plotting:

    # Orbits:

    fig = px.scatter(x=qx[-1],y=qz[-1],title='Orbits for non synchronous force',labels={'x','Amplitude in x axis [m]','z','Amplitude in z axis [m]'})
    fig.show()

    # Amplitude:

    return Amplitude_x, Amplitude_z, qx, qz

if __name__ == "__main__":

    F = 1
    m = 1
    a = 1
    k = 1
    omega = np.arange(0,1501,1)
    s = 0.5
    time_assync_params = [0,180,0.1]
    
    non_synchronous_force(F,m,a,k,omega,s,time_assync_params)

    leo = 1