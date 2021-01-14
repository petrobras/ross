# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 09:41:21 2020

@author: Ely Queiroz
"""

# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------- Bibliotecas Utilizadas ------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
from scipy.optimize import minimize
from numpy.linalg import inv
import scipy as sci
from scipy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
import time
import math
import sys

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

x0=[0.1, -0.1]

def hdf(x0):
    
    global Ytheta, Zdim, PPlot, T_mist_aux, Tdim, n_pad, ntheta, ngap, Pdim, Cr, war, R, betha_s, nZ, nY, dY, dZ, L, MI_e, MI_w, MI_s, MI_n, p, P, mi_ref, dy, dz, Wx, Wy, dtheta
    
    # ----------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------- Parâmetros de entrada ------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    L = float(0.263144)      # [metros]
    
    R = float(0.2)      # [metros]
    
    Cr = float(1.945e-4)     # [metros]
    
    mi_ref = float(0.02)      # [Ns/m²]
    
    wa = float(900)     # [RPM]
    
    Wx = float(0)    # [N]
    
    Wy = float(-112814.91)    # [N]
    
    kt=float(0.15327)     #Thermal conductivity [J/s.m.°C]
    
    Cp=float(1800.24)      #Specific heat [J/kg°C]
    
    rho=float(880)    #Specific mass [kg/m³]
    
    Treserv=float(50)      #Temperature of oil tank [ºC]
    
    T_ref=Treserv      #Reference temperature [ºC]
    
    fat_mist=float(0.8)    # Mixing factor. Used because the oil supply flow is not known.
      
    ntheta = int(38)
    nZ = int(30)
    nY = ntheta
    
    ngap= int(2) #    Number of volumes in recess zone
    
    n_pad=int(2) #    Number of pads
    
    betha_s = 170
    
    war = (wa * np.pi) / 30  # Transforma de rpm para rad/s
    betha_s = betha_s*np.pi/180 #[rad]
      
    
    theta1=0           #initial coordinate theta [rad]
    theta2=betha_s     #final coordinate theta [rad]
    dtheta=(theta2-theta1)/(ntheta)
    
    dY=1/nY
    dZ=1/nZ
    
    
    Ytheta=np.zeros(2*(ntheta+ngap)+2)

    
    Ytheta[1:-1]=np.arange(0.5*dtheta,2*np.pi,dtheta)
    Ytheta[0]=0
    Ytheta[-1]=2*np.pi
    
    Z1=0        #initial coordinate z dimensionless
    Z2=1
    Z = np.zeros((nZ+2))
    Z[0]=Z1 
    Z[nZ+1]=Z2   
    Z[1:nZ+1] = np.arange(Z1+0.5*dZ,Z2,dZ) #vector z dimensionless
    Zdim=Z*L
    
    
    #def THDEquilibrio(x0):
    #    global p
    # Dimensioless
    xr = x0[0] * Cr * np.sin(x0[1])  # Representa a posição do centro do eixo ao longo da direção "Y"
    yr = x0[0] * Cr * np.cos(x0[1])  # Representa a posição do centro do eixo ao longo da direção "X"
    Y = yr/Cr                        # Representa a posição em x adimensional
    X = xr/Cr   
    #h=Cr-(yr*np.cos(theta))-(xr*np.sin(theta))
    
    Xpt=0
    Ypt=0
    
    dz=dZ*L
    dy=dY*betha_s*R

    
    
    P=np.zeros((nZ,ntheta,n_pad))
    
    dPdy=np.zeros((nZ,ntheta,n_pad))
    
    dPdz=np.zeros((nZ,ntheta,n_pad))
    
    T = np.ones((nZ,ntheta,n_pad))
    
    T_new = np.ones((nZ,ntheta,n_pad))*1.2
    
    T_mist = np.ones((nZ,ntheta,n_pad))*1.2
    
    T_mist_aux = T_ref*np.ones(n_pad)
    
    mi_new = np.ones((nZ,ntheta,n_pad))
    
    for iii in np.arange(n_pad):
    
    
        for n_p in np.arange(n_pad):      
            
            T_ref = T_mist_aux[n_p-1]
    
            theta1=(n_p)*(betha_s)+(dtheta*ngap/2)+(n_p*dtheta*ngap)
            
            theta2=theta1+betha_s
            
            dtheta=(theta2-theta1)/(ntheta)
            
            mi_new = mi_new*mi_ref/mi_ref
            
            nk=(nZ)*(ntheta)
                
            Z1=0        # initial coordinate z dimensionless
            Z2=1        # final coordinate z dimensionless
            
    
            while np.linalg.norm(T_new[:,:,n_p]-T[:,:,n_p])/np.linalg.norm(T[:,:,n_p]) >= 1e-1:
                
    #            if iii >0:
    #                print('passei aqui')
                Mat_coef=np.zeros((nk,nk))
                Mat_coef_t=np.zeros((nk,nk))
                b=np.zeros((nk,1))
                b_t=np.zeros((nk,1))
            
                ki=0
                kj=0
        
                mi=mi_new
                
                T[:,:,n_p]=T_mist[:,:,n_p]
                
                k=0 #vectorization pressure index
                
                for ii in np.arange((Z1+0.5*dZ),Z2,dZ):      
                    for jj in np.arange(theta1+(dtheta/2),theta2,dtheta):
                 
                        hP=1-Y*np.cos(jj)-X*np.sin(jj)                     
                        he=1-Y*np.cos(jj+0.5*dY)-X*np.sin(jj+0.5*dY)
                        hw=1-Y*np.cos(jj-0.5*dY)-X*np.sin(jj-0.5*dY)
                        hn=hP
                        hs=hn
                        
                        if kj==0 and ki==0:
                            MI_e = 0.5*(mi[ki,kj]+mi[ki,kj+1])
                            MI_w = mi[ki,kj]
                            MI_s = mi[ki,kj]
                            MI_n = 0.5*(mi[ki,kj]+mi[ki+1,kj])
                  
                    
                        if kj==0 and ki>0 and ki<nZ-1:
                            MI_e= 0.5*(mi[ki,kj]+mi[ki,kj+1])
                            MI_w= mi[ki,kj]
                            MI_s= 0.5*(mi[ki,kj]+mi[ki-1,kj])
                            MI_n= 0.5*(mi[ki,kj]+mi[ki+1,kj])
                  
                    
                        if kj==0 and ki==nZ-1:
                            MI_e= 0.5*(mi[ki,kj]+mi[ki,kj+1])
                            MI_w= mi[ki,kj]
                            MI_s= 0.5*(mi[ki,kj]+mi[ki-1,kj])
                            MI_n= mi[ki,kj]
                   
                    
                        if ki==0 and kj>0 and kj<ntheta-1:
                            MI_e= 0.5*(mi[ki,kj]+mi[ki,kj+1])
                            MI_w= 0.5*(mi[ki,kj]+mi[ki,kj-1])
                            MI_s= mi[ki,kj]
                            MI_n= 0.5*(mi[ki,kj]+mi[ki+1,kj])
                    
                    
                        if kj>0 and kj<ntheta-1 and ki>0 and ki<nZ-1:
                            MI_e= 0.5*(mi[ki,kj]+mi[ki,kj+1])
                            MI_w= 0.5*(mi[ki,kj]+mi[ki,kj-1])
                            MI_s= 0.5*(mi[ki,kj]+mi[ki-1,kj])
                            MI_n= 0.5*(mi[ki,kj]+mi[ki+1,kj])
                    
                        if ki==nZ-1 and kj>0 and kj<ntheta-1:
                            MI_e= 0.5*(mi[ki,kj]+mi[ki,kj+1])
                            MI_w= 0.5*(mi[ki,kj]+mi[ki,kj-1])
                            MI_s= 0.5*(mi[ki,kj]+mi[ki-1,kj])
                            MI_n= mi[ki,kj]
                   
                    
                        if ki==0 and kj==ntheta-1:
                            MI_e= mi[ki,kj]
                            MI_w= 0.5*(mi[ki,kj]+mi[ki,kj-1])
                            MI_s= mi[ki,kj]
                            MI_n= 0.5*(mi[ki,kj]+mi[ki+1,kj])
                  
                    
                        if kj==ntheta-1 and ki>0 and ki<nZ-1:
                            MI_e= mi[ki,kj]
                            MI_w= 0.5*(mi[ki,kj]+mi[ki,kj-1])
                            MI_s= 0.5*(mi[ki,kj]+mi[ki-1,kj])
                            MI_n= 0.5*(mi[ki,kj]+mi[ki+1,kj])
                    
                    
                        if kj==ntheta-1 and ki==nZ-1:
                            MI_e= mi[ki,kj]
                            MI_w= 0.5*(mi[ki,kj]+mi[ki,kj-1])
                            MI_s= 0.5*(mi[ki,kj]+mi[ki-1,kj])
                            MI_n= mi[ki,kj]
                
                       
                        mi_e=MI_e[n_p]
                        mi_w=MI_w[n_p]
                        mi_n=MI_n[n_p]
                        mi_s=MI_s[n_p]
        

                        CE=(dZ*he**3)/(12*mi_e*dY*betha_s**2)
                        CW=(dZ*hw**3)/(12*mi_w*dY*betha_s**2)
                        CN=(dY*(R**2)*hn**3)/(12*mi_n*dZ*L**2)
                        CS=(dY*(R**2)*hs**3)/(12*mi_s*dZ*L**2)
                        CP=-(CE+CW+CN+CS)
                  
                        B=(dZ/2*betha_s)*(he-hw)-((Ypt*np.cos(jj)+Xpt*np.sin(jj))*dy*dZ)
                      
                        k=k+1
                        b[k-1,0]=B
                            
                        if ki==0 and kj==0:
                            Mat_coef[k-1,k-1]=CP-CS-CW
                            Mat_coef[k-1,k]=CE
                            Mat_coef[k-1,k+ntheta-1]=CN
                       
                        
                        elif kj==0 and ki>0 and ki<nZ-1: 
                            Mat_coef[k-1,k-1]=CP-CW
                            Mat_coef[k-1,k]=CE
                            Mat_coef[k-1,k-ntheta-1]=CS
                            Mat_coef[k-1,k+ntheta-1]=CN
                            
                            
                        elif kj==0 and ki==nZ-1:
                            Mat_coef[k-1,k-1]=CP-CN-CW
                            Mat_coef[k-1,k]=CE
                            Mat_coef[k-1,k-ntheta-1]=CS    
                
                
                        elif ki==0 and kj>0 and kj<nY-1:
                            Mat_coef[k-1,k-1]=CP-CS
                            Mat_coef[k-1,k]=CE
                            Mat_coef[k-1,k-2]=CW
                            Mat_coef[k-1,k+ntheta-1]=CN
                       
                      
                              
                        elif ki>0 and ki<nZ-1 and kj>0 and kj<nY-1:
                            Mat_coef[k-1,k-1]=CP
                            Mat_coef[k-1,k-2]=CW
                            Mat_coef[k-1,k-ntheta-1]=CS
                            Mat_coef[k-1,k+ntheta-1]=CN
                            Mat_coef[k-1,k]=CE
                        
                        
                        elif ki==nZ-1 and kj>0 and kj<nY-1:
                            Mat_coef[k-1,k-1]=CP-CN
                            Mat_coef[k-1,k]=CE
                            Mat_coef[k-1,k-2]=CW
                            Mat_coef[k-1,k-ntheta-1]=CS
                           
                           
                        elif ki==0 and kj==nY-1:
                            Mat_coef[k-1,k-1]=CP-CE-CS
                            Mat_coef[k-1,k-2]=CW
                            Mat_coef[k-1,k+ntheta-1]=CN
                                
                        elif kj==nY-1 and ki>0 and ki<nZ-1:
                            Mat_coef[k-1,k-1]=CP-CE
                            Mat_coef[k-1,k-2]=CW
                            Mat_coef[k-1,k-ntheta-1]=CS
                            Mat_coef[k-1,k+ntheta-1]=CN    
                  
                                       
                        elif ki==nZ-1 and kj==nY-1:
                            Mat_coef[k-1,k-1]=CP-CE-CN
                            Mat_coef[k-1,k-2]=CW
                            Mat_coef[k-1,k-ntheta-1]=CS
                         
                        kj=kj+1
                   
                    kj=0
                    ki=ki+1
                   
                #    %%%%%%%%%%%%%%%%%%%%%% Solution of pressure field %%%%%%%%%%%%%%%%%%%%
                
                p=np.linalg.solve(Mat_coef,b)
              
                cont=0
                
                for i in np.arange(nZ):
                    for j in np.arange(ntheta):
                               
                        P[i,j,n_p]=p[cont]
                        cont=cont+1
        
                #    %Boundary condiction of pressure
            
                for i in np.arange(nZ):
                    for j in np.arange(ntheta):
                        if P[i,j,n_p]<0:
                            P[i,j,n_p]=0
                  
                #    % Dimensional pressure fied [Pa]
                
                Pdim=(P*mi_ref*war*(R**2))/(Cr**2)
    

               #    %%%%%%%%%%%%%%%%%%%%%% Solution of temperature field %%%%%%%%%%%%%%%%%%%%
            
                ki=0
                kj=0
                k=0 #vectorization pressure index
                

                for ii in np.arange((Z1+0.5*dZ),(Z2),dZ):      
                    for jj in np.arange(theta1+(dtheta/2),theta2,dtheta):
                 
        #                  Pressure gradients
                     
    
                        if kj==0 and ki==0:
                            dPdy[ki,kj,n_p]= (P[ki,kj+1,n_p]-0)/(2*dY)
                            dPdz[ki,kj,n_p]= (P[ki+1,kj,n_p]-0)/(2*dZ)
        
         
                
                        if kj==0 and ki>0 and ki<nZ-1:
                            dPdy[ki,kj,n_p]= (P[ki,kj+1,n_p]-0)/(2*dY)
                            dPdz[ki,kj,n_p]= (P[ki+1,kj,n_p]-P[ki-1,kj,n_p])/(2*dZ)
              
                
                        if kj==0 and ki==nZ-1:
                            dPdy[ki,kj,n_p]= (P[ki,kj+1,n_p]-0)/(2*dY)
                            dPdz[ki,kj,n_p]= (0-P[ki-1,kj,n_p])/(2*dZ)
              
                
                        if ki==0 and kj>0 and kj<ntheta-1:
                            dPdy[ki,kj,n_p]= (P[ki,kj+1,n_p]-P[ki,kj-1,n_p])/(2*dY)
                            dPdz[ki,kj,n_p]= (P[ki+1,kj,n_p]-0)/(2*dZ)
             
                
                        if kj>0 and kj<ntheta-1 and ki>0 and ki<nZ-1:            
                            dPdy[ki,kj,n_p]= (P[ki,kj+1,n_p]-P[ki,kj-1,n_p])/(2*dY)
                            dPdz[ki,kj,n_p]= (P[ki+1,kj,n_p]-P[ki-1,kj,n_p])/(2*dZ)
                
                
                        if ki==nZ-1 and kj>0 and kj<ntheta-1:
                            dPdy[ki,kj,n_p]= (P[ki,kj+1,n_p]-P[ki,kj-1,n_p])/(2*dY)
                            dPdz[ki,kj,n_p]= (0-P[ki-1,kj,n_p])/(2*dZ)
               
                
                        if ki==0 and kj==ntheta-1:            
                            dPdy[ki,kj,n_p]= (0-P[ki,kj-1,n_p])/(2*dY)
                            dPdz[ki,kj,n_p]= (P[ki+1,kj,n_p]-0)/(2*dZ)
              
                
                        if kj==ntheta-1 and ki>0 and ki<nZ-1:            
                            dPdy[ki,kj,n_p]= (0-P[ki,kj-1,n_p])/(2*dY)
                            dPdz[ki,kj,n_p]= (P[ki+1,kj,n_p]-P[ki-1,kj,n_p])/(2*dZ)
             
                
                        if kj==ntheta-1 and ki==nZ-1:           
                            dPdy[ki,kj,n_p]= (0-P[ki,kj-1,n_p])/(2*dY)
                            dPdz[ki,kj,n_p]= (0-P[ki-1,kj,n_p])/(2*dZ)
    
#                        dPdy=dPdy
#                        dPdz=dPdz
                        
                        HP=1-Y*np.cos(jj)-X*np.sin(jj)     
                        hpt=(Ypt*np.sin(jj)+Xpt*np.cos(jj))                
        
        
                        mi_p=mi[ki,kj,n_p]
    
                        
                        
                        AE=-(kt*HP*dZ)/(rho*Cp*war*((betha_s*R)**2)*dY)
                        AW=(((HP**3)*dPdy[ki,kj,n_p]*dZ)/(12*mi_p*(betha_s**2)))-((HP)*dZ/2*betha_s)-((kt*HP*dZ)/(rho*Cp*war*((betha_s*R)**2)*dY))
                        AN=-(kt*HP*dY)/(rho*Cp*war*(L**2)*dZ)
                        AS=(((R**2)*(HP**3)*dPdz[ki,kj,n_p]*dY)/(12*(L**2)*mi_p))-((kt*HP*dY)/(rho*Cp*war*(L**2)*dZ))
                        AP=-(AE+AW+AN+AS)
                        
                        
                        auxB_t=(war*mi_ref)/(rho*Cp*T_ref*Cr)
                        B_tG=(mi_ref*war*(R**2)*dY*dZ*P[ki,kj,n_p]*hpt)/(rho*Cp*T_ref*(Cr**2))
                        B_tH=((war*mi_ref*(hpt**2)*4*mi_p*dY*dZ)/(rho*Cp*T_ref*3*HP))
                        B_tI=auxB_t*(mi_p*(R**2)*dY*dZ)/(HP*Cr)
                        B_tJ=auxB_t*((R**2)*(HP**3)*(dPdy[ki,kj,n_p]**2)*dY*dZ)/(12*Cr*(betha_s**2)*mi_p)
                        B_tK=auxB_t*((R**4)*(HP**3)*(dPdz[ki,kj,n_p]**2)*dY*dZ)/(12*Cr*(L**2)*mi_p)
                      
                        B_t = B_tG + B_tH + B_tI + B_tJ + B_tK
                        
                
                        k=k+1
        
                        b_t[k-1,0]=B_t
                            
                        if ki==0 and kj==0:
                            Mat_coef_t[k-1,k-1]=AP+AS-AW
                            Mat_coef_t[k-1,k]=AE
                            Mat_coef_t[k-1,k+ntheta-1]=AN
                            b_t[k-1,0]=b_t[k-1,0]-2*AW*(T_mist_aux[n_p]/Treserv)
                       
                        
                        elif kj==0 and ki>0 and ki<nZ-1: 
                            Mat_coef_t[k-1,k-1]=AP-AW
                            Mat_coef_t[k-1,k]=AE
                            Mat_coef_t[k-1,k-ntheta-1]=AS
                            Mat_coef_t[k-1,k+ntheta-1]=AN
                            b_t[k-1,0]=b_t[k-1,0]-2*AW*(T_mist_aux[n_p]/Treserv)
                
                        elif kj==0 and ki==nZ-1:
                            Mat_coef_t[k-1,k-1]=AP+AN-AW
                            Mat_coef_t[k-1,k]=AE
                            Mat_coef_t[k-1,k-ntheta-1]=AS            
                            b_t[k-1,0]=b_t[k-1,0]-2*AW*(T_mist_aux[n_p]/Treserv)
                
                        elif ki==0 and kj>0 and kj<nY-1:
                            Mat_coef_t[k-1,k-1]=AP+AS
                            Mat_coef_t[k-1,k]=AE
                            Mat_coef_t[k-1,k-2]=AW
                            Mat_coef_t[k-1,k+ntheta-1]=AN
                       
                        elif ki>0 and ki<nZ-1 and kj>0 and kj<nY-1:
                            Mat_coef_t[k-1,k-1]=AP
                            Mat_coef_t[k-1,k-2]=AW
                            Mat_coef_t[k-1,k-ntheta-1]=AS
                            Mat_coef_t[k-1,k+ntheta-1]=AN
                            Mat_coef_t[k-1,k]=AE
              
                        elif ki==nZ-1 and kj>0 and kj<nY-1:
                           Mat_coef_t[k-1,k-1]=AP+AN
                           Mat_coef_t[k-1,k]=AE
                           Mat_coef_t[k-1,k-2]=AW
                           Mat_coef_t[k-1,k-ntheta-1]=AS
                       
                
                        elif ki==0 and kj==nY-1:
                            Mat_coef_t[k-1,k-1]=AP+AE+AS
                            Mat_coef_t[k-1,k-2]=AW
                            Mat_coef_t[k-1,k+ntheta-1]=AN            
                              
                                                
                        elif kj==nY-1 and ki>0 and ki<nZ-1:
                            Mat_coef_t[k-1,k-1]=AP+AE
                            Mat_coef_t[k-1,k-2]=AW
                            Mat_coef_t[k-1,k-ntheta-1]=AS
                            Mat_coef_t[k-1,k+ntheta-1]=AN    
                  
                        
                        elif ki==nZ-1 and kj==nY-1:
                           Mat_coef_t[k-1,k-1]=AP+AE+AN
                           Mat_coef_t[k-1,k-2]=AW
                           Mat_coef_t[k-1,k-ntheta-1]=AS            
                         
                        kj=kj+1
                             
                    kj=0
                    ki=ki+1
                   
                           
                #    %%%%%%%%%%%%%%%%%%%%%% Solution of temperature field %%%%%%%%%%%%%%%%%%%%
                
                
                t=np.linalg.solve(Mat_coef_t,b_t)
                
                cont=0
                  
                for i in np.arange(nZ):
                    for j in np.arange(ntheta):
    
                        T_new[i,j,n_p]=t[cont]
                        cont=cont+1
                        
        
                #    % Dimensional Temperature fied [Pa]
                
                Tdim=T_new*T_ref
                
                T_end = np.sum(Tdim[:,-1,n_p])/nZ
                
                T_mist_aux[n_p] = (fat_mist*T_ref+(1-fat_mist)*T_end)
                
                T_mist = np.ones((nZ,ntheta,n_pad))*(T_mist_aux[n_p]/T_ref)
                
                for i in np.arange(nZ):
                    for j in np.arange(ntheta):
        
                        mi_new[i,j,n_p]=(6.4065*(Tdim[i,j,n_p])**-1.475)/mi_ref
        
                
                  
      
            
    PPlot=np.zeros(((nZ+2),(len(Ytheta))))
    
    cont=0
    for n_p in np.arange(n_pad):
        for ii in np.arange(1,nZ+1):
            cont=1+(n_p)*(ngap/2)+(n_p)*(ntheta+ngap/2)
            for jj in np.arange(1,ntheta+1):
    
                PPlot[ii,int(cont)]=Pdim[int(ii-1),int(jj-1),int(n_p)]
                cont=cont+1
           
    
    auxF=np.zeros((2,len(Ytheta)))
    
    auxF[0,:]=np.cos(Ytheta)
    auxF[1,:]=np.sin(Ytheta)
    
    
    dA=dy*dz 
    
    auxP=PPlot*dA
    
    vector_auxF_x=auxF[0,:]
    vector_auxF_y=auxF[1,:]
    
    
    auxFx=auxP*vector_auxF_x
    auxFy=auxP*vector_auxF_y
    
    fxj=-0.5*np.sum(auxFx)
    fyj=-0.5*np.sum(auxFy)

    
    Fhx=fxj
    Fhy=fyj

    
    score = np.sqrt(((Wx+Fhx)**2)+((Wy+Fhy)**2))
    
    print(f'Score: ', score)
    print('============================================')
    print(f'Força na direção x: ', Fhx)
    print('============================================')
    print(f'Força na direção y: ', Fhy)
    print('')
    return score


res = minimize(hdf, x0, method='Nelder-Mead', tol=10e-3, options={'maxiter': 1000})


print(res)

 

fig = plt.figure()
ax = fig.gca(projection='3d')
Ydim, Zdim = np.meshgrid(Ytheta, Zdim)
surf = ax.plot_surface(Ydim, Zdim, PPlot, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.show()

a=np.transpose(np.average(Tdim, axis=0))
aa=a[0,:]
bb=a[1,:]
tm1=[T_mist_aux[n_pad-1]]
tm2=[T_mist_aux[n_pad-2]]

ytheta=np.arange(2*ntheta)

fig = plt.figure()
cc=np.concatenate((aa,bb), axis=0)

dd=np.concatenate((aa,bb), axis=0)

plt.plot(ytheta, cc, 'ro')
plt.show()    

#sys.exit()
###############################################################################
###############################################################################
#
# Coeficientes de rigidez e amortecimento dos mancais

xeq=res.x[0]*Cr*np.cos(res.x[1])
yeq=res.x[0]*Cr*np.sin(res.x[1])

dE=0.001
epix=np.abs(dE*Cr*np.cos(res.x[1]))
epiy=np.abs(dE*Cr*np.sin(res.x[1]))

Va=war*(R)
epixpt=0.000001*np.abs(Va*np.sin(res.x[1]))
epiypt=0.000001*np.abs(Va*np.cos(res.x[1]))

def Hydroforces(x,y,xpt,ypt, resultados=False):
    global Pdim, dtheta, mi_new, F1, F2

        
    X=x/Cr
    Y=y/Cr

    Xpt=xpt/(Cr*war)
    Ypt=ypt/(Cr*war)
    
    
    for n_p in np.arange(n_pad):          
        
       
        theta1=(n_p)*(betha_s)+(dtheta*ngap/2)+(n_p*dtheta*ngap)
        theta2=theta1+betha_s


        nk=(nZ)*(ntheta)
        Mat_coef=np.zeros((nk,nk))
        b=np.zeros((nk,1))
        ki=0
        kj=0
        k=0 #vectorization pressure index
        
        Z1=0        #initial coordinate z dimensionless
        Z2=1        # final coordinate z dimensionless
        

        
        for ii in np.arange((Z1+0.5*dZ),(Z2),dZ):      
            for jj in np.arange(theta1+(dtheta/2),theta2,dtheta):
         
                hP=1-Y*np.cos(jj)-X*np.sin(jj)                     
                he=1-Y*np.cos(jj+0.5*dtheta)-X*np.sin(jj+0.5*dtheta)
                hw=1-Y*np.cos(jj-0.5*dtheta)-X*np.sin(jj-0.5*dtheta)
                hn=hP
                hs=hn
                
                mi_e = MI_e[n_p]
                mi_w = MI_w[n_p]
                mi_n = MI_n[n_p]
                mi_s = MI_s[n_p]
                
                CE=(dZ*he**3)/(12*mi_e*dY*betha_s**2)
                CW=(dZ*hw**3)/(12*mi_w*dY*betha_s**2)
                CN=(dY*(R**2)*hn**3)/(12*mi_n*dZ*L**2)
                CS=(dY*(R**2)*hs**3)/(12*mi_s*dZ*L**2)
                CP=-(CE+CW+CN+CS)
          
                B=(dZ/2*betha_s)*(he-hw)-((Ypt*np.cos(jj)+Xpt*np.sin(jj))*dY*dZ)
              
                k=k+1
                b[k-1,0]=B
                    
                if ki==0 and kj==0:
                    Mat_coef[k-1,k-1]=CP-CN-CW
                    Mat_coef[k-1,k]=CE
                    Mat_coef[k-1,k+ntheta-1]=CS
               
    
        
                elif ki==0 and kj>0 and kj<nY-1:
                    Mat_coef[k-1,k-1]=CP-CN
                    Mat_coef[k-1,k]=CE
                    Mat_coef[k-1,k-2]=CW
                    Mat_coef[k-1,k+ntheta-1]=CS
               
        
                elif ki==0 and kj==nY-1:
                    Mat_coef[k-1,k-1]=CP-CE-CN
                    Mat_coef[k-1,k-2]=CW
                    Mat_coef[k-1,k+ntheta-1]=CS            
                
        
                elif kj==0 and ki>0 and ki<nZ-1: 
                    Mat_coef[k-1,k-1]=CP-CW
                    Mat_coef[k-1,k]=CE
                    Mat_coef[k-1,k-ntheta-1]=CN
                    Mat_coef[k-1,k+ntheta-1]=CS
         
        
                elif ki>0 and ki<nZ-1 and kj>0 and kj<nY-1:
                    Mat_coef[k-1,k-1]=CP
                    Mat_coef[k-1,k-2]=CW
                    Mat_coef[k-1,k-ntheta-1]=CN
                    Mat_coef[k-1,k+ntheta-1]=CS
                    Mat_coef[k-1,k]=CE
                
                        
                elif kj==nY-1 and ki>0 and ki<nZ-1:
                    Mat_coef[k-1,k-1]=CP-CE
                    Mat_coef[k-1,k-2]=CW
                    Mat_coef[k-1,k-ntheta-1]=CN
                    Mat_coef[k-1,k+ntheta-1]=CS    
          
                
                elif kj==0 and ki==nZ-1:
                    Mat_coef[k-1,k-1]=CP-CS-CW
                    Mat_coef[k-1,k]=CE
                    Mat_coef[k-1,k-ntheta-1]=CN            
                
        
                elif ki==nZ-1 and kj>0 and kj<nY-1:
                   Mat_coef[k-1,k-1]=CP-CS
                   Mat_coef[k-1,k]=CE
                   Mat_coef[k-1,k-2]=CW
                   Mat_coef[k-1,k-ntheta-1]=CN
                  
              
                elif ki==nZ-1 and kj==nY-1:
                   Mat_coef[k-1,k-1]=CP-CE-CS
                   Mat_coef[k-1,k-2]=CW
                   Mat_coef[k-1,k-ntheta-1]=CN            
                 
                kj=kj+1
           
            kj=0
            ki=ki+1
           
        #    %%%%%%%%%%%%%%%%%%%%%% Solution of pressure field %%%%%%%%%%%%%%%%%%%%
        
        p=np.linalg.solve(Mat_coef,b)
      
        cont=0
        
        for i in np.arange(nZ):
            for j in np.arange(ntheta):
                       
                P[i,j,n_p]=p[cont]
                cont=cont+1

        #    %Boundary condiction of pressure
    
        for i in np.arange(nZ):
            for j in np.arange(ntheta):
                if P[i,j,n_p]<0:
                    P[i,j,n_p]=0
          
        #    % Dimensional pressure fied [Pa]
        
        Pdim=P*mi_ref*war*R**2/Cr**2

    
    PPlot=np.zeros(((nZ+2),(len(Ytheta))))
    cont=0
    for n_p in np.arange(n_pad):
        for ii in np.arange(1,nZ+1):
            cont=1+(n_p)*(ngap/2)+(n_p)*(ntheta+ngap/2)
            
            for jj in np.arange(1,ntheta+1):
                
                PPlot[ii,int(cont)]=Pdim[int(ii-1),int(jj-1),int(n_p)]
    #                print(cont)
                cont=cont+1
                
    auxF=np.zeros((2,len(Ytheta)))
    
    auxF[0,:]=np.cos(Ytheta)
    auxF[1,:]=np.sin(Ytheta)
    
    dA=dy*dz 
    
    auxP=PPlot*dA
    
    vector_auxF_x=auxF[0,:]
    vector_auxF_y=auxF[1,:]
    
    auxFx=auxP*vector_auxF_x
    auxFy=auxP*vector_auxF_y
    
    fxj=-0.5*np.sum(auxFx)
    fyj=-0.5*np.sum(auxFy)
    
    
    F1=fxj
    F2=fyj
    
    return(F1,F2)    
# Forças Hidrodinâmicas


Aux01 = Hydroforces(xeq+epix,yeq,0,0)
Aux02 = Hydroforces(xeq-epix,yeq,0,0)
Aux03 = Hydroforces(xeq,yeq+epiy,0,0) 
Aux04 = Hydroforces(xeq,yeq-epiy,0,0)


Aux05 = Hydroforces(xeq,yeq,epixpt,0)
Aux06 = Hydroforces(xeq,yeq,-epixpt,0)
Aux07 = Hydroforces(xeq,yeq,0,epiypt)
Aux08 = Hydroforces(xeq,yeq,0,-epiypt) 


# Coeficientes Adimensionais de Rigidez e Amortecimento dos Mancais


#S=(mi_ref*((R)**3)*L*war)/(np.pi*(Cr**2)*math.sqrt((Wx**2)+(Wy**2)))
S=1/(2*((L/(2*R))**2)*(np.sqrt((F1**2)+(F2**2))))
Ss=S*((L/(2*R))**2)


Kxx=-Ss*((Aux01[0]-Aux02[0])/(epix/Cr))
Kxy=-Ss*((Aux03[0]-Aux04[0])/(epiy/Cr))
Kyx=-Ss*((Aux01[1]-Aux02[1])/(epix/Cr))
Kyy=-Ss*((Aux03[1]-Aux04[1])/(epiy/Cr))


Cxx=-Ss*((Aux05[0]-Aux06[0])/(epixpt/Cr/war))
Cxy=-Ss*((Aux07[0]-Aux08[0])/(epiypt/Cr/war))
Cyx=-Ss*((Aux05[1]-Aux06[1])/(epixpt/Cr/war))
Cyy=-Ss*((Aux07[1]-Aux08[1])/(epiypt/Cr/war))



kxx=(math.sqrt((Wx**2)+(Wy**2))/Cr)*Kxx
kxy=(math.sqrt((Wx**2)+(Wy**2))/Cr)*Kxy
kyx=(math.sqrt((Wx**2)+(Wy**2))/Cr)*Kyx
kyy=(math.sqrt((Wx**2)+(Wy**2))/Cr)*Kyy


cxx=(math.sqrt((Wx**2)+(Wy**2))/(Cr*war))*Cxx
cxy=(math.sqrt((Wx**2)+(Wy**2))/(Cr*war))*Cxy
cyx=(math.sqrt((Wx**2)+(Wy**2))/(Cr*war))*Cyx
cyy=(math.sqrt((Wx**2)+(Wy**2))/(Cr*war))*Cyy


print('kxx = {}'.format(kxx))
print('kxy = {}'.format(kxy))
print('kyx = {}'.format(kyx))
print('kyy = {}'.format(kyy))


print('cxx = {}'.format(cxx))
print('cyx = {}'.format(cyx))
print('cxy = {}'.format(cxy))
print('cyy = {}'.format(cyy))