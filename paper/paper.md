---
title: 'ROSS - Rotordynamic Open Source Software'
tags:
  - rotordynamic
  - dynamics
  - Python
authors:
 - name: Raphael Timbó
   orcid: 0000-0001-7689-5486
   affiliation: 1
 - name: Rodrigo Martins
   orcid: 0000-0002-9996-6600
   affiliation: 2
 - name: Gabriel Bachmann
   orcid: 0000-0003-4401-8408
   affiliation: 4
 - name: Flavio Rangel
   orcid: 0000-0002-4852-8141
   affiliation: 5
 - name: Júlia Mota
   orcid: 0000-0002-9547-9629
   affiliation: 3   
 - name: Juliana Valério
   orcid: 0000-0002-6198-7932
   affiliation: 5
 - name: Thiago G Ritto
   orcid: 0000-0003-0649-6919
   affiliation: 2
affiliations:
 - name: Petrobras - Petróleo Brasileiro S.A.
   index: 1
 - name: Universidade Federal do Rio de Janeiro, Department of Mechanical Engineering, Rio de Janeiro, Brazil
   index: 2
 - name: Universidade Federal do Rio de Janeiro, Graduate Program in Informatics, Rio de Janeiro, Brazil
   index: 3
 - name: Universidade Federal do Rio de Janeiro, Department of Electrical Engineering, Rio de Janeiro, Brazil
   index: 4
 - name: Universidade Federal do Rio de Janeiro, Department of Computer Science, Rio de Janeiro, Brazil
   index: 5
date: 10 October 2019
bibliography: paper.bib
---

# Summary

There are several categories of critical rotating equipment crucial to industry, such as compressors, pumps, and turbines. Computational mechanical models aim to simulate the behavior of such mechanical systems and these models are used to support research and decision making. To this purpose, we present ROSS, an open source library written in Python for rotordynamic analysis.

Existing tools that have rotordynamic functionalities include commercial finite element software with rotordynamic modules [@comsol; @ansys], packages based on proprietary runtimes (MATLAB) [@madyn; @dynamicsrotating], and some standalone tools [@rotorinsa; @trcsoftware]. To use all of these options however requires the purchase of a license and they are not intended to be developed in an open, collaborative manner. Additionally for some of these commercial packages, the user is 'locked in' to the environment, interacting with the software only through a graphical user interface, which makes it harder (impossible sometimes) to automate analyses.

To our knowledge, ROSS is the first software being developed using the open source paradigm in the rotordynamic field, with the code being clearly licensed and fully available on code hosting platforms, issues tracked online, and the possibility of direct contribution by the community.

ROSS allows the construction of rotor models and their numerical simulation. Shaft elements, as a default, are modeled with the Timoshenko beam theory [@Hutchinson2001], which considers shear and rotary inertia effects, and discretized by means of the Finite Element Method [@friswell2010dynamics]. Disks (impellers, blades, or other equipment attached to the rotor) are assumed to be rigid bodies, thus their strain energy is not taken into account and only the kinetic energy due to translation and rotation is calculated. We can obtain the mass and gyroscopic matrices by applying Lagrange’s equations to the total kinetic energy.

The mass matrix is given by:

\begin{equation}
\mathbf{M_e} =  
  \begin{bmatrix}
    m_d & 0 & 0 & 0 \\
    0 & m_d & 0 & 0 \\
    0 & 0 & I_d & 0 \\
    0 & 0 & 0 & I_p  
  \end{bmatrix}
\end{equation}

The gyroscopic matrix is given by:

\begin{equation}
  \mathbf{G_e} =  
  \begin{bmatrix}
    0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 \\
    0 & 0 & 0 & I_p \\
    0 & 0 & -I_p & 0
  \end{bmatrix}
\end{equation}

Where:

\begin{itemize}
  \item $m_d$ is the disk mass;
  \item $I_d$ is the diametral moment of inertia;
  \item $I_p$ is the polar moment of inertia.
\end{itemize}

For most types of bearing, the load-deflection relationship is nonlinear. Furthermore, load-deflection relationships are
often a function of shaft speed (i.e $\mathbf{K_e} = \mathbf{K_e(\omega)}$ and $\mathbf{C_e} = \mathbf{C_e(\omega)}$).
To simplify dynamic analysis, one widely used approach is to assume that the bearing has a linear load-deflection relationship.
This assumption is reasonably valid, provided that the dynamic displacements are small [@friswell2010dynamics].
Thus, the relationship between the forces acting on the shaft due to the bearing and the resultant velocities and
displacements of the shaft may be approximated by:

\begin{equation}
    \begin{Bmatrix}
    f_x \\ f_y
    \end{Bmatrix} = -
    \begin{bmatrix}
    k_{xx} & k_{xy} \\ k_{yx} & k_{yy}
    \end{bmatrix}
    \begin{Bmatrix}
    u \\ v
    \end{Bmatrix} -  
    \begin{bmatrix}
    c_{xx} & c_{xy} \\ c_{yx} & c_{yy}
    \end{bmatrix}
    \begin{Bmatrix}
    \dot{u} \\ \dot{v}
    \end{Bmatrix}
\end{equation}

where $f_x$ and $f_y$ are the dynamic forces in the $x$ and $y$ directions, and $u$ and $v$ are the dynamic displacements
of the shaft journal relative to the bearing housing in the $x$ and $y$ directions.

After defining the element matrices, ROSS performs the assembling of the global matrices and the general form of the
equation of the system is

\begin{equation}\label{eq:general-form}
   \mathbf{M \ddot{q}}(t)
  + \mathbf{C}(\Omega) \mathbf{\dot{q}}(t)
  + \omega \mathbf{G} \mathbf{\dot{q}}(t)
  + \mathbf{K}(\Omega) \mathbf{{q}}(t)
  = \mathbf{f}(t)\,,
\end{equation}

where:
\begin{itemize}
  \item $\textbf{q}$ is the generalized coordinates of the system (displacements and rotations);
  \item $\mathbf{M}$ is the mass matrix;
  \item $\mathbf{K}$ is the stiffness matrix;
  \item $\mathbf{C}$ is the damping matrix;
  \item $\mathbf{G}$ is the gyroscopic matrix;
  \item $\Omega$ is the excitation frequency;
  \item $\omega$ is the rotor whirl speed;
  \item $\mathbf{f}$ is the generalized force vector.
\end{itemize}

After building a model with ROSS, the user can plot the rotor geometry,
run simulations, and obtain results in the form of graphics. ROSS can perform several analyses, such as static analysis,
whirl speed map, mode shapes, frequency response, and time response.

ROSS is extensible and new elements, such as different types of bearings or seals, can be added to the code. As an
example, one can add a class for a tapered roller bearing by inheriting from `BearingElement`. The implementation of
the `BallBearingElement` in our code uses this strategy.

Other elements that require more customization can be added by inheriting directly from `Element`, in this case it is
necessary to implement the required methods that should return the element's mass, stiffness, damping, and gyroscopic
matrices.

We have built the package using main Python packages such as NumPy [@van2011numpy] for multi-dimensional arrays,
SciPy [@2020SciPy-NMeth] for linear algebra, optimization, interpolation and other tasks and Bokeh [@bokeh2019] for creating interactive plots.
Developing the software using Python and its scientific ecosystem enables the user to also make use of this ecosystem,
making it easier to run rotordynamics analysis. It is also easier to integrate the code into other programs, since we
only use open source packages and do not depend on proprietary commercial platforms.

Besides the [documentation](https://ross-rotordynamics.github.io/ross-website/), a set of Jupyter Notebooks
are available for the tutorial and some examples. Users can also access these notebooks through a [Binder server](https://mybinder.org/v2/gh/ross-rotordynamics/ross/main).

As an example, Figure 1 shows a centrifugal compressor modeled with ROSS.

![Centrifugal Compressor modeled with ROSS.](rotor_plot.png)

The shaft elements are in gray,
the impellers represented as disks are in blue and the bearings are displayed as springs and dampers. This plot is generated with Bokeh,
and we can use the hover tool to get additional information for each element.

One analysis that can be carried out is the modal analysis. Figure 2 shows the whirl speed map (Campbell Diagram)
generated for this compressor; the natural frequencies and the log dec vary with the machine rotation speed.

![Whirl speed map (Campbell Diagram) for the Centrifugal Compressor.](campbell.png)

The whirl speed map is one of the results that can be obtained from the model, other types of analyses can be found
in the [documentation](https://ross-rotordynamics.github.io/ross-website/).

ROSS has been used to evaluate the impact of damper seal coefficients uncertainties in a compressor rotordynamics [@timbo2019]. 
Other projects are also using ROSS to test machine learning algorithms that can diagnose machine failure. Here we can create
a model with some problem such as unbalance or misalignment and then use the output data from this model to test the learning
algorithm and check the accuracy of this algorithm in diagnosing the problem.

# Acknowledgements

We acknowledge that ROSS development is supported by Petrobras, Universidade Federal do Rio de Janeiro (UFRJ) and
Agência Nacional de Petróleo, Gás Natural e Biocombustíveis (ANP).

# References
