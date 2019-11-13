---
title: 'ROSS - Rotordynamic Open Source Software'

tags:
  - rotordynamic
  - numerical calculus
  - Python
authors:
 - name: Raphael Timbó
   orcid: 0000-0001-7689-5486
   affiliation: 1
 - name: Rodrigo Martins
   orcid: 0000-0002-9996-6600
   affiliation: 2
 - name: Júlia Mota
   orcid: 0000-0002-9547-9629
   affiliation: 3
 - name: Gabriel Bachmann
   orcid: 0000-0003-4401-8408
   affiliation: 4
 - name: Flavio Rangel
   orcid: 0000-0002-4852-8141
   affiliation: 5
 - name: Juliana Valério
   orcid: 0000-0002-6198-7932
   affiliation: 5
 - name: Thiago Ritto
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

There are several critical rotating equipment crucial to the industry, such as compressors, 
pumps and turbines.
Computational mechanical models aim to simulate the behavior of such mechanical
systems, and to support decision making. To this purpose, we present ROSS, an open source
library written in Python for rotordynamic analysis.

ROSS allows modeling finite elements for components such as shafts, disks, and bearings, 
and join them in order to construct rotors. For shaft elements we can use Timoshenko beam 
theory as described by [@friswell2010dynamics], in this case shear and rotary inertia effects 
are considered. Disks are assumed to be rigid and we ignore the strain energy for their model.
For the bearings a linear load-deflection relationship is assumed.

After defining the elements and assembling the system, it is possible to plot the rotor geometry, 
run simulations, and obtain results in the form of graphics, by performing Static analysis, Campbell Diagram,
Frequency response, Forced response, and Mode Shapes.

The general form of the equation for the system, after matrix assembly is

\begin{equation}\label{eq:general-form}
    \mathbf{M \ddot{q}}
    + \mathbf{C(\Omega) \dot{q}}
    + \omega \mathbf{G \dot{q}}
    + \mathbf{K(\Omega) {q}}
    = \mathbf{f}\,,
\end{equation}

where $\textbf{q}$ represents the displacements and rotations at the
nodes, $\mathbf{M}$, $\mathbf{K}$, $\mathbf{C}$ and $\mathbf{G}$ are the mass, stiffness, damping and gyroscopic matrices, $\Omega$ is the rotor whirl speed and $\mathbf{f}$ is the generalized force vector.

The package has been built using main Python packages such as numpy [@walt2011numpy], scipy [@jones2014scipy] 
and bokeh [@bokeh2019].

In addition to the [documentation](https://ross-rotordynamics.github.io/ross-website/), a set of Jupyter Notebooks 
is available for the tutorial and some examples. These notebooks can also be accessed through a 
[binder server](https://mybinder.org/v2/gh/ross-rotordynamics/ross/master).

As an example, Figure 1 shows a centrifugal compressor modeled with ROSS. 

![Centrifugal Compressor modeled with ROSS.](rotor_plot.png)

The shaft elements are in gray, 
the disks are in blue and the bearings are displayed as springs and dampers. This plot is generated with bokeh, 
and the hover tool can be used to get additional information on each element.

One of the analyses that can be carried ou is the modal analysis. If the modal analysis is carried out for a range 
of different rotor speed we can assemble the Campbel Diagram. Figure 2 shows the Campbell Diagram generated 
for this compressor.

![Campbell Diagram for the Centrifugal Compressor.](campbell.png)

# Acknowledgements
We acknowledge that ROSS development is supported by Petrobras, Universidade Federal do Rio de Janeiro (UFRJ) and 
Agência Nacional de Petróleo, Gás Natural e Biocombustíveis (ANP).

# References