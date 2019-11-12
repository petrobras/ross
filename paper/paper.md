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
and join them in order to construct rotors. It is possible to plot the rotor geometry, 
run simulations, and obtain results in the form of graphics, by performing Static analysis, Campbell Diagram,
Frequency response, Forced response, and Mode Shapes.

As an example, Figure 1 shows a centrifugal compressor modeled with ROSS. The shaft elements are in gray, 
the disks are in blue and the bearings are displayed as springs and dampers.

![Centrifugal Compressor modeled with ROSS.](rotor_plot.png)

In this case we used Timoshenko beam theory to model the shaft elements as described by [@friswell2010dynamics].

Figure 2 shows the Campbell Diagram generated for this compressor.

![Campbell Diagram for the Centrifugal Compressor.](campbell.png)

# References