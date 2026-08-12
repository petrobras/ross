"""Thermo-elasto-hydrodynamic (TEHD) solver for fluid-film journal bearings.

Internal numerical core of ROSS's hydrodynamic journal-bearing classes --
not part of the public API. The solver models a tilting-pad or
fixed-geometry journal bearing. One analysis proceeds as:

==========================  ==================================================
Module                      Responsibility
==========================  ==================================================
``constants``               Physical constants and the legal values of every
                            string type flag; iteration limits and convergence
                            tolerances.
``state``                   The objects the modules below pass between them:
                            the two meshes, the pad geometry and the
                            lubricant. Frozen -- built once, never mutated.
``mesh``                    Finite-element meshes: the Reynolds (x-z) film
                            mesh, the energy (x-y) film+pad mesh, the pad
                            deformation mesh, and the 3-D film mesh that ties
                            the two orthogonal 2-D meshes together.
``hydrodynamics``           Film thickness, velocities, flow rates, turbulence
                            regime and effective viscosity; drives the journal
                            equilibrium search and the starvation iteration.
``pressure``                Solves the generalized Reynolds equation for the
                            film pressure on one pad.
``thermal``                 Film and pad temperature: the adiabatic 2-D energy
                            equation over the Reynolds mesh, and the full
                            energy equation over the film+pad mesh with
                            conduction and convection.
``deform``                  Elastic and thermal deformation of the pads and
                            the pivot flexibility.
``coefficients``            Perturbation solves for the stiffness and damping
                            coefficients, the journal-equilibrium Jacobian,
                            and the dynamic reduction over the pad tilt (and
                            pivot) degrees of freedom.
``driver``                 Orchestration: meshes once, then loops over speed
                            cases through the thermal / deformation /
                            equilibrium iterations and assembles the outputs.
                            :func:`~ross.bearings.fluid_film.driver.run_case` is the
                            entry point.
==========================  ==================================================

``_numba_kernels`` holds the ``@njit`` inner loops the modules above call; each
kernel documents the readable Python it replaces.

Everything is SI and every array is 0-based: arrays are allocated exactly as
large as they need to be, connectivity arrays store node and element numbers
that index the coordinate arrays directly, and loops run ``range(total_*)``.
"""
