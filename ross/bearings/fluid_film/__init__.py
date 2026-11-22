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
                            regime and effective viscosity; the pad-tilt root
                            finding (``PadTiltSearch``); drives the journal
                            equilibrium search and the starvation iteration.
``pressure``                Solves the generalized Reynolds equation for the
                            film pressure on one pad (no solve for a fully
                            cavitated pad).
``thermal``                 Film and pad temperature: the adiabatic 2-D energy
                            equation over the Reynolds mesh, and the full
                            energy equation over the film+pad mesh with
                            conduction and convection.
``deform``                  Elastic and thermal deformation of the pads and
                            the pivot flexibility.
``coefficients``            Perturbation solves for the stiffness and damping
                            coefficients and the journal-equilibrium Jacobian
                            (one factorisation per pad and perturbation set).
``padmatrices``             Unreduced journal, pad tilt and pivot matrices
                            (``PadMatrices``), the pad inertia about the
                            pivot, and their exact condensation at any whirl
                            frequency (``film.condense``).
``diagnostics``             Convergence records of every loop of a case
                            (``LoopMonitor``), the Reynolds-solve and time
                            budget (``CaseBudget``) and
                            ``FluidFilmConvergenceError``.
``driver``                  Orchestration: builds the speed-independent state
                            (meshes) and solves one speed case through the
                            thermal / deformation / equilibrium iterations,
                            then assembles the outputs.
                            :func:`~ross.bearings.fluid_film.driver.run_case`
                            is the single-case entry point.
``sweep``                   ``BearingSetup`` (meshes once, read-only),
                            ``solve_case``, ``solve_sweep`` (warm-started
                            chains of speeds, worker processes) and
                            :func:`~ross.bearings.fluid_film.sweep.film_force_at`
                            (the film force with the journal held at a given
                            position: finite-difference reference, force
                            maps).
==========================  ==================================================

``_numba_kernels`` holds the ``@njit`` inner loops the modules above call; each
kernel documents the readable Python it replaces.

Everything is SI and every array is 0-based: arrays are allocated exactly as
large as they need to be, connectivity arrays store node and element numbers
that index the coordinate arrays directly, and loops run ``range(total_*)``.
"""
