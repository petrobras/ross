Version 2.2.0
-------------

The following enhancements and bug fixes were implemented for this release:

Enhancements
~~~~~~~~~~~~

LLM/Agent-Oriented Documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Added comprehensive documentation and cookbook recipes for LLM and AI agent integration:

- Restructured ``CLAUDE.md`` with a quick-reference table for all ``run_*`` methods, units conventions, example rotors, and cookbook pointers
- Added 12 focused cookbook recipe files in ``docs/cookbook/`` covering modal analysis, Campbell diagram, critical speed, static analysis, unbalance response, frequency response, time response, UCS & Level 1, faults, bearings advanced, building rotors, and gotchas
- Added ``docs/cookbook/README.md`` as an index for all recipes
- Expanded ``AGENTS.md`` with a quick-start card for agent onboarding


Bug Fixes
~~~~~~~~~

Fix seal_leakage for list/array inputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fixed ``seal_leakage`` handling in bearing seal elements to correctly detect whether the value is a collection (list or NumPy array) or a single scalar (float/int). Previously, passing a list of leakage values would cause incorrect behavior (`#1270 <https://github.com/petrobras/ross/issues/1270>`_).


Removals
~~~~~~~~

Remove MCP Server Package
^^^^^^^^^^^^^^^^^^^^^^^^^^

Removed the ``ross/mcp/`` package, including the MCP server, ``__main__`` module, and ``__init__``. Also removed the ``mcp`` optional dependency, the ``ross-mcp`` entry point from ``pyproject.toml``, and all MCP-related references from documentation and tutorials.


Contributors
~~~~~~~~~~~~

This release includes contributions from: @raphaeltimbo, @ViniciusTxc3
