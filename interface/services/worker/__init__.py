# -*- coding: utf-8 -*-
"""The analysis worker: a second process that owns the ROSS objects.

Why a process at all, and why this shape -- every point below was measured
before it was decided:

* an analysis costs 6 to 12 seconds on a 40-element rotor, and the cost grows
  with the model -- so there is a wait worth managing;
* a thread cannot be interrupted mid-computation, so cancellation that actually
  stops the CPU needs a process;
* two of the twelve ROSS results do not pickle, so the worker cannot hand
  results back: it keeps them and returns the finished chart as text;
* a cold process costs about 9 seconds to spawn and import ROSS, plus up to 16
  seconds of numba compilation -- so the worker is persistent, not per job.

The split inside this package follows what can be tested and what cannot:

    protocol.py   one message in, one message out. A pure function, no I/O,
                  no process. The suite exercises the twelve through it.
    child.py      the loop that reads lines and writes lines. Thin on purpose.
    host.py       the parent's handle on the child: start, ask, restart.
"""
