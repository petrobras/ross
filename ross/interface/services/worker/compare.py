# -*- coding: utf-8 -*-
"""Do two charts say the same thing?

The worker check began by comparing charts **byte for byte**, and that was the
right way to start: the worker exists to move where the work happens, not to
change the answer, and the strictest possible test is the one that finds out
fastest whether that is true.

It found something else instead. Two of the twelve analyses -- `campbell` and
`modes`, the two that go through ARPACK with `sparse=True` -- came back
different from a second process on about one run in three, and the two
differences are not the same kind of thing at all:

* `campbell`'s critical speeds moved in the **last bit**: 1984.2799673184086
  against 1984.2799673183688, two parts in a hundred thousand billion. No
  screen, no unit, no report can tell those apart. Reporting it as "the worker
  changed the answer" is a false accusation, and a check that cries wolf twice a
  week stops being read.
* `modes` moved by **two hundred per cent**: 0.011971 against 0.036790. That
  one has to be reported loudly -- and reported *precisely*, which the first
  version of this file was not. Those numbers are `customdata`, and
  `customdata` in ROSS's `plot_mode_2d` is `angle_0`, the orbit angle for the
  hover box, in radians. So the disagreement is 0.025 rad -- about a degree and
  a half -- in an angle that happens to sit near zero, which is why the
  relative difference is enormous while the number is small. Reading it as the
  mode shape, and saying so, was a mistake this file's design made possible by
  stopping at the first difference; see `Comparison`.

  The cause is settled: `tools/degeneracy_probe.py` measured the rotor's modes
  in pairs -- 207.794467303244 and 207.794467304002, then 580.058527967929 and
  580.058527968867 -- the same frequency to twelve digits. At speed zero a
  symmetric rotor has no gyroscopic coupling to separate whirl in one plane
  from the other, and a repeated eigenvalue has no single eigenvector: every
  combination of the pair is one. Which representative a solver returns is
  decided by the last bits, so two processes disagree about the orientation
  while both are exactly right.

A byte comparison cannot tell those two apart, so it says the wrong thing about
both. This module can: it parses the two charts, decodes the numeric arrays
Plotly writes as base64, and compares them **numerically**, returning one of
three verdicts -- identical, close, different -- with the largest disagreement
and where it is.

## Why decoding is unavoidable

Plotly does not write a chart's numbers as JSON numbers. An array arrives as

    {"dtype": "f8", "bdata": "DrLArx4Bn0CNMP9FNQGfQA=="}

-- raw little-endian doubles in base64. Comparing that text is comparing the
bits, which is exactly the comparison being replaced: one bit of difference in
the last place of a float changes four characters of base64 and tells you
nothing about how far apart the two numbers are. The bytes have to become
numbers again before the question can even be asked.

## The tolerance, and what it is a tolerance *for*

The question this file answers is not "are these two computations equal" -- it
is **"would anyone see a different chart"**. So the threshold is set far below
anything a screen, an axis label or an exported figure could show, and far
above the noise of a reduction that summed its terms in another order:
`rtol=1e-9`. The 2e-14 above passes it by five orders of magnitude; the 25% of
`modes` fails it by seven.

`atol` exists for one case only: a coordinate that is zero on one side. There
is no relative scale at zero, so an absolute floor has to stand in for it, and
1e-12 is well under the resolution of every quantity this program plots.

## What it deliberately does not do

It does not compare only the numbers. Keys, list lengths, strings, booleans and
`null`s are compared **exactly**, and any disagreement in them is `different`
without appeal. A tolerance that quietly forgave a renamed trace or a missing
axis would be a tolerance for the wrong thing: the numerical noise is a known,
measured, bounded property of ARPACK; a changed structure is a bug.

Nothing here imports ROSS, numpy or Flask -- it is base64, struct and
arithmetic -- so the suite can drive it anywhere, including on a machine with
none of the scientific stack installed.
"""

import base64
import json
import math
import struct

IDENTICAL = "identical"
CLOSE = "close"
DIFFERENT = "different"

# The two thresholds, written once. They used to be default arguments in two
# places -- the class and the function -- which is one place too many: changing
# the tolerance in one of them would have left the other quietly deciding the
# verdicts, and no test would have noticed.
RTOL = 1e-9
ATOL = 1e-12

# The dtype tags Plotly puts next to `bdata`, and how `struct` spells each one.
# Everything is little-endian: that is the format's rule, not a guess about the
# machine, and writing "<" here is what keeps the comparison correct on a
# big-endian one.
FORMATS = {
    "f8": "d",
    "f4": "f",
    "i1": "b",
    "i2": "h",
    "i4": "i",
    "i8": "q",
    "u1": "B",
    "u2": "H",
    "u4": "I",
    "u8": "Q",
}


def decoded(node):
    """The numbers inside a Plotly binary block, or None if it is not one.

    Returning None for anything unrecognised is the safe direction: an unknown
    `dtype` then falls back to the exact comparison of the surrounding
    dictionary, which is stricter, rather than to no comparison at all.
    """
    if not isinstance(node, dict):
        return None
    if "bdata" not in node or "dtype" not in node:
        return None
    code = FORMATS.get(node.get("dtype"))
    if code is None:
        return None
    try:
        raw = base64.b64decode(node["bdata"])
    except Exception:
        return None
    size = struct.calcsize(code)
    if size == 0 or len(raw) % size:
        return None
    return list(struct.unpack("<%d%s" % (len(raw) // size, code), raw))


def _kind(value):
    if isinstance(value, dict):
        return "an object of %d keys" % len(value)
    if isinstance(value, list):
        return "a list of %d" % len(value)
    return repr(value)


def _array(path):
    """The array a value belongs to: its path with the index taken off."""
    if path.endswith("]") and "[" in path:
        return path[: path.rindex("[")]
    return path


class Comparison(object):
    """The running verdict, and it treats two kinds of disagreement differently.

    **A structural difference stops the walk.** A renamed key, a trace that
    disappeared, a string that changed: past that point the two charts are not
    two versions of the same drawing, and continuing would only pile up
    consequences of the first fact.

    **A numeric difference does not.** The first version of this class stopped
    at the first number out of tolerance too, and that cost a wrong answer.
    `modes` was reported as differing at `customdata[0]` and the report stopped
    there -- so I read the first array I saw as the mode shape and told Leonardo
    the picture had changed. It had not: `customdata` in `plot_mode_2d` is
    ROSS's `angle_0`, the orbit angle for the hover box, and whether the
    plotted deflection `y` also differed was a question the report had stopped
    before reaching. "Which parts of the drawing disagree" is exactly what a
    human needs and exactly what stopping early throws away, so the numeric
    sweep now runs to the end and reports per array: how many values are out of
    tolerance, the worst one, and which arrays agree.
    """

    def __init__(self, rtol=RTOL, atol=ATOL):
        self.rtol = rtol
        self.atol = atol
        self.worst = 0.0
        self.worst_at = ""
        self.structural = None
        # array path -> [values compared, values beyond tolerance, worst gap,
        # where the worst one was, the two numbers there]
        self.arrays = {}
        # Every value agreed and yet the two texts are not the same: key order,
        # spacing, or a number spelled two ways. It is not `identical` -- the
        # byte comparison would have failed -- and it is not a numerical gap
        # either, so it needs a state of its own rather than a fake gap.
        self.textual = False

    @property
    def apart(self):
        """The arrays holding at least one value beyond tolerance."""
        return {where: row for where, row in self.arrays.items() if row[1]}

    @property
    def verdict(self):
        if self.structural is not None or self.apart:
            return DIFFERENT
        if self.worst or self.textual:
            return CLOSE
        return IDENTICAL

    def unequal(self, path, one, other):
        if self.structural is None:
            self.structural = "%s: %s vs %s" % (path or "the chart", one, other)

    def numbers(self, path, one, other):
        one, other = float(one), float(other)
        where = _array(path)
        row = self.arrays.setdefault(where, [0, 0, 0.0, "", ("", "")])
        row[0] += 1
        if one == other or (math.isnan(one) and math.isnan(other)):
            return

        gap = abs(one - other)
        scale = max(abs(one), abs(other))
        if math.isinf(gap) or math.isnan(gap):
            relative = float("inf")
            allowed = False
        else:
            relative = gap / scale if scale else gap
            allowed = gap <= self.atol + self.rtol * scale

        if allowed:
            if relative > self.worst:
                self.worst, self.worst_at = relative, path
            return

        row[1] += 1
        if relative >= row[2]:
            row[2:] = [relative, path, ("%.17g" % one, "%.17g" % other)]

    def describe(self):
        if self.structural is not None:
            return self.structural
        apart = self.apart
        if apart:
            where = max(apart, key=lambda name: apart[name][2])
            compared, beyond, relative, at, (mine, theirs) = apart[where]
            agree = len(self.arrays) - len(apart)
            return (
                "%d of %d arrays differ; worst %.3g relative at %s "
                "(%s vs %s; %d of %d values in that array); %d agree"
                % (
                    len(apart),
                    len(self.arrays),
                    relative,
                    at or where,
                    mine,
                    theirs,
                    beyond,
                    compared,
                    agree,
                )
            )
        if self.worst:
            return "largest relative difference %.3g, at %s" % (
                self.worst,
                self.worst_at or "the chart",
            )
        if self.textual:
            return "every value agrees; the two texts differ only in formatting"
        return "byte for byte"


def _walk(one, other, path, state):
    if state.structural is not None:
        return

    mine, theirs = decoded(one), decoded(other)
    if mine is not None or theirs is not None:
        if mine is None or theirs is None:
            state.unequal(path, _kind(one), _kind(other))
        elif len(mine) != len(theirs):
            state.unequal(path, "%d numbers" % len(mine), "%d numbers" % len(theirs))
        elif one.get("shape") != other.get("shape"):
            state.unequal(path, one.get("shape"), other.get("shape"))
        else:
            for position, (a, b) in enumerate(zip(mine, theirs, strict=True)):
                state.numbers("%s[%d]" % (path, position), a, b)
        return

    if isinstance(one, dict) and isinstance(other, dict):
        if set(one) != set(other):
            missing = sorted(set(one) ^ set(other))
            state.unequal(path, "keys %s" % sorted(one), "differ in %s" % missing)
            return
        for key in sorted(one):
            _walk(one[key], other[key], "%s.%s" % (path, key), state)
            if state.structural is not None:
                return
        return

    if isinstance(one, list) and isinstance(other, list):
        if len(one) != len(other):
            state.unequal(path, "%d entries" % len(one), "%d entries" % len(other))
            return
        for position, (a, b) in enumerate(zip(one, other, strict=True)):
            _walk(a, b, "%s[%d]" % (path, position), state)
            if state.structural is not None:
                return
        return

    # `True` is an `int` in Python, and a chart where a flag became 1 is not the
    # same chart. Booleans are settled before the numeric branch, by identity.
    if isinstance(one, bool) or isinstance(other, bool):
        if one is not other:
            state.unequal(path, one, other)
        return

    if isinstance(one, (int, float)) and isinstance(other, (int, float)):
        state.numbers(path, one, other)
        return

    if one != other:
        state.unequal(path, _kind(one), _kind(other))


def compare(one, other, rtol=RTOL, atol=ATOL):
    """Compare two Plotly charts, as text, and say how far apart they are.

    Returns a `Comparison`: `.verdict` is one of `identical`, `close`,
    `different`, and `.describe()` is a line a human can read.

    Equal text short-circuits to `identical` without parsing anything, which is
    both the common case and the cheapest one.
    """
    state = Comparison(rtol=rtol, atol=atol)
    if one == other:
        return state
    try:
        mine = json.loads(one)
        theirs = json.loads(other)
    except ValueError as error:
        state.unequal("", "unreadable JSON (%s)" % error, "")
        return state
    _walk(mine, theirs, "", state)
    if state.verdict == IDENTICAL:
        state.textual = True
    return state
