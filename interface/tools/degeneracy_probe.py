# -*- coding: utf-8 -*-
"""Two modes at the same frequency: then neither shape is determined.

    python tools/degeneracy_probe.py

WHAT THIS IS FOR.

`--worker-check` found `modes` coming back from a second process with values two
hundred per cent apart, while `campbell`'s frequencies moved only in the last
bit. Two explanations were proposed and both were **measured and rejected**:
ARPACK does not start from a random vector (ROSS passes `v0=np.ones(...)`), and
pinning `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` to 1
changed nothing -- same two analyses, same one run in three.

This probe tests the third explanation, which is not about the arithmetic at
all. `modes` runs at **speed zero** by default. A rotor that is symmetric about
its axis has, at zero speed, no gyroscopic coupling to separate whirl in one
plane from whirl in the perpendicular one: the two are the **same frequency**.
An eigenvalue repeated twice does not have "the" eigenvector -- every
combination of the two is one, and they all satisfy the equations equally. Which
member of the pair a solver hands back is decided by the last bits of the
arithmetic, and any two processes are free to disagree about it while both are
exactly right.

The numbers below say whether that is what is happening:

* the damped natural frequencies come **in pairs**, each pair the same number to
  twelve digits or so -- not "close", the same eigenvalue twice;
* mode 0 and mode 1 are the two members of the first pair, and the value printed
  for each is `customdata` -- which in ROSS's `plot_mode_2d` is `angle_0`, the
  **orbit angle**, not the deflection. Expect the two to sit about ninety
  degrees apart. That is the pair being orthogonal, and it is what the
  disagreement between processes is a small rotation of.

WHAT IT MEANS IF IT IS TRUE, and this is the part that matters more than the
worker: the interface **already** draws the mode with an orientation that is not
reproducible between runs. It has nothing to do with slice 6 -- opening the same
rotor tomorrow can give the orbit turned by a degree or two, and there is no
wrong answer among them. What the screen should do about that is a question for
the interface, not for the pipe.
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from domain.analysis_catalog import default_params  # noqa: E402
from selftest import ROTOR  # noqa: E402
from services.analysis import get_runner  # noqa: E402
from services.analysis.pipeline import prepared_rotor  # noqa: E402
from services.worker.compare import decoded  # noqa: E402


def first_numbers(figure, how_many=4):
    """The first numeric array Plotly wrote into this figure.

    Reusing the worker's decoder rather than writing a second one: it is the
    same base64 the check compares, and two decoders would be two things to
    keep right.
    """
    payload = json.loads(figure.to_json())
    for trace in payload.get("data", []):
        for key in sorted(trace):
            numbers = decoded(trace[key])
            if numbers:
                return key, numbers[:how_many]
    return "", []


def frequencies(result):
    """The damped natural frequencies, whatever this ROSS release calls them."""
    for name in ("wd", "wn"):
        values = getattr(result, name, None)
        if values is not None:
            return name, [float(v) for v in values]
    return "", []


def main():
    params = default_params("modes")
    runner = get_runner("modes")
    rotor = prepared_rotor(ROTOR, "none")
    spec = runner.spec(params, rotor)

    print("DEGENERACY PROBE")
    print("python %s on %s" % (sys.version.split()[0], sys.platform))
    print("the spec the interface uses for `modes`: %s" % spec)
    print("")

    result = runner.compute(rotor, spec)
    name, values = frequencies(result)
    if not values:
        print("this ROSS release exposes neither `wd` nor `wn` on a modal result")
        return 1

    print(" mode  %-22s  distance from the mode before" % name)
    for position, value in enumerate(values[:8]):
        if position == 0:
            print("  %2d   %-22.15g" % (position, value))
            continue
        before = values[position - 1]
        scale = max(abs(value), abs(before)) or 1.0
        print(
            "  %2d   %-22.15g  %.3g relative"
            % (position, value, abs(value - before) / scale)
        )

    print("")
    for index in (0, 1):
        key, numbers = first_numbers(
            runner.plot(result, dict(params, plot_idx=index), rotor)
        )
        print(
            "  the chart for mode %d begins, in `%s`, with %s"
            % (index, key, ["%.6g" % v for v in numbers])
        )

    print("")
    print("Read it like this: if the frequencies come in pairs that agree to")
    print("twelve digits, each pair is one eigenvalue twice, and neither member")
    print("printed below is the one the equations pick -- both are, and so is")
    print("every mixture of them. A second process choosing a different mixture")
    print("is not a defect of the process. `customdata` is the orbit angle, so")
    print("the two members should sit about ninety degrees apart.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
