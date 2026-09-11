# -*- coding: utf-8 -*-
"""Undamped critical speed map (UCS)."""

from .base import Runner, register

# Asked for one end and not the other. Not a default and not a silent omission:
# a half-answered range is a question the user started and did not finish, and
# guessing the other end would answer a question nobody asked.
HALF_A_RANGE = (
    "The bearing frequency range needs both ends. Fill in 'Bearing Freq Min' and "
    "'Bearing Freq Max', or leave both empty -- with neither, ROSS derives the "
    "range from the bearing itself."
)


@register
class UcsRunner(Runner):
    """Critical speeds as a function of bearing stiffness.

    `k_min` and `k_max` are **powers of 10**, not stiffnesses: ROSS builds
    `np.logspace(k_min, k_max)`. The screen's label says so ("Min Stiffness
    (10^x N/m)"), and it is worth repeating here because `1e6` instead of `6`
    raises nothing -- it gives a logspace of 10**1e6, which is infinity.

    ## The bearing frequency range, and why it was away

    ROSS 2.3.0 at commit `631a249` raised on **any** value for
    `bearing_frequency_range`: `@check_units` turned the sequence into a numpy
    array and `run_ucs` then did `if bearing_frequency_range:`, a truth test
    numpy refuses for more than one element. The field was withdrawn from the
    form and refused here with an explanation, so that an analysis saved before
    that would not reach the user as numpy's error with no provenance.

    Commit `2a253e6` fixed it -- `if bearing_frequency_range is not None:`, the
    line this project's report proposed -- so the field is back. The premise is
    held by `test_ross_accepts_a_bearing_frequency_range`: install a ROSS without
    the fix and that test says so, instead of the user finding out.
    """

    name = "ucs"

    def spec(self, params, rotor):
        spec = {
            "k_min": self.number(params, "k_min", 4),
            "k_max": self.number(params, "k_max", 10),
            "num_modes": self.integer(params, "num_modes", 4),
            "synchronous": self.flag(params, "synchronous"),
        }

        # Read as text, because the question is "did they fill it in?" and
        # `number` cannot tell an empty field from a zero -- and zero is a
        # legitimate lower end of a frequency range.
        low = self.text(params, "bearing_freq_min")
        high = self.text(params, "bearing_freq_max")
        if (low is None) != (high is None):
            raise ValueError(HALF_A_RANGE)
        if low is not None:
            spec["bearing_frequency_range"] = (float(low), float(high))

        return spec

    def compute(self, rotor, spec):
        # Absent, and not passed as `None`: ROSS's own default derives the range
        # from the bearing, and an explicit `None` is only the same thing for as
        # long as nobody changes how the argument is read. The interface asked
        # nothing, so it says nothing.
        optional = {}
        if "bearing_frequency_range" in spec:
            optional["bearing_frequency_range"] = spec["bearing_frequency_range"]

        return rotor.run_ucs(
            stiffness_range=(spec["k_min"], spec["k_max"]),
            num=50,
            num_modes=spec["num_modes"],
            synchronous=spec["synchronous"],
            **optional,
        )

    def plot(self, result, params, rotor):
        return result.plot(**self.units(params, ["stiffness_units", "frequency_units"]))
