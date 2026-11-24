# Bearing Surrogate Models

Source: `docs/user_guide/tutorial_surrogates.ipynb`

A surrogate replaces a slow bearing solver (`PlainJournal`, `TiltingPad`, `ThrustPad`) by a metamodel of its stiffness and damping coefficients. The solver runs once per training sample; after that, `predict` returns a `BearingElement` in milliseconds. Use it when the same bearing is evaluated many times (sweeps, optimization, stochastic analysis).

The surrogates are an optional extra. Install them with `pip install "ross-rotordynamics[surrogates]"`. Without the extra, `import ross` works and using a surrogate class raises an `ImportError` with this command.

## Train a surrogate

```python
import ross as rs
import ross.surrogates as sg
from ross.units import Q_

surrogate = sg.PlainJournalSurrogate(
    metamodel="KGR",  # "KGR" (Kriging, ross.surrogates.kriging) or "gaussian" (scikit-learn)
    num_inital_samples=50,  # number of solver runs (sic: "inital")
    input_variable=["radial_clearance", "oil_supply_temperature"],  # at most two
    n=3,  # node of the predicted bearing
    frequency=Q_([900, 1100], "RPM"),  # [min, max] of the trained speed range
    pad_axial_length=0.263144,
    journal_diameter=0.4,
    radial_clearance=[1.75e-4, 2.05e-4],  # [min, max] because it is an input variable
    n_pads=2,
    pad_arc=Q_(176, "deg"),
    oil_supply_temperature=Q_([40, 60], "degC"),  # [min, max]
    fys_load=-112814.91,
    lubricant="ISOVG32",
    oil_flow_v=Q_(37.86, "l/min"),
)
```

- The constructor trains the model when `input_variable` is given. It runs the solver once per sample, so it takes as long as `num_inital_samples` solver runs.
- Every variable listed in `input_variable` is given as `[min, max]`; the other arguments are the fixed arguments of the solver (same names, meaning and SI units as `PlainJournal`, `TiltingPad` or `ThrustPad`).
- `TiltingPadSurrogate` and `ThrustPadSurrogate` take the arguments of `TiltingPad` and `ThrustPad`. `ThrustPadSurrogate` predicts only `kzz` and `czz`.
- `distribution="LHS"` (default) samples by Latin Hypercube; `"FullFactorial"` uses a regular grid. `corr` is the Kriging correlation (default `"squar_exp"`).
- `save_sample="samples.csv"` writes each sample as it is computed. If the file exists, its samples are reused and only the missing ones are computed, so an interrupted training can resume.
- `surrogate.cylidrical_bearing` holds the solver bearings of the training samples.

## Predict

```python
bearing = surrogate.predict(
    frequency=Q_(1000, "RPM"),  # one speed or an array of speeds (default unit rad/s)
    radial_clearance=1.9e-4,  # one value for each variable in input_variable
    oil_supply_temperature=Q_(50, "degC"),
)
bearing.kxx  # a BearingElement at node n, usable in rs.Rotor
```

`predict` extrapolates silently. Pass `verify_limits=True` to get a `UserWarning` when the speed or an input is outside the trained range, or call `surrogate.verify_limits_predict(frequency, args)` to get the values outside the range.

## Use the bearing in a rotor

```python
rotor = rs.Rotor(shaft_elements, disk_elements, [bearing, other_bearing])
modal = rotor.run_modal(speed=Q_(1000, "RPM"))
```

Predict again, inside a loop, for each operating point.

## Save and load

```python
surrogate.save("journal_surrogate")  # writes journal_surrogate.zip
surrogate = sg.PlainJournalSurrogate.load("journal_surrogate")
```

The file holds only what `predict` needs (a few MB, not the training matrices). `load` unpickles with an allowlist of modules; load only files you trust.

## Gotchas

- The argument is spelled `num_inital_samples`.
- `step_num_inital_samples` and `coef_R2` are accepted but not used.
- Keep predictions inside the trained `frequency` and input ranges; the metamodel is not valid outside them.
- Check a few predictions against the solver before trusting a surrogate; more samples are needed as the input ranges grow.
