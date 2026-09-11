# -*- coding: utf-8 -*-
"""Which analyses work on a converted rotor -- and why the others do not.

The "Rotor Model" selector offers converting the rotor to 4 DoF per node
(`rs.utils.convert_6dof_to_4dof`) or to the torsional degree of freedom
(`convert_6dof_to_torsional`). Not every ROSS analysis accepts those models,
and the library says so itself, in the torsional converter's docstring:

    Some Rotor class methods, such as `run_ucs`, and `run_unbalance_response`,
    may not work correctly with the modified rotor object. This is because
    these methods expect a rotor with 6 dofs or at least 4 dofs.

That "such as" leaves the list open, so it cannot be copied and called a table.
What is here has three origins, and every entry names its own:

* `documented` -- ROSS itself warns about it;
* `measured`   -- the `conversion_probe.py` probe ran all 36 combinations
                  through the interface's own route and recorded what happened;
* `domain`     -- it runs without error, but the ROSS model was not built for
                  that number of degrees of freedom. Running is not the same as
                  being validated, and no probe measures that: it is a judgement
                  call from someone who knows the library.

## The two ways to fail, and why both belong here

The obvious one is the analysis **raising**. The other is more dangerous: both
conversions `copy(rotor)` and swap the matrix methods (`M`, `K`, `C`, `G`,
`Ksdt`) for versions that drop degrees of freedom. An analysis that assembles
its own matrices, instead of calling `rotor.M()`, uses the 6 DoF ones **with no
error at all** -- and returns a chart badged "4 DoF" carrying full-model
numbers. A plausible, wrong result, which is the class of defect this whole
refactoring is chasing.

That is why the probe compares each conversion's figure against the 6 DoF one:
an identical figure means the conversion was ignored, and that counts as
unsupported.
"""

# The model names are the same ones the screen badges show, and they are
# deliberately language-neutral: they also appear in the exported Python
# script, which is written in English.
MODEL_NAMES = {"": "6 DoF", "4dof": "4 DoF", "torsional": "Torsional"}

RAISES = "raises"  # ROSS raises an exception
IGNORED = "ignored"  # runs, and returns the full-model chart
NOT_VALIDATED = "not_validated"  # runs, but the model was not built for it


def _I(analysis, conversion, failure, origin, reason_en, reason_pt):
    return (
        (analysis, conversion),
        {
            "analysis": analysis,
            "conversion": conversion,
            "failure": failure,
            "origin": origin,
            "reason": {"en": reason_en, "pt": reason_pt},
        },
    )


# --- what the probe measured, on ross 2.3.0 ----------------------------------
#
# Five analyses index lateral degrees of freedom the torsional model does not
# have: the error is always an IndexError whose axis size equals the number of
# nodes (or 1), because the reduced matrix keeps one degree per node.
_INDEXES_LATERAL_EN = (
    "This analysis addresses lateral degrees of freedom, which the torsional "
    "model does not have (it keeps one degree of freedom per node). Measured on "
    "ross 2.3.0: IndexError."
)
_INDEXES_LATERAL_PT = (
    "Esta analise enderecca graus de liberdade laterais, que o modelo torcional "
    "nao tem (ele guarda um grau por no). Medido no ross 2.3.0: IndexError."
)

_IGNORES_EN = (
    "This analysis does not go through the reduced matrices: it produced exactly "
    "the same chart as the full 6 DoF model. The conversion has no effect, so a "
    "chart labelled with the reduced model would be showing 6 DoF numbers."
)
_IGNORES_PT = (
    "Esta analise nao passa pelas matrizes reduzidas: ela produziu exatamente o "
    "mesmo grafico do modelo completo de 6 DoF. A conversao nao tem efeito, "
    "entao um grafico com o selo do modelo reduzido estaria mostrando numeros "
    "de 6 DoF."
)

UNSUPPORTED = dict(
    [
        # --- index lateral degrees the torsional model does not have -------------
        _I(
            "clearance",
            "torsional",
            RAISES,
            "measured",
            _INDEXES_LATERAL_EN,
            _INDEXES_LATERAL_PT,
        ),
        _I(
            "crack",
            "torsional",
            RAISES,
            "measured",
            _INDEXES_LATERAL_EN,
            _INDEXES_LATERAL_PT,
        ),
        _I(
            "misalignment",
            "torsional",
            RAISES,
            "measured",
            _INDEXES_LATERAL_EN,
            _INDEXES_LATERAL_PT,
        ),
        _I(
            "rubbing",
            "torsional",
            RAISES,
            "measured",
            _INDEXES_LATERAL_EN,
            _INDEXES_LATERAL_PT,
        ),
        _I(
            "unbalance",
            "torsional",
            RAISES,
            "measured",
            "Unbalance response is a lateral phenomenon and the torsional model keeps "
            "only the torsional degree of freedom. ROSS documents "
            "run_unbalance_response as incompatible with this conversion; measured on "
            "ross 2.3.0: IndexError.",
            "A resposta ao desbalanceamento e um fenomeno lateral, e o modelo "
            "torcional guarda so o grau de liberdade torcional. O ROSS documenta o "
            "run_unbalance_response como incompativel com esta conversao; medido no "
            "ross 2.3.0: IndexError.",
        ),
        # --- run, and ignore the conversion --------------------------------------
        _I("ucs", "4dof", IGNORED, "measured", _IGNORES_EN, _IGNORES_PT),
        _I(
            "ucs",
            "torsional",
            IGNORED,
            "measured",
            "The UCS diagram varies bearing stiffness in the lateral directions, "
            "which the torsional model does not have. ROSS documents run_ucs as "
            "incompatible with this conversion, and it does not raise: measured on "
            "ross 2.3.0 it produced exactly the 6 DoF chart.",
            "O diagrama UCS varia a rigidez dos mancais nas direcoes laterais, que o "
            "modelo torcional nao tem. O ROSS documenta o run_ucs como incompativel "
            "com esta conversao, e ele nao estoura: medido no ross 2.3.0, produziu "
            "exatamente o grafico de 6 DoF.",
        ),
        _I("static", "4dof", IGNORED, "measured", _IGNORES_EN, _IGNORES_PT),
        _I("static", "torsional", IGNORED, "measured", _IGNORES_EN, _IGNORES_PT),
        # --- fault models, built for 6 DoF ---------------------------------------
        #
        # These three **do run** on the 4 DoF model and return a chart that differs
        # from the 6 DoF one: they neither raise nor ignore the conversion. They are
        # here anyway, with origin `domain`: the ROSS fault models (crack,
        # misalignment, rubbing) were developed for the 6 DoF model, and running
        # without error is not the same as being validated. This is a judgement
        # call, made with the library at hand, and it is provisional: if a ROSS
        # release starts covering these cases, the entry goes away.
        #
        # `crack` has a symptom of its own on top of that -- with the default
        # values the Newton-Raphson iteration did not converge.
        _I(
            "crack",
            "4dof",
            NOT_VALIDATED,
            "domain",
            "The crack model in ROSS was developed for the 6 DoF rotor model; running "
            "it on the reduced model is not validated. With the default values the "
            "time integration also failed to converge (Newton-Raphson reached the "
            "maximum number of iterations).",
            "O modelo de trinca do ROSS foi desenvolvido para o rotor de 6 DoF; "
            "roda-lo no modelo reduzido nao e um caso validado. Com os valores "
            "padrao a integracao no tempo tambem nao convergiu (o Newton-Raphson "
            "chegou ao maximo de iteracoes).",
        ),
        _I(
            "misalignment",
            "4dof",
            NOT_VALIDATED,
            "domain",
            "The misalignment model in ROSS was developed for the 6 DoF rotor model; "
            "it runs on the reduced model but the result is not validated.",
            "O modelo de desalinhamento do ROSS foi desenvolvido para o rotor de 6 "
            "DoF; ele roda no modelo reduzido, mas o resultado nao e validado.",
        ),
        _I(
            "rubbing",
            "4dof",
            NOT_VALIDATED,
            "domain",
            "The rubbing model in ROSS was developed for the 6 DoF rotor model; it "
            "runs on the reduced model but the result is not validated.",
            "O modelo de roçamento do ROSS foi desenvolvido para o rotor de 6 DoF; "
            "ele roda no modelo reduzido, mas o resultado nao e validado.",
        ),
    ]
)


def reason(analysis, conversion, language="en"):
    """Say why this combination is unsupported, or None when it is."""
    entry = UNSUPPORTED.get((analysis, conversion or ""))
    if not entry:
        return None
    return entry["reason"].get(language) or entry["reason"]["en"]


def supported(analysis, conversion):
    return (analysis, conversion or "") not in UNSUPPORTED


def table(language="en"):
    """Return what the screen needs to warn before computing.

    One table, served alongside the catalog: the screen warns and the route
    refuses by reading the same place. Two copies would drift apart, and the
    one drifting silently would be the screen's -- the user would be told
    "allowed" and then get "not allowed".
    """
    output = {}
    for (analysis, conversion), entry in UNSUPPORTED.items():
        output.setdefault(analysis, {})[conversion] = {
            "failure": entry["failure"],
            "reason": entry["reason"].get(language) or entry["reason"]["en"],
        }
    return output
