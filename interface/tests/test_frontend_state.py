# -*- coding: utf-8 -*-
"""The analysis state lives in a store, not in the DOM.

Until this point the DOM was the analyses' database. The parameters lived in
`div.rossParams`, the type in `div.rossType`, the frames in `div.rossFrames`,
and the title was **read back** from the header's `innerText` -- with a
`.replace(' (Loaded)', '')` to undo what the screen itself had written.

That is not only inelegant: what was saved to disk came from
`p.data`/`p.layout` of the chart div. An analysis whose card had not rendered
-- because the request failed, for instance -- vanished from what was saved,
with no warning.

The behaviour of the store is in `tests/js/test_store.js`, which exercises the
real functions. Here are the structural properties: that the DOM reads are gone,
and that the state and the screen cannot be cleared separately."""

import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from frontend_source import source as _code
from frontend_source import code_lines as _code_lines


OLD_PROPERTIES = ("rossType", "rossParams", "rossFrames", "rossConversion")


@pytest.mark.parametrize("property_name", OLD_PROPERTIES)
def test_the_dom_no_longer_carries_the_analysis_state(property_name):
    """None of the four properties hung off DOM nodes is left."""
    found_items = [
        "%s:%d" % (mod, number)
        for mod, number, line in _code_lines()
        if property_name in line
    ]
    assert not found_items, "%s ainda aparece em %s" % (property_name, found_items)


def test_nothing_reads_the_cards_as_a_database():
    """`querySelectorAll('.analysis-card')` was the query to the database."""
    found_items = [
        "%s:%d" % (mod, number)
        for mod, number, line in _code_lines()
        if "querySelectorAll('.analysis-card')" in line
    ]
    assert not found_items, "DOM read as a database in %s" % found_items


def test_the_title_is_no_longer_read_back_from_the_screen():
    """Reading the header's `innerText` back as data was the most literal case.

    The screen still writes " (Loaded)" in the title of a restored analysis --
    that is drawing. What went away is the read-back, with the `.replace` that
    undid its own decoration to recover the value."""
    js = _code()
    assert "replace(' (Loaded)', '')" not in js
    assert "titleEl.innerText" not in js


def test_the_store_functions_exist_with_one_definition_each():
    """A silent second definition would overwrite the first."""
    js = _code()
    for func in (
        "registerAnalysis",
        "recordResult",
        "forgetAnalysis",
        "analysesInScreenOrder",
        "analysesToSave",
        "forgetAllAnalyses",
    ):
        assert len(re.findall(r"function %s\(" % func, js)) == 1, func


def test_the_list_and_the_state_are_always_emptied_together():
    """Clearing only one of the two would leave one rotor's analyses alive in another.

    The two places that empty the card list -- entering a rotor and going back to
    the Hub -- have to forget the state as well."""
    js = _code()
    lines = js.split("\n")

    # Counting the calls did not work: the i18n slice added a third legitimate
    # place (redrawing the cards in the new language) and the guard accused the
    # right code. What matters is the pair, not how many pairs exist.
    # Only the card list: `container.innerHTML` is also the element list and the
    # Hub's, which have no state to forget.
    empty_together = []
    for i, line in enumerate(lines):
        if "nnerHTML = ''" not in line:
            continue
        # Removing the empty-list notice deletes no card: there is no state
        # to forget alongside it.
        if "dashboards-empty" in line:
            continue
        before = "\n".join(lines[max(0, i - 25) : i + 1])
        if "analysis-list" in before or "analysisContainer" in line:
            empty_together.append(i)
    assert len(empty_together) >= 2, (
        empty_together
    )  # control: the sweep found the places

    for i in empty_together:
        neighbourhood = "\n".join(lines[max(0, i - 4) : i + 2])
        assert "forgetAllAnalyses();" in neighbourhood, lines[i]

    # and the inverse: forgetting without emptying would leave orphan cards on screen
    for i, line in enumerate(lines):
        if "forgetAllAnalyses();" not in line:
            continue
        neighbourhood = "\n".join(lines[max(0, i - 3) : i + 4])
        assert (
            "nnerHTML = ''" in neighbourhood or "analysesToSave()" in neighbourhood
        ), lines[i]


def test_deleting_a_card_also_forgets_the_analysis():
    """A card with no record would be an orphan; a record with no card, a ghost analysis."""
    excerpt = _code()
    excerpt = excerpt[excerpt.index("async function deleteAnalysis(") :]
    excerpt = excerpt[: excerpt.index("async function addAnalysis(")]
    assert "document.getElementById(cardId).remove()" in excerpt
    assert "forgetAnalysis(" in excerpt


# --- slice 2: persistence (FE-09) ------------------------------------------
#
# The behaviour is in tests/js/test_persistence.js, with a fake localStorage.
# Here are the properties you only see by looking at the whole file.


def test_the_save_is_periodic_and_not_scattered_across_mutations():
    """A clock does not forget; dozens of mutation points would.

    Many places touch the project -- every field of every form, every analysis,
    every Hub operation. Firing the save at each of them would be a hand-kept list,
    and whichever one was missed would produce a silent loss."""
    js = _code()
    assert "setInterval(saveState, SAVE_INTERVAL)" in js
    assert "window.addEventListener('beforeunload', saveState)" in js


def test_the_state_is_restored_when_the_page_loads():
    """Anchored on the bootstrap block, not on the first DOMContentLoaded.

    There are two in the file: the first is the guard for whoever opens index.html
    straight from disk. Anchoring on the first found the wrong block.

    The anchor has changed once already -- it was the language selector, which the
    i18n slice took out of the bootstrap. Now it is the call that translates the
    page, which is the first thing the bootstrap does."""
    js = _code()
    excerpt = js[js.index("    applyLanguage();") :]
    excerpt = excerpt[: excerpt.index("});")]
    assert "restoreState()" in excerpt
    assert "startPersistence()" in excerpt


def test_the_stored_key_carries_a_version():
    """A state from another version has to be recognised, not interpreted."""
    js = _code()
    assert "ross_interface_state_v1" in js
    # Without the name of the local variable: it has changed once already (it was
    # called `state` and shadowed the shared state), and the guard broke because
    # of that, not because of a defect.
    assert ".version !== STATE_VERSION" in js


def test_a_refused_state_is_kept_instead_of_dropped():
    """An unreadable state is the user's work: deleting it without trace would lose it."""
    js = _code()
    excerpt = js[js.index("function refuseState(") :]
    excerpt = excerpt[: excerpt.index("function restoreState(")]
    assert "setItem(REFUSED_KEY" in excerpt
    assert "removeItem(STATE_KEY)" in excerpt


def test_the_figures_are_stripped_by_a_named_function():
    """Not storing the figure was a decision, not an oversight -- and the name says so."""
    js = _code()
    assert "function analysisWithoutChart(" in js
    assert "function rotorWithoutCharts(" in js
    # e o strip alcanca os dois rotores de um MultiRotor
    excerpt = js[js.index("function rotorWithoutCharts(") :]
    excerpt = excerpt[: excerpt.index("function stateToDisk(")]
    assert "driving_rotor" in excerpt and "driven_rotor" in excerpt


def test_a_card_without_a_chart_invites_a_recalculation():
    """An empty box looks like a defect; the invitation says it was a choice.

    The charts are deliberately not stored -- yesterday's chart, from an earlier
    version of the rotor, would look current. But the restored card called Plotly
    with `data: []`, and the result was a blank rectangle with no explanation at
    all. Now the chart's space says what happened and offers the button."""
    js = _code()
    assert "function inviteToRecompute(" in js
    assert js.count("inviteToRecompute(divNode") == 2  # the two restores
    assert "chartNotKept" in js and "recalculate" in js

    for language in ("chartNotKept", "recalculate"):
        assert js.count("%s:" % language) == 2, (
            "%s missing in one of the languages" % language
        )


def test_the_hub_exports_the_rotors_own_analyses():
    """The empty list was the remains of a decision that stopped making sense.

    In Phase 1 slice 4, exporting from the Hub sent `[]` on purpose: the only
    source of analyses was the DOM, holding the cards of the **open** project --
    which belong to another rotor and would come out with nodes this one does not
    even have. When the state left the DOM (Phase 3 slice 1), each rotor started
    carrying its own in savedAnalyses and the reason evaporated. The empty list
    stayed."""
    excerpt = _code()
    excerpt = excerpt[excerpt.index("function generatePythonFromHub(") :]
    excerpt = excerpt[: excerpt.index("function switchScreen(")]
    assert "generatePythonFile(rotorLibrary[index], [])" not in excerpt
    assert "savedAnalyses" in excerpt


def test_the_conversion_comes_from_the_analyses_not_from_the_selector():
    """A script carries one rotor, and a rotor has one conversion.

    Before, `conversion_type` came from the screen selector -- the same value for
    every card, including the ones computed under another conversion. The script
    came out with numbers that did not match the charts, with nothing to warn."""
    js = _code()
    assert "function unanimousConversion(" in js
    assert (
        "conversionNode.value"
        not in js[js.index("async function generatePythonFile(") :]
    )
    assert js.count("mixedConversions:") == 2  # os dois idiomas


def test_the_export_follows_the_settings_not_the_drawing():
    """A restored card lost its figure and kept its parameters.

    Filtering the export by "has a figure" would leave out precisely the analysis
    the user has just seen reappear on screen."""
    excerpt = _code()
    excerpt = excerpt[excerpt.index("function collectActiveAnalyses(") :]
    excerpt = excerpt[: excerpt.index("function unanimousConversion(")]
    assert "record.params" in excerpt
    assert "record.figure" not in excerpt


def _body(name, end):
    """The body of a frontend function, without the whole-line comments."""
    js = _code()
    start = js.index(name)
    return js[start : js.index(end, start)]


def test_the_recalculation_uses_the_cards_own_rotor_model():
    """The screen selector chooses the model of the card about to be born, and no more.

    After a page reload the selector goes back to its default, and `runCardAnalysis`
    re-read it on every computation: three cards of the same rotor -- 6 DoF, 4 DoF
    and torsional -- all recomputed as 6 DoF. It was the same selector that had
    already produced the export defect: the third place reading from the screen a
    value that belongs to the analysis.

    The behaviour is in `tests/js/test_conversion.js`; here is the structural
    property."""
    # The end has to be code: `_code()` deletes the comments, and a guard in this
    # suite has already anchored on a comment the filter ate.
    body = _body("async function runCardAnalysis(", "function isModeShape(")
    assert "rotor-conversion-type" not in body, (
        "the computation still reads the screen selector"
    )
    assert "cardConversion(uniqueId)" in body


def test_only_the_creation_of_a_card_reads_the_selector():
    """Two reads of the selector, and both about a card that does not exist yet.

    One is `addAnalysis`, which stamps the model on the new card; the other is the
    `cardConversion` fallback, for an id with no record. A third read would be the
    defect coming back, by another path."""
    found_items = [
        "%s:%d" % (mod, number)
        for mod, number, line in _code_lines()
        if "rotor-conversion-type" in line
    ]
    assert len(found_items) == 2, "selector reads in %s" % found_items


def test_every_rotor_model_carries_a_badge():
    """With no badge, "6 DoF" and "badge lost along the way" were the same screen.

    The badge only existed in the HTML `addAnalysis` wrote, and only for 4 DoF and
    torsional. A card restored from memory or read from a file came back with no
    badge -- and there was no way to tell whether that was the full model or a
    piece of information the restoration had dropped."""
    js = _code()
    assert "function conversionBadge(" in js
    assert "const CONVERSION_BADGES = {" in js

    for model in ("'4dof'", "'torsional'"):
        assert (
            model
            in js[
                js.index("const CONVERSION_BADGES") : js.index(
                    "function conversionBadge("
                )
            ]
        )

    mapping = js[
        js.index("const CONVERSION_BADGES") : js.index("function conversionBadge(")
    ]
    for hint in ("conv6dof", "conv4dof", "convTorsional"):
        assert "'%s'" % hint in mapping, "%s is not in the badge map" % hint
        assert js.count("%s:" % hint) == 2, "%s missing in one of the languages" % hint


def test_the_badge_is_written_in_one_place_only():
    """Three paths build a card; a badge written by hand in one of them disappears.

    `addAnalysis`, restoring from memory and reading from a file build the same
    header. While the badge was literal HTML inside `addAnalysis`, the other two
    had no way of carrying a badge at all."""
    js = _code()
    assert js.count("conversionBadge(") == 4  # the definition and the three headers
    assert "Model reduced to 4 Degrees of Freedom" not in js

    # The rotor model badge comes from one place only. The third `badge-conversion`
    # in the file is the MultiRotor badge, another subject and another colour.
    body = _body("function conversionBadge(", "function cardConversion(")
    assert (
        body.count('class="badge-conversion') == 2
    )  # o ramo conhecido e o desconhecido
    assert js.count('class="badge-conversion') == 3
    assert "MultiRotor" in js[js.index('style="background:#8b5cf6;"') :][:120]
