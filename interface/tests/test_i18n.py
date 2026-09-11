# -*- coding: utf-8 -*-
"""The whole interface speaks both languages.

Until this point only the element form was translated -- its labels come from
the backend -- plus 18 interface texts. The rest was fixed English: **80
visible texts and 13 attributes** in `index.html`, ~45 sentences written by the
JS and 1240 words of help. And the language selector did not exist:
`preferredLanguage()` read a `localStorage` value nothing in the interface
could write. Dead configuration deciding what the user saw -- that is how
Leonardo came to see the interface in Portuguese without having chosen it.

What the slice did, and what these tests hold:

* `index.html` marked with `data-i18n`, `data-i18n-title` and
  `data-i18n-placeholder`; `applyLanguage()` walks it and substitutes;
* the ~45 sentences of the JS went through `t()`;
* the 17 help entries gained a Portuguese version;
* the twelve analysis titles left the HTML and the `typeNames` and come from the
  catalogue, already translated;
* the selector is on all three top bars."""

import io
import json
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import repository
from frontend_source import FRONTEND, source, code_lines

LANGUAGES = ("en", "pt")


def _index():
    with io.open(
        os.path.join(FRONTEND, "index.html"), encoding="utf-8", newline=""
    ) as handle:
        return handle.read()


def _without_script(html):
    return re.sub(r"<script[\s\S]*?</script>|<!--[\s\S]*?-->", "", html)


def _dictionary():
    """The keys of each language, read from the `UI_TEXT` of `core/i18n.js`."""
    with io.open(
        os.path.join(FRONTEND, "core", "i18n.js"), encoding="utf-8", newline=""
    ) as handle:
        js = handle.read()
    block = js[js.index("const UI_TEXT = {") : js.index("\n};")]
    output = {}
    for language in LANGUAGES:
        body = block[block.index("    %s: {" % language) :]
        body = body[: body.index("\n    },")]
        output[language] = set(re.findall(r"^        (\w+):", body, re.M))
    return output


# --- the HTML is marked -------------------------------------------------------
#
# The only words of the interface that are NOT translated: each language's name
# appears written in that language, otherwise someone who only reads Portuguese
# looks for "Portuguese" in a list that says "Português". They are exempted in
# writing, with the option value alongside -- replacing the label with "Inglês"
# breaks the test.
ENDONYMS = {"en": "English", "pt": "Português"}


def test_the_language_names_are_written_in_their_own_language():
    """Controle da isencao acima: se os rotulos mudassem, ninguem notaria."""
    html = _index()
    for value, name in ENDONYMS.items():
        expected = '<option value="%s">%s</option>' % (value, name)
        assert html.count(expected) == 3, "faltou %r nas tres barras" % expected


def test_every_visible_text_in_the_page_is_marked():
    """A text with no `data-i18n` stays in English forever, and silently."""
    without_key = _without_script(_index())
    missing = []
    for found in re.finditer(r"(<[a-z0-9]+[^<>]*>)\s*([^<>]+?)\s*<", without_key):
        text = found.group(2).strip()
        if not text or not re.search(r"[A-Za-z]{3}", text):
            continue
        opening = found.group(1)
        if (
            re.match(r'<option value="(en|pt)">$', opening)
            and text in ENDONYMS.values()
        ):
            continue
        if "data-i18n" not in opening:
            missing.append(text)
    assert missing == [], "no data-i18n: %s" % missing


def test_every_title_and_placeholder_is_marked():
    without_key = _without_script(_index())
    missing = []
    for found in re.finditer(r'(placeholder|title)="([^"]+)"([^<>]*)', without_key):
        attribute, value, rest = found.groups()
        if "data-i18n-%s=" % attribute not in rest:
            # may come before the attribute, on the same element
            element = without_key[max(0, found.start() - 400) : found.end() + 200]
            if "data-i18n-%s=" % attribute not in element:
                missing.append("%s=%r" % (attribute, value))
    assert missing == [], "attribute with no key: %s" % missing


@pytest.mark.parametrize(
    "attribute", ["data-i18n", "data-i18n-title", "data-i18n-placeholder"]
)
def test_every_key_used_in_the_page_exists_in_both_languages(attribute):
    """A wrong key in the HTML gives back the key itself on screen."""
    dictionary = _dictionary()
    used = set(re.findall(r'%s="([^"]+)"' % attribute, _index()))
    assert used, "nenhum %s no index -- a varredura parou de funcionar" % attribute
    for language in LANGUAGES:
        missing = sorted(used - dictionary[language])
        assert missing == [], "%s with no translation in %s: %s" % (
            attribute,
            language,
            missing,
        )


# --- o dicionario ------------------------------------------------------------


def test_both_languages_have_the_same_keys():
    """A key in only one language falls back to English with nobody noticing."""
    dictionary = _dictionary()
    assert dictionary["en"] == dictionary["pt"], "only in en: %s | only in pt: %s" % (
        sorted(dictionary["en"] - dictionary["pt"]),
        sorted(dictionary["pt"] - dictionary["en"]),
    )


def test_the_badge_tooltip_is_read_through_the_seal_table():
    """The dictionary's only indirection, written down here so the test below can
    count on it. The three conversion tooltips do not appear as `t('conv6dof')`:
    the key lives in `CONVERSION_BADGES` and reaches `t()` through the property."""
    with io.open(
        os.path.join(FRONTEND, "core", "analysis_store.js"),
        encoding="utf-8",
        newline="",
    ) as handle:
        js = handle.read()
    assert "t(badge.tooltip)" in js, "the badge indirection changed shape"
    assert sorted(re.findall(r"tooltip: '(\w+)'", js)) == [
        "conv4dof",
        "conv6dof",
        "convTorsional",
    ]


def _keys_by_indirection():
    """The keys that reach `t()` through a property, not through a literal."""
    with io.open(
        os.path.join(FRONTEND, "core", "analysis_store.js"),
        encoding="utf-8",
        newline="",
    ) as handle:
        return set(re.findall(r"tooltip: '(\w+)'", handle.read()))


def test_no_key_in_the_dictionary_is_unused():
    """Text nobody shows is dead weight -- and a translation to maintain for nothing."""
    dictionary = _dictionary()
    js = source() + _index()
    indirect = _keys_by_indirection()
    orphans = []
    for key in sorted(dictionary["en"]):
        used_keys = (
            "t('%s')" % key in js
            or 't("%s")' % key in js
            or '"%s"' % key in _index()
            or key in indirect
        )
        if not used_keys:
            orphans.append(key)
    assert orphans == [], "unused key: %s" % orphans


# --- the HTML the JS builds ---------------------------------------------------

# `help.js` and `i18n.js` ARE the dictionaries: by definition they have both
# languages written out. The other guards in this suite are what look after them.
DICTIONARIES = ("components/help.js", "core/i18n.js")

# Terms that stay the same in both languages. Each needs a reason:
NOT_TRANSLATED = {
    "MultiRotor",  # feature name, written the same way in Portuguese
    "Default (Steel)",  # the option VALUE (goes to the backend), not the label
    "BASIC",
    "LIST",  # model names, coming from the ROSS schema
}


# `[^=<!-]>` because `=>` and `-->` also end in `>`: without it the sweep
# accused `b.classList.remove('active` as screen text. The `\x00` marks where
# an interpolation was: it cuts the sentence, which is what we want, and does
# not match as a letter.
_PATTERNS = (
    re.compile(
        r"(?:^|[^=<!-])>\s*((?:<i [^<>]*>\s*</i>\s*)?[A-Za-z][A-Za-z ,.:()'/-]{2,}?)\s*[<\x00]"
    ),
    re.compile(r'(?:title|placeholder)="([^"+\'\x00]*[A-Za-z]{3}[^"+\'\x00]*)'),
)


def _visible_literals():
    """(module, number, text) of fixed English text in the HTML the JS builds.

    This guard was born late: `index.html` was marked and the dictionary had both
    languages, and the analysis screen still wrote "Generated dashboards will
    appear here." in English -- the key existed in the dictionary and nobody
    called it. Marking the static HTML does not cover the HTML the JS writes, and
    that was half the screen."""
    visible, attribute = _PATTERNS
    for module, number, line in code_lines():
        if module in DICTIONARIES:
            continue
        # The interpolated part is removed before the sweep, not the whole line.
        # While the line with `${` was ignored, three English texts hid behind an
        # interpolation: the modeling tab title with its raw key,
        # `title="Help about ${category}"` and the `Driving:` of the multirotor
        # selector -- and the modeling tab title stayed in English while in
        # Portuguese, with the guard green.
        cleaned = re.sub(r"\$\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", "\x00", line)
        for found in visible.finditer(cleaned):
            text = re.sub(r"<i [^<>]*>\s*</i>", "", found.group(1)).strip()
            if text and text not in NOT_TRANSLATED:
                yield module, number, text
        for found in attribute.finditer(cleaned):
            value = found.group(1).strip()
            if value and value not in NOT_TRANSLATED:
                yield module, number, value


def test_the_html_built_by_the_javascript_has_no_english_literal():
    found_items = ["%s:%d %r" % item for item in _visible_literals()]
    assert found_items == [], "fixed text in the JS-built HTML: %s" % found_items


def _sweep(line):
    """The guard's sweep, applied to one line, for the control below."""
    visible, attribute = _PATTERNS
    cleaned = re.sub(r"\$\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", "\x00", line)
    return [a.group(1).strip() for a in visible.finditer(cleaned)] + [
        a.group(1).strip() for a in attribute.finditer(cleaned)
    ]


# What the guard MUST accuse. The last three came from the defect Leonardo
# found on screen: while the line with `${` was discarded whole, the modeling
# tab title showed the raw key and stayed in English even in Portuguese -- and
# this guard was green.
MUST_FLAG = [
    '<button onclick="x()">Go to Modeling</button>',
    '`<i class="fas fa-save"></i> Save JSON</button>`',
    '<button title="Delete"><i class="fas fa-trash"></i>',
    '<button class="b" onclick="ajuda(\'${category}\')" title="Help about ${category}">',
    '<option value="driving" ${sel}>Driving: ${escapeHtml(name)}</option>',
    "`<h4>Select Model</h4>`",
]

# What it must NOT accuse: a dictionary call, an option value, an expression.
MUST_NOT_FLAG = [
    "title=\"${escapeHtml(t('delete'))}\"",
    "<span>${escapeHtml(name)}</span>",
    "sel.innerHTML = `<option value=\"Default (Steel)\">${escapeHtml(t('defaultSteel'))}</option>`;",
    "document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));",
    "placeholder=\"' + escapeHtml(dica) + '\"",
]


def test_the_scan_for_literals_actually_sees_them():
    """Control: the sweep is a regular expression over loose lines. If it stopped
    matching, the test would pass with the whole screen in English."""
    blind_spots = [line for line in MUST_FLAG if not _sweep(line)]
    assert blind_spots == [], "the sweep does not see: %s" % blind_spots

    false_hits = [
        (line, _sweep(line))
        for line in MUST_NOT_FLAG
        if [t for t in _sweep(line) if t not in NOT_TRANSLATED]
    ]
    assert false_hits == [], "the sweep accuses what it should not: %s" % false_hits


# --- the JS writes no English sentence ----------------------------------------

DIALOGS = ("openCustomAlert", "openCustomConfirm", "openCustomPrompt")


@pytest.mark.parametrize("dialog", DIALOGS)
def test_no_dialog_is_opened_with_an_english_literal(dialog):
    """A dialog's message comes from the dictionary, never written in the call.

    It was the easiest way to reintroduce fixed English: an `openCustomAlert` with
    the sentence in quotes passes every other test."""
    found_items = []
    for module, number, line in code_lines():
        for found in re.finditer(r"%s\(\s*(['\"])([^'\"]{4,})\1" % dialog, line):
            found_items.append("%s:%d %r" % (module, number, found.group(2)[:50]))
    assert found_items == [], "literal em dialogo: %s" % found_items


# --- a ajuda -----------------------------------------------------------------


def test_the_help_has_every_entry_in_both_languages():
    with io.open(
        os.path.join(FRONTEND, "components", "help.js"), encoding="utf-8", newline=""
    ) as handle:
        js = handle.read()
    block = js[js.index("const HelpContent = {") : js.index("\n};")]
    entries = re.findall(r"^    (\w+): \{", block, re.M)
    assert len(entries) == 17, entries

    for helpEntry in entries:
        excerpt = block[block.index("    %s: {" % helpEntry) :]
        end = re.search(r"\n    \w+: \{", excerpt)
        excerpt = excerpt[: end.start()] if end else excerpt
        for field in ("title", "body"):
            for language in LANGUAGES:
                assert re.search(r"%s: \{[\s\S]*?%s: " % (field, language), excerpt), (
                    "%s: falta %s em %s" % (helpEntry, field, language)
                )


def test_the_help_text_is_not_english_in_portuguese():
    """Controle: se o `pt` fosse copia do `en`, os testes acima passariam."""
    with io.open(
        os.path.join(FRONTEND, "components", "help.js"), encoding="utf-8", newline=""
    ) as handle:
        js = handle.read()
    assert "Parâmetros principais" in js
    assert "Key Parameters" in js
    assert js.count("Parâmetros principais") >= 10


# --- the selector -------------------------------------------------------------


def test_the_language_selector_is_on_every_screen():
    """Changing the language only on the Hub would force leaving the screen to change it."""
    html = _index()
    assert html.count('class="ui-language"') == 3
    assert html.count('onchange="changeLanguage(this.value)"') == 3
    for language in LANGUAGES:
        assert 'value="%s"' % language in html


def test_changing_the_language_translates_the_page_before_returning_early():
    """The order has bitten before: the translation comes before the early return.

    `changeLanguage` returns early when there is no element tab open. With
    `applyLanguage()` after that `return`, changing the language on the Hub loaded
    the new schema and left the screen in English."""
    js = source()
    body = js[js.index("export async function changeLanguage(") :]
    body = body[: body.index("\nexport ", 1)] if "\nexport " in body[1:] else body
    assert body.index("applyLanguage()") < body.index("if (!state.currentTab) return")


def test_no_element_rewritten_by_the_javascript_is_marked():
    """`data-i18n` on an element the JS rewrites swaps the content for the label.

    `#tab-title` (the big title above the element list) was like that. What writes
    it is `openTab`, with the category name and the help button; `applyLanguage()`
    wrote over all of it and left the bare word "Category" in place of the name.
    Changing tabs made `openTab` rewrite it and the title came back -- which made
    the defect look intermittent."""
    rewritten = set()
    for _, _, line in code_lines():
        for found in re.finditer(
            r"getElementById\(\s*['\"]([\w-]+)['\"]\s*\)\.innerHTML\s*=", line
        ):
            rewritten.add(found.group(1))
    assert rewritten, "the sweep found no innerHTML by id"

    html = _index()
    marked = []
    for identifier in sorted(rewritten):
        element = re.search(
            r"<[a-z0-9]+[^<>]*id=\"%s\"[^<>]*>" % re.escape(identifier), html
        )
        if element and "data-i18n" in element.group(0):
            marked.append(identifier)
    assert marked == [], "the JS rewrites it, but it is marked: %s" % marked


def test_no_branch_decides_by_comparing_visible_text():
    """Translating a text breaks whoever uses it as a marker -- with no error at all.

    Two `if`s read the card list looking for "Generated dashboards will appear".
    In Portuguese the `includes` stopped matching: the empty-list notice stayed on
    screen and the first card went in underneath it. Nothing threw.
    The marker is now the class `dashboards-empty`, which does not change language."""
    sentences = set()
    for language in LANGUAGES:
        with io.open(
            os.path.join(FRONTEND, "core", "i18n.js"), encoding="utf-8", newline=""
        ) as handle:
            js = handle.read()
        block = js[js.index("    %s: {" % language) :]
        block = block[: block.index("\n    },")]
        for value in re.findall(r"^        \w+: [\"'](.{12,}?)[\"'],$", block, re.M):
            sentences.add(value)
    assert len(sentences) > 50, len(sentences)  # controle: a leitura achou o dicionario

    found_items = []
    for module, number, line in code_lines():
        if module == "core/i18n.js":
            continue
        if ".includes(" not in line and "===" not in line:
            continue
        for sentence in sentences:
            for chunk in (sentence, sentence[:30]):
                if len(chunk) >= 12 and chunk in line:
                    found_items.append("%s:%d %r" % (module, number, chunk))
    assert found_items == [], "comparison against screen text: %s" % found_items


# --- what comes from the server ----------------------------------------------


def _analyses_route(language):
    pytest.importorskip("flask", reason="the route needs Flask installed")
    from api import create_app
    from api.security import SESSION_TOKEN

    application = create_app()
    application.config["TESTING"] = True
    client = application.test_client()
    response = client.get(
        "/api/schema/analyses?lang=%s" % language,
        headers={"X-ROSS-Token": SESSION_TOKEN},
    )
    assert response.status_code == 200, response.status_code
    return json.loads(response.data.decode("utf-8"))


def test_the_schema_route_answers_in_the_language_asked():
    """Half the screen's texts are not in the JS dictionary: field labels, analysis
    titles and the incompatibility reasons come from here. Translating only the
    frontend would leave the screen half and half -- which is what Leonardo saw."""
    english, portuguese = _analyses_route("en"), _analyses_route("pt")

    assert english["titles"]["modes"] == "Modal Analysis"
    assert portuguese["titles"]["modes"] != english["titles"]["modes"], (
        "the title did not change language"
    )

    identical = [
        name
        for name, title in english["titles"].items()
        if portuguese["titles"][name] == title
    ]
    assert identical == [], "titulos identicos nos dois idiomas: %s" % identical

    labels_en = [field["label"] for field in english["fields"]["campbell"]]
    labels_pt = [field["label"] for field in portuguese["fields"]["campbell"]]
    assert labels_en != labels_pt, "the field labels did not change language"

    reason_en = english["unsupported"]["unbalance"]["torsional"]["reason"]
    reason_pt = portuguese["unsupported"]["unbalance"]["torsional"]["reason"]
    assert reason_pt != reason_en, "the refusal reason did not change language"


def test_an_unknown_language_falls_back_to_english():
    """A language that does not exist must not give back an empty field on screen."""
    unknown = _analyses_route("de")
    assert unknown["titles"] == _analyses_route("en")["titles"]


def test_the_analysis_titles_come_from_the_catalogue():
    """The twelve names were in the HTML and again in a `typeNames` in the JS."""
    js = source()
    assert "typeNames" not in js
    assert "analysisTitle(" in js
    assert "fillAnalysisTypes" in js

    from domain.analysis_catalog import TITLES

    for names in TITLES.values():
        for language in LANGUAGES:
            assert names.get(language), names


# --- A function name must not leak into what the user reads -------------------
#
# Translating the code into English renamed identifiers with a table and
# `\b...\b`. One of the entries was `mostrar -> showHelpEntry` -- a function
# name on one side, a common Portuguese word on the other. The substitution
# reached the Portuguese help content and the screen started saying "para
# esconder ou showHelpEntry o conteudo". No test saw it: the code stayed
# coherent, both halves of the dictionary went on existing, and the text only
# appears on screen.
#
# The mark of the defect is the shape of the word, not the name: prose has no
# camelCase.

CAMEL_CASE = re.compile(r"\b[a-z]+[A-Z][A-Za-z]*\b")


def _without_markup(cut):
    """Strips tags and entities: the help entries are HTML, and `data-i18n` and
    `fa-chevron-down` are not text anyone reads."""
    cut = re.sub(r"<[^>]*>", " ", cut)
    return re.sub(r"&[a-z]+;", " ", cut)


def _block(text_value, opening, file_name):
    start_at = text_value.find(opening)
    assert start_at >= 0, "%s no longer has %r" % (file_name, opening)
    end_at = text_value.index("\n};", start_at)
    return text_value[start_at:end_at]


def _file(relative_path):
    with io.open(
        os.path.join(FRONTEND, relative_path), encoding="utf-8", newline=""
    ) as handle:
        return handle.read()


def _screen_texts():
    """Every text the interface writes on screen that comes from a dictionary."""
    help_text = _block(_file("components/help.js"), "const HelpContent = {", "help.js")
    languages = _block(_file("core/i18n.js"), "const UI_TEXT = {", "i18n.js")
    entries = [
        body for _, body in re.findall(r"([\"'`])((?:[^\\]|\\.)*?)\1", help_text, re.S)
    ]
    sentences = [
        body for _, body in re.findall(r":\s*([\"'`])(.*?)\1", languages, re.S)
    ]
    return entries + sentences


def test_the_camel_case_detector_catches_a_leaked_name():
    """Control: without this, a broken regex would leave the guard below green."""
    assert CAMEL_CASE.findall("para esconder ou showHelpEntry o conteudo")
    assert CAMEL_CASE.findall(_without_markup("<i class='fa-x'></i> ou saveState o"))
    assert not CAMEL_CASE.findall("para esconder ou mostrar o conteudo")
    assert not CAMEL_CASE.findall(_without_markup('<b data-i18n="helpTitle">Ok</b>'))


def test_no_function_name_leaked_into_what_the_user_reads():
    texts = _screen_texts()
    assert len(texts) > 200, "the sweep stopped finding text: %d" % len(texts)

    leaked = sorted(
        {
            key_name
            for text_value in texts
            for key_name in CAMEL_CASE.findall(_without_markup(text_value))
        }
    )
    assert leaked == [], "identifier name inside screen text: %s" % leaked


# --- the code speaks English; only the dictionaries speak Portuguese ----------
#
# Slice 2 of phase 4 translated the codebase and was declared finished. The ruff
# pass of slice 3 then walked the same files and found forty leftovers: eight
# `TEM_ROSS`, eight `reason="exige o ROSS instalado"`, a user-facing
# `"Erro inesperado (%s): %s"`, half-translated comments, and five references to
# test files the reorganisation had already renamed away.
#
# They survived because the sweep that declared the slice done was written by
# hand each time and never became a test. This is that test.

PORTUGUESE = frozenset(
    """
    nao que uma umas uns dos das pelo pela pelos pelas para por como sobre entre
    depois antes desde tambem apenas muito assim entao porque quando onde cada
    todos todas mesmo mesma este esta esse essa aquele aquela isso isto aquilo
    seu sua seus suas ser sao foi foram eram tem temos tinha havia estao
    exige exigem instalado instalada sumiu sumiram mudou mudaram gancho ganchos
    chamado chamados chamadas teste testes controle ajudantes compartilhados
    existe existem tamanho divergiram divergiu precisa precisam quebra quebrou
    arquivo arquivos linha linhas nome nomes campo campos valor valores
    guarda guardas erro erros falha falhas chave chaves vazio vazia
    declarados dentro ambiente configura acima abaixo meio aspa fechava
    resposta respostas desbalanceamento balanco harmonico introspeccao
    fase fatia rotas sessao ver velocidade frequencia mancal mancais
    disco discos eixo eixos analise analises grafico graficos tela usuario
    elemento elementos entrada saidas recusada versao dominio importa
    ponte pontes caso casos
    entrada saida largura altura profundidade numero primeiro primeira
    ultimo ultima antigo antiga
    """.split()
)
PORTUGUESE_ENDING = re.compile(r"[a-z]{3,}(?:cao|coes|mente|ando|endo|idade|agem)\b")
ACCENTED = re.compile("[À-ſ]")

# Where Portuguese is the content and not a leftover: the `pt` half of each
# catalogue, and this file, which has to spell the words out to look for them.
# Portuguese that is there on purpose: the three guards that quote the
# identifier they replaced. Keyed by the text
# that has to be present -- a line number rots on the next edit, and the control
# below fails if one of these stops being true.
DELIBERATE = (
    "`quando` until the translation",
    "`quando[A-Z]` pattern",
    "still using `eixo`",
)

TRANSLATED_CONTENT = (
    os.path.join("domain", "field_catalog.py"),
    os.path.join("domain", "analysis_catalog.py"),
    os.path.join("domain", "compatibility.py"),
    os.path.join("tests", "test_i18n.py"),
)


def _unaccented(text_value):
    import unicodedata

    broken_up = unicodedata.normalize("NFD", text_value)
    return "".join(c for c in broken_up if unicodedata.category(c) != "Mn")


def _portuguese_in(text_value):
    """The Portuguese words in a piece of text, or an empty set.

    Every word on the list above is one that English does not have, so a single
    hit is enough. The first version of this sweep needed two, and that is
    exactly how `TEM_ROSS` and `"divergiram em %r"` got through.
    """
    plain = _unaccented(text_value).lower()
    found = {w for w in re.findall(r"[a-z]+", plain) if w in PORTUGUESE}
    found |= {m.group(0) for m in PORTUGUESE_ENDING.finditer(plain)}
    if ACCENTED.search(text_value):
        found.add("<accent>")
    return found


def _our_python_files():
    found_files = []
    for folder, folders, names in os.walk(ROOT):
        folders[:] = repository.ours(folder, folders)
        for name in sorted(names):
            if not name.endswith(".py"):
                continue
            full = os.path.join(folder, name)
            relative = os.path.relpath(full, ROOT)
            if relative not in TRANSLATED_CONTENT:
                found_files.append((relative, full))
    return found_files


def test_the_portuguese_detector_catches_what_slipped_through():
    """Control: the four shapes that actually got past slice 2."""
    assert _portuguese_in("exige o ROSS instalado")
    assert _portuguese_in("divergiram em %r: js=%r python=%r")
    assert _portuguese_in("TEM ROSS")
    assert _portuguese_in("Ângulo do Pivô")
    assert _portuguese_in("derivado por introspeccao")
    assert not _portuguese_in("requires ROSS installed")
    assert not _portuguese_in("HAS ROSS")
    assert not _portuguese_in("the interface serves plotly.js from inside")


def test_every_deliberate_mention_is_still_there():
    """Control: an exception that stopped being needed drops off the list.

    Without this, `DELIBERATE` only grows, and each entry silences a line
    forever -- including, one day, a line that went back to being a leftover.
    """
    everything = []
    for _, full in _our_python_files():
        with io.open(full, encoding="utf-8") as handle:
            everything.append(handle.read())
    joined = "\n".join(everything)
    for mention in DELIBERATE:
        assert mention in joined, (
            "%r is no longer written anywhere: take it off DELIBERATE" % mention
        )


def test_no_portuguese_is_left_in_the_documentation():
    """The README is read before the code, and nothing was reading the README.

    The sweep below walks `.py` files, so the documentation was never measured.
    It carried two references to files the reorganisation had renamed away --
    `domain/compatibilidade.py` and `tests/test_fase4.py` -- and the detector
    found both on its first pass over `.md`. Same rot as the five stale
    references slice 3 found in the code, in the file a ROSS maintainer opens
    first.
    """
    leftovers = []
    for folder, folders, names in os.walk(ROOT):
        folders[:] = repository.ours(folder, folders)
        for name in sorted(n for n in names if n.endswith(".md")):
            full = os.path.join(folder, name)
            with io.open(full, encoding="utf-8") as handle:
                for number, line in enumerate(handle, 1):
                    words = _portuguese_in(line)
                    if words:
                        leftovers.append(
                            "%s:%d %s"
                            % (os.path.relpath(full, ROOT), number, sorted(words))
                        )
    assert leftovers == [], "Portuguese left in the documentation:\n  " + "\n  ".join(
        leftovers
    )


def test_no_portuguese_is_left_in_the_python_code():
    import ast
    import tokenize

    found_files = _our_python_files()
    assert len(found_files) > 50, "the sweep stopped finding files: %d" % len(
        found_files
    )

    leftovers = []
    for relative, full in found_files:
        with io.open(full, "rb") as handle:
            for token in tokenize.tokenize(handle.readline):
                if token.type in (tokenize.COMMENT, tokenize.STRING):
                    for offset, one_line in enumerate(token.string.split("\n")):
                        if any(mention in one_line for mention in DELIBERATE):
                            continue
                        words = _portuguese_in(one_line)
                        if words:
                            leftovers.append(
                                "%s:%d %s"
                                % (relative, token.start[0] + offset, sorted(words))
                            )
        with io.open(full, encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        for node in ast.walk(tree):
            written_name = None
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                written_name = node.name
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                written_name = node.id
            elif isinstance(node, ast.arg):
                written_name = node.arg
            if not written_name:
                continue
            words = _portuguese_in(" ".join(re.split(r"[_\W]+", written_name)))
            if words:
                leftovers.append(
                    "%s:%d name %r %s"
                    % (
                        relative,
                        getattr(node, "lineno", 0),
                        written_name,
                        sorted(words),
                    )
                )

    assert leftovers == [], "Portuguese left in the code:\n  " + "\n  ".join(
        sorted(set(leftovers))
    )
