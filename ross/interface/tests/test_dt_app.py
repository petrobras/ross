# -*- coding: utf-8 -*-
"""The extension inside the ROSS interface: page, files, routes and token."""

from ross.interface.tests.dt_app import HEADERS, create_app

from ross.interface.extensions.digital_twin.extension import EXTENSION


def test_page_carries_the_screens_and_the_buttons():
    page = create_app().test_client().get("/").get_data(as_text=True)
    for screen in ("screen-dt", "screen-dt-analysis", "screen-balance", "screen-cfd"):
        assert 'id="%s"' % screen in page
        assert 'data-screen="%s"' % screen in page
    home = page.index('id="screen-home"')
    hub = page.index('id="screen-rotor-hub"')
    hub_list = page.index('id="rotor-hub-list"')
    for screen in ("screen-balance", "screen-cfd"):
        assert home < page.index('data-screen="%s"' % screen) < hub
    assert page.count('data-screen="screen-dt"') == 3
    dt_button = page.index('data-screen="screen-dt"')
    assert hub < dt_button < hub_list
    analysis = page.index('id="screen-analysis"')
    dt_button = page.index('data-screen="screen-dt"', analysis)
    assert dt_button < page.index('class="analysis-layout"', analysis)
    assert '<script type="module" src="ext/dt/main.js?v=' in page
    assert '<link rel="stylesheet" href="ext/dt/style.css?v=' in page
    assert "<!-- extensions:" not in page


def test_the_rotor_figure_leads_then_the_sensors_then_the_balancing_result():
    # The rotor figure is the top of the panel, the sensor inputs of step 4
    # follow it, and the balancing result they lead to comes last.
    page = create_app().test_client().get("/").get_data(as_text=True)
    screen = page.index('id="screen-dt"')
    assert (
        screen
        < page.index('id="dt-plot"', screen)
        < page.index('id="dt-sensors-card"', screen)
        < page.index('id="dt-balancing-card"', screen)
    )


def test_frontend_files_are_public():
    client = create_app().test_client()
    for name in ("main.js", "style.css", "features/balance.js"):
        response = client.get("/ext/dt/" + name)
        assert response.status_code == 200, name
        response.close()


def test_routes_require_the_session_token():
    client = create_app().test_client()
    assert client.get("/api/dt/equipment").status_code == 403
    assert client.post("/api/dt/session").status_code == 403
    assert client.post("/api/balance/calculate", json={}).status_code == 403
    assert client.post("/api/dt/session", headers=HEADERS).status_code == 200


def test_every_route_of_the_extension_is_registered():
    rules = {rule.rule for rule in create_app().url_map.iter_rules()}
    for blueprint in EXTENSION.blueprints:
        assert blueprint.name.startswith("dt_")
    assert {
        "/api/dt/equipment",
        "/api/dt/features",
        "/api/dt/session",
        "/api/dt/process-data",
        "/api/dt/model",
        "/api/dt/model-settings",
        "/api/dt/balancing",
        "/api/dt/balancing-settings",
        "/api/dt/balancing-figure",
        "/api/dt/balancing-parameters",
        "/api/dt/project",
        "/api/dt/analysis",
        "/api/dt/campbell/mode-shape",
        "/api/balance/calculate",
        "/api/balance/report",
        "/api/balance/save",
        "/api/balance/open",
    } <= rules
