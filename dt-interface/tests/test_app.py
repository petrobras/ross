# -*- coding: utf-8 -*-
"""The shell answers the page, and closed routes stay closed."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import SESSION_TOKEN, create_app


def test_index_is_served_without_a_token():
    client = create_app().test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"Digital Twin" in response.data
    assert SESSION_TOKEN.encode("ascii") in response.data


def test_health_requires_the_session_token():
    client = create_app().test_client()
    refused = client.get("/api/health")
    assert refused.status_code == 403

    accepted = client.get("/api/health", headers={"X-ROSS-Token": SESSION_TOKEN})
    assert accepted.status_code == 200
    assert accepted.get_json()["application"] == "ross-digital-twin"
