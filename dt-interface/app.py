# -*- coding: utf-8 -*-
"""Entry point of the Digital Twin interface.

Same shape as the ROSS graphical interface (`interface/app.py` in the
reference tree): this file only starts the server. Routes live in `api/`,
what the application does will live in `services/`, and what it knows will
live in `domain/`. The page is `frontend/`.

The Streamlit application in `a17404-digital-twin-app` stays the source of
behavior until each screen is brought across. Nothing here imports Streamlit.
"""

import sys

from api import HOST, PORT, create_app

app = create_app()


def main():
    print("ROSS Digital Twin interface: http://%s:%s/" % (HOST, PORT))
    app.run(host=HOST, port=PORT, use_reloader=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
