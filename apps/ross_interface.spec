# PyInstaller spec for the ROSS Graphical Interface (Flask + static frontend).
#
# Requires Python 3.10+ (matches ROSS optional dependency stack on modern installs).
# Build from repo root:
#   pip install -e . && pip install -r apps/requirements.txt pyinstaller
#   pyinstaller apps/ross_interface.spec
#
# Output: dist/ROSS-Interface/ (folder bundle — distribute as ZIP for end users)

import os

from PyInstaller.utils.hooks import collect_all

block_cipher = None
SPEC_ROOT = os.path.dirname(os.path.abspath(SPEC))

datas = [(os.path.join(SPEC_ROOT, "frontend"), "frontend")]

plotly_datas, plotly_binaries, plotly_hiddenimports = collect_all("plotly")
datas += plotly_datas

# Dev-only / heavy optional stacks — avoid pulling pytest hooks into the binary
excludes = [
    "pytest",
    "_pytest",
    "pytest_cov",
    "sphinx",
    "IPython",
    "jupyter",
    "notebook",
    "ipykernel",
    "matplotlib.tests",
    "numpy.tests",
    "scipy.tests",
]

hiddenimports = plotly_hiddenimports + [
    "flask",
    "flask_cors",
    "werkzeug",
    "pint",
    "toml",
    "numba",
    "ccp_performance",
    "control",
    "ross",
]

a = Analysis(
    [os.path.join(SPEC_ROOT, "app.py")],
    pathex=[SPEC_ROOT],
    binaries=plotly_binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ROSS-Interface",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_selector=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="ROSS-Interface",
)
