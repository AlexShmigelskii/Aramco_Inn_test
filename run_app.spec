# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(
    ['run_app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ("data/monthly-australian-wine-sales.csv", "monthly-australian-wine-sales.csv"),
        ("venv/lib/python3.10/site-packages/streamlit/static", "/streamlit/static"),
        ("venv/lib/python3.10/site-packages/streamlit/runtime", "/streamlit/runtime"),
    ],
    hiddenimports=[],
    hookspath=['./hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='run_app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)