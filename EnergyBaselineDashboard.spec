# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import copy_metadata
import streamlit
import os

streamlit_dir = os.path.dirname(streamlit.__file__)

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('seu_mapping.csv', '.'),
        (streamlit_dir, 'streamlit'),
    ] + copy_metadata('streamlit'),
    hiddenimports=[
        'streamlit',
        'streamlit.version',
        'importlib.metadata',
        'importlib_metadata',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

# Exclude unnecessary modules to reduce size
a.binaries = TOC([x for x in a.binaries if not x[0].startswith('tcl') and not x[0].startswith('tk')])

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='EnergyBaselineDashboard',
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
