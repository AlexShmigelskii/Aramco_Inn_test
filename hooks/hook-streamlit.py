from PyInstaller.utils.hooks import collect_all

hiddenimports = []
binaries = []
datas = []

data, binary, hiddenimport = collect_all('streamlit')

hiddenimports += hiddenimport
binaries += binary
datas += data

data, binary, hiddenimport = collect_all('etna')

hiddenimports += hiddenimport
binaries += binary
datas += data

data, binary, hiddenimport = collect_all('pyarrow')

hiddenimports += hiddenimport
binaries += binary
datas += data

data, binary, hiddenimport = collect_all('pandas')

hiddenimports += hiddenimport
binaries += binary
datas += data

