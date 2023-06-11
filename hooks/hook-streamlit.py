from PyInstaller.utils.hooks import copy_metadata, collect_all, collect_submodules

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

# data, binary, hiddenimport = collect_all('etna.etna.pipeline')
#
# hiddenimports += hiddenimport
# binaries += binary
# datas += data
#
# data, binary, hiddenimport = collect_all('etna.etna.etna.datasets.tsdataset')
#
# hiddenimports += hiddenimport
# binaries += binary
# datas += data
#
# data, binary, hiddenimport = collect_all('etna.transforms')
#
# hiddenimports += hiddenimport
# binaries += binary
# datas += data
#
# data, binary, hiddenimport = collect_all('etna.models')
#
# hiddenimports += hiddenimport
# binaries += binary
# datas += data
#
# data, binary, hiddenimport = collect_all('etna.metrics')
#
# hiddenimports += hiddenimport
# binaries += binary
# datas += data
