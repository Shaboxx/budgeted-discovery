# Third-party notices

The repository distributes original Python code, synthetic data, and preserved notices. Dependencies are installed from their upstream distributions; this repository does not bundle their binaries. Each dependency's own license and bundled notices control its use.

Full preserved texts and copyright notices are under [third_party/notices](third_party/notices). Exact portable runtime/test versions are in [requirements.txt](requirements.txt). The runtime includes NumPy, SciPy, pandas, NetworkX, BM25S, Gymnasium, River, PyYAML, PyArrow, DuckDB, psutil, and their recorded dependencies. The test environment includes pytest and Hypothesis. These dependencies include BSD, MIT, Apache-2.0, and other permissive or file-level license terms; consult the full texts instead of treating this inventory as a replacement license.

River's BSD notice, PyArrow's Apache license and NOTICE, the SciPy bundled-library licenses, and all selected dependency copyright notices are retained. NumPy's wheel contains additional library notices; its installed distribution remains the authoritative source for binary redistribution. Colorama is used only on Windows by pytest and is covered by its upstream BSD license distributed in the package.

The basic NOL reference adaptation retains Tim LaRock's 2019 MIT notice in [its source](src/frontier_bench/nol_reference.py) and [the separate notice](third_party/notices/nol-source/LICENSE). The TADC-SBM Apache-2.0 license and Google Research simulator attribution are retained under `third_party/notices/tadc-sbm-source/` because the optional generation module interfaces with that software.

The optional generator needs native graph-tool and additional generation packages, whose licenses and binary/source obligations must be reviewed when distributing such an environment. They are not bundled or installed by this release's portable quick start. No research papers, proprietary datasets, or third-party media are included.

These notices do not select a license for this project's original code. The first-party license decision is pending.
