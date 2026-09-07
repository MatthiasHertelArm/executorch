# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Host tests for the PDSC release-history generator.

Verify that only published releases carrying both a .pdsc and a
PyTorch.ExecuTorch pack asset become <release> entries, that entries are
ordered newest-first (semver, prereleases before their release), and that
the rendered XML carries version/date/tag/url as the PDSC expects.

"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import generate_release_history as grh  # type: ignore[import-not-found]  # noqa: E402


def _release(tag, published, asset_names, draft=False):
    return {
        "tag_name": tag,
        "published_at": published,
        "draft": draft,
        "assets": [
            {
                "name": name,
                "browser_download_url": (
                    f"https://github.com/pytorch/executorch/releases/download/{tag}/{name}"
                ),
            }
            for name in asset_names
        ],
    }


_RELEASES = [
    # Wheel-only release: no pack assets -> excluded.
    _release("v1.3.0", "2026-05-01T10:00:00Z", ["executorch-1.3.0.whl"]),
    # Pack without pdsc -> excluded (broken upload).
    _release("v1.3.1", "2026-06-01T10:00:00Z", ["PyTorch.ExecuTorch.1.3.1.pack"]),
    # Draft with both assets -> excluded.
    _release(
        "v1.5.0",
        "2026-09-01T10:00:00Z",
        ["PyTorch.ExecuTorch.pdsc", "PyTorch.ExecuTorch.1.5.0.pack"],
        draft=True,
    ),
    # Valid releases, deliberately unordered.
    _release(
        "v1.4.0",
        "2026-08-07T19:06:34Z",
        ["PyTorch.ExecuTorch.pdsc", "PyTorch.ExecuTorch.1.4.0.pack"],
    ),
    _release(
        "v1.4.1",
        "2026-08-20T09:00:00Z",
        ["PyTorch.ExecuTorch.pdsc", "PyTorch.ExecuTorch.1.4.1.pack"],
    ),
    _release(
        "v1.4.0-rc2",
        "2026-07-30T09:00:00Z",
        ["PyTorch.ExecuTorch.pdsc", "PyTorch.ExecuTorch.1.4.0-rc2.pack"],
    ),
]


def test_filters_to_pack_releases():
    entries = grh.pack_releases(_RELEASES, exclude_version=None)
    assert [e["version"] for e in entries] == ["1.4.1", "1.4.0", "1.4.0-rc2"]


def test_entry_fields_come_from_release_and_asset():
    entry = grh.pack_releases(_RELEASES, exclude_version=None)[1]
    assert entry == {
        "version": "1.4.0",
        "date": "2026-08-07",
        "tag": "v1.4.0",
        "url": "https://github.com/pytorch/executorch/releases/download/"
        "v1.4.0/PyTorch.ExecuTorch.1.4.0.pack",
    }


def test_exclude_version_drops_the_version_being_built():
    entries = grh.pack_releases(_RELEASES, exclude_version="1.4.1")
    assert [e["version"] for e in entries] == ["1.4.0", "1.4.0-rc2"]


def test_prerelease_sorts_before_its_release():
    key = grh._version_sort_key
    assert key("1.4.0-rc2") < key("1.4.0")
    assert key("1.4.0-rc2") > key("1.4.0-rc1")
    assert key("1.4.0-rc10") > key("1.4.0-rc9")  # numeric, not lexicographic
    assert key("1.4.0") < key("1.4.1")


def test_rendered_xml_shape():
    xml = grh.render_history(grh.pack_releases(_RELEASES, exclude_version=None))
    assert (
        '<release version="1.4.0" date="2026-08-07" tag="v1.4.0" '
        'url="https://github.com/pytorch/executorch/releases/download/'
        'v1.4.0/PyTorch.ExecuTorch.1.4.0.pack">' in xml
    )
    assert xml.count("<release ") == xml.count("</release>") == 3
    # Newest first, as the PDSC schema expects.
    assert xml.index('version="1.4.1"') < xml.index('version="1.4.0"')


def test_empty_input_renders_empty():
    assert grh.render_history(grh.pack_releases([], None)) == ""


def test_published_two_release_world():
    """Mirror the live repository state after the 1.4.1 release: both 1.4.0
    and 1.4.1 carry pack assets, and a 1.5.0 build's history must list them
    newest-first while excluding the version being built.
    """
    live = [
        _release(
            "v1.4.0",
            "2026-08-07T19:06:34Z",
            ["PyTorch.ExecuTorch.pdsc", "PyTorch.ExecuTorch.1.4.0.pack"],
        ),
        _release(
            "v1.4.1",
            "2026-08-18T16:37:18Z",
            ["PyTorch.ExecuTorch.pdsc", "PyTorch.ExecuTorch.1.4.1.pack"],
        ),
    ]
    entries = grh.pack_releases(live, exclude_version="1.5.0")
    assert [e["version"] for e in entries] == ["1.4.1", "1.4.0"]
    assert entries[0]["date"] == "2026-08-18"
    assert entries[0]["tag"] == "v1.4.1"
