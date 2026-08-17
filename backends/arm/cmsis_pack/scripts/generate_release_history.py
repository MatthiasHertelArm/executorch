#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Generate the PDSC <releases> history from published GitHub releases.

A CMSIS pack's PDSC must list every published pack version so tools can
offer updates and resolve older versions. Instead of hand-maintaining that
list, this script derives it from the GitHub releases of the repository:
every non-draft release that has both a .pdsc and a PyTorch.ExecuTorch
.pack asset attached becomes one <release> entry:

    <release version="1.4.0" date="2026-08-07" tag="v1.4.0"
             url="https://github.com/pytorch/executorch/releases/download/v1.4.0/PyTorch.ExecuTorch.1.4.0.pack">

The version comes from the pack asset's filename, the date from the
release publication time, the tag from the release's git tag, and the url
from the asset's download location. Entries are ordered newest-first as
the PDSC schema expects.

The output feeds the %{HISTORY}% placeholder of the PDSC template (via
generate_components.py --release-history). The entry for the version being
built is emitted separately by the template, so that version is excluded
here when it is already published (a rebuild of a released version).

Network failures degrade to an empty history with a warning so offline
pack builds keep working; CI can pass --strict to fail instead.

"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from datetime import datetime
from xml.sax.saxutils import escape, quoteattr

_PACK_ASSET_RE = re.compile(r"^PyTorch\.ExecuTorch\.(.+)\.pack$")

# CMSIS pack versions are semver: MAJOR.MINOR.PATCH with an optional
# -prerelease suffix. A prerelease sorts before its release (1.4.0-rc1 <
# 1.4.0); prerelease identifiers compare numerically when both are numeric.
_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:-(.+))?$")


def _version_sort_key(version: str) -> tuple:
    m = _VERSION_RE.match(version)
    if not m:
        # Unparseable versions sort last; they still appear in the history.
        return (-1, 0, 0, ())
    major, minor, patch, prerelease = m.groups()
    if prerelease is None:
        # Releases sort after any prerelease; same nesting depth as the
        # identifier keys below so tuple comparison stays homogeneous.
        pre_key: tuple = (((2, ""),),)
    else:
        # Natural ordering within each dot-separated identifier, so rc10 >
        # rc9 (digit runs compare numerically, the rest as text).
        pre_key = tuple(
            tuple(
                (0, int(run)) if run.isdigit() else (1, run)
                for run in re.findall(r"\d+|\D+", part)
            )
            for part in prerelease.split(".")
        )
    return (int(major), int(minor), int(patch), pre_key)


def fetch_releases(repo: str, token: str | None, timeout: float) -> list[dict]:
    """Return the repository's releases from the GitHub API (paginated)."""
    releases: list[dict] = []
    page = 1
    while True:
        url = f"https://api.github.com/repos/{repo}/releases?per_page=100&page={page}"
        request = urllib.request.Request(  # noqa: S310 - fixed https host
            url, headers={"Accept": "application/vnd.github+json"}
        )
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            batch = json.load(response)
        releases.extend(batch)
        if len(batch) < 100:
            return releases
        page += 1


def pack_releases(releases: list[dict], exclude_version: str | None) -> list[dict]:
    """Filter to published releases carrying a .pdsc and a pack asset.

    Returns one dict per pack release: version, date, tag, url; newest
    version first.
    """
    entries = []
    for release in releases:
        if release.get("draft"):
            continue
        assets = release.get("assets") or []
        has_pdsc = any(a["name"].endswith(".pdsc") for a in assets)
        pack_assets = [a for a in assets if _PACK_ASSET_RE.match(a["name"])]
        if not has_pdsc or not pack_assets:
            continue
        for asset in pack_assets:
            version = _PACK_ASSET_RE.match(asset["name"]).group(1)
            if version == exclude_version:
                continue
            published = release.get("published_at") or ""
            entries.append(
                {
                    "version": version,
                    "date": datetime.fromisoformat(
                        published.replace("Z", "+00:00")
                    ).strftime("%Y-%m-%d")
                    if published
                    else "",
                    "tag": release.get("tag_name") or "",
                    "url": asset["browser_download_url"],
                }
            )
    entries.sort(key=lambda e: _version_sort_key(e["version"]), reverse=True)
    return entries


def render_history(entries: list[dict], indent: str = "    ") -> str:
    """Render <release> XML entries for the PDSC <releases> section."""
    lines = []
    for e in entries:
        lines.append(
            f"{indent}<release version={quoteattr(e['version'])} "
            f"date={quoteattr(e['date'])} tag={quoteattr(e['tag'])} "
            f"url={quoteattr(e['url'])}>"
        )
        lines.append(f"{indent}  ExecuTorch {escape(e['version'])} pack release")
        lines.append(f"{indent}</release>")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default="pytorch/executorch",
        help="GitHub repository whose releases carry the pack assets",
    )
    parser.add_argument(
        "--releases-json",
        help="read the GitHub releases from this JSON file instead of the API "
        "(offline builds, tests)",
    )
    parser.add_argument(
        "--exclude-version",
        help="omit this version from the history (the entry for the version "
        "being built is emitted by the template)",
    )
    parser.add_argument(
        "--output", "-o", help="write the XML here (default: stdout)"
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail on API errors instead of emitting an empty history",
    )
    args = parser.parse_args()

    if args.releases_json:
        with open(args.releases_json) as f:
            releases = json.load(f)
    else:
        try:
            releases = fetch_releases(
                args.repo, os.environ.get("GITHUB_TOKEN"), args.timeout
            )
        except Exception as exc:  # noqa: BLE001 - degrade to empty history
            if args.strict:
                raise
            print(
                f"warning: could not fetch releases for {args.repo}: {exc}; "
                "emitting empty release history",
                file=sys.stderr,
            )
            releases = []

    history = render_history(pack_releases(releases, args.exclude_version))
    if args.output:
        with open(args.output, "w") as f:
            f.write(history + ("\n" if history else ""))
        print(
            f"Release history ({history.count('<release ')} entries) "
            f"written to: {args.output}"
        )
    else:
        print(history)
    return 0


if __name__ == "__main__":
    sys.exit(main())
