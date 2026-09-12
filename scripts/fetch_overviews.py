"""
fetch_overviews.py
Fills the gap behind the content model's 36.4% catalogue coverage.

data/overview_plot.csv currently holds 4,800 TMDB overviews, covering 3,536
of the 9,724 rated movies. 6,197 of the uncovered movies have a perfectly
valid tmdbId in links.csv and were simply never fetched — only 8 movies in
movies.csv have no tmdbId at all. So the ceiling on the content model (and
on the hybrid's cold-start reach, which is capped at the 15.6% of cold
items that have text) is a data-collection gap, not a modelling one.

This script fetches the missing overviews and writes a merged file. It does
NOT overwrite the existing one until you tell it to — see --output.

Usage, from the repo root with the venv active:

    export TMDB_API_KEY="your key here"

    python scripts/fetch_overviews.py --limit 10        # smoke test first
    python scripts/fetch_overviews.py                   # the real run
    python scripts/fetch_overviews.py --install         # swap it in when happy

Get a free key at themoviedb.org -> Settings -> API (v3 auth, the short one).
The key is read from the environment on purpose so it never lands in a file
that git might pick up.
"""

import argparse
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[1]
LINKS = REPO / "data/ml-latest-small/links.csv"
MOVIES = REPO / "data/ml-latest-small/movies.csv"
CURRENT = REPO / "data/overview_plot.csv"
CHECKPOINT = REPO / "data/.overview_fetch_checkpoint.csv"

API = "https://api.themoviedb.org/3/movie/{tmdb_id}"
WORKERS = 8          # TMDB tolerates far more; 8 keeps it polite and stable
TIMEOUT = 15
RETRIES = 3


def fetch_one(session, api_key, tmdb_id):
    """
    Returns a dict row, or None when TMDB has nothing usable for this id.
    A 404 means the id is stale (movies do get merged or removed upstream) —
    that is a real answer, not an error to retry.
    """
    for attempt in range(RETRIES):
        try:
            resp = session.get(
                API.format(tmdb_id=tmdb_id),
                params={"api_key": api_key, "language": "en-US"},
                timeout=TIMEOUT,
            )
        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1))
            continue

        if resp.status_code == 404:
            return None
        if resp.status_code == 429:
            # Respect the server's own backoff hint when it sends one.
            time.sleep(float(resp.headers.get("Retry-After", 2)) + 0.5)
            continue
        if resp.status_code == 401:
            raise SystemExit("TMDB rejected the API key (401). Check TMDB_API_KEY.")
        if not resp.ok:
            time.sleep(1.0 * (attempt + 1))
            continue

        data = resp.json()
        overview = (data.get("overview") or "").strip()
        if not overview:
            return None  # some titles genuinely have no synopsis on TMDB
        return {
            "tmdbId": int(tmdb_id),
            "title": data.get("title") or "",
            "overview": overview,
            # Same pipe-separated shape as the existing file's genres column.
            "genres": "|".join(g["name"] for g in data.get("genres", [])),
        }
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="fetch only this many (smoke test)")
    ap.add_argument("--output", default="data/overview_plot_full.csv",
                    help="where to write the merged file")
    ap.add_argument("--install", action="store_true",
                    help="after merging, back up and replace data/overview_plot.csv")
    args = ap.parse_args()

    api_key = os.environ.get("TMDB_API_KEY", "").strip()
    if not api_key:
        raise SystemExit('TMDB_API_KEY is not set.  export TMDB_API_KEY="..."')

    links = pd.read_csv(LINKS)
    movies = pd.read_csv(MOVIES)
    current = pd.read_csv(CURRENT)

    rated_or_listed = set(movies.movieId)
    have = set(current.tmdbId.dropna().astype(int))
    wanted = links[links.movieId.isin(rated_or_listed) & links.tmdbId.notna()]
    missing = sorted({int(t) for t in wanted.tmdbId} - have)

    print(f"movies.csv                 : {len(movies)}")
    print(f"with a tmdbId in links.csv : {len(wanted)}")
    print(f"already in overview_plot   : {len(have)}")
    print(f"to fetch                   : {len(missing)}")

    # Resume a previous interrupted run rather than paying for those calls again.
    fetched = []
    if CHECKPOINT.exists():
        ck = pd.read_csv(CHECKPOINT)
        done = set(ck.tmdbId.astype(int))
        fetched = ck.to_dict("records")
        missing = [t for t in missing if t not in done]
        print(f"resuming from checkpoint   : {len(done)} already done, {len(missing)} left")

    if args.limit:
        missing = missing[:args.limit]
        print(f"--limit {args.limit}: fetching only the first {len(missing)}")

    if not missing:
        print("\nNothing to fetch.")
    else:
        t0 = time.time()
        no_overview = 0
        with requests.Session() as session:
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                futures = {pool.submit(fetch_one, session, api_key, t): t for t in missing}
                for n, fut in enumerate(as_completed(futures), 1):
                    row = fut.result()
                    if row:
                        fetched.append(row)
                    else:
                        no_overview += 1
                    if n % 250 == 0 or n == len(missing):
                        rate = n / max(time.time() - t0, 1e-9)
                        print(f"  {n}/{len(missing)}  ({rate:.0f}/s)  "
                              f"got {len(fetched)}, no overview {no_overview}", flush=True)
                        # Checkpoint as we go, so a dropped connection at 90%
                        # costs a restart, not 5,000 wasted requests.
                        pd.DataFrame(fetched).to_csv(CHECKPOINT, index=False)
        print(f"\nfetched in {time.time() - t0:.0f}s")

    if not fetched:
        print("Nothing new was fetched; leaving files alone.")
        return

    new_df = pd.DataFrame(fetched)
    merged = (
        pd.concat([current, new_df], ignore_index=True)
        .drop_duplicates(subset="tmdbId", keep="first")   # keep the original rows
        .sort_values("tmdbId")
        .reset_index(drop=True)
    )

    out = REPO / args.output
    merged.to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(REPO)}  ({len(current)} -> {len(merged)} rows)")

    # --- what this actually buys the content model ---
    ratings = pd.read_csv(REPO / "data/ml-latest-small/ratings.csv")
    grid = set(ratings.movieId.unique())
    mv = movies.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
    for label, frame in (("before", current), ("after", merged)):
        ids = set(frame.tmdbId.dropna().astype(int))
        covered = set(mv.loc[mv.tmdbId.isin(ids), "movieId"]) & grid
        rating_cov = ratings.movieId.isin(covered).mean()
        print(f"  {label:6s}: {len(covered):5d}/{len(grid)} movies "
              f"({100 * len(covered) / len(grid):.1f}% of catalogue), "
              f"{100 * rating_cov:.1f}% of ratings")

    if args.install:
        backup = CURRENT.with_suffix(".csv.bak")
        shutil.copy2(CURRENT, backup)
        shutil.copy2(out, CURRENT)
        print(f"\ninstalled: {CURRENT.name} replaced (backup at {backup.name})")
        print("Now restart the kernel and run the notebook end to end.")
    else:
        print(f"\nNot installed yet. When the numbers above look right:")
        print(f"    python scripts/fetch_overviews.py --install")


if __name__ == "__main__":
    main()
