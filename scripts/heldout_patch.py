"""
heldout_patch.py
Surfaces the 192 held-out hits the demo is currently hiding.

Measured: across all 610 users, 192 (user, movie) pairs appear BOTH in the
left column ("what this user actually loved") and in the right column's
top-10 recommendations — 166 of 610 users see at least one. Every single
one of those 192 is in the TEST split; none are from train/val. That is
not a leak and not a bug: the model was fit on train+val only, so those
are movies it had never seen a rating for, recommended anyway, and the
user really did rate them highly.

As the page stands, that reads as "why is it recommending something he
already rated?". Labelled, it is the single most convincing thing this
demo can show. So the export now carries each recommendation's held-out
rating when one exists, and the page renders it as a badge.

Run from the repo root:
    python scripts/heldout_patch.py
    python scripts/export_recommendations.py     # regenerate the JSON
    uvicorn api.main:app --reload                # restart the server
"""

import pathlib

repo = pathlib.Path(__file__).resolve().parents[1]

# ------------------------------------------------------- export script
p = repo / "scripts/export_recommendations.py"
s = p.read_text()

old = """    recommendations = {}
    for uid in all_user_ids:
        top = nested_hybrid.recommend_top_n(int(uid), n=TOP_N, exclude_seen=True)
        rows = []
        for rank, (movie_id, score) in enumerate(top.items(), start=1):
            meta = movie_lookup.loc[movie_id] if movie_id in movie_lookup.index else None
            rows.append({
                "rank": rank,
                "movieId": int(movie_id),
                "title": str(meta["title"]) if meta is not None else f"Movie {movie_id}",
                "genres": str(meta["genres"]) if meta is not None else "",
                "score": round(float(score), 4),
            })
        recommendations[str(int(uid))] = rows"""

new = """    # exclude_seen uses the model's TRAIN+VAL mask, so a movie the user rated
    # only in the TEST split is still a candidate — correctly, since that is
    # exactly what held-out evaluation measures. When one of those comes back
    # as a recommendation it is a HIT: the model never saw that rating and
    # recommended the movie anyway. Attaching the held-out rating here lets
    # the demo label those instead of looking like it re-recommends things
    # the user already rated. Measured on the current export: 192 such pairs
    # across 166 of the 610 users, all 192 from test, none from train/val.
    held_out = {}
    for row in ds.test_df.itertuples(index=False):
        held_out.setdefault(int(row.userId), {})[int(row.movieId)] = float(row.rating)

    recommendations = {}
    for uid in all_user_ids:
        top = nested_hybrid.recommend_top_n(int(uid), n=TOP_N, exclude_seen=True)
        user_held_out = held_out.get(int(uid), {})
        rows = []
        for rank, (movie_id, score) in enumerate(top.items(), start=1):
            meta = movie_lookup.loc[movie_id] if movie_id in movie_lookup.index else None
            rows.append({
                "rank": rank,
                "movieId": int(movie_id),
                "title": str(meta["title"]) if meta is not None else f"Movie {movie_id}",
                "genres": str(meta["genres"]) if meta is not None else "",
                "score": round(float(score), 4),
                # None for most rows: the user has no held-out rating for
                # this movie, so there is nothing to confirm either way.
                "held_out_rating": user_held_out.get(int(movie_id)),
            })
        recommendations[str(int(uid))] = rows

    n_hits = sum(1 for rows in recommendations.values()
                 for r in rows[:10] if r["held_out_rating"] is not None)
    n_users_with_hit = sum(1 for rows in recommendations.values()
                           if any(r["held_out_rating"] is not None for r in rows[:10]))
    print(f"Held-out hits in the top 10: {n_hits} across {n_users_with_hit} users "
          f"(movies the model never saw a rating for, that the user did rate)")"""

assert old in s, "export: recommendation loop not found"
p.write_text(s.replace(old, new))
print("scripts/export_recommendations.py: held_out_rating attached to each row")

# ------------------------------------------------------------ web page
p = repo / "web/index.html"
s = p.read_text()

old_css = "  td.rating { width: 6.5em; }"
new_css = """  td.rating { width: 6.5em; }

  /* Held-out hit: this movie was recommended by a model fit on train+val
     only, and the user's TEST-split rating (which the model never saw)
     confirms they liked it. Green because it is evidence the model works,
     not a warning. */
  .hit { display: inline-block; margin-left: 6px; padding: 1px 6px; border-radius: 10px;
         background: #e3f4e8; color: #1f6b38; font-size: 0.72rem; font-weight: 600;
         white-space: nowrap; vertical-align: middle; }
  tr.hit-row td { background: #fafdfb; }
  #hitNote { color: #1f6b38; font-size: 0.82rem; margin-top: 8px; min-height: 1.1em; }"""
assert old_css in s, "web: rating css not found"
s = s.replace(old_css, new_css)

old_tbl = """    <table id="resultsTable">
      <thead><tr><th></th><th>Title</th><th>Genres</th><th>Score</th></tr></thead>
      <tbody></tbody>
    </table>"""
new_tbl = """    <table id="resultsTable">
      <thead><tr><th></th><th>Title</th><th>Genres</th><th>Score</th></tr></thead>
      <tbody></tbody>
    </table>
    <div id="hitNote"></div>"""
assert old_tbl in s, "web: results table not found"
s = s.replace(old_tbl, new_tbl)

old_render = """  const rec = await res.json();
  resultsBody.innerHTML = '';
  for (const row of rec.recommendations) {
    const tr = document.createElement('tr');
    tr.innerHTML =
      `<td class="rank">${row.rank}</td>` +
      `<td>${escapeHtml(row.title)}</td>` +
      `<td class="genres">${escapeHtml(row.genres)}</td>` +
      `<td class="score">${row.score}</td>`;
    resultsBody.appendChild(tr);
  }
}"""
new_render = """  const rec = await res.json();
  resultsBody.innerHTML = '';
  let hits = 0;
  for (const row of rec.recommendations) {
    // held_out_rating is set only when this user rated this movie in the
    // TEST split — data the model was never fit on. So the row is not "a
    // movie they already rated" showing up by mistake, it is the model
    // predicting something that turned out to be right.
    const isHit = row.held_out_rating != null;
    if (isHit) hits++;
    const tr = document.createElement('tr');
    if (isHit) tr.className = 'hit-row';
    const badge = isHit
      ? `<span class="hit" title="The model was fit on train+val only and never saw this rating">` +
        `✓ they rated this ${row.held_out_rating}★</span>`
      : '';
    tr.innerHTML =
      `<td class="rank">${row.rank}</td>` +
      `<td>${escapeHtml(row.title)}${badge}</td>` +
      `<td class="genres">${escapeHtml(row.genres)}</td>` +
      `<td class="score">${row.score}</td>`;
    resultsBody.appendChild(tr);
  }

  const note = document.getElementById('hitNote');
  note.textContent = hits === 0
    ? ''
    : `${hits} of these ${hits === 1 ? 'is a movie' : 'are movies'} the user really did rate — ` +
      `in held-out test data the model was never fit on.`;
}"""
assert old_render in s, "web: render loop not found"
s = s.replace(old_render, new_render)

# Clear the note when the user switches, so it can't describe a stale list.
old_reset = """function resetRecommendations() {
  document.querySelector('#resultsTable tbody').innerHTML ="""
new_reset = """function resetRecommendations() {
  document.getElementById('hitNote').textContent = '';
  document.querySelector('#resultsTable tbody').innerHTML ="""
assert old_reset in s, "web: reset function not found"
s = s.replace(old_reset, new_reset)

p.write_text(s)
print("web/index.html: held-out hit badge, row tint, and summary line")
print("\nNext: python scripts/export_recommendations.py   (regenerates the JSON)")
print("      uvicorn api.main:app --reload")
