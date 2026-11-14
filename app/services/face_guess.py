"""Who an unnamed face probably is.

Read-side companion to `brain.py`. Given the brain already loaded in memory
and the embeddings sitting in a photograph's `.faces.json`, score each unnamed
face against every person who has been named often enough to have a brain, and
return the one good answer if there is one.

Nothing is stored. A guess is an opinion of the brain as it stands this second
— confirm one face and every other guess shifts — so writing guesses into the
sidecars would mean rewriting 358k files to keep them honest, and `intake-sync`
would eventually import an opinion into the graph as a fact. The embedding is
already in the sidecar and the brain is already cached in process, so the guess
costs one small matmul at read time and is always current.

The gate is a lead, not a score. The numbers here are cosine similarity
between face embeddings, not probabilities, and the absolute value mostly
tracks how good the photograph is and how old the person was — not how sure
the answer is. A picture of Amelia scored 0.746 against her own brain with the
runner-up at 0.336: unmistakably her, and a 0.75 floor threw it away over four
thousandths.

What actually says "nobody else is close" is the distance to the second-best
person. So a guess has to clear a modest floor AND lead the runner-up by a
wide margin. These numbers come from a hold-out: each person's centroids rebuilt from half
their confirmed faces, then scored against the other half. 71,292 faces over
350 people. The margin is where the accuracy lives, and the floor hardly
matters at all:

    rule                    recall   precision   stranger errors
    floor 0.45  margin 0.20  83.2%     99.88%         0.79%
    floor 0.45  margin 0.25  74.0%     99.92%         0.34%   <- here
    floor 0.45  margin 0.30  59.8%     99.91%         0.24%
    floor 0.70  margin 0.25  33.6%     99.83%         0.23%

"Stranger errors" is the one that matters and the one a naive test misses.
Precision above is measured on faces of people the brain already knows, but
most unnamed faces belong to strangers, and a stranger cannot be got right —
only left alone. So that column re-runs the same faces with the true person
masked out of the brain entirely, which is exactly what a stranger looks like,
and asks how often the rule still confidently names somebody. Raising the
margin from 0.20 to 0.25 halves it. Raising the floor instead costs enormous
recall and buys almost nothing.

Dating was considered and measured: buckets carry an `age:N-M` or
`decade:NNNNs` key, and the photograph's year is known here, so only the
matching era could be compared. It moves recall by 0.9 points and stranger
errors by 0.02 — because the deliberately era-agnostic `appearance:N`
centroids win the maximum anyway. Not worth the complexity.

For scale on the floor: faces already confirmed by hand score a median of
0.765 against the brain, and that is measured generously since those same
faces helped build it. So 0.85 — which sounds like a modest ask — is stricter
than a typical *correct* match and matches almost nothing.
"""
import numpy as np

# Low enough to admit a bad angle, a child at a different age, or a face
# sampled from a video frame — those run far lower than a photograph: a
# night-time clip of Guillaume scored 0.60 at his best and 0.45 at his worst
# while being unmistakably him. On its own this would be far too loose;
# MARGIN is what does the work.
FLOOR = 0.45

# How far ahead of the runner-up the winner has to be, and the real gate. A
# quarter of the scale is a landslide: everybody else is a stranger by
# comparison. This archive is full of people who look like each other, and
# Amelia and Margaret are exactly the pair a confident near-tie would
# mislabel, so a close contest is a question for a person rather than a
# one-click confirmation.
MARGIN = 0.25


def guess_people(brain, photo_path: str, faces: list[dict]) -> dict[int, dict]:
    """Best candidate per face, keyed by face_index.

    `brain` is the tuple from the brain loader, or None when no brain files
    exist yet. `photo_path` is relative (`archive/...`), matching the way
    rejections are recorded. `faces` are sidecar entries carrying `face_index`
    and `embedding`.

    Faces with no embedding are skipped: the sidecar predates the field, or
    the detector wrote a box it could not describe. Either way there is
    nothing to compare.
    """
    if not brain or not faces:
        return {}
    centroids, _bucket_person_ix, persons_order, persons, _meta, seg_starts, seg_persons, sort_idx = brain
    if centroids is None or not len(centroids):
        return {}

    rows, indexes = [], []
    for f in faces:
        emb = f.get("embedding")
        fi = f.get("face_index")
        if fi is None or not emb:
            continue
        rows.append(emb)
        indexes.append(int(fi))
    if not rows:
        return {}

    X = np.asarray(rows, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != centroids.shape[1]:
        return {}
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # A zero vector normalises to NaN and then wins every comparison.
    norms[norms == 0] = 1.0
    X = X / norms

    sims = X @ centroids.T                                   # [n_faces, n_buckets]
    # Best bucket per person. The buckets are pre-sorted by person and
    # `seg_starts` marks where each person's run begins, so one reduceat
    # replaces a loop over 471 people.
    per_person = np.maximum.reduceat(sims[:, sort_idx], seg_starts, axis=1)

    out: dict[int, dict] = {}
    for i, fi in enumerate(indexes):
        row = per_person[i].copy()

        # A face somebody has explicitly said is NOT this person stays not
        # this person, however much the brain likes it.
        for k, pix in enumerate(seg_persons):
            pid = persons_order[pix]
            if (photo_path, fi) in (persons.get(pid, {}).get("negatives") or ()):
                row[k] = -1.0

        order = np.argsort(-row)
        top = float(row[order[0]])
        if top < FLOOR:
            continue
        if len(order) > 1 and top - float(row[order[1]]) < MARGIN:
            continue

        pid = persons_order[seg_persons[order[0]]]
        out[fi] = {
            "person_id": pid,
            "name":      (persons.get(pid) or {}).get("name") or pid,
            "score":     round(top, 4),
        }
    return out
