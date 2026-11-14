"""Rejected citations and re-measurement fallbacks."""

import pickle, numpy as np
cases = [c for c in pickle.load(open("outputs/test_cache.pkl", "rb")) if c.get("P")]
plus = {c["image"]: c for c in pickle.load(open("outputs/tbv_cache.pkl", "rb"))}
same = lambda a, b: np.allclose(np.asarray(a, float)[:4], np.asarray(b, float)[:4], atol=0.5)
cited = sum(len(c["P"]) for c in cases)
kept = sum(len(plus[c["image"]]["P_ea0_keep"]) for c in cases)
fb = sum(any(same(k, p) for p in c["P"]) for c in cases for k in plus[c["image"]]["P_ea0_keep"])
print(f"cited follicles: {cited} | rejected: {cited - kept} ({100*(cited-kept)/cited:.1f}%) | kept: {kept}")
print(f"fallback (original box kept) among kept follicles: {fb}/{kept} ({100*fb/kept:.1f}%), "
      f"i.e. {100*fb/cited:.1f}% of all cited follicles")
