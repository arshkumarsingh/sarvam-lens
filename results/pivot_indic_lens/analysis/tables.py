import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
L = [str(l) for l in range(9, 18)]
names = {"logit": "logit", "j_indic_27p": "J-Indic-27", "j_wiki_40p": "J-Wiki-40", "j_200p": "J-200"}
for role, ctrl in (("english", "english_shuffled"), ("hindi", "hindi_shuffled"), ("target", None), ("chinese", "chinese_shuffled")):
    print(f"\n**{role}**: top-10 % (mean prob)" + (f"; shuffled control `{ctrl}` max top-10 over L9-17 in the last column" if ctrl else ""))
    print("| lens | " + " | ".join(f"L{l}" for l in L) + (" | ctrl max |" if ctrl else " |"))
    print("|---" * (len(L) + 1 + (1 if ctrl else 0)) + "|")
    for lens, nm in names.items():
        cells = []
        for l in L:
            x = d["by_layer"][lens][role][l]
            cells.append(f"{100 * x['top10'][0]:.0f} ({x['prob'][0]:.3f})")
        row = f"| {nm} | " + " | ".join(cells)
        if ctrl:
            cmax = max(d["by_layer"][lens][ctrl][l]["top10"][0] for l in L)
            pmax = max(d["by_layer"][lens][ctrl][l]["prob"][0] for l in L)
            row += f" | {100 * cmax:.0f} ({pmax:.4f})"
        print(row + " |")
print("\n**Band L14-16 (per-item mean), and paired difference vs J-Wiki-40** (top-10 pts / prob, 95% CI, concept bootstrap)")
print("| lens | English | Hindi | target | Chinese |")
print("|---|---|---|---|---|")
for lens, nm in names.items():
    cells = []
    for role in ("english", "hindi", "target", "chinese"):
        b = d["band"][lens][role]
        s = f"{100 * b['top10'][0]:.0f}% p={b['prob'][0]:.3f}"
        if lens != "j_wiki_40p":
            t = d["paired_vs_ref"][lens][role]["top10"]["band"]; p = d["paired_vs_ref"][lens][role]["prob"]["band"]
            s += f"; Δ {100 * t[0]:+.1f} [{100 * t[1]:+.1f}, {100 * t[2]:+.1f}], Δp {p[0]:+.3f} [{p[1]:+.3f}, {p[2]:+.3f}]"
        cells.append(s)
    print(f"| {nm} | " + " | ".join(cells) + " |")
print("\nitems:", d["n_items"], " English top-10 onset (first layer >= 50%):", d["onset"])
