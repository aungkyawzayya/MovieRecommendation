# Report source (CVPR template, APA 7 references)

    pdflatex report && pdflatex report      # twice: cross-references

`report.tex` is the draft — it carries the yellow TODO boxes. To produce the
submission PDF, change one word near the top:

    \draftnotestrue   ->   \draftnotesfalse

`report_submit.tex` is that file with the word already flipped, kept only so
the page count of the real submission can be checked without editing anything.
Delete it once you are editing `report.tex` directly.

## Page budget

The brief allows 6–8 pages plus an uncounted appendix. As it stands:

| | pages |
|---|---|
| Body (title → references) | ~7.5 |
| Appendix A–E | ~1.5 |
| **Total PDF** | **9** |

The body is inside the limit with roughly half a page of room. If you add
material and go over, cut from Section 5.7 (ablations) or 5.9 (error
analysis) into the appendix first — appendix pages do not count.

## Figures

Regenerate any of them with `python docs/figures/<name>.py`:

| File | Used in |
|---|---|
| `fig1_architecture.pdf` | Figure 1, Methods |
| `fig3_beyond.pdf` | Figure 2, Section 5.5 |
| `fig2_serving.pdf` | Figure D1, Appendix D |

## References

APA 7, typeset by hand in the `apareferences` environment rather than by
BibTeX, because the template ships `ieee.bst` (numeric) and the brief allows
either style. Every in-text citation is narrative — "Sarwar et al. (2001)" —
so there are no `\cite` commands to keep in sync.

**Verify every entry before submitting.** The DOIs for Adomavicius & Kwon,
Burke, Cremonesi et al. and Sedhain et al. were checked against publisher
listings; the other seven were not.

## report.docx

An editable Word copy of exactly the same content, produced with
`pandoc report.tex -o report.docx`. Use it to draft wording; the submission
PDF still comes from `report.tex`.

Two things it does not carry over:

- **Layout.** Word renders it single-column, so it runs to 16 pages. That
  says nothing about the submission, which is 9 in the two-column template.
- **Cross-references.** `\ref{app:grids}` becomes "Appendix 8" rather than
  "Appendix B". Ignore it in the Word file, or fix by hand if you decide to
  submit from Word rather than LaTeX.

If you edit wording in `report.docx`, copy the change back into `report.tex` —
the two do not sync themselves.
