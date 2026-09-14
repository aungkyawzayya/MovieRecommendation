# Report figures

Regenerate either figure with `python docs/figures/<name>.py` (needs matplotlib
only — no models are fitted, the numbers are typed in from the notebook's
outputs). Both are emitted as PDF for `\includegraphics` and PNG for previewing.

| File | Use |
|---|---|
| `fig1_architecture.{pdf,png}` | Methodology section — data, split, three base models, the nested composition, and the evaluation battery |
| `fig2_serving.{pdf,png}` | Implementation / deployment section — the batch/serving split and the cold-start fallback path |

Colours match the notebook's model palette (SVD blue, AutoRec tan, Content
green, Nested Hybrid orange), so a reader recognises each model in the figures
and in the results charts. Fills are light tints of the same hue so the
figures stay readable printed in greyscale.

`../architecture_diagram_PROPOSAL_superseded.png` is the September proposal
diagram. **Do not use it in the report.** It states 9,742 movies rather than
the 9,724 that are actually rated, 3,537 plot texts rather than 9,603, and it
draws a user feedback loop and a "Periodic Retraining Trigger" that were never
built — the README's own Limitations section says there is no refresh layer,
so using that diagram would contradict the report's own text.
