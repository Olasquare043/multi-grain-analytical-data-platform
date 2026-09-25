"""Task 8 (paper repair pass): renumber all 25 figures sequentially by order
of first mention, plus two new figures as numbers 26 (month-by-month V0
predictions vs actual) and 27 (V0 response curve). Copies (never
moves or edits) the existing 300 dpi PNGs from outputs/figures/ into
outputs/paper/figures_renumbered/ under new fig_NN_<slug>.png names, and
writes outputs/paper/figure_map.csv.

Pure filesystem + csv work; no docker/duckdb/lightgbm needed. Captions are
reproduced verbatim from outputs/figures/figure_manifest.csv (for the A- and
docs-series figures) and from the notebook cell that generated
fig_ladder_comparison.png (its caption was never written to figure_manifest.csv).
Figures 23 and 24 are the regenerated ladder figures (src/paper/ladder_figures.py).

Two orderings (A1a/A1b: heatmap then lines; A12a/b/c/d: national,
state_premiums, rank_stability, zones -- both assigned in the order
src/viz/figures.py's fig_a1()/fig_a12() generate them) were initially
flagged as unverified assumptions in the `label_assumption` column. The user
has since verified both against the paper's own text (2026-09-20) and
confirmed they are correct as assigned; the flags have been cleared.
"""
from __future__ import annotations

import shutil

import pandas as pd

from config import settings
from src.paper.ladder_captions import nigeria_caption, nyc_caption

FIGURES_DIR = settings.OUTPUTS_DIR / "figures"
PAPER_DIR = settings.OUTPUTS_DIR / "paper"
OUT_DIR = PAPER_DIR / "figures_renumbered"

MANIFEST = FIGURES_DIR / "figure_manifest.csv"

# The two ladder figures were regenerated from the corrected tables by
# src/paper/ladder_figures.py (into outputs/paper/figures/); their captions come from
# src/paper/ladder_captions.py, which the figures embed too.
REGENERATED = {
    "fig_model_ladder_nigeria.png": nigeria_caption,
    "fig_model_ladder_nyc.png": nyc_caption,
}

LADDER_CAPTIONS = {
    "fig_ladder_comparison.png": (
        "Both ladders on one axis, and which steps survived five repeats. Bars are the mean of 5 "
        "repeats, shown as % change in MAPE against each task's own V0, so two incomparable error "
        "scales (naira/litre, seconds) sit on one axis. Markers report whether the step from the "
        "previous rung into this one held its direction across ALL five repeats, taken within seed: "
        "a difference that flipped sign (cross) is not an established effect however large its bar "
        "looks. Bars and markers both show each task's ORIGINAL hyperparameter variant only, so "
        "NYC's single inconsistent comparison, which occurs under the capacity-controlled variant "
        "(V4b to V5, 4/5), is not visible here; the full set for both variants is in "
        "ladder_paired_comparisons.csv. The same fixed model and hyperparameters are used at every "
        "rung of a given ladder; only the input features change. V3a is computed with future "
        "months present and is shown only for contrast with V3b."
    ),
}

# (current_filename, current_label_in_paper, section, slug, label_assumption)
SEQUENCE = [
    ("fig_erd_conceptual.png", "Figure 4.1", "4.1", "erd_conceptual", ""),
    ("fig_erd_star_schema_dimensions.png", "Figure 4.2 (dimensions)", "4.2", "erd_star_schema_dimensions", ""),
    ("fig_erd_star_schema_facts.png", "Figure 4.2 (facts)", "4.2", "erd_star_schema_facts", ""),
    ("fig_architecture_ingestion.png", "Figure 4.4 (ingestion)", "4.4", "architecture_ingestion", ""),
    ("fig_architecture_serving.png", "Figure 4.4 (serving)", "4.4", "architecture_serving", ""),
    ("fig_cloud_architecture.png", "Figure 4.5", "4.5", "cloud_architecture", ""),
    ("A1_demand_profile_heatmap.png", "Figure A1a", "6.2", "a1a_demand_profile_heatmap", ""),
    ("A1_demand_profile_lines.png", "Figure A1b", "6.2", "a1b_demand_profile_lines", ""),
    ("A2_od_corridors.png", "Figure A2", "6.2", "a2_od_corridors", ""),
    ("A3_congestion_proxy.png", "Figure A3", "6.2", "a3_congestion_proxy", ""),
    ("A5_revenue_concentration.png", "Figure A5", "6.2", "a5_revenue_concentration", ""),
    ("A4_tipping_behaviour.png", "Figure A4", "6.2", "a4_tipping_behaviour", ""),
    ("A11_missingness_structure.png", "Figure A11", "6.2", "a11_missingness_structure", ""),
    ("A6_food_price_index.png", "Figure A6", "6.3", "a6_food_price_index", ""),
    ("A8_price_dispersion.png", "Figure A8", "6.3", "a8_price_dispersion", ""),
    ("A12_petrol_price_national.png", "Figure A12a", "6.3", "a12a_petrol_price_national", ""),
    ("A12_petrol_price_state_premiums.png", "Figure A12b", "6.3", "a12b_petrol_price_state_premiums", ""),
    ("A12_petrol_price_rank_stability.png", "Figure A12c", "6.3", "a12c_petrol_price_rank_stability", ""),
    ("A12_petrol_price_zones.png", "Figure A12d", "6.3", "a12d_petrol_price_zones", ""),
    ("A7_subsidy_structural_break.png", "Figure A7", "6.4", "a7_subsidy_structural_break", ""),
    ("A9_fuel_food_passthrough.png", "Figure A9", "6.4", "a9_fuel_food_passthrough", ""),
    ("A10_grain_comparison.png", "Figure A10", "6.5", "a10_grain_comparison", ""),
    ("fig_model_ladder_nigeria.png", "Figure 6.1", "6.7.1", "ladder_nigeria", ""),
    ("fig_model_ladder_nyc.png", "Figure 6.2", "6.7.2", "ladder_nyc", ""),
    ("fig_ladder_comparison.png", "Figure 6.3", "6.10", "ladder_comparison", ""),
]


# Figure 26: must match the caption embedded in the PNG by
# src/paper/extrapolation_ceiling.py (FIGURE_CAPTION), preceded by a one-line description.
FIG26_CAPTION = (
    "Nigeria: national mean petrol price by month, actual versus the V0 predictions and the "
    "random walk. Each point is the unweighted mean over the 37 states' test rows for one target "
    "month. The two V0 series are the mean of 20 seeds (LightGBM random_state 1-20) for the "
    "original configuration (300 trees) and the capacity-controlled configuration (12 trees). "
    "The random walk predicts each state's next-month price as its current-month price. "
    "outputs/paper/extrapolation_ceiling.csv carries the underlying series."
)


def fig27_caption() -> str:
    """Rebuilds the caption embedded by src/paper/nigeria_inference_v2.py from v0_plateau_summary.csv."""
    p = pd.read_csv(PAPER_DIR / "v0_plateau_summary.csv").set_index("configuration")
    o = p.loc["original (untuned, 300 trees)"]
    c = p.loc["capacity-controlled (CV-selected on V0)"]
    return (
        "Fitted V0 response curve (next-month price as a function of current-month price) for the "
        "original and capacity-controlled configurations. "
        "Each curve is one model (V0, seed 1) whose test errors were verified to reproduce the saved "
        "seed-1 errors exactly. The dotted line is the random walk. The rug marks the training values "
        "of price_ngn; the dashed vertical line is the highest of them "
        f"(NGN {o['train_price_ngn_max']:,.2f}). Points are the 185 test observations. "
        f"Original curve: exactly flat above price_ngn {o['last_split_threshold_price_ngn']:,.2f}, "
        f"at NGN {o['plateau_level_above_last_threshold']:,.2f}; {int(o['n_test_actuals_above_plateau_level'])} "
        "of 185 actual test values exceed that level. Capacity-controlled curve: flat above "
        f"{c['last_split_threshold_price_ngn']:,.2f}, at NGN {c['plateau_level_above_last_threshold']:,.2f}; "
        f"{int(c['n_test_actuals_above_plateau_level'])} of 185 exceed it. "
        f"{int(o['n_test_rows_price_above_train_price_max'])} test observations have a current price "
        "above the highest training price."
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST).set_index("filename")["caption"].to_dict()

    rows = []
    for i, (fname, label, section, slug, assumption) in enumerate(SEQUENCE, start=1):
        src_path = (PAPER_DIR / "figures" if fname in REGENERATED else FIGURES_DIR) / fname
        if not src_path.exists():
            raise FileNotFoundError(f"expected figure not found: {src_path}")
        new_name = f"fig_{i:02d}_{slug}.png"
        dest_path = OUT_DIR / new_name
        shutil.copy2(src_path, dest_path)
        if fname in REGENERATED:
            caption = REGENERATED[fname]()
        else:
            caption = LADDER_CAPTIONS.get(fname) or manifest.get(fname, "")
        rows.append({
            "current_filename": fname,
            "current_label_in_paper": label,
            "current_section_first_mentioned": section,
            "new_number": i,
            "new_filename": new_name,
            "caption_as_currently_written": caption,
            "label_assumption": assumption,
        })

    # Additional figures, numbered after the 25 renumbered ones.
    for number, fname, slug, caption, note in (
        (26, "fig_extrapolation_ceiling.png", "extrapolation_ceiling", FIG26_CAPTION,
         "additional figure introduced by this repair pass (Task 7), not a renumbering of an existing one."),
        (27, "fig_v0_response_curve.png", "v0_response_curve", fig27_caption(),
         "additional figure introduced by a follow-up analysis, not a renumbering of an existing one."),
    ):
        extra_src = PAPER_DIR / "figures" / fname
        if not extra_src.exists():
            raise FileNotFoundError(f"expected figure not found: {extra_src}")
        new_name = f"fig_{number}_{slug}.png"
        shutil.copy2(extra_src, OUT_DIR / new_name)
        rows.append({
            "current_filename": f"{fname} (new)",
            "current_label_in_paper": "(not previously in the paper)",
            "current_section_first_mentioned": "(new)",
            "new_number": number,
            "new_filename": new_name,
            "caption_as_currently_written": caption,
            "label_assumption": note,
        })

    out = pd.DataFrame(rows)
    out.to_csv(PAPER_DIR / "figure_map.csv", index=False)
    print(f"wrote outputs/paper/figure_map.csv ({len(out)} rows)")
    print(f"copied {len(rows)} PNGs into {OUT_DIR}")


if __name__ == "__main__":
    main()
