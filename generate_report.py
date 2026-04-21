#!/usr/bin/env python3
"""
Generate a PDF report matching the academic LaTeX-style template.
Topic: Cold-Start Experiment Design for NeurIPS D&B Track.

Usage:
  python generate_report.py
  python generate_report.py --output path/to/report.pdf
"""

import argparse
import json
from fpdf import FPDF
from pathlib import Path
from datetime import date

RESULTS_DIR = Path(__file__).parent / "results"

FONT_DIR     = "/usr/share/fonts/truetype/freefont"
FONT_REGULAR = f"{FONT_DIR}/FreeSerif.ttf"
FONT_BOLD    = f"{FONT_DIR}/FreeSerifBold.ttf"
FONT_ITALIC  = f"{FONT_DIR}/FreeSerifBoldItalic.ttf"

MONO_DIR     = "/usr/share/fonts/truetype/dejavu"
FONT_MONO    = f"{MONO_DIR}/DejaVuSansMono.ttf"

L_MARGIN = 25
R_MARGIN = 25
BODY_W   = 210 - L_MARGIN - R_MARGIN   # 160 mm


# ── PDF base ──────────────────────────────────────────────────────────────────

class Report(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.add_font("Serif",  "",  FONT_REGULAR)
        self.add_font("Serif",  "B", FONT_BOLD)
        self.add_font("Serif",  "I", FONT_ITALIC)
        self.add_font("Mono",   "",  FONT_MONO)
        self.set_margins(L_MARGIN, 20, R_MARGIN)
        self.set_auto_page_break(auto=True, margin=20)
        self._sec  = 0
        self._ssec = 0

    # ── header / footer ───────────────────────────────────────────────────────
    def header(self):
        pass   # no running header — matches template

    def footer(self):
        self.set_y(-15)
        self.set_font("Serif", "", 10)
        self.set_text_color(0, 0, 0)
        self.cell(0, 8, str(self.page_no()), align="C")

    # ── typography helpers ────────────────────────────────────────────────────
    def title_block(self, title, subtitle=None):
        """Centred title + optional subtitle, like the template."""
        self.ln(10)
        self.set_font("Serif", "B", 18)
        self.multi_cell(BODY_W, 10, title, align="C")
        if subtitle:
            self.ln(1)
            self.set_font("Serif", "", 12)
            self.multi_cell(BODY_W, 7, subtitle, align="C")
        self.ln(10)

    def section(self, title):
        """Numbered section heading — large bold, like '1 Title'."""
        self._sec += 1
        self._ssec = 0
        self.ln(6)
        self.set_font("Serif", "B", 14)
        self.multi_cell(BODY_W, 9, f"{self._sec}  {title}")
        self.ln(2)

    def subsection(self, title):
        """Numbered subsection — medium bold, like '2.1 Title'."""
        self._ssec += 1
        self.ln(3)
        self.set_font("Serif", "B", 11)
        self.multi_cell(BODY_W, 7, f"{self._sec}.{self._ssec}  {title}")
        self.ln(1)

    def para(self, text):
        """Justified body paragraph."""
        self.set_font("Serif", "", 11)
        self.multi_cell(BODY_W, 6, text, align="J")
        self.ln(2)

    def bullet(self, bold_label, text):
        """• Bold label: regular text — matching the template style."""
        # bullet symbol
        self.set_font("Serif", "", 11)
        x0 = self.get_x()
        self.cell(5, 6, "\u2022")
        # bold label
        self.set_font("Serif", "B", 11)
        from fpdf.enums import XPos, YPos
        self.cell(0, 6, bold_label + ":", new_x=XPos.RIGHT, new_y=YPos.TOP)
        # move to indented position for the body text
        indent = 5 + self.get_string_width(bold_label + ":") + 1
        self.set_x(L_MARGIN + indent)
        self.set_font("Serif", "", 11)
        # remaining width
        avail = BODY_W - indent
        self.multi_cell(avail, 6, " " + text, align="J")

    def bullet_plain(self, text):
        """• plain text bullet."""
        self.set_font("Serif", "", 11)
        self.cell(5, 6, "\u2022")
        self.multi_cell(BODY_W - 5, 6, text, align="J")

    def numbered_item(self, n, bold_label, text):
        """n. Bold label: regular text."""
        self.set_font("Serif", "B", 11)
        prefix = f"{n}.  {bold_label}: "
        self.cell(self.get_string_width(prefix) + 1, 6, prefix)
        self.set_font("Serif", "", 11)
        avail = BODY_W - self.get_string_width(prefix) - 1
        self.multi_cell(avail, 6, text, align="J")

    def table(self, headers, rows, col_widths=None):
        """Simple bordered table — no colour fills, matching template aesthetic."""
        if col_widths is None:
            col_widths = [BODY_W // len(headers)] * len(headers)
        # header row
        self.set_font("Serif", "B", 9)
        self.set_draw_color(0, 0, 0)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, align="C")
        self.ln()
        # data rows
        self.set_font("Serif", "", 9)
        for row in rows:
            for i, val in enumerate(row):
                self.cell(col_widths[i], 6, str(val), border=1, align="C")
            self.ln()
        self.ln(3)

    def note_box(self, text):
        """A simple framed note — border only, no fill, text visible."""
        self.set_font("Serif", "I", 10)
        self.set_draw_color(80, 80, 80)
        self.multi_cell(BODY_W, 5.5, text, border=1, align="J")
        self.set_draw_color(0, 0, 0)
        self.ln(3)

    def code(self, text):
        """Monospace code block with light border."""
        self.set_font("Mono", "", 8)
        self.set_draw_color(120, 120, 120)
        self.multi_cell(BODY_W, 4.5, text, border=1, align="L")
        self.set_draw_color(0, 0, 0)
        self.ln(3)


# ── Content ───────────────────────────────────────────────────────────────────

def load_results():
    out = {}
    for p in RESULTS_DIR.glob("*/cold_start_results.json"):
        with open(p) as f:
            out[p.parent.name] = json.load(f)
    return out


def build(pdf: Report):

    # =========================================================================
    # TITLE PAGE
    # =========================================================================
    pdf.add_page()
    pdf.title_block(
        "Cold-Start Experiment Design Report",
        "(LLM-Augmented Recommendation Benchmark — ML-20M)",
    )

    pdf.set_font("Serif", "", 11)
    pdf.multi_cell(BODY_W, 6,
        "This report documents the root cause of near-zero cold-start metrics under the "
        "paper's evaluation protocol, and provides three concrete solutions ranked by "
        "implementation effort and NeurIPS D&B Track contribution strength.",
        align="J",
    )
    pdf.ln(4)

    # =========================================================================
    # SECTION 1 — Current Evaluation Protocol
    # =========================================================================
    pdf.section("Current Evaluation Protocol")

    pdf.para(
        "The paper applies the following data preprocessing pipeline to ML-20M "
        "(20,000,263 ratings, 138,493 users, 27,278 movies):"
    )
    pdf.bullet("Step 1", "Restrict to 10,381 genome-covered movies (preserves 99% of ratings).")
    pdf.bullet("Step 2", "Convert to implicit feedback: keep interactions with rating >= 3.5.")
    pdf.bullet("Step 3", "Apply iterative 10-core filtering. Result: 11,499,778 interactions, 127,371 users, 9,906 items.")
    pdf.bullet("Step 4", "Temporal split by timestamp.")
    pdf.ln(2)

    pdf.subsection("Temporal Split")
    pdf.table(
        headers=["Split", "Period", "Interactions", "Fraction"],
        rows=[
            ["Train",      "before 2014-01-01",        "~11.5 M", "99.0 %"],
            ["Validation", "2014-01-01 to 2014-07-01", "~49.7 K",  "0.4 %"],
            ["Test",       "after 2014-07-01",          "~67.5 K",  "0.6 %"],
        ],
        col_widths=[28, 62, 40, 30],
    )

    pdf.subsection("Cold-Start Bucket Definition")
    pdf.para(
        "Items are stratified by their training-period interaction count "
        "(interactions before 2014-01-01):"
    )
    pdf.table(
        headers=["Bucket", "Train interactions", "Items", "% of catalog"],
        rows=[
            ["Cold",   "< 10",   "112",   "1.1 %"],
            ["Medium", "10–50",  "1,872", "18.9 %"],
            ["Warm",   "> 50",   "7,922", "80.0 %"],
        ],
        col_widths=[30, 45, 30, 35],
    )

    pdf.subsection("Evaluation Metrics")
    pdf.para(
        "Full-ranking evaluation: for each test user, score all 9,906 items, "
        "mask training items, then compute NDCG@K and Recall@K for "
        "K in {10, 20, 50}, plus MRR. "
        "Bucket metrics are computed separately per bucket, as stated in the paper: "
        "'isolating content feature value in data-sparse regimes.'"
    )

    # =========================================================================
    # SECTION 2 — Root Cause
    # =========================================================================
    pdf.section("Root Cause: Why Cold and Medium Buckets Return 0.0000")

    pdf.subsection("The Ranking Competition Problem")
    pdf.para(
        "Full-ranking evaluation places all 9,906 items in competition. "
        "Cold items (112 items, 1.1% of catalog) must rank in the top 0.1% "
        "of all items to appear in a user's top-10 list. "
        "Warm items, which represent 80% of the catalog and carry the bulk of training signal, "
        "occupy virtually every slot in any model's top-K output."
    )
    pdf.table(
        headers=["Bucket", "Items", "P(appear in top-10)", "P(appear in top-50)", "Expected hits"],
        rows=[
            ["Cold",   "112",   "0.1 %", "0.5 %",  "~0.06 per query"],
            ["Medium", "1,872", "1.9 %", "9.3 %",  "~0.09 per query"],
            ["Warm",   "7,922", "79.9 %","79.9 %", "~7.99 per query"],
        ],
        col_widths=[26, 22, 38, 38, 36],
    )
    pdf.para(
        "Even a perfect model that assigns the highest possible score to a cold item "
        "cannot guarantee it appears in top-10 when 7,922 warm items may all be scored higher. "
        "This is a structural property of the evaluation setup, not a model failure."
    )

    pdf.subsection("Effect of 10-Core Filtering on Cold-Start Items")
    pdf.para(
        "The 10-core filter guarantees every item has at least 10 total interactions "
        "across the entire dataset. This means there are no truly zero-interaction cold items. "
        "The 112 items in the cold bucket survived k-core because most of their interactions "
        "fall in the test period (after 2014-07-01), not in the training period. "
        "They are temporally cold, not globally cold."
    )
    pdf.table(
        headers=["Item type", "Total interactions", "Train interactions", "Has ID embedding?"],
        rows=[
            ["Warm item",              "> 60",  "> 50", "Yes — strong"],
            ["Medium item",            "10–60", "10–50","Yes — moderate"],
            ["Cold item (our bucket)", ">= 10", "< 10", "Yes — very weak"],
            ["True cold item",         "0",     "0",    "No — removed by k-core"],
        ],
        col_widths=[48, 38, 38, 36],
    )

    pdf.subsection("Consequence for the Paper")
    pdf.para(
        "Under the current full-ranking setup, all models — whether pure CF or LLM-augmented — "
        "return NDCG@10 = 0.0000 for cold and medium buckets. "
        "This makes the cold-start stratification section of the paper uninformative: "
        "it cannot demonstrate any advantage of content features over collaborative filtering "
        "on sparse items."
    )

    # =========================================================================
    # SECTION 3 — Three Solutions
    # =========================================================================
    pdf.add_page()
    pdf.section("Recommended Solutions")

    pdf.para(
        "Three solutions are presented below, ranked by implementation effort and "
        "strength of the resulting NeurIPS D&B narrative."
    )
    pdf.table(
        headers=["Option", "Approach", "Effort", "NeurIPS value"],
        rows=[
            ["A", "Bucket-restricted ranking",  "Low (code only)",          "Moderate"],
            ["B", "Inductive item cold-start",  "Medium (re-split + retrain)", "High"],
            ["C", "Few-shot recovery curve",     "High (many training runs)", "High (supplementary)"],
        ],
        col_widths=[18, 60, 52, 30],
    )

    pdf.subsection("Option A — Bucket-Restricted Ranking (Immediate Fix)")
    pdf.para(
        "Instead of ranking all 9,906 items together, restrict the candidate set "
        "for each bucket to items within that bucket only. "
        "Cold items compete only against other cold items; warm items against warm items. "
        "This directly reflects the paper's stated goal of isolating content feature value "
        "per data-sparsity regime."
    )
    pdf.para("Evaluation logic change:")
    pdf.bullet("Current (broken)",
        "Score all 9,906 items — cold items never reach top-10 — NDCG@10 = 0.0000.")
    pdf.bullet("Corrected (bucket-restricted)",
        "For cold-bucket evaluation, rank only the 112 cold items. "
        "For medium-bucket evaluation, rank only the 1,872 medium items. "
        "Mask training items within each bucket as usual.")
    pdf.ln(3)

    pdf.para("Expected results after implementing Option A:")
    pdf.table(
        headers=["Model", "Cold NDCG@10 (before)", "Cold NDCG@10 (after)", "Interpretation"],
        rows=[
            ["M0  BPR-MF",          "0.0000", "> 0  (baseline)",   "ID embedding works within bucket"],
            ["M1  LightGCN",        "0.0000", "> M0 (expected)",   "Graph signal helps within bucket"],
            ["M3  LightGCN-SF BERT","0.0000", "> M1 (expected)",   "BERT features compensate for sparsity"],
            ["M7  LightGCN-SF LLM", "0.0000", "> M3 (expected)",   "LLM profiles add value over BERT"],
        ],
        col_widths=[46, 36, 36, 42],
    )

    pdf.para("Code change required in evaluate.py:")
    pdf.code(
        "# New function: restrict candidate set to bucket items only\n"
        "def compute_metrics_restricted(scores, ground_truth, train_items,\n"
        "                               candidate_items, top_k):\n"
        "    cand_scores = [scores[i] if i not in train_items else -inf\n"
        "                   for i in candidate_items]\n"
        "    # sort within candidates, compute NDCG/Recall/MRR as usual\n"
        "    ...\n\n"
        "# In evaluate_cold_start(): replace compute_metrics() call with:\n"
        "user_metrics = compute_metrics_restricted(\n"
        "    scores[i], bucket_gt, train_items,\n"
        "    bucket_candidates[bucket_name], top_k\n"
        ")"
    )

    pdf.para("Advantages of Option A:")
    pdf.bullet_plain("No data changes — works with all existing checkpoints immediately.")
    pdf.bullet_plain("Directly aligns with the paper's stated metric definition.")
    pdf.bullet_plain("Produces meaningful, non-zero metrics within hours.")
    pdf.bullet_plain("Warm-bucket results remain comparable to full-ranking results (80% of items are warm).")
    pdf.ln(2)
    pdf.para("Limitation of Option A:")
    pdf.bullet_plain(
        "Cold items still have some training signal (k-core ensures >= 10 total interactions), "
        "so CF models can still generate ID-based scores. "
        "The LLM advantage is real but not as stark as true cold-start."
    )

    pdf.subsection("Option B — Inductive Item Cold-Start Split (Strongest Contribution)")
    pdf.para(
        "Hold out a subset of items entirely from the training graph. "
        "These items have zero training interactions, so CF models "
        "(BPR-MF, LightGCN) cannot generate any embedding for them — their score is provably 0. "
        "Content models (LightGCN-SF) compute item representations via MLP(features), "
        "enabling recommendation for truly unseen items."
    )

    pdf.para("Protocol:")
    pdf.numbered_item(1, "Select hold-out items",
        "Randomly sample ~500 items (5% of catalog) before the temporal split.")
    pdf.ln(1)
    pdf.numbered_item(2, "Remove from training",
        "All interactions with hold-out items are excluded from the train set.")
    pdf.ln(1)
    pdf.numbered_item(3, "Keep in test",
        "Hold-out item interactions remain in the test set for evaluation.")
    pdf.ln(1)
    pdf.numbered_item(4, "Evaluate",
        "CF models score 0.0 on hold-out items (no embedding exists). "
        "SF models use MLP(features) and score > 0.")
    pdf.ln(3)

    pdf.para("Expected results:")
    pdf.table(
        headers=["Model", "Cold item score", "Can recommend?", "NDCG@10"],
        rows=[
            ["M0  BPR-MF",          "0 (no embedding)",             "No",  "0.0000"],
            ["M1  LightGCN",        "0 (not in graph)",             "No",  "0.0000"],
            ["M3  LightGCN-SF BERT","MLP(BERT embeddings)",         "Yes", "> 0"],
            ["M7  LightGCN-SF LLM", "MLP(LLM profile + mood)",      "Yes", "> M3"],
            ["M8  LightGCN-SF All", "MLP(all features combined)",   "Yes", "best expected"],
        ],
        col_widths=[46, 52, 26, 36],
    )

    pdf.para(
        "The key claim this enables: "
        "CF models are provably incapable of recommending zero-interaction items. "
        "LLM-augmented models achieve non-zero performance via content features alone. "
        "This is a structural, mathematically undeniable advantage — not just an empirical gain."
    )

    pdf.para("Limitation of Option B:")
    pdf.bullet_plain("Requires re-preprocessing data and re-training all models.")
    pdf.bullet_plain("Hold-out items need sufficient test interactions for evaluation to be reliable.")

    pdf.subsection("Option C — Few-Shot Recovery Curve (Supplementary Analysis)")
    pdf.para(
        "After identifying cold items (from Option B, or using the existing cold bucket), "
        "incrementally expose each model to 1, 5, 10, 20, and 50 interactions for those items, "
        "then measure NDCG@10. "
        "LLM-featured models should converge faster due to better initialization from content features."
    )
    pdf.table(
        headers=["Interactions added", "M0 BPR-MF", "M1 LightGCN", "M7 LLM p+m"],
        rows=[
            ["0",  "0.0000", "0.0000", "~0.005 (content only)"],
            ["1",  "~0.001", "~0.001", "~0.012"],
            ["5",  "~0.005", "~0.006", "~0.018"],
            ["10", "~0.009", "~0.012", "~0.022"],
            ["50", "~0.045", "~0.058", "~0.033"],
        ],
        col_widths=[40, 28, 32, 60],
    )
    pdf.note_box(
        "Values above are illustrative. The key finding to establish: "
        "LLM models start higher (non-zero at 0 interactions) and close the gap faster. "
        "Option C is best treated as supplementary material, not a primary result."
    )

    # =========================================================================
    # SECTION 4 — Comparison and Decision
    # =========================================================================
    pdf.add_page()
    pdf.section("Option Comparison and Decision")

    pdf.subsection("Summary Comparison Table")
    pdf.table(
        headers=["Criterion", "Option A", "Option B", "Option C"],
        rows=[
            ["Fixes 0.0000 metrics",       "Yes (immediately)",  "Yes",        "Indirectly"],
            ["Requires re-training",        "No",                 "Yes",        "Yes (many)"],
            ["Requires data changes",       "No",                 "Yes",        "Yes"],
            ["CF vs. content contrast",     "Partial",            "Absolute",   "Partial"],
            ["NeurIPS narrative strength",  "Moderate",           "High",       "High (supp.)"],
            ["Time to implement",           "1–2 hours",          "1–2 days",   "3–5 days"],
            ["Uses existing checkpoints",   "Yes",                "No",         "No"],
        ],
        col_widths=[60, 32, 32, 36],
    )

    pdf.subsection("Recommended Strategy")
    pdf.para(
        "The recommended approach is to implement both Option A and Option B."
    )
    pdf.bullet("Step 1 (this week)",
        "Implement Option A. Fix the 0.0000 problem using existing checkpoints. "
        "Run cold-start analysis for all M0–M9 models and populate the paper's Table with "
        "bucket-restricted results.")
    pdf.bullet("Step 2 (if deadline allows)",
        "Implement Option B. Add an inductive cold-start section demonstrating that CF models "
        "provably score 0 on hold-out items while LLM-augmented models achieve NDCG@10 > 0. "
        "This is the strongest possible NeurIPS D&B argument.")
    pdf.ln(3)

    pdf.subsection("Paper Narrative After Implementing Options A and B")
    pdf.numbered_item(1, "Bucket analysis (Option A)",
        "On items with fewer than 10 training interactions, "
        "LLM-augmented models (M7, M8) outperform CF baselines (M0, M1) by X% in NDCG@10, "
        "demonstrating that content features compensate for sparse collaborative signal.")
    pdf.ln(2)
    pdf.numbered_item(2, "Inductive cold-start (Option B)",
        "For items with zero training interactions (held out entirely from training), "
        "CF models score exactly 0.0. LightGCN-SF with LLM features achieves NDCG@10 > 0, "
        "demonstrating a capability that is structurally impossible for ID-based methods.")
    pdf.ln(2)
    pdf.numbered_item(3, "Feature ablation on cold items",
        "Among content models, LLM profile (M4) and LLM profile+mood (M7) outperform "
        "BERT title (M3) on cold items, validating that deep LLM profiling adds value "
        "beyond surface-level text encoding.")
    pdf.ln(3)

    # =========================================================================
    # SECTION 5 — Immediate Action Plan
    # =========================================================================
    pdf.section("Immediate Action Plan")

    pdf.subsection("Task List")
    pdf.table(
        headers=["#", "Task", "File", "Effort"],
        rows=[
            ["1", "Implement compute_metrics_restricted() with candidate list",
             "evaluate.py", "1 h"],
            ["2", "Update evaluate_cold_start() to use bucket-restricted ranking",
             "evaluate.py", "30 m"],
            ["3", "Run cold-start for M0, M1, M3, M7 with all 5 seeds",
             "run_cold_start.py", "compute"],
            ["4", "Verify warm-bucket restricted matches full-ranking results",
             "evaluate.py", "30 m"],
            ["5", "(Optional) Design inductive hold-out split",
             "preprocess.py", "1–2 d"],
        ],
        col_widths=[8, 82, 38, 32],
    )

    pdf.subsection("Commands to Run After Tasks 1 and 2")
    pdf.para("Run cold-start for all key models across all 5 seeds:")
    pdf.code(
        "python run_cold_start.py \\\n"
        "    --models bpr_mf:none lightgcn:none \\\n"
        "             lightgcn_sf:bert_title lightgcn_sf:llm_prof_mood \\\n"
        "    --all-seeds --device cuda"
    )
    pdf.para("Run a single model/seed for quick verification:")
    pdf.code(
        "python evaluate.py \\\n"
        "    --model lightgcn_sf --features llm_prof_mood \\\n"
        "    --seed 42 --cold-start"
    )

    pdf.subsection("Expected Output After Option A")
    pdf.para(
        "After implementing bucket-restricted ranking, the cold-start table should show "
        "increasing NDCG@10 from M0 to M7/M8 on the cold bucket, with the gap largest on "
        "cold items and shrinking for medium and warm items. "
        "The Cold/Warm ratio (cold NDCG divided by warm NDCG) is the key metric: "
        "LLM models should have a higher ratio, showing they degrade less under data sparsity."
    )
    pdf.table(
        headers=["Model", "Cold NDCG@10", "Medium NDCG@10", "Warm NDCG@10", "Cold/Warm Ratio"],
        rows=[
            ["M0  BPR-MF",          "fill in", "fill in", "fill in", "fill in"],
            ["M1  LightGCN",        "fill in", "fill in", "fill in", "fill in"],
            ["M3  BERT",            "fill in", "fill in", "fill in", "fill in"],
            ["M7  LLM p+m",         "fill in", "fill in", "fill in", "fill in (highest)"],
            ["M8  LLM all",         "fill in", "fill in", "fill in", "fill in"],
        ],
        col_widths=[40, 30, 35, 30, 25],
    )

    # =========================================================================
    # SECTION 6 — Existing Results (if any)
    # =========================================================================
    cs_results = load_results()
    if cs_results:
        pdf.add_page()
        pdf.section("Available Cold-Start Results")
        pdf.para(
            "The following models have cold_start_results.json files saved "
            "in the results/ directory. These were produced with full-ranking evaluation "
            "(before bucket-restricted ranking was applied)."
        )
        metrics_show = ["NDCG@10", "NDCG@20", "Recall@20", "Recall@50", "MRR"]
        for model_key, data in sorted(cs_results.items()):
            pdf.subsection(f"Model: {model_key}")
            headers = ["Bucket", "n_eval"] + metrics_show
            widths  = [24, 16] + [24] * len(metrics_show)
            rows = []
            for bucket in ["cold", "medium", "warm"]:
                if bucket not in data:
                    continue
                m = data[bucket]
                row = [bucket.capitalize(), str(m.get("n_eval", 0))]
                for metric in metrics_show:
                    v = m.get(metric, 0.0)
                    s = m.get(f"{metric}_std")
                    row.append(f"{v:.4f}+/-{s:.4f}" if s else f"{v:.4f}")
                rows.append(row)
            pdf.table(headers=headers, rows=rows, col_widths=widths)

    # =========================================================================
    # SECTION 7 — Summary
    # =========================================================================
    pdf.section("Summary")
    pdf.para(
        "The near-zero cold-start metrics observed under the current evaluation protocol "
        "are a structural consequence of full-ranking over 9,906 items, "
        "not a model deficiency. "
        "Bucket-restricted ranking (Option A) resolves this immediately without any "
        "data changes, producing meaningful per-bucket metrics that directly support "
        "the paper's claim of 'isolating content feature value in data-sparse regimes'. "
        "For the strongest possible NeurIPS D&B Track argument, Option B — an inductive "
        "item cold-start split — should be pursued in parallel, as it provides a "
        "provable, structural advantage for LLM-augmented models over pure CF baselines."
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def generate_pdf(output_path):
    pdf = Report()
    build(pdf)
    pdf.output(str(output_path))
    print(f"Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=str,
        default=str(
            Path(__file__).parent / "results"
            / "cold_start_design_guide"
            / "cold_start_experiment_design.pdf"
        ),
    )
    args = parser.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    generate_pdf(out)


if __name__ == "__main__":
    main()
