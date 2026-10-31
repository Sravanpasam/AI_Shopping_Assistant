"""
run_eval.py
-----------
Sends every question in eval/test_questions.csv to a running chatbot API (POST /api/chat),
saves the answers, and scores what can be scored automatically. The rest is scored by hand.

Usage (start the Flask server first, in another terminal):
    python eval/run_eval.py --system A                      # full run against http://127.0.0.1:5000
    python eval/run_eval.py --system A --limit 10           # quick smoke test
    python eval/run_eval.py --system B                      # same server, http://127.0.0.1:5000
    python eval/run_eval.py --system A --translate-ar       # translate Arabic replies to English for the checks
    python eval/run_eval.py --system A --language ar        # only the Arabic questions
    python eval/run_eval.py --system A --category warranty  # only one category
    python eval/run_eval.py --system A --language ar --tag ar_questions
                                                            # -> results_A_ar_questions.csv

After you fill the manual_* columns in the results file, recompute the summary:
    python eval/run_eval.py --summarize eval/results_A.csv

Outputs:
    eval/results_<system>[<tag>].csv   (and .xlsx if openpyxl is installed)
    eval/summary_<system>[<tag>].csv

Manual columns (fill with 1 or 0):
    manual_correct       1 = the answer is correct and useful (overrides the automatic check)
    manual_hallucinated  1 = the answer states something not supported by the data/documents
"""
import argparse
import re
import sqlite3
import time
import uuid
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

POLICY_CATEGORIES = {"returns", "warranty", "shipping", "payment", "order_policy"}
ORDER_STATUSES = ["processing", "shipped", "out for delivery", "delivered", "cancelled", "returned"]
NOT_FOUND_RX = re.compile(
    r"not found|couldn.t find|could not find|no order|can.t find|cannot find|don.t (see|have)|"
    r"unable to find|doesn.t exist|does not exist|no record|invalid|not able to find|no matching",
    re.I,
)
ESCALATION_RX = re.compile(
    r"human|live agent|customer support|customer service|representative|connect you with|hand(ing)? (you )?over",
    re.I,
)


# ----------------------------------------------------------------------------- API
def chat(base_url, message, session_id, timeout=180):
    t0 = time.time()
    try:
        r = requests.post(
            f"{base_url}/api/chat",
            json={"message": message, "session_id": session_id},
            timeout=timeout,
        )
        data = r.json()
        err = "" if r.ok else f"HTTP {r.status_code}"
    except Exception as exc:  # network error, bad JSON, timeout
        data, err = {}, str(exc)[:200]
    return data, round(time.time() - t0, 2), err


_translator = None


def to_english(text):
    global _translator
    try:
        if _translator is None:
            from deep_translator import GoogleTranslator
            _translator = GoogleTranslator(source="auto", target="en")
        return _translator.translate(text[:4500]) or text
    except Exception:
        return text


# ----------------------------------------------------------------------------- helpers
def numbers_in(s):
    return set(re.findall(r"\d+(?:\.\d+)?", str(s).replace(",", "")))


def product_text(p):
    return " ".join(str(p.get(k, "")) for k in ("title", "category_path", "features_text", "description_text")).lower()


def relevance_hits(products, rx, k):
    return [bool(re.search(rx, product_text(p), re.I)) for p in products[:k]]


def order_truth(db_path, order_id):
    try:
        con = sqlite3.connect(db_path)
        row = con.execute(
            "SELECT status, estimated_delivery, tracking_number FROM orders WHERE order_id=?", (order_id,)
        ).fetchone()
        con.close()
        return row
    except Exception:
        return None


def num_or_none(v):
    try:
        return float(v) if str(v).strip() != "" and str(v) != "nan" else None
    except ValueError:
        return None


# ----------------------------------------------------------------------------- scoring
def score_row(row, resp, text_en, setup_resp, db_path):
    """Return (auto_pass: True/False/None, detail: str). None means 'score by hand'."""
    cat = row["category"]
    q = str(row["question"])
    products = resp.get("products") or []
    rtype = resp.get("response_type", "")
    intent = str(resp.get("intent", ""))
    n = len(products)

    if resp.get("error") or rtype == "error":
        return False, "server error"

    if cat == "product_search":
        rx = str(row["relevance_regex"])
        h5 = relevance_hits(products, rx, 5)
        p5 = sum(h5) / len(h5) if h5 else 0.0
        hit3 = any(h5[:3])
        return (n > 0 and hit3), f"n={n} p@5={p5:.2f} hit@3={hit3}"

    if cat == "price_filter":
        rx = str(row["relevance_regex"])
        lo, hi = num_or_none(row["min_price_usd"]), num_or_none(row["max_price_usd"])
        prices = [p.get("price") or 0 for p in products]
        price_ok = n > 0 and all(
            pr > 0 and (lo is None or pr >= lo - 1e-9) and (hi is None or pr <= hi + 1e-9) for pr in prices
        )
        hit3 = any(relevance_hits(products, rx, 3))
        return (price_ok and hit3), f"n={n} price_ok={price_ok} hit@3={hit3}"

    if cat == "comparison":
        if "compare" in q.lower():
            headers = ((resp.get("comparison") or {}).get("table") or {}).get("headers") or []
            return (rtype == "comparison" and len(headers) >= 3), f"type={rtype} columns={max(len(headers) - 1, 0)}"
        return None, "score by hand (which product is cheaper)"

    if cat == "follow_up":
        if "how much" in q.lower():
            first = ((setup_resp or {}).get("products") or [{}])[0]
            price = first.get("price")
            if price:
                found = any(s in text_en for s in (f"{price:.2f}", f"{price:g}"))
                return found, f"expected price {price:.2f}"
        return None, "score by hand"

    if cat == "order_status":
        m = re.search(r"ORD-\d+", q)
        if not m:
            return None, "no order id"
        truth = order_truth(db_path, m.group(0))
        low = text_en.lower()
        if "cancel" in q.lower():
            return None, f"score by hand; DB status={truth[0] if truth else 'not found'}"
        if truth is None:
            invented = any(s in low for s in ORDER_STATUSES)
            return (bool(NOT_FOUND_RX.search(text_en)) and not invented), "expected: order not found"
        status = truth[0].lower()
        return (status in low), f"DB status={truth[0]} est={truth[1]}"

    if cat in POLICY_CATEGORIES:
        want = numbers_in(row["expected_answer"])
        if not want:
            return None, "score by hand (yes/no or text answer)"
        got = numbers_in(text_en)
        found = want & got
        return (found == want), f"facts {len(found)}/{len(want)} ({sorted(want - got)} missing)"

    if cat == "out_of_scope":
        refused = n == 0 and rtype != "product_list"
        return refused, f"products_shown={n}"

    if cat == "escalation":
        esc = bool(resp.get("escalated")) or intent in ("escalate", "escalation", "human_handoff") \
            or rtype in ("escalate", "escalation") or bool(ESCALATION_RX.search(text_en))
        return esc, f"intent={intent} type={rtype}"

    if cat == "greeting":
        return (bool(text_en.strip()) and n == 0), ""

    return None, "score by hand"


# ----------------------------------------------------------------------------- summary
def summarize(df, out_path):
    df = df.copy()

    def final(r):
        m = str(r.get("manual_correct", "")).strip()
        if m in ("0", "1", "0.0", "1.0"):
            return int(float(m))
        a = str(r.get("auto_pass", "")).strip()
        if a in ("True", "False"):
            return 1 if a == "True" else 0
        return None

    df["final_pass"] = df.apply(final, axis=1)
    df["latency_s"] = pd.to_numeric(df["latency_s"], errors="coerce")
    scored = df[df["final_pass"].notna()]

    by_cat = scored.groupby("category")["final_pass"].agg(["count", "mean"]).rename(
        columns={"count": "scored", "mean": "pass_rate"}
    )
    by_cat["total_rows"] = df.groupby("category").size()
    by_cat["pass_rate"] = (by_cat["pass_rate"] * 100).round(1)
    by_lang = scored.groupby("language")["final_pass"].agg(["count", "mean"])
    by_lang["mean"] = (by_lang["mean"] * 100).round(1)

    hall = pd.to_numeric(df.get("manual_hallucinated", pd.Series(dtype=float)), errors="coerce").dropna()

    print("\n=== Pass rate by category (scored rows only) ===")
    print(by_cat[["total_rows", "scored", "pass_rate"]].to_string())
    print("\n=== Pass rate by language ===")
    print(by_lang.rename(columns={"count": "scored", "mean": "pass_rate"}).to_string())
    print(f"\nOverall: {scored['final_pass'].mean() * 100:.1f}% over {len(scored)} scored of {len(df)} rows")
    print(f"Mean latency: {df['latency_s'].mean():.2f}s   Max: {df['latency_s'].max():.2f}s")
    errs = int((df["error"].fillna("") != "").sum())
    print(f"Request errors: {errs}")
    if len(hall):
        print(f"Hallucination rate (manual): {hall.mean() * 100:.1f}% over {len(hall)} rows marked")
    else:
        print("Hallucination rate: fill the manual_hallucinated column (1/0) to get this.")
    unscored = len(df) - len(scored)
    if unscored:
        print(f"{unscored} rows still need manual_correct (1/0).")

    by_cat.to_csv(out_path)
    print(f"\nSaved {out_path}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="A", help="label for this system, e.g. A or B")
    ap.add_argument("--base-url", default="http://127.0.0.1:5000")
    ap.add_argument("--questions", default=str(HERE / "test_questions.csv"))
    ap.add_argument("--db", default=str(ROOT / "data" / "orders.db"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--language", default="", choices=["", "en", "ar"], help="only run questions in this language")
    ap.add_argument("--category", default="", help="only run questions in this category")
    ap.add_argument("--tag", default="", help="suffix for output filenames, e.g. 'ar' -> results_A_ar.csv")
    ap.add_argument("--translate-ar", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.0, help="pause between questions (seconds)")
    ap.add_argument("--resume", action="store_true",
                    help="keep completed rows in the existing results file and skip them")
    ap.add_argument("--summarize", default="", help="only recompute the summary from a results CSV")
    args = ap.parse_args()

    if args.summarize:
        path = Path(args.summarize)
        summarize(pd.read_csv(path, dtype=str, keep_default_na=False), path.with_name(path.name.replace("results", "summary")))
        return

    qs = pd.read_csv(args.questions, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if args.language:
        qs = qs[qs["language"] == args.language]
    if args.category:
        qs = qs[qs["category"] == args.category]
    if args.limit:
        qs = qs.head(args.limit)

    if qs.empty:
        print("No questions matched the given filters; nothing to do.")
        return

    try:
        requests.get(f"{args.base_url}/api/health", timeout=10)
    except Exception:
        print(f"Cannot reach {args.base_url}. Start the Flask server first.")
        return

    suffix = f"_{args.tag}" if args.tag else ""
    out_csv = HERE / f"results_{args.system}{suffix}.csv"

    existing_rows = []
    if args.resume and out_csv.exists():
        previous = pd.read_csv(out_csv, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        done = previous[
            (previous["response_text"].str.strip() != "")
            | (previous["error"].str.strip() != "")
        ]
        existing_rows = done.to_dict("records")
        completed_ids = set(done["id"])
        qs = qs[~qs["id"].isin(completed_ids)]
        print(f"Resume: {len(completed_ids)} completed rows kept, {len(qs)} questions left.")
        if qs.empty:
            print("Nothing left to run.")
            summarize(done, HERE / f"summary_{args.system}{suffix}.csv")
            return

    out_rows = list(existing_rows)
    stopped_reason = ""
    total = len(qs) + len(existing_rows)
    for i, row in qs.iterrows():
        session_id = f"eval-{args.system}-{uuid.uuid4().hex[:8]}"
        setup_resp = None
        if row["setup_question"]:
            setup_resp, _, _ = chat(args.base_url, row["setup_question"], session_id)

        resp, latency, err = chat(args.base_url, row["question"], session_id)
        text = str(resp.get("text", ""))
        text_en = to_english(text) if (args.translate_ar and row["language"] == "ar" and text) else text

        auto, detail = score_row(row, resp, text_en, setup_resp, args.db)
        products = resp.get("products") or []
        degraded = bool(resp.get("degraded")) or resp.get("error") == "quota_exhausted"
        out_rows.append({
            "id": row["id"], "language": row["language"], "category": row["category"],
            "setup_question": row["setup_question"], "question": row["question"],
            "expected_behavior": row["expected_behavior"], "expected_answer": row["expected_answer"],
            "response_text": text, "response_text_en": text_en if text_en != text else "",
            "intent": resp.get("intent", ""), "confidence": resp.get("confidence", ""),
            "response_type": resp.get("response_type", ""), "n_products": len(products),
            "top_products": " | ".join(str(p.get("title", ""))[:50] for p in products[:3]),
            "latency_s": latency, "error": err, "degraded": degraded,
            "auto_pass": "" if auto is None else str(bool(auto)), "auto_detail": detail,
            "manual_correct": "", "manual_hallucinated": "", "notes": row["notes"],
        })
        flag = {"True": "PASS", "False": "FAIL", "": "hand"}["" if auto is None else str(bool(auto))]
        print(f"[{len(out_rows):>3}/{total}] {row['id']} {row['category']:<14} {flag:<4} {latency:>5.1f}s  {row['question'][:55]}")
        pd.DataFrame(out_rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
        if degraded:
            stopped_reason = (
                f"Stopped cleanly at {row['id']}: the LLM is unavailable/quota-exhausted "
                f"(degraded answer). Results are saved; rerun with --resume."
            )
            break
        if args.sleep:
            time.sleep(args.sleep)

    df = pd.DataFrame(out_rows)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    try:
        df.to_excel(HERE / f"results_{args.system}{suffix}.xlsx", index=False)
    except Exception:
        pass
    print(f"\nSaved {out_csv}")
    if stopped_reason:
        print(stopped_reason)
    summarize(df.astype(str), HERE / f"summary_{args.system}{suffix}.csv")


if __name__ == "__main__":
    main()
