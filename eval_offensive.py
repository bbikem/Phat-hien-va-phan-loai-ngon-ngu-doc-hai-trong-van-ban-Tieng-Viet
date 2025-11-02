# -*- coding: utf-8 -*-
"""
Đánh giá mô hình phát hiện & highlight ngôn ngữ xúc phạm.
- So sánh 3 mô hình: Lexicon, ML (TFIDF+LR), Hybrid (OR).
- K-fold cross-validation và tối ưu ngưỡng theo F1 trên train.
- Nếu có cột 'spans' dạng "start-end|start-end" sẽ chấm thêm F1 ký tự.

Chạy:
    python eval_offensive.py --csv data_eval.csv --text-col text --label-col label --spans-col spans --k 5
"""

import argparse, re, json, math, statistics as stats
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_fscore_support, roc_auc_score, roc_curve, confusion_matrix
)

# ==== Import hàm & từ điển từ app.py để đúng y logic trong app ====
import importlib.util, os, sys

def import_app_module():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "app.py")
    if not os.path.exists(path):
        raise FileNotFoundError("Không tìm thấy app.py bên cạnh script.")
    spec = importlib.util.spec_from_file_location("app_module", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["app_module"] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod

app = import_app_module()

# ==== Chuẩn hoá text giống app ====
def normalize_text(s: str) -> str:
    t = str(s).lower().strip()
    for abb, meaning in app.norm_dict.items():
        t = re.sub(r'\b' + re.escape(abb) + r'\b', str(meaning), t, flags=re.IGNORECASE)
    return t

# ==== Dự đoán theo Lexicon/Abbrev ====
def lexicon_spans(text: str):
    spans = []
    spans += app.find_spans_lexicon(text)
    spans += app.find_spans_abbrev(text, app.norm_dict)
    return merge_spans(spans, text)

# (tuỳ chọn) ML gợi ý span giống app
def ml_spans(text: str, model, vec, top_k=3):
    return app.find_spans_ml(text, normalize_text(text), model, vec, top_k=top_k)

def merge_spans(spans, text):
    if not spans: return []
    spans_sorted = sorted(spans, key=lambda s: (s["start"], s["end"]))
    merged = [dict(spans_sorted[0])]
    for s in spans_sorted[1:]:
        last = merged[-1]
        if s["start"] <= last["end"]:
            last["end"] = max(last["end"], s["end"])
            last["source"] = set(last.get("source", set())) | set(s.get("source", set()))
        else:
            merged.append(dict(s))
    for m in merged:
        m["text"] = text[m["start"]:m["end"]]
        m["source"] = sorted(list(m["source"])) if isinstance(m["source"], set) else m.get("source", [])
    return merged

# ==== Chấm F1 ký tự cho spans (char-level) ====
def spans_to_charset(spans: List[Dict[str, Any]]) -> set:
    chars = set()
    for s in spans:
        a, b = int(s["start"]), int(s["end"])
        for i in range(max(0,a), max(0,b)):
            chars.add(i)
    return chars

def parse_gt_spans(s: str) -> List[Dict[str,int]]:
    # "10-15|20-25" -> [{"start":10,"end":15},{"start":20,"end":25}]
    out = []
    if not isinstance(s, str) or not s.strip():
        return out
    parts = s.split("|")
    for p in parts:
        if "-" in p:
            try:
                a, b = p.split("-")
                out.append({"start": int(a), "end": int(b)})
            except Exception:
                pass
    return out

def char_f1(pred_spans: List[Dict[str,int]], gt_spans: List[Dict[str,int]]) -> Tuple[float,float,float]:
    P = spans_to_charset(pred_spans)
    G = spans_to_charset(gt_spans)
    if len(P) == 0 and len(G) == 0:
        return 1.0, 1.0, 1.0
    if len(P) == 0:
        return 0.0, 0.0, 0.0
    tp = len(P & G); fp = len(P - G); fn = len(G - P)
    prec = tp / (tp + fp) if (tp+fp)>0 else 0.0
    rec  = tp / (tp + fn) if (tp+fn)>0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    return prec, rec, f1

# ==== Tối ưu ngưỡng theo F1 trên tập train ====
def best_threshold_by_f1(y_true, scores):
    # quét 101 ngưỡng 0..1
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0,1,101):
        yhat = (scores >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_true, yhat, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)

# ==== Đánh giá một fold ====
def eval_fold(train_df, test_df, text_col, label_col, spans_col=None, topk_span=3):
    # Train ML trên train_df
    X_train = [normalize_text(s) for s in train_df[text_col].astype(str).tolist()]
    y_train = train_df[label_col].astype(int).to_numpy()

    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=0.95)
    Xtr = vec.fit_transform(X_train)

    clf = LogisticRegression(max_iter=200)
    clf.fit(Xtr, y_train)

    # Tối ưu ngưỡng theo F1 (dựa trên train)
    train_scores = clf.predict_proba(Xtr)[:,1]
    thr, _ = best_threshold_by_f1(y_train, train_scores)

    # ---- Dự đoán trên test ----
    texts_test = test_df[text_col].astype(str).tolist()
    y_true = test_df[label_col].astype(int).to_numpy()

    Xte = vec.transform([normalize_text(s) for s in texts_test])
    ml_scores = clf.predict_proba(Xte)[:,1]
    ml_pred   = (ml_scores >= thr).astype(int)

    # Lexicon predictions
    lex_pred = []
    for t in texts_test:
        spans = lexicon_spans(t)
        lex_pred.append(1 if len(spans)>0 else 0)
    lex_pred = np.array(lex_pred, dtype=int)
    lex_scores = lex_pred.astype(float)  # 0/1 để vẫn tính AUC

    # Hybrid
    hyb_pred = ((ml_scores >= thr) | (lex_pred==1)).astype(int)
    hyb_scores = np.maximum(ml_scores, lex_scores)

    # --- Metrics: Precision, Recall, F1, AUC ---
    def four_metrics(y, scores, yhat):
        p, r, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
        try:
            auc = roc_auc_score(y, scores)
        except Exception:
            auc = float("nan")
        return p, r, f1, auc

    lex_m = four_metrics(y_true, lex_scores, lex_pred)
    ml_m  = four_metrics(y_true, ml_scores, ml_pred)
    hyb_m = four_metrics(y_true, hyb_scores, hyb_pred)

    # --- Span F1 ký tự (nếu có) ---
    span_res = None
    if spans_col and spans_col in test_df.columns:
        lex_f1s, ml_f1s, hyb_f1s = [], [], []
        gt_list = [parse_gt_spans(s) for s in test_df[spans_col].fillna("").astype(str).tolist()]
        for t, gt in zip(texts_test, gt_list):
            s_lex = lexicon_spans(t)
            s_ml  = ml_spans(t, clf, vec, top_k=topk_span)
            # union cho hybrid
            s_hyb = merge_spans(s_lex + s_ml, t)
            _, _, f1_lex = char_f1(s_lex, gt)
            _, _, f1_ml  = char_f1(s_ml,  gt)
            _, _, f1_hyb = char_f1(s_hyb, gt)
            lex_f1s.append(f1_lex); ml_f1s.append(f1_ml); hyb_f1s.append(f1_hyb)
        span_res = {
            "lex_charF1": float(np.mean(lex_f1s)),
            "ml_charF1":  float(np.mean(ml_f1s)),
            "hyb_charF1": float(np.mean(hyb_f1s)),
        }

    return {
        "lex": {"precision":lex_m[0], "recall":lex_m[1], "f1":lex_m[2], "auc":lex_m[3]},
        "ml":  {"precision":ml_m[0],  "recall":ml_m[1],  "f1":ml_m[2],  "auc":ml_m[3], "thr":thr},
        "hyb": {"precision":hyb_m[0], "recall":hyb_m[1], "f1":hyb_m[2], "auc":hyb_m[3]},
        "span": span_res
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Đường dẫn CSV nhãn")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--spans-col", default=None)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.text_col not in df.columns or args.label_col not in df.columns:
        raise ValueError("Thiếu cột text/label. Dùng --text-col và --label-col nếu tên khác.")

    y = df[args.label_col].astype(int).to_numpy()
    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=42)

    results = {"lex": [], "ml": [], "hyb": [], "span": []}

    for fold, (tr, te) in enumerate(skf.split(df, y), start=1):
        res = eval_fold(
            df.iloc[tr].reset_index(drop=True),
            df.iloc[te].reset_index(drop=True),
            text_col=args.text_col, label_col=args.label_col, spans_col=args.spans_col
        )
        for key in ["lex","ml","hyb"]:
            results[key].append(res[key])
        results["span"].append(res["span"])

        print(f"[Fold {fold}]",
              f"Lex F1={res['lex']['f1']:.3f} |",
              f"ML F1={res['ml']['f1']:.3f} (thr={res['ml']['thr']:.2f}) |",
              f"Hybrid F1={res['hyb']['f1']:.3f}")

    # Tổng hợp trung bình ± std
    def agg(key):
        P = [r["precision"] for r in results[key]]
        R = [r["recall"] for r in results[key]]
        F = [r["f1"] for r in results[key]]
        A = [r["auc"] for r in results[key]]
        return {
            "precision": (float(np.mean(P)), float(np.std(P))),
            "recall":    (float(np.mean(R)), float(np.std(R))),
            "f1":        (float(np.mean(F)), float(np.std(F))),
            "auc":       (float(np.nanmean(A)), float(np.nanstd(A))),
        }

    lex_agg = agg("lex"); ml_agg = agg("ml"); hyb_agg = agg("hyb")

    def fmt(m):  # "mean ± sd"
        return f"{m[0]:.3f} ± {m[1]:.3f}"

    print("\n=== Kết quả trung bình ({}-fold) ===".format(args.k))
    print("| Mô hình | Precision | Recall | F1 | AUC |")
    print("|---|---:|---:|---:|---:|")
    print(f"| Lexicon | {fmt(lex_agg['precision'])} | {fmt(lex_agg['recall'])} | {fmt(lex_agg['f1'])} | {fmt(lex_agg['auc'])} |")
    print(f"| ML      | {fmt(ml_agg['precision'])}  | {fmt(ml_agg['recall'])}  | {fmt(ml_agg['f1'])}  | {fmt(ml_agg['auc'])}  |")
    print(f"| Hybrid  | {fmt(hyb_agg['precision'])} | {fmt(hyb_agg['recall'])} | {fmt(hyb_agg['f1'])} | {fmt(hyb_agg['auc'])} |")

    # Span summary (nếu có)
    if any(results["span"]):
        valid = [s for s in results["span"] if s is not None]
        if valid:
            lex_span = [s["lex_charF1"] for s in valid]
            ml_span  = [s["ml_charF1"] for s in valid]
            hyb_span = [s["hyb_charF1"] for s in valid]
            print("\n[Span] F1 ký tự (trung bình các fold):")
            print(f"Lexicon: {np.mean(lex_span):.3f}  |  ML: {np.mean(ml_span):.3f}  |  Hybrid: {np.mean(hyb_span):.3f}")

    # Lưu summary
    summary = {
        "lex": lex_agg, "ml": ml_agg, "hyb": hyb_agg,
        "notes": "Điền số (cột mean) vào bảng báo cáo; bạn có thể giữ phần ±std ở slide phụ lục."
    }
    with open("metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
