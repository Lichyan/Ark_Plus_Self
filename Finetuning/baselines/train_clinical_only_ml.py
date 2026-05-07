import argparse, os, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, cohen_kappa_score, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TAB_COLS=["age_abs","age_topcoded","sex_bin","bmi_stat","bmi_missing"]
LEGAL={(0,0),(1,1),(1,2),(2,1),(2,2),(3,2)}


def safe_auc(y, s):
    y=np.asarray(y)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, s))


def load_df(p):
    return pd.read_csv(p)


def fit_norm(df):
    return {
        "age_mean": float(df["age_abs"].mean()),
        "age_std": float(df["age_abs"].std() + 1e-8),
        "bmi_mean": float(df["bmi_stat"].mean()),
        "bmi_std": float(df["bmi_stat"].std() + 1e-8),
    }


def xform(df, s):
    x = df[TAB_COLS].copy()
    x["age_abs"] = (x["age_abs"] - s["age_mean"]) / s["age_std"]
    x["bmi_stat"] = (x["bmi_stat"] - s["bmi_mean"]) / s["bmi_std"]
    return x.values.astype(np.float32)


def make_model(kind):
    if kind == "lr":
        return LogisticRegression(max_iter=3000, multi_class="multinomial")
    if kind == "rf":
        return RandomForestClassifier(n_estimators=500, random_state=42)
    if kind == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(n_estimators=500, random_state=42)
        except Exception:
            return HistGradientBoostingClassifier(random_state=42)
    if kind == "hgb":
        return HistGradientBoostingClassifier(random_state=42)
    raise ValueError(kind)


def plot_confmat(cm, labels, title, savep):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center')
    fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(savep, dpi=200); plt.close(fig)


def eval_one(df, pg, ps, outdir):
    os.makedirs(outdir, exist_ok=True)
    y_g = df['grade'].astype(int).values
    y_s = df['stage'].astype(int).values
    pred_g = pg.argmax(1)
    pred_s = ps.argmax(1)

    p_any_g = 1.0 - pg[:, 0]
    p_ge2 = pg[:, 2] + pg[:, 3]
    p_ge3 = pg[:, 3]
    p_any_s = 1.0 - ps[:, 0]
    p_sge2 = ps[:, 2]

    invalid = np.array([(int(g), int(s)) not in LEGAL for g, s in zip(pred_g, pred_s)], dtype=int)
    invalid_type = [f"g{int(g)}_s{int(s)}" if iv else "" for g, s, iv in zip(pred_g, pred_s, invalid)]

    cm_g = confusion_matrix(y_g, pred_g, labels=[0, 1, 2, 3])
    cm_s = confusion_matrix(y_s, pred_s, labels=[0, 1, 2])
    plot_confmat(cm_g, ['0','1','2','3'], 'Grade Confmat', os.path.join(outdir, 'Confmat_grade.png'))
    plot_confmat(cm_s, ['0','1','2'], 'Stage Confmat', os.path.join(outdir, 'Confmat_stage.png'))

    cnt = {}
    for t in invalid_type:
        if t:
            cnt[t] = cnt.get(t, 0) + 1
    fig, ax = plt.subplots(figsize=(6, 4))
    if cnt:
        ax.bar(list(cnt.keys()), list(cnt.values())); ax.tick_params(axis='x', rotation=45)
    ax.set_title('Invalid type histogram'); fig.tight_layout(); fig.savefig(os.path.join(outdir, 'invalid_type_hist.png'), dpi=200); plt.close(fig)

    metrics = {
        'QWK_grade': float(cohen_kappa_score(y_g, pred_g, weights='quadratic')),
        'QWK_stage': float(cohen_kappa_score(y_s, pred_s, weights='quadratic')),
        'AUROC_grade_any_htn': safe_auc((y_g > 0).astype(int), p_any_g),
        'AUROC_grade_ge2': safe_auc((y_g >= 2).astype(int), p_ge2),
        'AUROC_grade_ge3': safe_auc((y_g >= 3).astype(int), p_ge3),
        'AUROC_stage_any_htn': safe_auc((y_s > 0).astype(int), p_any_s),
        'AUROC_stage_ge2': safe_auc((y_s >= 2).astype(int), p_sge2),
        'invalid_rate': float(invalid.mean()),
    }

    pd.DataFrame({
        'Path': df.get('Path', pd.Series([''] * len(df))),
        'grade_gt': y_g, 'stage_gt': y_s,
        'grade_pred': pred_g, 'stage_pred': pred_s,
        'prob_grade_any_htn': p_any_g, 'prob_stage_any_htn': p_any_s,
        'invalid_flag': invalid, 'invalid_type': invalid_type,
    }).to_csv(os.path.join(outdir, 'predictions.csv'), index=False)

    with open(os.path.join(outdir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(os.path.join(outdir, 'result.txt'), 'w', encoding='utf-8') as f:
        f.write('[scalar metrics]\n')
        for k, v in metrics.items():
            f.write(f'{k}={v}\n')
        f.write('\n[Confmat_grade labels=0,1,2,3]\n')
        f.write(np.array2string(cm_g, separator=', ') + '\n\n')
        f.write('[Confmat_stage labels=0,1,2]\n')
        f.write(np.array2string(cm_s, separator=', ') + '\n\n')
        f.write('[invalid_type_count]\n')
        f.write(json.dumps(cnt, ensure_ascii=False, sort_keys=True) + '\n\n')
        f.write('[generated_figures]\nConfmat_grade.png\nConfmat_stage.png\ninvalid_type_hist.png\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_csv', required=True)
    ap.add_argument('--valid_csv', required=True)
    ap.add_argument('--test_csv', required=True)
    ap.add_argument('--external_csvs', nargs='*', default=[])
    ap.add_argument('--external_names', nargs='*', default=[])
    ap.add_argument('--model_type', choices=['lr', 'rf', 'lightgbm', 'hgb'], default='lr')
    ap.add_argument('--out_root', required=True)
    args = ap.parse_args()

    tr = load_df(args.train_csv)
    te = load_df(args.test_csv)
    stats = fit_norm(tr)
    Xtr = xform(tr, stats)
    Xte = xform(te, stats)

    yg = tr['grade'].astype(int).values
    ys = tr['stage'].astype(int).values
    mg, ms = make_model(args.model_type), make_model(args.model_type)
    mg.fit(Xtr, yg)
    ms.fit(Xtr, ys)

    os.makedirs(args.out_root, exist_ok=True)
    with open(os.path.join(args.out_root, 'norm_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    eval_one(te, mg.predict_proba(Xte), ms.predict_proba(Xte), os.path.join(args.out_root, 'internal'))
    for i, csvp in enumerate(args.external_csvs):
        name = args.external_names[i] if i < len(args.external_names) else f'external_{i}'
        d = load_df(csvp)
        X = xform(d, stats)
        eval_one(d, mg.predict_proba(X), ms.predict_proba(X), os.path.join(args.out_root, name))


if __name__ == '__main__':
    main()
