import os, json, warnings
import numpy as np
import pandas as pd
import joblib
import mlflow, mlflow.sklearn, mlflow.xgboost, mlflow.lightgbm
import shap
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, precision_recall_curve
)
from imblearn.combine import SMOTETomek
import xgboost  as xgb
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from data_pipeline import run_pipeline

MODEL_DIR       = 'models'
RESULTS_PATH    = os.path.join(MODEL_DIR, 'results.json')
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pkl')
os.makedirs(MODEL_DIR, exist_ok=True)

mlflow.set_tracking_uri('mlflow_runs')
mlflow.set_experiment('fraud_detection')


def evaluate(model, X_test, y_test, model_name, threshold=0.5):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    auc_roc     = roc_auc_score(y_test, y_prob)
    auc_pr      = average_precision_score(y_test, y_prob)
    precision   = precision_score(y_test, y_pred, zero_division=0)
    recall      = recall_score(y_test, y_pred, zero_division=0)
    f1          = f1_score(y_test, y_pred, zero_division=0)
    k           = min(100, len(y_test))
    top_k_idx   = np.argsort(y_prob)[-k:]
    precision_k = y_test.iloc[top_k_idx].sum() / k
    metrics = dict(model=model_name, threshold=round(threshold,3),
                   auc_roc=round(auc_roc,4), auc_pr=round(auc_pr,4),
                   precision=round(precision,4), recall=round(recall,4),
                   f1_score=round(f1,4), precision_k=round(precision_k,4))
    print(f"\n{'='*50}\n  {model_name}  (threshold={threshold:.2f})\n{'='*50}")
    for k_, v in metrics.items():
        if k_ != 'model': print(f"  {k_:<15} {v}")
    return metrics


def optimal_threshold(model, X_val, y_val):
    y_prob = model.predict_proba(X_val)[:, 1]
    precs, recs, thrs = precision_recall_curve(y_val, y_prob)
    f1s  = 2*precs*recs / (precs+recs+1e-9)
    best = thrs[np.argmax(f1s[:-1])]
    print(f"[INFO] Optimal threshold: {best:.3f}")
    return float(best)


def cross_validate_model(model, X, y, model_name):
    print(f"[INFO] 3-Fold CV for {model_name}...")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_validate(model, X, y, cv=cv,
                            scoring=['f1','roc_auc','average_precision'],
                            n_jobs=-1)
    cv_results = {
        'cv_f1_mean':     round(scores['test_f1'].mean(), 4),
        'cv_f1_std':      round(scores['test_f1'].std(),  4),
        'cv_auc_mean':    round(scores['test_roc_auc'].mean(), 4),
        'cv_auc_pr_mean': round(scores['test_average_precision'].mean(), 4),
    }
    print(f"  CV F1: {cv_results['cv_f1_mean']} ± {cv_results['cv_f1_std']}")
    return cv_results


def compute_shap(model, X_sample, model_name):
    try:
        print(f"[INFO] Computing SHAP for {model_name}...")
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list): shap_values = shap_values[1]
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
        path = os.path.join(MODEL_DIR, f'shap_{model_name.lower()}.png')
        plt.tight_layout(); plt.savefig(path, dpi=100, bbox_inches='tight'); plt.close()
        mean_abs = np.abs(shap_values).mean(axis=0)
        feat_imp = dict(zip(X_sample.columns.tolist(), mean_abs.tolist()))
        feat_imp = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:15])
        with open(os.path.join(MODEL_DIR, 'shap_importance.json'), 'w') as f:
            json.dump(feat_imp, f, indent=2)
        print(f"[INFO] SHAP saved → {path}")
        return path, feat_imp
    except Exception as e:
        print(f"[WARN] SHAP failed: {e}")
        return None, {}


def plot_cm(model, X_test, y_test, model_name, threshold=0.5):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax,
                xticklabels=['Normal','Fraud'], yticklabels=['Normal','Fraud'])
    ax.set_title(f'{model_name}'); ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
    path = os.path.join(MODEL_DIR, f'cm_{model_name.lower()}.png')
    plt.tight_layout(); plt.savefig(path); plt.close()
    return path


def tune_xgboost(X_train, y_train, X_val, y_val, n_trials=5):
    print(f"[INFO] Optuna tuning XGBoost ({n_trials} trials)...")
    def objective(trial):
        p = dict(
            n_estimators     = trial.suggest_int('n_estimators', 100, 300),
            max_depth        = trial.suggest_int('max_depth', 3, 7),
            learning_rate    = trial.suggest_float('learning_rate', 0.01, 0.2),
            subsample        = trial.suggest_float('subsample', 0.6, 1.0),
            colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0),
            scale_pos_weight = (y_train==0).sum()/(y_train==1).sum(),
            random_state=42, n_jobs=-1, eval_metric='aucpr',
        )
        m = xgb.XGBClassifier(**p)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return average_precision_score(y_val, m.predict_proba(X_val)[:,1])
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"[INFO] Best XGBoost params: {study.best_params}")
    return study.best_params


def tune_lightgbm(X_train, y_train, X_val, y_val, n_trials=5):
    print(f"[INFO] Optuna tuning LightGBM ({n_trials} trials)...")
    def objective(trial):
        p = dict(
            n_estimators     = trial.suggest_int('n_estimators', 100, 400),
            max_depth        = trial.suggest_int('max_depth', 3, 8),
            learning_rate    = trial.suggest_float('learning_rate', 0.01, 0.2),
            num_leaves       = trial.suggest_int('num_leaves', 20, 63),
            subsample        = trial.suggest_float('subsample', 0.6, 1.0),
            colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0),
            class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1,
        )
        m = lgb.LGBMClassifier(**p)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(-1)])
        return average_precision_score(y_val, m.predict_proba(X_val)[:,1])
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"[INFO] Best LightGBM params: {study.best_params}")
    return study.best_params


def train_all(X_train, y_train, X_val, y_val, X_test, y_test):
    all_results = {}

    # 1. Logistic Regression
    print("\n[MODEL 1] Logistic Regression (Baseline)...")
    with mlflow.start_run(run_name='LogisticRegression'):
        lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        thr = optimal_threshold(lr, X_val, y_val)
        m   = evaluate(lr, X_test, y_test, 'LogisticRegression', thr)
        cv  = cross_validate_model(lr, X_train, y_train, 'LR')
        m.update(cv)
        mlflow.log_metrics({k:v for k,v in m.items() if k!='model'})
        mlflow.sklearn.log_model(lr, 'lr_model')
        joblib.dump(lr, os.path.join(MODEL_DIR, 'lr_model.pkl'))
        all_results['logistic_regression'] = m

    # 2. Random Forest
    print("\n[MODEL 2] Random Forest...")
    with mlflow.start_run(run_name='RandomForest'):
        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                    random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        thr = optimal_threshold(rf, X_val, y_val)
        m   = evaluate(rf, X_test, y_test, 'RandomForest', thr)
        cv  = cross_validate_model(rf, X_train, y_train, 'RF')
        m.update(cv)
        mlflow.log_metrics({k:v for k,v in m.items() if k!='model'})
        plot_cm(rf, X_test, y_test, 'RandomForest', thr)
        mlflow.sklearn.log_model(rf, 'rf_model')
        joblib.dump(rf, os.path.join(MODEL_DIR, 'rf_model.pkl'))
        all_results['random_forest'] = m

    # 3. XGBoost + Optuna
    print("\n[MODEL 3] XGBoost + Optuna...")
    best_xgb = tune_xgboost(X_train, y_train, X_val, y_val, n_trials=5)
    with mlflow.start_run(run_name='XGBoost_Optuna'):
        best_xgb.update({'random_state':42,'n_jobs':-1,'eval_metric':'aucpr'})
        xgb_m = xgb.XGBClassifier(**best_xgb)
        xgb_m.fit(X_train, y_train, eval_set=[(X_val,y_val)], verbose=False)
        thr = optimal_threshold(xgb_m, X_val, y_val)
        m   = evaluate(xgb_m, X_test, y_test, 'XGBoost', thr)
        cv  = cross_validate_model(xgb_m, X_train, y_train, 'XGB')
        m.update(cv); m['optuna_trials'] = 5
        mlflow.log_params(best_xgb)
        mlflow.log_metrics({k:v for k,v in m.items() if k!='model'})
        plot_cm(xgb_m, X_test, y_test, 'XGBoost', thr)
        mlflow.xgboost.log_model(xgb_m, 'xgb_model')
        joblib.dump(xgb_m, os.path.join(MODEL_DIR, 'xgboost_model.pkl'))
        all_results['xgboost'] = m

    # 4. LightGBM + Optuna
    print("\n[MODEL 4] LightGBM + Optuna...")
    best_lgb = tune_lightgbm(X_train, y_train, X_val, y_val, n_trials=5)
    with mlflow.start_run(run_name='LightGBM_Optuna'):
        best_lgb.update({'class_weight':'balanced','random_state':42,'n_jobs':-1,'verbose':-1})
        lgb_m = lgb.LGBMClassifier(**best_lgb)
        lgb_m.fit(X_train, y_train, eval_set=[(X_val,y_val)],
                  callbacks=[lgb.early_stopping(30,verbose=False), lgb.log_evaluation(-1)])
        thr = optimal_threshold(lgb_m, X_val, y_val)
        m   = evaluate(lgb_m, X_test, y_test, 'LightGBM', thr)
        cv  = cross_validate_model(lgb_m, X_train, y_train, 'LGB')
        m.update(cv); m['optuna_trials'] = 5
        mlflow.log_params(best_lgb)
        mlflow.log_metrics({k:v for k,v in m.items() if k!='model'})
        plot_cm(lgb_m, X_test, y_test, 'LightGBM', thr)
        mlflow.lightgbm.log_model(lgb_m, 'lgb_model')
        joblib.dump(lgb_m, os.path.join(MODEL_DIR, 'lightgbm_model.pkl'))
        all_results['lightgbm'] = m

    return all_results, xgb_m, lgb_m


def main():
    print("\n[STEP 1] Data pipeline...")
    X_train, X_test, y_train, y_test, scaler = run_pipeline()

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42)

    print("\n[STEP 2] SMOTETomek...")
    smt = SMOTETomek(random_state=42)
    X_tr_res, y_tr_res = smt.fit_resample(X_tr, y_tr)
    print(f"[INFO] After resample — Fraud:{y_tr_res.sum():,} Normal:{(y_tr_res==0).sum():,}")

    print("\n[STEP 3] Training all models...")
    all_results, xgb_m, lgb_m = train_all(X_tr_res, y_tr_res, X_val, y_val, X_test, y_test)

    best_name  = max(all_results, key=lambda k: all_results[k]['f1_score'])
    best_model = xgb_m if 'xgboost' in best_name else lgb_m
    joblib.dump(best_model, BEST_MODEL_PATH)
    print(f"\n[BEST] {best_name} (F1={all_results[best_name]['f1_score']})")

    print("\n[STEP 4] SHAP explainability...")
    X_shap = X_test.sample(min(300, len(X_test)), random_state=42)
    shap_path, shap_imp = compute_shap(best_model, X_shap, best_name)

    final = {
        'models':            all_results,
        'best_model':        best_name,
        'shap_top_features': shap_imp,
    }
    with open(RESULTS_PATH, 'w') as f:
        json.dump(final, f, indent=2)

    print(f"\n[DONE] Results  → {RESULTS_PATH}")
    print(f"[DONE] Best model → {BEST_MODEL_PATH}")
    print(f"[DONE] MLflow UI  → mlflow ui --backend-store-uri mlflow_runs")


if __name__ == '__main__':
    main()
