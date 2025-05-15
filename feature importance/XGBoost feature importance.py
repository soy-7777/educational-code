booster = model.get_booster()
"""
.booster_：LightGBM の 内部の Booster クラス（純粋な学習器）へのアクセス。
"""
"""
feature_importance(...)：←こういう関数
    importance_type	 "split" または "gain" を指定。
    "split"	          その特徴量が木の分岐で使われた「回数」。
    "gain"	          その特徴量の使用によって得られた「情報利得（目的関数の改善量の合計）」。
結果は List[float]（各特徴量に対応するスコアのリスト）
"""
fscore = model.get_score(importance_type="total_gain")
fscore = sorted([(k, v) for k, v in fscore.items()], key=lambda tpl: tpl[1], reverse=True)
print(f"\n {name} importance")
print(fscore)
