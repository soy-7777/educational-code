model.fit(x_train, y_train)
importances = model.booster_.feature_importance(importance_type="gain")
"""
feature_importance(...)：←こういう関数
    importance_type	 "split" または "gain" を指定。
    "split"	          その特徴量が木の分岐で使われた「回数」。
    "gain"	          その特徴量の使用によって得られた「情報利得（目的関数の改善量の合計）」。
結果は List[float]（各特徴量に対応するスコアのリスト）
"""
feature_names = model.booster_.feature_name()
fscore = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
print(f"\n {name} importance")
print(fscore)
