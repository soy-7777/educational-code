model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=100)
booster = model.get_booster()
"""
.get_booster()：LightGBM の 内部の Booster クラス（純粋な学習器）へのアクセス。
"""
fscore = booster.get_score(importance_type="total_gain")
fscore = sorted([(k, v) for k, v in fscore.items()], key=lambda tpl: tpl[1], reverse=True)
print(f"\n {name} importance")
print(fscore)
