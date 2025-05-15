booster = model.get_booster()
"""
.booster_：LightGBM の 内部の Booster クラス（純粋な学習器）へのアクセス。
"""
fscore = model.get_score(importance_type="total_gain")
fscore = sorted([(k, v) for k, v in fscore.items()], key=lambda tpl: tpl[1], reverse=True)
print(f"\n {name} importance")
print(fscore)
