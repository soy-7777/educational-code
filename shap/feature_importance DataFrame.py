import shap
from collections import defaultdict
#元々設定してあるpredict calorieの私のモデル
models = {
    "CatBoost": CatBoostRegressor(verbose=0, random_seed=42, cat_features=["Sex"],
                                  early_stopping_rounds=100),
    "XGBoost": XGBRegressor(max_depth=10, colsample_bytree=0.7, subsample=0.9,
                            n_estimators=2000, learning_rate=0.02, gamma=0.01,
                            max_delta_step=2, early_stopping_rounds=100,
                            eval_metric="rmse", enable_categorical=True, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=2000, learning_rate=0.02, max_depth=10,
                                colsample_bytree=0.7, subsample=0.9, random_state=42, verbose=-1)
    }
FOLDS = 5
all_shape_values = defaultdict(lambda: defaultdict(list))
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

for name, model in models.items():
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            x_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            x_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            x_train["Sex"] = x_train["Sex"].astype(int)
            x_val["Sex"] = x_val["Sex"].astype(int)
            if name == "XGBoost":
                model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=100)
            elif name == "CatBoost":
                model.fit(x_train, y_train, eval_set=(x_val, y_val))
            else:
                model.fit(x_train, y_train)
              
     #すべてのデータについて計算すると、時間が爆発しちゃうから各Foldから2000個ずつ抜き出して計算         
            x_val_for_shap = x_val.sample(n=2000, random_state=42+ fold)
            explainer = shap.Explainer(
                lambda x: model.predict(pd.DataFrame(x, columns=x_val_for_shap.columns).assign(Sex=lambda df: df["Sex"].astype("str"))),
                x_val_for_shap)
      """
      ここでのコードはSHAP の Explainer に model.predict を直接渡すと、内部処理で x が numpy.array になり、"Sex" のようなカテゴリ変数が float に変換されてしまうことを回避するために書いた。ただ、私の技術力では
      どうすればいいのかさっぱりだったので、ChatGPTが教えてくれたのをそのまま写した。それだけじゃ何も身にならないので、ここで解説
       lambda x: ...
　　　　  SHAP は model.predict に似た形の関数を必要とするので、それを満たすための関数を匿名関数（lambda）で作っている。lambdaっていうのはdef関数の名前なしバージョン。簡単に関数を書きたいときに使う。
　　　  　この関数は、SHAP が内部的にデータ x（＝特徴量）を渡してくるときに呼ばれる。
       pd.DataFrame(x, columns=x_val_for_shap.columns)
         SHAP が渡してくる x は numpy.array 型になっているため、元の特徴量名がなくなる。
         それを DataFrame に戻すことで、CatBoost に渡すときに正しい列名と順番が復元される。
         そのあとのassign()の所で"Sex"列をstr型に変換している。

         ここまででちゃんと整形されたデータをpredict()に渡してる。なので、explainer()を通してもエラーにならない。
      """
      """
      explainer = shap.Explainer(model.predict, x_val_for_shap) の説明：
      　　　　データを入力するとその予測に対するSHAP値（どの特徴が予測にどれくらい影響したか）を計算してくれる関数になる。
        model.predictは今使ってる機械学習モデルの予測関数。
        x_val_for_shapはSHAP値を計算したいデータのサンプル
　　　 """
            shap_values = explainer(x_val_for_shap)
            for i, col in enumerate(x_val_for_shap.columns):
                mean_abs_shap = np.abs(shap_values[:, i].values).mean()
                all_shap_values[name][col].append(mean_abs_shap)
      """
      shap_values[:, i]で列ごとに特徴量のSHAP値を取り出す（ベクトルになる）
      .valuesでSHAP値をnumpy.arrayに変換
      np.abs(...)で正負がある（プラスに効く、マイナスに効く）SHAP値を絶対値に変換
      .mean()で特徴量 i が持つ SHAP 値の絶対値の「平均」を出す（今まで各データでSHAP値を計算していたから
      """
