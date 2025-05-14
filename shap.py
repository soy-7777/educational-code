shap_df = pd.DataFrame(shap_values.values, columns=x_val.sample(500).columns)
mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
print(mean_abs_shap.head(200))
