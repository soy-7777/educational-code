importance = model.feature_importances_

import matplotlib.pyplot as plt
import seaborn as sns

feat_imp = pd.Series(importance, index=x_val.columns)
plt.figure(figsize=(10, 20))
sns.barplot(x=feat_imp.values, y=feat_imp.index)
plt.title("Feature Importance")
plt.show()
