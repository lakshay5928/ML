import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('mouse_viral_study.csv')
print(df.head())
#X = df.drop('Virus Present', axis=1)
#y = df['Virus Present']
features=["Med_1_mL", "Med_2_mL"]
X = df[features]
Y = df['Virus Present']
sns.scatterplot(x='Med_1_mL', y='Med_2_mL', hue='Virus Present', data=df)
plt.show()

from sklearn.svm import SVC
svm_model = SVC(kernel='linear')
svm_model.fit(X, Y)
pred_y = svm_model.predict(X)

from sklearn.metrics import classification_report, confusion_matrix

performance= classification_report(Y, pred_y)
print(performance)
confusion = confusion_matrix(Y, pred_y)
print(confusion)

