import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Load Dataset
df = pd.read_csv('mouse_viral_study.csv')

# Show first 5 rows
print(df.head())

# Features
features = ["Med_1_mL", "Med_2_mL"]

X = df[features]

# Target Variable
y = df['Virus Present']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scatter Plot
plt.figure(figsize=(6,4))

sns.scatterplot(
    x="Med_1_mL",
    y="Med_2_mL",
    hue=y_train,
    data=X_train
)

plt.title("Medicine Data Visualization")

plt.show()

# Create SVM Model
svm_model = SVC(kernel='linear')

# Train Model
svm_model.fit(X_train, y_train)

# Prediction
pred_y = svm_model.predict(X_test)

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, pred_y))

# Confusion Matrix
print("\nConfusion Matrix:\n")
cm = confusion_matrix(y_test, pred_y)

print(cm)