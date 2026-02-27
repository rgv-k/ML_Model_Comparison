

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


df = pd.read_csv("Churn_Modelling.csv")


df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)


le = LabelEncoder()

df["Gender"] = le.fit_transform(df["Gender"])
df["Geography"] = le.fit_transform(df["Geography"])

X = df.drop("Exited", axis=1)
y = df["Exited"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Preprocessing complete ✅")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])



from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

print("XGBoost trained ✅")



svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

print("SVM trained ✅")



knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)

print("KNN trained ✅")


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt


metrics = {
    "Model": ["XGBoost", "SVM", "KNN"],
    "Accuracy": [
        accuracy_score(y_test, xgb_pred),
        accuracy_score(y_test, svm_pred),
        accuracy_score(y_test, knn_pred)
    ],
    "Precision": [
        precision_score(y_test, xgb_pred),
        precision_score(y_test, svm_pred),
        precision_score(y_test, knn_pred)
    ],
    "Recall": [
        recall_score(y_test, xgb_pred),
        recall_score(y_test, svm_pred),
        recall_score(y_test, knn_pred)
    ],
    "F1 Score": [
        f1_score(y_test, xgb_pred),
        f1_score(y_test, svm_pred),
        f1_score(y_test, knn_pred)
    ]
}


metrics_df = pd.DataFrame(metrics)
print("\nModel Comparison:\n")
print(metrics_df)

metrics_df.set_index("Model").plot(kind="bar", figsize=(8,5))
plt.title("Model Comparison Metrics")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("model_comparison.png", dpi=300)
print("Graph saved as model_comparison.png ✅")




from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

models_preds = {
    "XGBoost": xgb_pred,
    "SVM": svm_pred,
    "KNN": knn_pred
}

for name, pred in models_preds.items():
    cm = confusion_matrix(y_test, pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig(f"{name}_confusion.png", dpi=300)
    print(f"{name} confusion matrix saved ✅")

    plt.close()



from sklearn.metrics import roc_curve, auc

# probabilities
xgb_prob = xgb_model.predict_proba(X_test)[:,1]
svm_prob = svm_model.predict_proba(X_test)[:,1]
knn_prob = knn_model.predict_proba(X_test)[:,1]

# compute curves
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_prob)
fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_prob)
fpr_knn, tpr_knn, _ = roc_curve(y_test, knn_prob)

# AUC
auc_xgb = auc(fpr_xgb, tpr_xgb)
auc_svm = auc(fpr_svm, tpr_svm)
auc_knn = auc(fpr_knn, tpr_knn)

# plot
plt.figure()
plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC={auc_xgb:.2f})")
plt.plot(fpr_svm, tpr_svm, label=f"SVM (AUC={auc_svm:.2f})")
plt.plot(fpr_knn, tpr_knn, label=f"KNN (AUC={auc_knn:.2f})")
plt.plot([0,1], [0,1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()

plt.savefig("ROC_curve.png", dpi=300)
print("ROC curve saved ✅")

plt.close()