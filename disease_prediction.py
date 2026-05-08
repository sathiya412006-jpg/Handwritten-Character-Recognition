import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


data = pd.read_csv("diabetes.csv")

print(data.head())


data = data.dropna()


X = data.drop("Outcome", axis=1)

y = data["Outcome"]


scaler = StandardScaler()

X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


def evaluate_model(model_name, model):

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    print(model_name)

    print(f"Accuracy  : {accuracy:.2f}")

    print(f"Precision : {precision:.2f}")

    print(f"Recall    : {recall:.2f}")

    print(f"F1 Score  : {f1:.2f}")

    print(classification_report(y_test, y_pred))

    print(confusion_matrix(y_test, y_pred))

    return accuracy


lr_model = LogisticRegression()

lr_accuracy = evaluate_model(
    "LOGISTIC REGRESSION MODEL",
    lr_model
)


svm_model = SVC()

svm_accuracy = evaluate_model(
    "SUPPORT VECTOR MACHINE MODEL",
    svm_model
)


rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_accuracy = evaluate_model(
    "RANDOM FOREST MODEL",
    rf_model
)


models = [
    "Logistic Regression",
    "SVM",
    "Random Forest"
]

accuracies = [
    lr_accuracy,
    svm_accuracy,
    rf_accuracy
]


plt.figure(figsize=(8, 5))

plt.bar(models, accuracies)

plt.title("Model Accuracy Comparison")

plt.xlabel("Models")

plt.ylabel("Accuracy")

plt.show()


best_accuracy = max(accuracies)

best_model = models[
    accuracies.index(best_accuracy)
]

print(f"Best Model    : {best_model}")

print(f"Best Accuracy : {best_accuracy:.2f}")


sample_patient = np.array([
    [6,148,72,35,0,33.6,0.627,50]
])

sample_patient = scaler.transform(sample_patient)

prediction = rf_model.predict(sample_patient)


if prediction[0] == 1:
    print("Patient has HIGH POSSIBILITY of Disease")
else:
    print("Patient has LOW POSSIBILITY of Disease")