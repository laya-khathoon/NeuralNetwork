
from evaluation import evaluate_classification


y_true = [0, 0, 1, 1]
y_pred = [0.1, 0.4, 0.8, 0.9]

metrics = evaluate_classification(y_true, y_pred)

print("Evaluation Results")
print("------------------")

for name, value in metrics.items():
    print(f"{name}: {value}")

assert metrics["accuracy"] == 1.0
assert metrics["precision"] == 1.0
assert metrics["recall"] == 1.0
assert metrics["f1_score"] == 1.0

print("\nEvaluation tests passed!")

