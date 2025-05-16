import json

def evaluate_results(results, ground_truths):
    correct = 0
    total = len(results)
    for res, gt in zip(results, ground_truths):
        if res.strip().lower() == gt.strip().lower():
            correct += 1
    return {"accuracy": correct / total}
