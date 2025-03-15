import json
import os


def load_ground_truth(file_path):
    """Load the list of ground truth records from a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def load_prediction(pred_file_path):
    """Load a predicted JSON result from a file."""
    with open(pred_file_path, "r") as f:
        return json.load(f)


def evaluate(ground_truth, field_list, base_path="./base_src/output/"):
    """
    Evaluate predictions against ground truth.

    Returns overall precision, recall, f1 and per-field stats.
    """
    total_correct = 0   # total number of fields correctly predicted
    total_predicted = 0  # total number of fields predicted
    total_actual = 0    # total number of fields present in ground truth

    per_field_stats = {field: {"correct": 0, "predicted": 0, "actual": 0} for field in field_list}

    for record in ground_truth:
        file_name = record.get("file")
        actual_data = record.get("data", {})
        total_actual += len(actual_data)
        for field, value in actual_data.items():
            if field in per_field_stats:
                per_field_stats[field]["actual"] += 1

        # Compute predicted filename from ground truth filename.
        # For example: "License 1.png" -> "License 1.json"
        base_filename = os.path.splitext(os.path.basename(file_name))[0]
        pred_filename = f"{base_path}{base_filename}.json"
        if not os.path.exists(pred_filename):
            print(f"Warning: Prediction file {pred_filename} not found for {file_name}")
            continue

        predicted_data = load_prediction(pred_filename)

        if "expiration_date" in predicted_data:
            predicted_data["expiration_date"] = predicted_data.pop("expiration_date")

        # Count predicted fields.
        total_predicted += len(predicted_data)
        for field, value in predicted_data.items():
            if field in per_field_stats:
                per_field_stats[field]["predicted"] += 1

        # Compare each field's values in a case-insensitive way.
        for field in field_list:
            if field in actual_data and field in predicted_data:
                actual_val = str(actual_data[field]).strip().lower()
                predicted_val = str(predicted_data[field]).strip().lower()
                if predicted_val == actual_val:
                    total_correct += 1
                    per_field_stats[field]["correct"] += 1

    precision = total_correct / total_predicted if total_predicted > 0 else 0
    recall = total_correct / total_actual if total_actual > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    return precision, recall, f1, per_field_stats


def evaluate_run(base_path="./base_src/output/"):
    ground_truth_file = "./eval/actual_results.json"
    ground_truth = load_ground_truth(ground_truth_file)

    field_list = [
        "document_type",
        "document_number",
        "first_name",
        "last_name",
        "state",
        "sex",
        "date_of_birth",
        "issue_date",
        "expiration_date",
        "height",
        "address",
        "hair",
        "eyes",
        "weight",
        "class"
    ]

    precision, recall, f1, per_field_stats = evaluate(ground_truth, field_list, base_path)

    print("\n")
    print("Per-field evaluation:")
    for field, stats in per_field_stats.items():
        field_precision = stats["correct"] / stats["predicted"] if stats["predicted"] > 0 else 0
        field_recall = stats["correct"] / stats["actual"] if stats["actual"] > 0 else 0
        field_f1 = (2 * field_precision * field_recall / (field_precision + field_recall)
                    if (field_precision + field_recall) > 0 else 0)
        print(f"{field:15}: Precision: {field_precision:.3f}, Recall: {field_recall:.3f}, F1: {field_f1:.3f}")

    print("\n")
    print("Overall evaluation:")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}\n")
