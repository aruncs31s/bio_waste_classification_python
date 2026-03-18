import csv
import os


# Create or append to the CSV file
def write_to_csv(save_dir, image_name, prediction, confidence):
    csv_path = save_dir / "predictions.csv"
    """Writes prediction data for an image to a CSV file, appending if the file exists."""
    data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)
