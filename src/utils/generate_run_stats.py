import json
import glob
import os

def generate_stats():
    # Use glob to find all triplet-hardmining-* directories
    base_dir = os.path.join("experiments", "runs")
    run_dirs = glob.glob(os.path.join(base_dir, "triplet-hardmining-*"))

    for run_dir in run_dirs:
        if not os.path.isdir(run_dir):
            continue

        history_path = os.path.join(run_dir, "training_history.json")
        if not os.path.exists(history_path):
            print(f"Skipping {run_dir}, no training_history.json found.")
            continue

        with open(history_path, 'r') as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                print(f"Error parsing JSON in {history_path}")
                continue
            
        if not history:
            continue

        best_epoch_data = None
        best_monitor_value = -1.0

        for epoch_data in history:
            monitor_val = epoch_data.get("monitor_value", -1.0)
            if monitor_val > best_monitor_value:
                best_monitor_value = monitor_val
                best_epoch_data = epoch_data

        if best_epoch_data and "val" in best_epoch_data and "map_at_k" in best_epoch_data["val"]:
            map_at_k = best_epoch_data["val"]["map_at_k"]
            mAP_1 = map_at_k.get("1", "N/A")
            mAP_5 = map_at_k.get("5", "N/A")
            mAP_10 = map_at_k.get("10", "N/A")
            epoch_num = best_epoch_data["val"].get("epoch", "N/A")

            stats_content = f"""# Run Summary

**Best Epoch:** {epoch_num}

## Metrics (mAP@k)
- **mAP@1:** {mAP_1}
- **mAP@5:** {mAP_5}
- **mAP@10:** {mAP_10}
"""
            stats_path = os.path.join(run_dir, "stats.md")
            with open(stats_path, 'w') as f:
                f.write(stats_content)
            print(f"Created stats.md in {run_dir}")

if __name__ == "__main__":
    generate_stats()
