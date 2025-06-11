import os
import json

def parse_logs_and_find_best(log_dir=".venv/Logs/Epoches"):
    total_fitness = {}  # {player_index: total_fitness}
    last_seen = {}      # {player_index: log_file_name}

    # Walk through all files in the directory
    for root, _, files in os.walk(log_dir):
        for file in sorted(files):  # Sort to ensure consistent "last seen"
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                        for agent in data.get("top_agents", []):
                            idx = agent["index"]
                            fitness = agent["fitness"]

                            total_fitness[idx] = total_fitness.get(idx, 0) + fitness
                            last_seen[idx] = file  # Overwrites with most recent seen
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    if not total_fitness:
        return "No agent data found."

    # Find best player
    best_player = max(total_fitness.items(), key=lambda x: x[1])[0]
    best_fitness = total_fitness[best_player]
    last_log = last_seen[best_player]

    return {
        "best_player_index": best_player,
        "total_fitness": best_fitness,
        "last_seen_in_log": last_log
    }

# Example usage
if __name__ == "__main__":
    result = parse_logs_and_find_best("Logs")
    print(result)
