import os
import pandas as pd

"""
Merge `final` and `Season_11` databases into one.
"""

test_folder = "./Season_11"
training_folder = "./raw"
output_folder = "./merged"

def get_all_files(folder_path):
    return [
        os.path.join(folder_path, entry)
        for entry in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, entry))
    ]


def main():
    training_files = get_all_files(training_folder)
    test_files = get_all_files(test_folder)
    os.makedirs(output_folder, exist_ok=True)

    for training_file in training_files:
        print(training_file)
        training_basename = os.path.basename(training_file)
        merged_path = os.path.join(output_folder, training_basename)

        # Find corresponding test file by basename
        matching_test_file = next(
            (tf for tf in test_files if os.path.basename(tf) == training_basename),
            None,
        )
        print("Test:", matching_test_file)

        if matching_test_file:
            try:
                training_df = pd.read_csv(training_file)
                test_df = pd.read_csv(matching_test_file)
                merged_df = pd.concat([training_df, test_df], ignore_index=True)
                merged_df.to_csv(merged_path)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to merge '{training_file}' and '{matching_test_file}': {e}"
                )
        else:
            training_df = pd.read_csv(training_file)
            merged_df = training_df.to_csv(merged_path)

if __name__ == "__main__":
    main()
