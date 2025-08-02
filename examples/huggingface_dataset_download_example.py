import os
import json
import argparse
from datasets import load_dataset, get_dataset_config_names, Image as HFImage
from PIL import Image
import uuid

def save_dataset_to_folders(dataset_name: str, base_dir: str, specified_subset: str = "all"):
    """
    Load a dataset from Hugging Face Hub and save each subset and split to a local folder structure.
    Capable of handling multiple image columns in each data row.
    If the target JSON file already exists, it will be skipped, enabling "resume from checkpoint" functionality.

    Args:
        dataset_name (str): Dataset name on Hugging Face Hub (e.g., 'TrustGen/Trustgen_dataset').
        base_dir (str): Root directory for storing all output folders.
        specified_subset (str): Specify a single subset to download, or "all" to download all subsets.
    """
    print(f"Processing dataset: {dataset_name}")

    # 1. Create the root directory for all outputs
    os.makedirs(base_dir, exist_ok=True)
    print(f"All files will be saved to root directory: '{base_dir}'")

    try:
        # 2. Get all subset (configuration) names in the dataset
        all_subset_names = get_dataset_config_names(dataset_name)
        print(f"Available subsets: {all_subset_names}")
    except Exception as e:
        print(f"Unable to get subset list: {e}")
        print("Will attempt to load the dataset without specifying a subset.")
        all_subset_names = [None]

    # 3. Determine which subsets to process
    if specified_subset.lower() == "all":
        subset_names = all_subset_names
        print(f"Will process all {len(subset_names)} subsets")
    else:
        if specified_subset in all_subset_names:
            subset_names = [specified_subset]
            print(f"Will process only the '{specified_subset}' subset")
        else:
            print(f"Error: Specified subset '{specified_subset}' not found in available subsets.")
            print(f"Available subsets are: {all_subset_names}")
            return

    # 4. Iterate through each subset
    for subset_name in subset_names:
        print("-" * 50)
        current_subset_name = subset_name if subset_name else "default"
        print(f"Loading subset: '{current_subset_name}'...")

        try:
            # datasets library automatically caches data, so repeated loading is fast
            dataset_subset = load_dataset(dataset_name, name=subset_name)
        except Exception as e:
            print(f"Failed to load subset '{current_subset_name}': {e}")
            continue

        # Create a folder for the current subset
        subset_dir = os.path.join(base_dir, current_subset_name)
        os.makedirs(subset_dir, exist_ok=True)
        print(f"Created folder for subset '{current_subset_name}': '{subset_dir}'")

        # 5. Iterate through all splits in the subset (e.g., 'train', 'test')
        for split_name, split_data in dataset_subset.items():

            # --- Checkpoint resume logic ---
            json_file_path = os.path.join(subset_dir, f"{split_name}.json")
            if os.path.exists(json_file_path):
                print(f"  File '{json_file_path}' already exists, skipping this split.")
                continue
            # --- End of check ---

            print(f"  Processing split: '{split_name}' ({len(split_data)} rows)")

            # Find all image feature columns in this split
            image_keys = [key for key, feature in split_data.features.items() if isinstance(feature, HFImage)]

            images_dir = None
            if image_keys:
                images_dir = os.path.join(subset_dir, "images")
                os.makedirs(images_dir, exist_ok=True)
                print(f"  Found image columns: {image_keys}. Images will be saved to: '{images_dir}'")

            records_to_save = []

            # 6. Iterate through each row in the split
            for index, item in enumerate(item for item in split_data):
                record = {}
                for key, value in item.items():
                    # Check if current key is an image column and value is not empty
                    if key in image_keys and value:
                        image_filename = f"{current_subset_name}_{split_name}_{index}_{key}_{uuid.uuid4()}.png"
                        image_path_absolute = os.path.join(images_dir, image_filename)

                        try:
                            value.save(image_path_absolute)
                            record[key] = os.path.join("images", image_filename).replace("\\", "/")
                        except Exception as e:
                            print(f"    Warning: Cannot save image index:{index} (column: {key}): {e}")
                            record[key] = None
                    else:
                        record[key] = value

                records_to_save.append(record)

            # 7. Write the processed data list to a JSON file
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(records_to_save, f, ensure_ascii=False, indent=4)

            print(f"  Split '{split_name}' successfully saved to: '{json_file_path}'")

    print("-" * 50)
    print("All requested subset processing completed!")


if __name__ == '__main__':
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Download and process Hugging Face datasets')
    parser.add_argument('--dataset', type=str, default="TrustGen/Trustgen_dataset",
                        help='Dataset name o Hugging Face Hub (default: TrustGen/Trustgen_dataset)')
    parser.add_argument('--output_dir', type=str, default="./",
                        help='Output directory for the processed dataset (default: dataset_output)')
    parser.add_argument('--subset', type=str, default="all",
                        help='Specific subset to download, or "all" to download all subsets (default: all)')

    args = parser.parse_args()

    # Execute the function with command line arguments
    save_dataset_to_folders(args.dataset, args.output_dir, args.subset)