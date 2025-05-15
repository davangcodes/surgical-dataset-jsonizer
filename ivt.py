import torch
import json
import base64
from io import BytesIO
import os
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

# ==================== Dataset Definition ====================

class CholecT50():
    """
    Manages dataset splitting for CholecT45 using predefined cross-validation folds.
    Converts numeric video indices to standardized format (e.g., VID01).
    """
    def __init__(self, dataset_dir, dataset_variant="cholect45-crossval", test_fold=1):
        self.dataset_dir = dataset_dir
        self.dataset_variant = dataset_variant
        self.test_fold = test_fold

        # Select the split for training, validation, and testing
        video_split = self.split_selector()
        test_videos = video_split[self.test_fold]
        train_videos = [vid for k, vids in video_split.items() if k != self.test_fold for vid in vids]

        val_videos = train_videos[-5:]  # Use last 5 videos as validation
        train_videos = train_videos[:-5]

        self.train_records = [f'VID{str(v).zfill(2)}' for v in train_videos]
        self.val_records = [f'VID{str(v).zfill(2)}' for v in val_videos]
        self.test_records = [f'VID{str(v).zfill(2)}' for v in test_videos]

    def split_selector(self):
        """Defines the official cross-validation splits."""
        return {
            'cholect45-crossval': {
                1: [79, 2, 51, 6, 25, 14, 66, 23, 50],
                2: [80, 32, 5, 15, 40, 47, 26, 48, 70],
                3: [31, 57, 36, 18, 52, 68, 10, 8, 73],
                4: [42, 29, 60, 27, 65, 75, 22, 49, 12],
                5: [78, 43, 62, 35, 74, 1, 56, 4, 13],
            }
        }[self.dataset_variant]

    def get_transform(self):
        """Preprocess images: Resize and convert to tensor."""
        return transforms.Compose([
            transforms.Resize((256, 448)),
            transforms.ToTensor(),
        ])

    def build_dataset(self, video_list):
        """Build dataset for each video in the split."""
        datasets = []
        for video in video_list:
            datasets.append(T50(
                os.path.join(self.dataset_dir, 'data', video),
                os.path.join(self.dataset_dir, 'instrument', f'{video}.txt'),
                os.path.join(self.dataset_dir, 'verb', f'{video}.txt'),
                os.path.join(self.dataset_dir, 'target', f'{video}.txt'),
                self.get_transform()
            ))
        return datasets

    def build(self):
        """Returns: train, val, and test datasets."""
        return self.build_dataset(self.train_records), \
               self.build_dataset(self.val_records), \
               self.build_dataset(self.test_records)


class T50():
    """
    Dataset class for a single video. Handles image loading and label parsing.
    """
    def __init__(self, img_dir, instrument_file, verb_file, target_file, transform=None):
        self.instrument_labels = np.loadtxt(instrument_file, dtype=np.int32, delimiter=',')
        self.verb_labels = np.loadtxt(verb_file, dtype=np.int32, delimiter=',')
        self.target_labels = np.loadtxt(target_file, dtype=np.int32, delimiter=',')
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.instrument_labels)

    def __getitem__(self, index):
        """Returns: transformed image, 3 label lists, image path."""
        instrument_label = self.instrument_labels[index, 1:]
        verb_label = self.verb_labels[index, 1:]
        target_label = self.target_labels[index, 1:]
        basename = f"{str(self.instrument_labels[index, 0]).zfill(6)}.png"
        img_path = os.path.join(self.img_dir, basename)
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, instrument_label.tolist(), verb_label.tolist(), target_label.tolist(), img_path

# ==================== Label Maps ====================

# Maps class indices to names for readability in output JSON
INSTRUMENT_LABEL_MAP = {
    0: "grasper", 1: "bipolar", 2: "hook", 3: "scissors",
    4: "clipper", 5: "irrigator"
}

VERB_LABEL_MAP = {
    0: "grasp", 1: "retract", 2: "dissect", 3: "coagulate",
    4: "clip", 5: "cut", 6: "aspirate", 7: "irrigate",
    8: "pack", 9: "null_verb"
}

TARGET_LABEL_MAP = {
    0: "gallbladder", 1: "cystic_plate", 2: "cystic_duct", 3: "cystic_artery",
    4: "cystic_pedicle", 5: "blood_vessel", 6: "fluid", 7: "abdominal_wall_cavity",
    8: "liver", 9: "adhesion", 10: "omentum", 11: "peritoneum",
    12: "gut", 13: "specimen_bag", 14: "null_target"
}

# ==================== Utility Functions ====================

def image_to_base64(image):
    """Converts a PIL image to base64-encoded PNG."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def labels_to_names(labels, label_map):
    """Returns comma-separated string of active label names."""
    return ", ".join([label_map[i] for i, val in enumerate(labels) if val == 1])

def extract_video_and_image_no(image_path):
    """Extracts video and image ID from full path."""
    parts = image_path.split("/")
    video_no = parts[-2]
    image_no = parts[-1].split(".")[0]
    return video_no, image_no

# ==================== JSON Creator ====================

def create_json(dataset_list, filename):
    """
    Iterates over dataset, converts each frame to base64 + label format,
    and writes the full list to a JSON file.
    """
    data_list = []

    for dataset in dataset_list:
        for index in tqdm(range(len(dataset)), desc=f"Processing {filename}"):
            img, inst_lbl, verb_lbl, target_lbl, img_path = dataset[index]
            img_pil = transforms.ToPILImage()(img)
            base64_image = image_to_base64(img_pil)

            instrument_labels = labels_to_names(inst_lbl, INSTRUMENT_LABEL_MAP)
            verb_labels = labels_to_names(verb_lbl, VERB_LABEL_MAP)
            target_labels = labels_to_names(target_lbl, TARGET_LABEL_MAP)
            video_no, image_no = extract_video_and_image_no(img_path)

            data_list.append({
                "video_no": video_no,
                "image_no": image_no,
                "image": base64_image,
                "instrument_labels": inst_lbl,
                "instrument": instrument_labels,
                "verb_labels": verb_lbl,
                "verb": verb_labels,
                "target_labels": target_lbl,
                "target": target_labels
            })

    with open(filename, "w") as f:
        json.dump(data_list, f, indent=4)

# ==================== Main Execution ====================

if __name__ == "__main__":
    # Change path to where CholecT45 is located on your system
    dataset_dir = "../../../../../../../../../mount/Data1/Davang/CholecT45"
    
    # Initialize dataset and build splits
    cholecT50 = CholecT50(dataset_dir)
    train_datasets, val_datasets, test_datasets = cholecT50.build()

    # Convert each split into JSON files
    create_json(train_datasets, "ivt_train.json")
    create_json(val_datasets, "ivt_val.json")
    create_json(test_datasets, "ivt_test.json")

    print("✅ JSON files saved: ivt_train.json, ivt_val.json, ivt_test.json")
