import os

ROOT_DIR = r"C:\Users\Ankit kumar\Downloads\model"

print("\nFolders found in root:")
for f in os.listdir(ROOT_DIR):
    print(f)
import os
import shutil
import yaml

ROOT_DIR = r"C:\Users\Ankit kumar\Downloads\model"
FINAL_DIR = os.path.join(ROOT_DIR, "VisionSafe_Dataset")

DATASETS = [
    "unsafe equipment use.v1i.yolov8",
    "sleeping.v7i.yolov8",
    "smoking.v1-smoker1.yolov8",
    "PPE DETECTION.v14i.yolov8",
    "Phone Call Usage.v1i.yolov8",
    "ppe-factory.v8i.yolov8",
    "Hard Worker.v1i.yolov8"
    "Phone usage.v3i.yolov8"   
]

FINAL_CLASSES = [
    "helmet",
    "vest",
    "gloves",
    "safety_boots",
    "ear_protection",
    "safety_glasses",
    "mask",
    "phone_usage",
    "smoking",
    "sleeping",
    "unsafe_equipment"
]

# Create folders
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(FINAL_DIR, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(FINAL_DIR, split, "labels"), exist_ok=True)

def remap_label(name):
    mapping = {
        "helmet": "helmet",
        "Vest": "vest",
        "gloves": "gloves",
        "boots": "safety_boots",
        "earmuffs": "ear_protection",
        "glasses": "safety_glasses",
        "Mask": "mask",
        "phone": "phone_usage",
        "phone-person": "phone_usage",
        "smoker": "smoking",
        "sleep": "sleeping",
        "sleeping": "sleeping",
        "sleepping": "sleeping",
        "unsafe equipment use": "unsafe_equipment"
    }
    return mapping.get(name, None)

for dataset in DATASETS:
    print(f"Processing {dataset}...")
    yaml_path = os.path.join(ROOT_DIR, dataset, "data.yaml")
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
        names = config["names"]

    for split in ["train", "valid", "test"]:
        if split not in config:
            continue

        src_img = os.path.join(ROOT_DIR, dataset, split, "images")
        src_lbl = os.path.join(ROOT_DIR, dataset, split, "labels")

        if not os.path.exists(src_img):
            continue

        dst_img = os.path.join(FINAL_DIR, split.replace("valid", "val"), "images")
        dst_lbl = os.path.join(FINAL_DIR, split.replace("valid", "val"), "labels")

        for img_file in os.listdir(src_img):
            shutil.copy(os.path.join(src_img, img_file), os.path.join(dst_img, img_file))

        for lbl_file in os.listdir(src_lbl):
            old_path = os.path.join(src_lbl, lbl_file)
            with open(old_path, "r") as lf:
                lines = lf.readlines()

            new_lines = []
            for ln in lines:
                cls, *box = ln.split()
                cls = int(cls)
                new_name = remap_label(names[cls])
                if new_name is None: continue
                new_index = FINAL_CLASSES.index(new_name)
                new_lines.append(" ".join([str(new_index)] + box) + "\n")

            with open(os.path.join(dst_lbl, lbl_file), "w") as lf:
                lf.writelines(new_lines)

print("\n✔ Dataset merged successfully!")
print("Final path:", FINAL_DIR)
