import face_recognition
import os
import shutil
from collections import defaultdict



# === CONFIG ===
KNOWN_DIR = "known_people"
UNSORTED_DIR = r"F:\My phthos\2025"
OUTPUT_DIR = r"F:\My phthos\sorted images"

def load_known_faces():
    name_to_encodings = defaultdict(list)

    for person_name in os.listdir(KNOWN_DIR):
        person_path = os.path.join(KNOWN_DIR, person_name)
        if not os.path.isdir(person_path):
            continue

        for image_file in os.listdir(person_path):
            image_path = os.path.join(person_path, image_file)
            try:
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    name_to_encodings[person_name].append(encodings[0])
                    print(f"✅ Loaded encoding for {person_name} from {image_file}")
            except Exception as e:
                print(f"❌ Failed to load {image_path}: {e}")

    # Flatten into a list of (encoding, name) pairs
    known_encodings = []
    known_names = []
    for name, enc_list in name_to_encodings.items():
        for enc in enc_list:
            known_encodings.append(enc)
            known_names.append(name)

    return known_encodings, known_names

def sort_images(known_encodings, known_names):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for image_name in os.listdir(UNSORTED_DIR):
        image_path = os.path.join(UNSORTED_DIR, image_name)
        try:
            image = face_recognition.load_image_file(image_path)
        except Exception as e:
            print(f"❌ Cannot load {image_path}: {e}")
            continue

        face_encodings = face_recognition.face_encodings(image)

        matches_found = set()

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
            for idx, is_match in enumerate(matches):
                if is_match:
                    matches_found.add(known_names[idx])

        # Determine folder based on number of matched faces
        if len(matches_found) == 0:
            target_folder = os.path.join(OUTPUT_DIR, "unknown")
        elif len(matches_found) == 1:
            person_name = list(matches_found)[0]
            target_folder = os.path.join(OUTPUT_DIR, person_name)
        else:
            # Sort alphabetically to ensure folder names are consistent
            group_name = "_".join(sorted(matches_found))
            target_folder = os.path.join(OUTPUT_DIR, group_name)

        os.makedirs(target_folder, exist_ok=True)
        shutil.copy(image_path, os.path.join(target_folder, image_name))
        print(f"✅ Sorted '{image_name}' → {os.path.basename(target_folder)}")

if __name__ == "__main__":
    print("🔍 Loading known faces...")
    known_encodings, known_names = load_known_faces()

    print(f"✅ Loaded encodings for {len(set(known_names))} known people.")
    print("📂 Sorting unsorted images...")
    sort_images(known_encodings, known_names)
    print("🎉 Done!")
