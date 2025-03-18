import os
import uuid
import random
from PIL import Image, ImageDraw, ImageFont
from faker import Faker
import albumentations as A
import cv2
import numpy as np
import json

fake = Faker('en_US')

TEMPLATE_PATH = "images/passport-1-mod.jpg"
OUTPUT_DIR = "synthetic_passports"
FONT_PATH = "images/Arial.ttf"
JSONL_PATH = "synthetic_passports/passports_dataset.jsonl"

os.makedirs(OUTPUT_DIR, exist_ok=True)

coords = {
    "Passport Number": (556, 577),
    "Surname": (289, 608),
    "Given Names": (290, 646),
    "Date of Birth": (295, 719),
    "Sex": (641, 758),
    "Place of Birth": (296, 757),
    "Issue Date": (298, 795),
    "Expiration Date": (300, 832),
}

font = ImageFont.truetype(FONT_PATH, 24)

# Albumentations transform (minor augmentation)
transform = A.Compose([
    A.Rotate(limit=2, p=0.4),
    A.RandomBrightnessContrast(p=0.3),
    A.MotionBlur(blur_limit=3, p=0.2),
])


def generate_passport_data():
    return {
        "Passport Number": fake.bothify("#########"),
        "Surname": fake.last_name().upper(),
        "Given Names": f"{fake.first_name()} {fake.first_name()}".upper(),
        "Date of Birth": fake.date_of_birth(minimum_age=18, maximum_age=70).strftime("%d %b %Y").upper(),
        "Sex": random.choice(["M", "F"]),
        "Place of Birth": f"{fake.city().upper()}, USA",
        "Issue Date": fake.date_between(start_date="-8y", end_date="-1y").strftime("%d %b %Y").upper(),
        "Expiration Date": fake.date_between(start_date="+1y", end_date="+9y").strftime("%d %b %Y").upper(),
    }


def create_passport_image(data):
    # Open and copy template
    img = Image.open(TEMPLATE_PATH).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Overlay the text onto the image
    for field, position in coords.items():
        draw.text(position, data[field], font=font, fill="black")

    # Convert PIL image to OpenCV format for augmentation
    cv_img = np.array(img)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

    # Apply Albumentations augmentation
    augmented = transform(image=cv_img)["image"]

    # Generate filename
    filename = f"passport_{uuid.uuid4()}.jpg"
    image_path = os.path.join(OUTPUT_DIR, filename)

    # Save the augmented image
    cv2.imwrite(image_path, augmented)

    return filename, data


if __name__ == "__main__":
    total_images = 1000
    with open(JSONL_PATH, "w") as jsonl_file:
        for i in range(total_images):
            data = generate_passport_data()
            filename, passport_data = create_passport_image(data)

            json_line = {
                "image": os.path.join(OUTPUT_DIR, filename),
                "prompt": "Extract Passport Number, Surname, Given Names, DOB, Expiration Date, Issue Date, Place of Birth, Sex, Nationality",
                "response": {
                    "Passport Number": passport_data["Passport Number"],
                    "Surname": passport_data["Surname"],
                    "Given Names": passport_data["Given Names"],
                    "Date of Birth": passport_data["Date of Birth"],
                    "Place of Birth": passport_data["Place of Birth"],
                    "Issue Date": passport_data["Issue Date"],
                    "Expiration Date": passport_data["Expiration Date"],
                    "Sex": passport_data["Sex"],
                    "Nationality": "USA"
                }
            }

            jsonl_file.write(json.dumps(json_line) + "\n")

            print(f"[{i+1}/{total_images}] Generated {filename} with ground truth.")