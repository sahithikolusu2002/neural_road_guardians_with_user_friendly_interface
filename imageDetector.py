import cv2 as cv
import os
import csv
import random
import torch
import numpy as np
import exifread
from flask import Flask, render_template, request
import pytesseract
import re

# Path to Tesseract executable (update this with your installation path)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = Flask(__name__)

# Load the MiDaS model only once
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")


def extract_lat_lon_from_exif(image_filename):
    with open(image_filename, "rb") as f:
        tags = exifread.process_file(f)
        latitude = tags.get("GPS GPSLatitude")
        longitude = tags.get("GPS GPSLongitude")

        if latitude and longitude:
            lat_degrees, lat_minutes, lat_seconds = [
                float(x)
                for x in str(latitude).replace(" ", "").replace("/", "").split(",")
            ]
            lon_degrees, lon_minutes, lon_seconds = [
                float(x)
                for x in str(longitude).replace(" ", "").replace("/", "").split(",")
            ]

            # Convert degrees, minutes, seconds to decimal degrees
            latitude_decimal = lat_degrees + lat_minutes / 60 + lat_seconds / 3600
            longitude_decimal = lon_degrees + lon_minutes / 60 + lon_seconds / 3600

            return latitude_decimal, longitude_decimal
        else:
            return None


def create_csv_file(csv_filename):
    # Create a CSV file if it doesn't exist and write the header row
    with open(csv_filename, mode="w", newline="") as csv_file:
        fieldnames = [
            "Pothole ID",
            "X-coordinate",
            "Y-coordinate",
            "Latitude",
            "Longitude",
            "Confidence Score",
            "Area",
            "Width",
            "Height",
            "Area_Classification",
            "Depth_mono",
            "Pixel Distance",
            "Scale Factor",
            "Depth_Classification",
            "Actual_Depth",
            "Date",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()


def classify_area(area):
    if area < 10000:
        return "Low"
    elif 10000 <= area < 25000:
        return "Medium"
    else:
        return "High"


def classify_depth(actual_depth):
    if actual_depth <= 25:
        return "Shallow"
    elif 25 < actual_depth <= 50:
        return "Moderate"
    else:
        return "Deep"


def calculate_pixel_distance(centroid_x, centroid_y, x, y, w, h):
    # Calculate the pixel distance from the centroid to the right side of the rectangle
    side_x = x + w  # X-coordinate of the right side
    side_y = centroid_y  # Y-coordinate of the centroid

    pixel_distance = np.sqrt((centroid_x - side_x) ** 2 + (centroid_y - side_y) ** 2)

    return pixel_distance


def estimate_depth_for_potholes_in_frame(frame, pothole_bboxes):
    # Resize the frame to match the model's input size
    frame = cv.resize(frame, (384, 384))

    # Preprocess the frame for depth estimation
    frame = frame[:, :, :3]  # Remove alpha channel if present
    frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # Perform inference to get the depth map
    with torch.no_grad():
        depth_map = midas(frame)

    # List to store depth values for each pothole
    pothole_depths = []

    # Iterate over the bounding boxes for potholes
    for pothole_bbox in pothole_bboxes:
        x1, y1, x2, y2 = pothole_bbox

        # Calculate the average depth value for the pothole region
        pothole_depth = torch.mean(depth_map[0, y1:y2, x1:x2])

        # Append the depth value to the list
        pothole_depths.append(pothole_depth.item())

    # Convert depth values from meters to arbitrary units (Depth_mono), round to 1 decimal place
    pothole_depths_mono = ["{:.1f}".format(depth * 0.1) for depth in pothole_depths]

    return pothole_depths_mono


def extract_text_from_image(image_filename):
    # Read the image using OpenCV
    image = cv.imread(image_filename)

    # Convert the image to grayscale
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Apply any necessary preprocessing steps (e.g., resizing, thresholding, denoising)

    # Perform OCR on the preprocessed image
    text = pytesseract.image_to_string(gray_image)

    return text


def extract_area_lat_lon_from_text(extracted_text):
    # Extract area name, latitude, and longitude from the OCR text
    area_name_match = re.search(
        r"([\w\s]+),\s*([\w\s]+),\s*([\w\s]+)\s*Lat ([\d.]+),\s*Long ([\d.]+)",
        extracted_text,
    )
    latitude_match = re.search(r"Lat ([\d.]+)", extracted_text)
    longitude_match = re.search(r"Long ([\d.]+)", extracted_text)
    date_match = re.search(r"(\d{1,2}/\d{1,2}/\d{4})", extracted_text)

    area_name = area_name_match.group(1) if area_name_match else "Unknown Area"
    latitude = float(latitude_match.group(1)) if latitude_match else 0.0
    longitude = float(longitude_match.group(1)) if longitude_match else 0.0
    date = date_match.group(1) if date_match else "Unknown Date"
    return area_name, latitude, longitude, date


def detectPotholeonImage(image_filename):
    # Create an empty list to store pothole data
    pothole_data_list = []

    # Reading label names from obj.names file
    class_name = []
    with open(os.path.join("project_files", "obj.names"), "r") as f:
        class_name = [cname.strip() for cname in f.readlines()]

    # Importing model weights and config file
    net1 = cv.dnn.readNet(
        "project_files/yolov4_tiny.weights", "project_files/yolov4_tiny.cfg"
    )
    net1.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net1.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
    model1 = cv.dnn_DetectionModel(net1)
    model1.setInputParams(size=(640, 480), scale=1 / 255, swapRB=True)

    # Load the image
    frame = cv.imread(image_filename)

    # Process the image
    classes, scores, boxes = model1.detect(frame, confThreshold=0.1, nmsThreshold=0.4)

    # Get the base file name without extension
    file_name = os.path.splitext(os.path.basename(image_filename))[0]

    # Extract text from the image
    extracted_text = extract_text_from_image(image_filename)

    # Extract area name, latitude, and longitude from text
    area_name, latitude, longitude, date = extract_area_lat_lon_from_text(
        extracted_text
    )

    # Iterate over detected potholes
    for classid, score, box in zip(classes, scores, boxes):
        label = "Pothole"
        x, y, w, h = box
        area = w * h  # Calculate the area of the bounding box

        # Ensure the detected object is a pothole
        if score >= 0.1 and y < 600:
            x_coordinate = x
            y_coordinate = y

            # Classify the pothole based on area
            area_classification = classify_area(area)

            # Perform depth estimation for the pothole
            pothole_depth = estimate_depth_for_potholes_in_frame(
                frame, [(x, y, x + w, y + h)]
            )

            # Calculate the pixel distance from the centroid to the right side of the rectangle
            centroid_x = x + w // 2
            centroid_y = y + h // 2
            pixel_distance = calculate_pixel_distance(
                centroid_x, centroid_y, x, y, w, h
            )

            # Generate a random known_distance_mm between 25mm and 75mm
            known_distance_mm = random.uniform(25, 75)

            # Calculate the scale factor
            scale_factor = known_distance_mm / pixel_distance

            # Check if depth is NaN
            if not np.isnan(float(pothole_depth[0])):
                # Calculate the product of depth_mono and scale_factor
                depth_mm = float(pothole_depth[0])
                actual_depth_mm = depth_mm * scale_factor

                # Classify the pothole based on actual depth
                depth_classification = classify_depth(actual_depth_mm)

                # Append pothole data to the list
                pothole_data = {
                    "Pothole ID": area_name,  # Use area name as Pothole ID
                    "X-coordinate": x_coordinate,
                    "Y-coordinate": y_coordinate,
                    "Latitude": latitude,
                    "Longitude": longitude,
                    "Confidence Score": score,
                    "Area": area,
                    "Width": w,
                    "Height": h,
                    "Area_Classification": area_classification,
                    "Depth_mono": depth_mm,
                    "Pixel Distance": pixel_distance,
                    "Scale Factor": scale_factor,
                    "Depth_Classification": depth_classification,
                    "Actual_Depth": actual_depth_mm,
                    "Date": date,
                }
                pothole_data_list.append(pothole_data)

                # Draw bounding box on the frame
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Write the pothole data to the CSV file
    csv_filename = "./templates/pothole_data.csv"
    create_csv_file(csv_filename)
    save_to_csv(pothole_data_list, csv_filename)

    # Display the image with bounding boxes
    cv.imshow("Pothole Detection (Press any key to close)", frame)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return pothole_data_list


def save_to_csv(data_list, csv_filename):
    # Write the pothole data to the CSV file
    with open(csv_filename, mode="a", newline="") as csv_file:
        fieldnames = [
            "Pothole ID",
            "X-coordinate",
            "Y-coordinate",
            "Latitude",
            "Longitude",
            "Confidence Score",
            "Area",
            "Width",
            "Height",
            "Area_Classification",
            "Depth_mono",
            "Pixel Distance",
            "Scale Factor",
            "Depth_Classification",
            "Actual_Depth",
            "Date",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        for pothole_data in data_list:
            writer.writerow(pothole_data)


# Example usage:
# detectPotholeonImage('your_image_filename.jpg')

if __name__ == "__main__":
    app.run(debug=True)
