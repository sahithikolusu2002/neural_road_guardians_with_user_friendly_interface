import cv2 as cv
import os
import time
import geocoder
import csv
import random
import torch
import numpy as np

# Load the MiDaS model only once
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")

# Add this variable at the beginning of the script
frame_skip_factor = 6  # Adjust the value as needed to skip frames


def create_csv_file(csv_filename):
    # Create a CSV file if it doesn't exist and write the header row
    with open(csv_filename, mode="w", newline="") as csv_file:
        fieldnames = [
            "Pothole ID",
            "X-coordinate",
            "Y-coordinate",
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


def save_to_csv(data_list, csv_filename, video_name):
    # Write the pothole data to the CSV file
    with open(csv_filename, mode="a", newline="") as csv_file:
        fieldnames = [
            "Pothole ID",
            "X-coordinate",
            "Y-coordinate",
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
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        for pothole_data in data_list:
            pothole_data["Pothole ID"] = video_name
            writer.writerow(pothole_data)


def detectPotholeonVideo(filename):
    # Create an empty set to store unique areas
    unique_areas = set()

    # Create an empty list to store pothole data
    pothole_data_list = []

    # reading label name from obj.names file
    class_name = []
    with open(os.path.join("project_files", "obj.names"), "r") as f:
        class_name = [cname.strip() for cname in f.readlines()]

    # importing model weights and config file
    # defining the model parameters
    net1 = cv.dnn.readNet(
        "project_files/yolov4_tiny.weights", "project_files/yolov4_tiny.cfg"
    )
    net1.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net1.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
    model1 = cv.dnn_DetectionModel(net1)
    model1.setInputParams(size=(640, 480), scale=1 / 255, swapRB=True)

    # defining initial values for some parameters in the script
    g = geocoder.ip("me")
    result_path = "pothole_coordinates"
    starting_time = time.time()
    Conf_threshold = 0.5
    NMS_threshold = 0.4
    frame_counter = 0
    i = 0
    b = 0

    # Specify the CSV file name
    csv_filename = "./templates/pothole_data.csv"

    # Get the video name without the extension
    video_name = (
        "live_camera_feed"
        if filename == "live"
        else os.path.splitext(os.path.basename(filename))[0]
    )

    # Create a CSV file (if it doesn't exist)
    create_csv_file(csv_filename)

    # Video capture for live feed
    cap = cv.VideoCapture(0) if filename == "live" else cv.VideoCapture(filename)

    # Video writer only for non-live video
    width = cap.get(3)
    height = cap.get(4)
    result = None

    if filename != "live":
        result = cv.VideoWriter(
            "PotholeVideoResult.mp4",
            cv.VideoWriter_fourcc(*"MP4V"),
            10,
            (int(width), int(height)),
        )

    # Process the entire video
    while True:
        try:
            ret, frame = cap.read()
            frame_counter += 1
            if ret is False:
                break

            if frame_counter % frame_skip_factor != 0:
                continue  # Skip frames if they don't meet the skip factor condition

            # Analysis the stream with the detection model
            classes, scores, boxes = model1.detect(frame, Conf_threshold, NMS_threshold)

            # Clear the set for each new frame to track unique areas
            unique_areas.clear()
            for classid, score, box in zip(classes, scores, boxes):
                label = "Pothole"
                x, y, w, h = box
                area = w * h  # Calculate the area of the bounding box
                width_pothole = w
                height_pothole = h

                # Ensure the detected object is a pothole and meets area criteria
                if score >= 0.95 and (area / (width * height)) <= 0.1 and y < 600:
                    x_coordinate = x
                    y_coordinate = y

                    # Mark potholes with rectangles on the frame
                    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    confidence_text = f"Confidence: {score:.2f}"
                    cv.putText(
                        frame,
                        confidence_text,
                        (x, y - 10),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                        cv.LINE_AA,
                    )

                    # Check if the area is unique within this frame
                    if area not in unique_areas:
                        unique_areas.add(area)

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
                                "Pothole ID": video_name,
                                "X-coordinate": x_coordinate,
                                "Y-coordinate": y_coordinate,
                                "Confidence Score": score,
                                "Area": area,
                                "Width": width_pothole,
                                "Height": height_pothole,
                                "Area_Classification": area_classification,
                                "Depth_mono": depth_mm,
                                "Pixel Distance": pixel_distance,
                                "Scale Factor": scale_factor,
                                "Depth_Classification": depth_classification,
                                "Actual_Depth": actual_depth_mm,
                            }
                            pothole_data_list.append(pothole_data)

            # Writing FPS on frame
            endingTime = time.time() - starting_time
            fps = frame_counter / endingTime
            cv.putText(
                frame,
                f"FPS: {fps}",
                (20, 50),
                cv.FONT_HERSHEY_COMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            # Showing and saving result
            cv.imshow("Pothole Detection (Press Q to Close)", frame)
            if result is not None:
                result.write(frame)
            key = cv.waitKey(1)
            if key == ord("q"):
                break

        except Exception as e:
            print(f"An error occurred: {e}")
            break

    # Save the pothole data to the CSV file
    save_to_csv(pothole_data_list, csv_filename, video_name)

    # Shuffle the pothole_data_list
    random.shuffle(pothole_data_list)

    # Ensure the sample size is within the valid range
    sample_size = min(len(pothole_data_list), random.randint(1, 7))

    # Randomly select rows and save them to the CSV file
    selected_rows = random.sample(pothole_data_list, sample_size)
    save_to_csv(selected_rows, csv_filename, video_name)

    cap.release()
    if result is not None:
        result.release()
    cv.destroyAllWindows()
