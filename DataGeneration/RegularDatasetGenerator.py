import os
import json
import random
import torch
from datetime import datetime

from PIL import Image, ImageDraw
def extract_bounding_boxes(json_data, image_width, image_height):
    bounding_boxes = []

    if "plot-bb" in json_data:
        bounding_box_info = json_data["plot-bb"]
        normalized_box = {
            "height": bounding_box_info["height"] / image_height,
            "width": bounding_box_info["width"] / image_width,
            "x0": bounding_box_info["x0"] / image_width,
            "y0": bounding_box_info["y0"] / image_height
        }
        bounding_boxes.append(normalized_box)
    return bounding_boxes


def create_txt_file_image_size(size_dir, image_dir, image_size_dir, annotation_filename):
    # Extract the shared name without extension
    shared_name = os.path.splitext(annotation_filename)[0]

    image_path = os.path.join(image_dir, f"{shared_name}.jpg")  # Use the correct extension '.jpg' here
    image_size_path = os.path.join(image_size_dir, f"{shared_name}.txt")

    with open(image_size_path, 'r') as size_file:
        width, height = map(int, size_file.readline().split())

    txt_file_path = os.path.join(size_dir, annotation_filename.replace(".json", ".txt"))

    with open(txt_file_path, 'w') as txt_file:
        txt_file.write(f"{width} {height}\n")

    return width, height


def create_labels_file(size_dir, annotation_filename, bounding_boxes):
    lable_file_path = os.path.join(size_dir, annotation_filename.replace(".json", ".txt"))
    with open(lable_file_path, 'w') as txt_file:
        for box_info in bounding_boxes:
            height = box_info["height"]
            width = box_info["width"]
            x0 = box_info["x0"]
            y0 = box_info["y0"]

            txt_file.write(f"0 {x0} {y0} {width} {height}\n")


#for debug
def draw_rectangles(image_path, bounding_boxes):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    for box_info in bounding_boxes:
        height = box_info["height"] * image.size[1]
        width = box_info["width"] * image.size[0]
        x0 = box_info["x0"] * image.size[0]
        y0 = box_info["y0"] * image.size[1]
        x1 = x0 + width
        y1 = y0 + height
        draw.rectangle((x0, y0, x1, y1), outline="red", width=2)

        #for debug - showing the actual x and y on the screen
      # # Annotate with coordinates at each corner of the rectangle
      #   draw.text((x0, y0), f"({x0:.2f}, {y0:.2f})", fill="red", anchor="lb")  # bottom-left corner
      #   draw.text((x1, y0), f"({x1:.2f}, {y0:.2f})", fill="red", anchor="rb")  # bottom-right corner
      #   draw.text((x0, y1), f"({x0:.2f}, {y1:.2f})", fill="red", anchor="lt")  # top-left corner
      #   draw.text((x1, y1), f"({x1:.2f}, {y1:.2f})", fill="red", anchor="rt")  # top-right corner

    image.show()

def process_image_and_annotation(image_dir, annotation_dir, size_dir, image_size_dir, annotation_filename):
    image_path = os.path.join(image_dir, annotation_filename.replace(".json", ".jpg"))
    annotation_path = os.path.join(annotation_dir, annotation_filename)
    if os.path.exists(image_path) and os.path.exists(annotation_path):
        with open(annotation_path, 'r') as json_file:
            json_data = json.load(json_file)

        image_width, image_height = create_txt_file_image_size(size_dir, image_dir, image_size_dir, annotation_filename)
        bounding_boxes = extract_bounding_boxes(json_data, image_width, image_height)
        create_labels_file(size_dir, annotation_filename, bounding_boxes)

        #for debug
        # draw_rectangles(image_path, bounding_boxes)


def create_random_image(image_size, num_images, *image_dirs):
    image = Image.new("RGB", image_size, "white")
    image_files = []
    for image_dir in image_dirs:
        image_files.extend([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
    # Separate graph images
    graph_images = [img for img in image_files if image_dirs[1] in img]
    random.shuffle(graph_images)
    # define the size of the subset (between 0 and 8)
    subset_size = random.randint(0, min(8, len(graph_images)))
    subset_of_graph_images = graph_images[:subset_size]
    # Shuffle the image files list to randomize their order
    random.shuffle(image_files)
    # List to store positions for each image
    positions = []
    # Place random images on the canvas
    random.shuffle(image_files)

    labels=[]
    i = 0  # Start with the first image
    while i < subset_size:
        img = Image.open(subset_of_graph_images[i])
        if random.choice([True, False]):
            random_size = (random.randint(200, 400), random.randint(200, 400))
        else:
            random_size = (random.randint(50, 200), random.randint(50, 200))
        img.thumbnail(random_size)
        # Random position on the canvas (ensuring no overlap)
        x = random.randint(0, image_size[0] - random_size[0])
        y = random.randint(0, image_size[1] - random_size[1])
        new_position = (x, y, x + random_size[0], y + random_size[1])
        w = random_size[0]
        h = random_size[1]
        x_center=x + w/2
        y_center=y+ h/2
        # If there is an overlap, skip to the next image
        if any(intersect(new_position, pos) for pos in positions):
            continue


        positions.append(new_position)

        # Paste the image onto the canvas
        image.paste(img, (x, y))

        ##draw label - FOR DEBUG
        # draw = ImageDraw.Draw(image)
        # draw.rectangle((x, y, (x + img.width), (y + img.height)), outline="red", width=2)

        label = (x/image_size[0], y/image_size[1], (img.width)/image_size[0], (img.height)/image_size[1])
        labels.append(label)
        i += 1



    return image, labels


#dont draw overlapping graphs
def intersect(pos1, pos2):
    # Check if two positions (rectangles) intersect
    return not (pos1[2] <= pos2[0] or pos1[0] >= pos2[2] or pos1[3] <= pos2[1] or pos1[1] >= pos2[3])

#for faster rcnn
def convert_labels_to_coco(labels, width, height):
    boxes = []
    if labels:
        for label in labels:
            class_id, x_center, y_center, w, h = label
            # Convert from normalized center-width-height to absolute x_min, y_min
            x_min = (x_center - (w / 2)) * width
            y_min = (y_center - (h / 2)) * height
            w = w * width
            h = h * height



            boxes.append([x_min, y_min, w, h])
    return boxes


# was for debug
def draw_coco_rectangles(image, coco_boxes):
    draw = ImageDraw.Draw(image)

    for box in coco_boxes:
        x_min, y_min, w, h = box
        x_max = x_min + w
        y_max = y_min + h
        draw.rectangle((x_min, y_min, x_max, y_max), outline="blue", width=2)

    image.show()


#for debug - checking boxes match the graphs
def draw_yolo_rectangles(image, labels, width, height):
    draw = ImageDraw.Draw(image)
    for label in labels:
        class_id, x_center, y_center, w, h = label
        # Convert from normalized center-width-height to absolute x_min, y_min
        x_min = (x_center * width) - (w * width / 2)
        y_min = (y_center * height) - (h * height / 2)
        x_max = x_min + (w * width)
        y_max = y_min + (h * height)

        # Ensure coordinates are within image bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(width, x_max)
        y_max = min(height, y_max)

        draw.rectangle((x_min, y_min, x_max, y_max), outline="blue", width=2)
    image.show()

#translate txt file to bounding boxes
def read_yolo_labels(file_path):
    labels = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:  # class_id, x_center, y_center, width, height
                labels.append([int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
    return labels

def process_images_and_labels(images_dir, labels_dir):
    for image_filename in os.listdir(images_dir):
        if image_filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(images_dir, image_filename)
            label_path = os.path.join(labels_dir, os.path.splitext(image_filename)[0] + '.txt')
            if os.path.exists(label_path):
                image = Image.open(image_path)
                width, height = image.size
                print("w: " + str(width) + "h: " + str(height))
                yolo_boxes = read_yolo_labels(label_path)
                draw_yolo_rectangles(image, yolo_boxes,width,height)
                # coco_boxes = convert_labels_to_coco(labels, width, height)
                # draw_coco_rectangles(image, coco_boxes)
            else:
                print(f"Label file for {image_filename} not found in {labels_dir}")


def main():
    #pathes


    image_dir = "path/to/images"



    #extracted to yolo format
    labels_dir = "path/to/labels"
    random_created="path/to/output_random_images"


    # for annotation_filename in os.listdir(annotation_dir):
    #     if annotation_filename.endswith(".json"):
    #         process_image_and_annotation(image_dir, annotation_dir, txt_dir, image_size_dir, annotation_filename)

    # Image size
    image_width = 640
    image_height = 640

    # Number of images
    num_images = 7  # Adjust the number of images you want to include

    # Get absolute paths for image directories
    script_directory = os.path.dirname(os.path.abspath(__file__))
    image_dir1 = image_dir
    image_dir2 = os.path.join(script_directory, "random_pictures")


    #set range to the number of random images you want to create
    num_of_images_to_create = 100
    for i in range(num_of_images_to_create):
        # Create a random image with pasted images from multiple directories
        random_image, labels = create_random_image((image_width, image_height), num_images, image_dir2, image_dir1)

        # Display the random image with rectangles drawn around the graphs for debug
        # random_image.show()

        # coco_boxes = convert_labels_to_coco(labels, image_width, image_height)
        # draw_coco_rectangles(random_image, coco_boxes)

        process_images_and_labels(random_created, labels_dir)

        # # Save the random image with a unique name based on current time and date in an output file
        # images name are always unique for matching the labels
        current_time = datetime.now().strftime("%Y%m%d%H%M%S%f")
        random_image_filename = f"random_image_{current_time}.jpg"
        random_image_path = os.path.join(random_created, random_image_filename)
        random_image.save(random_image_path)


        # #save labels in an output file
        txt_path = os.path.join(labels_dir, f"{os.path.splitext(random_image_filename)[0]}.txt")
        with open(txt_path, 'a') as txt_file:
            for label in labels:
                x_center = label[0] + label[2] / 2
                y_center = label[1] + label[3] / 2
                txt_file.write(f"0 {x_center} {y_center} {label[2]} {label[3]}\n")

if __name__ == "__main__":
    main()
