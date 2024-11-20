import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import DiffusionPipeline

# Step 1: Load an uploaded image
uploaded_image = "C:/Users/Signity_Laptop/Pictures/fashion/model2.jpg" # Replace with the path to your image

# Step 2: YOLOv5 for Clothing Detection
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # Load YOLOv5

# Perform detection
results = model(uploaded_image)
results.show()  # Show image with detected bounding boxes

# Extract detected items and bounding boxes
detected_objects = results.pandas().xyxy[0]  # Get bounding box data
print(detected_objects[['name', 'confidence']])

# Load the image for OpenCV operations
image = cv2.imread(uploaded_image)

# Step 3: Detect Colors for Each Detected Clothing Item
def detect_color(image, bounding_box):
    x_min, y_min, x_max, y_max = int(bounding_box[0]), int(bounding_box[1]), int(bounding_box[2]), int(bounding_box[3])
    cropped_image = image[y_min:y_max, x_min:x_max]  # Crop the detected clothing region
    
    # Convert cropped image to HSV color space
    hsv_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)

    # Define more extended color ranges
    color_ranges = {
        'red': ([0, 50, 50], [10, 255, 255]),
        'blue': ([100, 50, 50], [130, 255, 255]),
        'black': ([0, 0, 0], [180, 255, 30]),
        'white': ([0, 0, 200], [180, 30, 255]),
        'green': ([40, 50, 50], [80, 255, 255]),
        'yellow': ([20, 100, 100], [30, 255, 255]),
        'orange': ([10, 100, 100], [25, 255, 255]),
        'purple': ([130, 50, 50], [160, 255, 255]),
        'pink': ([160, 50, 50], [170, 255, 255])
    }

    # Detect color in the cropped image
    detected_colors = []
    for color_name, (lower, upper) in color_ranges.items():
        lower_bound = np.array(lower, dtype="uint8")
        upper_bound = np.array(upper, dtype="uint8")
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

        if cv2.countNonZero(mask) > 0:
            detected_colors.append(color_name)

    return detected_colors if detected_colors else ["Unknown"]

# Detect colors for all detected objects (clothing items)
clothing_items = []
for index, row in detected_objects.iterrows():
    label = row['name']
    bounding_box = row[['xmin', 'ymin', 'xmax', 'ymax']].values
    colors = detect_color(image, bounding_box)
    clothing_items.append({'label': label, 'colors': colors})

    print(f"{label} detected with color: {colors}")

# Step 4: Use Hugging Face "KappaNeuro/color-palette" Model to Suggest Color Combinations
# Initialize the diffusion pipeline for color palette suggestions
pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
pipeline.load_lora_weights("KappaNeuro/color-palette")

# Function to suggest palette-based clothing combinations
def suggest_palette_based_outfit(clothing_items):
    # Iterate over detected clothing items and get color combinations
    for item in clothing_items:
        for color in item['colors']:
            # Generate a color palette for the detected clothing color
            prompt = f"Generate a matching color palette for {color} clothing."
            try:
                palette = pipeline(prompt).images  # This returns a generated image representing the palette
                palette[0].show()  # Display the palette inline (for notebook environments)
                palette[0].save(f"{color}_palette.png")
                print(f"Suggested palette for {item['label']} ({color}): Check {color}_palette.png")
            except Exception as e:
                print(f"Error generating palette for {color}: {e}")

# Suggest color combinations based on palettes
suggest_palette_based_outfit(clothing_items)

# Step 5: Provide suggestions based on standard rules + palette guidance
def suggest_outfit(clothing_items):
    # Simple suggestion rules for outfit combinations
    suggestions = {
        ('blue', 'black'): 'Blue top looks great with black pants.',
        ('white', 'blue'): 'White top goes well with blue jeans.',
        ('black', 'white'): 'Black top and white pants make a classic outfit.',
        ('red', 'black'): 'Red top and black pants make a bold statement.',
        ('green', 'white'): 'Green top works nicely with white bottoms.',
        ('yellow', 'blue'): 'Yellow top looks vibrant with blue bottoms.',
        ('pink', 'black'): 'Pink top looks great with black pants.',
        ('orange', 'blue'): 'Orange top contrasts well with blue bottoms.',
        # Add more combinations as needed
    }

    # Loop through detected items and pair them to make suggestions
    for i in range(len(clothing_items)):
        for j in range(i + 1, len(clothing_items)):
            item1 = clothing_items[i]
            item2 = clothing_items[j]
            
            # Iterate over possible colors for each item
            for color1 in item1['colors']:
                for color2 in item2['colors']:
                    suggestion = suggestions.get((color1, color2)) or suggestions.get((color2, color1))
                    if suggestion:
                        print(f"Suggested outfit: {item1['label']} ({color1}) and {item2['label']} ({color2}) -> {suggestion}")
                    else:
                        print(f"No suggestion available for {item1['label']} ({color1}) and {item2['label']} ({color2}).")

# Call the suggestion function with detected items
suggest_outfit(clothing_items)