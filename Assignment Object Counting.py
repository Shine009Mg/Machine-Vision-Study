import cv2
import numpy as np
import os

# ==========================================
# OBJECT COUNTING USING OPENCV
# Techniques:
# 1. Thresholding
# 2. Morphological Operations
# 3. Contour Detection
# ==========================================

# List of 5 original images
image_files = [
    "test1.jpg",
    "test2.jpg",
    "test3.jpg",
    "test4.jpg",
    "test5.jpg"
]

# Minimum contour area
MIN_AREA = 500


def count_objects(image_path):

    # --------------------------------------
    # 1. Read image
    # --------------------------------------
    image = cv2.imread(image_path)

    if image is None:
        print("Cannot open:", image_path)
        return None

    result = image.copy()

    # --------------------------------------
    # 2. Convert to grayscale
    # --------------------------------------
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --------------------------------------
    # 3. Thresholding
    # --------------------------------------
    # OTSU automatically determines threshold
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # --------------------------------------
    # 4. Morphological Operations
    # --------------------------------------
    kernel = np.ones((5, 5), np.uint8)

    # Remove small noise
    opening = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    # Fill small holes/gaps
    morph = cv2.morphologyEx(
        opening,
        cv2.MORPH_CLOSE,
        kernel
    )

    # --------------------------------------
    # 5. Contour Detection
    # --------------------------------------
    contours, hierarchy = cv2.findContours(
        morph,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # --------------------------------------
    # 6. Count objects
    # --------------------------------------
    object_count = 0

    for contour in contours:

        area = cv2.contourArea(contour)

        # Ignore small noise
        if area > MIN_AREA:

            object_count += 1

            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Draw bounding box
            cv2.rectangle(
                result,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Object number
            cv2.putText(
                result,
                f"Object {object_count}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # --------------------------------------
    # 7. Display total number
    # --------------------------------------
    cv2.putText(
        result,
        f"Total Objects: {object_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3
    )

    return image, gray, binary, morph, result, object_count


# ==========================================
# Process all 5 test images
# ==========================================

correct_count = 0
total_images = 0

for image_file in image_files:

    output = count_objects(image_file)

    if output is None:
        continue

    image, gray, binary, morph, result, count = output

    total_images += 1

    print("--------------------------------")
    print("Image:", image_file)
    print("Detected objects:", count)

    # Display results
    cv2.imshow("Original - " + image_file, image)
    cv2.imshow("Threshold - " + image_file, binary)
    cv2.imshow("Morphological Result - " + image_file, morph)
    cv2.imshow("Final Detection - " + image_file, result)

    # Save final result
    output_name = "result_" + image_file
    cv2.imwrite(output_name, result)

    print("Saved:", output_name)

    print("Press any key for next image...")
    cv2.waitKey(0)

    # Close windows
    cv2.destroyAllWindows()


print("\n================================")
print("Testing Completed")
print("Images processed:", total_images)
print("================================")