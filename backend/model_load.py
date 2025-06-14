import tensorflow as tf
import numpy as np
from urllib.request import urlopen
from PIL import Image
import io
import cv2
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

image = tf.keras.preprocessing.image

# Load trained models
try:
    male_model = tf.keras.models.load_model("model/hairfall_stage_male.h5")
    female_model = tf.keras.models.load_model("model/hairfall_stage_female.h5")
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise

# Class Labels
male_labels = ["non-scalp", "Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5", "Stage 6"]
female_labels = ["non-scalp", "Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5"]

def predict_stage(image_url, gender):
    if gender == "male":
        model = male_model
        class_labels = male_labels
    elif gender == "female":
        model = female_model
        class_labels = female_labels
    else:
        logger.error(f"Invalid gender: {gender}")
        return {"error": "Invalid gender selection. Choose 'male' or 'female'."}

    try:
        logger.info(f"Loading image from: {image_url}")
        with urlopen(image_url) as response:
            img_data = response.read()
        img = Image.open(io.BytesIO(img_data)).convert("RGB").resize((224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        logger.info("Image preprocessed successfully")
    except Exception as e:
        logger.error(f"Error loading image from URL: {str(e)}")
        return {"error": f"Error loading image from URL: {str(e)}"}

    try:
        predictions = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))
        logger.info(f"Prediction: {class_labels[predicted_class]}, Confidence: {confidence}")

        # Handle non-scalp class (index 0)
        if predicted_class == 0:
            logger.warning("Detected non-scalp image. Marking as invalid.")
            return {
                "error": "Invalid image! Please upload a scalp image.",
                "stage": class_labels[predicted_class],
                "confidence": confidence,
                "image_url": image_url
            }

        return {
            "stage": class_labels[predicted_class],
            "confidence": confidence,
            "image_url": image_url
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {"error": f"Prediction failed: {str(e)}"}

def analyze_hair_density(image_urls, timestamps):
    if len(image_urls) != 4 or len(timestamps) != 4:
        logger.error("Exactly 4 images and timestamps are required")
        return {"error": "Exactly 4 images and timestamps are required"}

    density_scores = []
    processed_urls = []

    try:
        for url in image_urls:
            try:
                logger.info(f"Loading image from: {url}")
                with urlopen(url) as response:
                    img_data = response.read()
                img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"Failed to decode image from {url}")
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
                hair_pixels = np.sum(thresh == 255)
                total_pixels = thresh.size
                density = (hair_pixels / total_pixels) * 100
                density_scores.append(density)
                processed_urls.append(url)
                logger.info(f"Density for {url}: {density:.2f}%")
            except Exception as e:
                logger.error(f"Error processing image {url}: {str(e)}")
                return {"error": f"Failed to process image {url}: {str(e)}"}

        if len(density_scores) != 4:
            logger.error("Not all images were processed successfully")
            return {"error": "Failed to process all images"}

        sorted_indices = np.argsort(timestamps)
        sorted_densities = [density_scores[i] for i in sorted_indices]
        sorted_urls = [processed_urls[i] for i in sorted_indices]

        baseline_density = sorted_densities[0]
        results = []
        for i in range(1, 4):
            current_density = sorted_densities[i]
            if baseline_density == 0:
                percentage_change = 0.0
            else:
                percentage_change = ((current_density - baseline_density) / baseline_density) * 100
            status = "Improved" if percentage_change > 0 else "Worsened" if percentage_change < -5 else "Stable"
            results.append({
                "week": i + 1,
                "percentageChange": round(percentage_change, 1),
                "status": status
            })

        avg_change = np.mean([r["percentageChange"] for r in results])
        overall_status = "Improved" if avg_change > 0 else "Worsened" if avg_change < -5 else "Stable"

        return {
            "results": results,
            "status": overall_status,
            "comparisonImageUrl": None  # Optional placeholder
        }
    except Exception as e:
        logger.error(f"Density analysis error: {str(e)}")
        return {"error": f"Density analysis failed: {str(e)}"}
