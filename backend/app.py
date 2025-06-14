from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from model_load import predict_stage, analyze_hair_density
from flask_cors import CORS
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime
import logging
import re
import os
from bson import ObjectId
from bson.errors import InvalidId

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Flask App Init
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000"]}})

# Cloudinary Configuration
try:
    cloudinary.config(
        cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
        api_key=os.getenv("CLOUDINARY_API_KEY"),
        api_secret=os.getenv("CLOUDINARY_API_SECRET"),
        secure=True
    )
    logger.info("Cloudinary configured successfully")
except Exception as e:
    logger.error(f"Error configuring Cloudinary: {str(e)}")
    raise Exception("Cloudinary configuration failed")

# MongoDB Atlas Configuration
mongo_client = None
db = None
predictions_collection = None
hair_tracking_collection = None
forum_posts_collection = None

def init_mongo():
    global mongo_client, db, predictions_collection, hair_tracking_collection, forum_posts_collection
    try:
        mongo_uri = "mongodb+srv://anvithakotian7:Sevanthi%4013@cluster0.d944hrr.mongodb.net/haircaredb?retryWrites=true&w=majority"
        database_name = "haircaredb"
        
        mongo_client = MongoClient(mongo_uri)
        mongo_client.admin.command("ping")
        db = mongo_client[database_name]
        predictions_collection = db["Predictions"]
        hair_tracking_collection = db["HairTracking"]
        forum_posts_collection = db["ForumPosts"]
        logger.info("MongoDB Atlas connected successfully")
    except Exception as e:
        logger.error(f"Error connecting to MongoDB Atlas: {str(e)}")
        raise

try:
    init_mongo()
except Exception as e:
    logger.error(f"Failed to initialize MongoDB: {str(e)}")
    exit(1)

def sanitize_folder_name(name):
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files or "gender" not in request.form or "user_id" not in request.form:
        return jsonify({"error": "Missing file, gender, or user_id"}), 400

    file = request.files["file"]
    gender = request.form["gender"].lower()
    user_id = request.form.get("user_id")

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    allowed_types = {"image/jpeg", "image/png"}
    if file.content_type not in allowed_types:
        return jsonify({"error": "Invalid file type. Only JPEG and PNG are allowed"}), 400

    sanitized_user_id = sanitize_folder_name(user_id)
    try:
        upload_result = cloudinary.uploader.upload(
            file,
            resource_type="image",
            folder=f"hairfall_predictions/{sanitized_user_id}"
        )
        image_url = upload_result.get("secure_url")
        logger.info(f"Uploaded image to Cloudinary: {image_url}")
    except Exception as e:
        logger.error(f"Cloudinary upload error: {str(e)}")
        return jsonify({"error": f"Cloudinary upload failed: {str(e)}"}), 500

    try:
        result = predict_stage(image_url, gender)
        logger.info(f"Prediction result: {result}")

        if predictions_collection is None:
            logger.error("MongoDB is not initialized")
            return jsonify({"error": "MongoDB is not initialized"}), 500

        try:
            import json
            answers = json.loads(request.form.get("answers", "{}"))
            if not isinstance(answers, dict):
                raise ValueError("Invalid answers format")
        except (ValueError, json.JSONDecodeError) as e:
            logger.error(f"Invalid questionnaire answers: {str(e)}")
            return jsonify({"error": f"Invalid questionnaire answers: {str(e)}"}), 400

        metadata = {
            "user_id": user_id,
            "image_url": image_url,
            "stage": result.get("stage"),
            "confidence": result.get("confidence"),
            "gender": gender,
            "timestamp": datetime.utcnow(),
            "questionnaire_answers": answers
        }
        predictions_collection.insert_one(metadata)
        logger.info(f"Metadata stored in MongoDB: {metadata}")

        return jsonify(result)
    except Exception as e:
        logger.error(f"Prediction or MongoDB error: {str(e)}")
        return jsonify({"error": f"Prediction or database error: {str(e)}"}), 500

@app.route("/predictions/<user_id>", methods=["GET"])
def get_predictions(user_id):
    try:
        if predictions_collection is None:
            logger.error("MongoDB is not initialized")
            return jsonify({"error": "MongoDB is not initialized"}), 500

        predictions = list(predictions_collection.find({"user_id": user_id}).sort("timestamp", -1))
        for pred in predictions:
            pred["_id"] = str(pred["_id"])
        logger.info(f"Retrieved {len(predictions)} predictions for user_id: {user_id}")
        return jsonify(predictions)
    except Exception as e:
        logger.error(f"Error retrieving predictions: {str(e)}")
        return jsonify({"error": f"Failed to retrieve predictions: {str(e)}"}), 500

@app.route("/analyze-hair-growth", methods=["POST"])
def analyze_hair_growth():
    if not all(f"image{i}" in request.files for i in range(4)) or "user_id" not in request.form:
        return jsonify({"error": "Missing images or user_id"}), 400

    user_id = request.form.get("user_id")
    sanitized_user_id = sanitize_folder_name(user_id)
    images = []
    timestamps = []

    try:
        for i in range(4):
            file = request.files[f"image{i}"]
            if file.filename == "":
                return jsonify({"error": f"No file selected for image{i}"}), 400

            upload_result = cloudinary.uploader.upload(
                file,
                resource_type="image",
                folder=f"hairfall_predictions/{sanitized_user_id}/tracking"
            )
            images.append(upload_result.get("secure_url"))
            timestamps.append(request.form.get(f"timestamp{i}", datetime.utcnow().isoformat()))
        logger.info(f"Uploaded {len(images)} images to Cloudinary for user_id: {user_id}")
    except Exception as e:
        logger.error(f"Cloudinary upload error: {str(e)}")
        return jsonify({"error": f"Cloudinary upload failed: {str(e)}"}), 500

    try:
        results = analyze_hair_density(images, timestamps)
        logger.info(f"Analysis results: {results}")

        if hair_tracking_collection is None:
            logger.error("MongoDB is not initialized")
            return jsonify({"error": "MongoDB is not initialized"}), 500

        metadata = {
            "user_id": user_id,
            "images": images,
            "timestamps": timestamps,
            "results": results["results"],
            "status": results["status"],
            "comparison_image_url": results.get("comparisonImageUrl"),
            "timestamp": datetime.utcnow()
        }
        hair_tracking_collection.insert_one(metadata)
        logger.info(f"Tracking metadata stored in MongoDB: {metadata}")

        return jsonify(results)
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

@app.route("/test-upload", methods=["POST"])
def test_upload():
    if "file" not in request.files or "user_id" not in request.form:
        return jsonify({"error": "No file or user_id provided"}), 400
    file = request.files["file"]
    user_id = request.form.get("user_id")

    sanitized_user_id = sanitize_folder_name(user_id)
    try:
        result = cloudinary.uploader.upload(
            file,
            resource_type="image",
            folder=f"hairfall_predictions/{sanitized_user_id}/tracking"
        )
        logger.info(f"Test upload successful: {result['secure_url']}")
        return jsonify({"url": result["secure_url"]})
    except Exception as e:
        logger.error(f"Test upload error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/forum/post", methods=["POST"])
def create_forum_post():
    if "user_id" not in request.form or "text" not in request.form:
        return jsonify({"error": "Missing user_id or text"}), 400

    user_id = request.form.get("user_id")
    text = request.form.get("text")
    username = request.form.get("username", "Anonymous")
    sanitized_user_id = sanitize_folder_name(user_id)
    image_url = None
    cloudinary_public_id = None

    if "image" in request.files:
        file = request.files["image"]
        if file.filename:
            try:
                upload_result = cloudinary.uploader.upload(
                    file,
                    resource_type="image",
                    folder=f"hairfall_predictions/{sanitized_user_id}/forum_images"
                )
                image_url = upload_result.get("secure_url")
                cloudinary_public_id = upload_result.get("public_id")
                logger.info(f"Uploaded forum image to Cloudinary: {image_url}")
            except Exception as e:
                logger.error(f"Cloudinary upload error: {str(e)}")
                return jsonify({"error": f"Cloudinary upload failed: {str(e)}"}), 500

    try:
        post = {
            "user_id": user_id,
            "username": username,
            "text": text,
            "image_url": image_url,
            "cloudinary_public_id": cloudinary_public_id,
            "likes": [],
            "comments": [],
            "timestamp": datetime.utcnow()
        }
        result = forum_posts_collection.insert_one(post)
        post["_id"] = str(result.inserted_id)
        logger.info(f"Forum post created: {post}")
        return jsonify(post)
    except Exception as e:
        logger.error(f"Error creating forum post: {str(e)}")
        return jsonify({"error": f"Database error: {str(e)}"}), 500

@app.route("/forum/posts", methods=["GET"])
def get_forum_posts():
    try:
        posts = list(forum_posts_collection.find().sort("timestamp", -1))
        for post in posts:
            post["_id"] = str(post["_id"])
            for comment in post.get("comments", []):
                if "_id" in comment:
                    comment["_id"] = str(comment["_id"])
                else:
                    logger.warning(f"Comment without _id in post {post['_id']}: {comment}")
        logger.info(f"Retrieved {len(posts)} forum posts")
        return jsonify(posts)
    except Exception as e:
        logger.error(f"Error retrieving forum posts: {str(e)}")
        return jsonify({"error": f"Database error: {str(e)}"}), 500

@app.route("/forum/comment", methods=["POST"])
def add_comment():
    data = request.get_json()
    if not data or not all(key in data for key in ["post_id", "user_id", "text"]):
        return jsonify({"error": "Missing post_id, user_id, or text"}), 400

    post_id = data["post_id"]
    user_id = data["user_id"]
    text = data["text"]
    username = data.get("username", "Anonymous")

    try:
        # Validate post_id
        try:
            post_object_id = ObjectId(post_id)
        except InvalidId:
            logger.error(f"Invalid post_id format: {post_id}")
            return jsonify({"error": "Invalid post_id format"}), 400

        comment = {
            "_id": ObjectId(),  # Generate a unique ID for the comment
            "user_id": user_id,
            "username": username,
            "text": text,
            "timestamp": datetime.utcnow()
        }
        result = forum_posts_collection.update_one(
            {"_id": post_object_id},
            {"$push": {"comments": comment}}
        )
        if result.modified_count == 0:
            logger.error(f"Post not found: {post_id}")
            return jsonify({"error": "Post not found"}), 404
        comment["_id"] = str(comment["_id"])
        logger.info(f"Comment added to post {post_id} by user {user_id}")
        return jsonify(comment)
    except Exception as e:
        logger.error(f"Error adding comment: {str(e)}")
        return jsonify({"error": f"Database error: {str(e)}"}), 500

@app.route("/forum/comment", methods=["DELETE"])
def delete_comment():
    data = request.get_json()
    if not data or not all(key in data for key in ["post_id", "comment_id", "user_id"]):
        return jsonify({"error": "Missing post_id, comment_id, or user_id"}), 400

    post_id = data["post_id"]
    comment_id = data["comment_id"]
    user_id = data["user_id"]

    try:
        # Validate post_id and comment_id
        try:
            post_object_id = ObjectId(post_id)
            comment_object_id = ObjectId(comment_id)
        except InvalidId as e:
            logger.error(f"Invalid ID format: post_id={post_id}, comment_id={comment_id}")
            return jsonify({"error": "Invalid post_id or comment_id format"}), 400

        # Find the post
        post = forum_posts_collection.find_one({"_id": post_object_id})
        if not post:
            logger.error(f"Post not found: {post_id}")
            return jsonify({"error": "Post not found"}), 404

        # Check if the comment exists and belongs to the user
        comment = next((c for c in post.get("comments", []) if str(c.get("_id")) == comment_id), None)
        if not comment:
            logger.error(f"Comment not found: {comment_id} in post {post_id}")
            return jsonify({"error": "Comment not found"}), 404
        if comment["user_id"] != user_id:
            logger.error(f"Unauthorized attempt to delete comment {comment_id} by user {user_id}")
            return jsonify({"error": "Unauthorized: You can only delete your own comments"}), 403

        # Remove the comment
        result = forum_posts_collection.update_one(
            {"_id": post_object_id},
            {"$pull": {"comments": {"_id": comment_object_id}}}
        )
        if result.modified_count == 0:
            logger.error(f"Failed to delete comment {comment_id} from post {post_id}")
            return jsonify({"error": "Failed to delete comment: No changes made"}), 500

        logger.info(f"Deleted comment {comment_id} from post {post_id} by user {user_id}")
        return jsonify({"message": "Comment deleted successfully", "comment_id": comment_id})
    except Exception as e:
        logger.error(f"Error deleting comment: post_id={post_id}, comment_id={comment_id}, error={str(e)}")
        return jsonify({"error": f"Database error: {str(e)}"}), 500

@app.route("/forum/like", methods=["POST"])
def toggle_like():
    data = request.get_json()
    if not data or not all(key in data for key in ["post_id", "user_id"]):
        return jsonify({"error": "Missing post_id or user_id"}), 400

    post_id = data["post_id"]
    user_id = data["user_id"]

    try:
        post_object_id = ObjectId(post_id)
        post = forum_posts_collection.find_one({"_id": post_object_id})
        if not post:
            return jsonify({"error": "Post not found"}), 404

        likes = post.get("likes", [])
        if user_id in likes:
            likes.remove(user_id)
        else:
            likes.append(user_id)

        forum_posts_collection.update_one(
            {"_id": post_object_id},
            {"$set": {"likes": likes}}
        )
        logger.info(f"Like toggled for post {post_id}")
        return jsonify({"likes": likes})
    except InvalidId:
        return jsonify({"error": "Invalid post_id format"}), 400
    except Exception as e:
        logger.error(f"Error toggling like: {str(e)}")
        return jsonify({"error": f"Database error: {str(e)}"}), 500

@app.route("/forum/post/<post_id>", methods=["DELETE"])
def delete_forum_post(post_id):
    data = request.get_json()
    if not data or "user_id" not in data:
        return jsonify({"error": "Missing user_id"}), 400

    user_id = data["user_id"]
    image_url = data.get("image_url")

    try:
        post_object_id = ObjectId(post_id)
        post = forum_posts_collection.find_one({"_id": post_object_id})
        if not post:
            return jsonify({"error": "Post not found"}), 404
        if post["user_id"] != user_id:
            return jsonify({"error": "Unauthorized: You can only delete your own posts"}), 403

        # Delete image from Cloudinary if it exists
        if image_url and post.get("cloudinary_public_id"):
            try:
                cloudinary.uploader.destroy(post["cloudinary_public_id"])
                logger.info(f"Deleted image from Cloudinary: {post['cloudinary_public_id']}")
            except Exception as e:
                logger.error(f"Cloudinary delete error: {str(e)}")
                # Continue with post deletion even if image deletion fails

        result = forum_posts_collection.delete_one({"_id": post_object_id})
        if result.deleted_count == 0:
            return jsonify({"error": "Failed to delete post"}), 500

        logger.info(f"Deleted post {post_id} by user {user_id}")
        return jsonify({"message": "Post deleted successfully"})
    except InvalidId:
        return jsonify({"error": "Invalid post_id format"}), 400
    except Exception as e:
        logger.error(f"Error deleting post: {str(e)}")
        return jsonify({"error": f"Database error: {str(e)}"}), 500

if __name__ == "__main__":
    required_env_vars = ["CLOUDINARY_CLOUD_NAME", "CLOUDINARY_API_KEY", "CLOUDINARY_API_SECRET"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing environment variables: {missing_vars}")
        exit(1)

    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=5000)