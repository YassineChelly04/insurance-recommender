"""
Flask routes for the insurance recommender API
"""
from flask import Blueprint, render_template, request, jsonify
from app.inference import predict_bundle

api_bp = Blueprint('api', __name__)

@api_bp.route('/')
def index():
    """Serve the frontend"""
    return render_template('index.html')

@api_bp.route('/api/predict', methods=['POST'])
def predict():
    """
    API endpoint for insurance bundle prediction.
    
    Expected JSON input:
    {
        "Adult_Dependents": 2,
        "Child_Dependents": 1,
        "Estimated_Annual_Income": 75000,
        ...
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        result = predict_bundle(data)
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500

@api_bp.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "Insurance Recommender API"}), 200

@api_bp.route('/api/bundle-info', methods=['GET'])
def bundle_info():
    """
    Return information about the 10 insurance bundles.
    """
    bundles = {
        0: {"name": "Basic Coverage", "description": "Essential protection for individuals"},
        1: {"name": "Standard Coverage", "description": "Balanced protection and value"},
        2: {"name": "Premium Coverage", "description": "Comprehensive protection with extras"},
        3: {"name": "Family Bundle", "description": "Protection for families"},
        4: {"name": "Family Plus", "description": "Enhanced family protection"},
        5: {"name": "Business Coverage", "description": "Commercial/business protection"},
        6: {"name": "Senior Coverage", "description": "Tailored for seniors"},
        7: {"name": "High-Value Coverage", "description": "Premium protection for high earners"},
        8: {"name": "Flex Bundle", "description": "Customizable insurance options"},
        9: {"name": "Elite Coverage", "description": "Top-tier comprehensive protection"},
    }
    return jsonify({"bundles": bundles}), 200
