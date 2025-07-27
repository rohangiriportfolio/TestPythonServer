from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.cluster import DBSCAN
from itertools import combinations
import numpy as np
from collections import defaultdict
import requests
import time
import os



app = Flask(__name__)
CORS(app)

# Replace with your real OpenRouteService API key
ORS_API_KEY = 'eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjNlZmJkMWU4ZTkwZDRjNTFiNDJkNGY0NjEzYzViOWZhIiwiaCI6Im11cm11cjY0In0='

@app.route('/', methods=['GET'])
def home():
    return "Python clustering server with OpenRouteService is running!", 200

def get_ors_distance(coord1, coord2):
    url = 'https://api.openrouteservice.org/v2/directions/driving-car'
    headers = {
        'Authorization': ORS_API_KEY,
        'Content-Type': 'application/json'
    }
    body = {
        "coordinates": [
            [coord1[1], coord1[0]],
            [coord2[1], coord2[0]]
        ],
        "units": "km"
    }

    try:
        response = requests.post(url, json=body, headers=headers)
        response.raise_for_status()
        data = response.json()
        distance_km = data['routes'][0]['summary']['distance']
        return distance_km
    except Exception as e:
        print(f"⚠️ Error fetching ORS distance between {coord1} and {coord2}: {e}")
        return 9999

def run_clustering(coordinates, radius_km=10.0):
    n = len(coordinates)
    distance_matrix = np.zeros((n, n))

    print("\n🚗 Calculating road distances using OpenRouteService...")

    for i, j in combinations(range(n), 2):
        distance_km = get_ors_distance(coordinates[i], coordinates[j])
        print(f"✓ Distance between {coordinates[i]} and {coordinates[j]}: {distance_km:.2f} km")
        distance_matrix[i][j] = distance_km
        distance_matrix[j][i] = distance_km
        time.sleep(0.1)

    db = DBSCAN(eps=radius_km, min_samples=1, metric='precomputed')
    labels = db.fit_predict(distance_matrix)

    groups = defaultdict(list)
    for label, coord in zip(labels, coordinates):
        groups[label].append(coord)

    result = {
        'clusters': labels.tolist(),
        'total_clusters': len(set(labels)),
        'noise_points': labels.tolist().count(-1),
        'grouped_coordinates': {f"group_{k+1}": v for k, v in groups.items()}
    }

    return result

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        coordinates = data.get('coordinates', [])
        radius_km = data.get('radius', 10.0)

        if not coordinates or not isinstance(coordinates, list):
            return jsonify({'error': 'No valid coordinates provided'}), 400

        coordinates = [(float(lat), float(lon)) for lat, lon in coordinates]
        result = run_clustering(coordinates, radius_km)

        return jsonify(result)
    except Exception as e:
        print("❌ Error:", e)
        return jsonify({'error': str(e)}), 500

# 🔧 Test route with hardcoded coordinates (for development/testing)
@app.route('/predict-test', methods=['GET'])
def predict_test():
    test_coords = [
        (28.6139, 77.2090),   # Delhi
        (28.5355, 77.3910),   # Noida
        (28.7041, 77.1025),   # Delhi
        (28.4595, 77.0266),   # Gurgaon
        (19.0760, 72.8777),   # Mumbai (should be far enough to form separate cluster)
        (19.2183, 72.9781)    # Navi Mumbai
    ]
    radius_km = 10.0
    result = run_clustering(test_coords, radius_km)
    return jsonify(result)

# if __name__ == '__main__':
#     app.run(port=5001, debug=True)
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Render sets this automatically
    app.run(host='0.0.0.0', port=port)
