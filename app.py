from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load and preprocess data
file_path = r'E:\datasets\recipe_final (1).csv'
recipe_df = pd.read_csv(file_path)

# Preprocessing
vectorizer = TfidfVectorizer()
X_ingredients = vectorizer.fit_transform(recipe_df['ingredients_list'])

# Normalize numerical features
scaler = StandardScaler()
X_numerical = scaler.fit_transform(recipe_df[['calories', 'fat', 'carbohydrates', 'protein', 'cholesterol', 'sodium']])

# Combine features
X_combined = np.hstack([X_numerical, X_ingredients.toarray()])

# Train KNN model
knn = NearestNeighbors(n_neighbors=3, metric='euclidean')
knn.fit(X_combined)

def recommend_recipes(input_features):
    try:
        # Extract numerical features (first 6 values)
        numerical_features = input_features[:6]
        # Extract ingredients (remaining values)
        ingredients = ' '.join(input_features[6:])
        
        # Scale numerical features
        input_features_scaled = scaler.transform([numerical_features])
        # Transform ingredients
        input_ingredients_transformed = vectorizer.transform([ingredients])
        
        # Combine features
        input_combined = np.hstack([input_features_scaled, input_ingredients_transformed.toarray()])
        
        # Get recommendations
        distances, indices = knn.kneighbors(input_combined)
        recommendations = recipe_df.iloc[indices[0]]
        return recommendations[['recipe_name', 'ingredients_list', 'image_url', 'aver_rate']]
    except Exception as e:
        print(f"Error in recommendation: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get form data
        calories = float(request.form['calories'])
        fat = float(request.form['fat'])
        carbs = float(request.form['carbohydrates'])
        protein = float(request.form['protein'])
        cholesterol = float(request.form['cholesterol'])
        sodium = float(request.form['sodium'])
        ingredients = request.form['ingredients'].split(',')
        
        # Prepare input features
        input_features = [calories, fat, carbs, protein, cholesterol, sodium] + ingredients
        
        # Get recommendations
        recommendations = recommend_recipes(input_features)
        
        if recommendations is not None:
            # Convert recommendations to list of dicts for template
            recipes = []
            for idx, row in recommendations.iterrows():
                recipes.append({
                    'name': row['recipe_name'],
                    'ingredients': row['ingredients_list'],
                    'image_url': row['image_url'],
                    'rating': row['aver_rate']
                })
            return render_template('results.html', recipes=recipes)
        else:
            return render_template('index.html', error="Could not generate recommendations. Please try again.")
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)