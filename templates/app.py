import csv
import json
import os
import random
from flask import Flask, render_template, request, jsonify

# --- Configuration ---
app = Flask(__name__)
# IMPORTANT: Set to True to enable CSV writing
ENABLE_CSV_WRITING = True 
CSV_FILE = 'user_inputs.csv'

# --- Food Lists (Same as before) ---
# NOTE: These lists are designed to have items that fit multiple categories 
# (e.g., oats, eggs, fish are generally heart-healthy and low GI).
breakfast_foods = [
    'Oats porridge with berries (whole grain)', 'Scrambled eggs with spinach (low sodium)', 
    'Whole-wheat toast with avocado', 'Fruit smoothie with Greek yogurt (unsweetened)'
]
lunch_foods = [
    'Grilled chicken salad with low-fat dressing', 'Lentil soup with whole-grain bread (low sodium)', 
    'Turkey and vegetable wrap (whole-wheat)', 'Baked fish fillet with steamed vegetables'
]
dinner_foods = [
    'Baked salmon with quinoa and asparagus', 'Chicken khichdi with turmeric and spices (low sodium)', 
    'Vegetable stir-fry with brown rice', 'Minced lean beef with baked sweet potato'
]

# --- Core Recommendation Logic ---
def get_recommendations(user_input):
    """
    Generates food recommendations based on user health data and filters.
    """
    
    # 1. PARSE AND VALIDATE INPUTS
    try:
        # Convert necessary values to numbers
        bmi_num = float(user_input.get('bmi', 0))
        bp_num = int(user_input.get('bp', 0))
        cholesterol_num = int(user_input.get('cholesterol', 0))
        sugar_num = int(user_input.get('sugar', 0))
    except ValueError:
        return {'Breakfast': 'Error: Invalid numeric data received.'}
    
    # Categorize status
    bmi_cat = 'Normal'
    if bmi_num >= 30:
        bmi_cat = 'Obese'
    elif bmi_num >= 25:
        bmi_cat = 'Overweight'
        
    bp_cat = 'Normal'
    if bp_num >= 140:
        bp_cat = 'High'
        
    cholesterol_cat = 'Normal'
    if cholesterol_num >= 200:
        cholesterol_cat = 'High'
        
    sugar_cat = 'Normal'
    if sugar_num >= 125:
        sugar_cat = 'High'

    # Extract remaining categorical strings (ensure they are lowercase for comparison)
    chronic_disease = user_input.get('chronicDisease', 'None').lower()
    stress_eating = user_input.get('stressEating', 'no').lower()
    diet = user_input.get('dietaryHabits', 'Balanced').lower()
    sodium_sensitive = user_input.get('sodiumSensitivity', 'no').lower()
    sugar_sensitive = user_input.get('sugarSensitivity', 'no').lower()
    
    # 2. FILTERING LOGIC (CRUCIAL CHANGE HERE)
    
    meal_lists = [list(breakfast_foods), list(lunch_foods), list(dinner_foods)]
    filtered_meals = []

    for meal_foods_original in meal_lists:
        # Start with a copy of the original list for filtering
        current_list = list(meal_foods_original) 
        
        # 1. HIGH BLOOD PRESSURE/SODIUM SENSITIVITY (Filter high-sodium/processed foods)
        if bp_cat == 'high' or sodium_sensitive == 'yes':
            # Strict filter: remove items clearly marked as higher in sodium/processed
            current_list = [f for f in current_list if 'processed' not in f.lower() and 'canned' not in f.lower() and 'beef' not in f.lower()]
            
        # 2. HIGH SUGAR/DIABETES/SUGAR SENSITIVITY (Filter high-sugar items)
        if sugar_cat == 'high' or sugar_sensitive == 'yes' or chronic_disease == 'diabetes':
             # Strict filter: remove items clearly marked as high sugar/white carbs
             current_list = [f for f in current_list if 'sweet' not in f.lower() and 'white bread' not in f.lower() and 'smoothie' not in f.lower()]

        # 3. DIET (e.g., Vegetarian filter)
        if diet == 'vegetarian':
            # Strict filter: remove all meat/fish items
            current_list = [f for f in current_list if 'fish' not in f.lower() and 'chicken' not in f.lower() and 'beef' not in f.lower() and 'turkey' not in f.lower()]

        # 4. STRESS EATING (Encourage comfort/warm foods)
        if stress_eating == 'yes':
            # Soft filter: prioritize warm items, but only if the list isn't empty after other filters
            comfort_foods = [f for f in current_list if 'soup' in f.lower() or 'porridge' in f.lower() or 'oats' in f.lower() or 'khichdi' in f.lower() or 'stew' in f.lower()]
            if comfort_foods:
                current_list = comfort_foods
            
        # FINAL GUARDRAIL: If all filters resulted in an empty list, revert to the original list 
        # (or provide a safe default like "Plain Whole Grain Porridge") to ensure a selection can be made.
        if not current_list:
            current_list = ["No specific recommendation found. Please consult a dietitian."]
            
        filtered_meals.append(current_list)

    # 3. SELECT RECOMMENDATIONS
    
    # Use index 0 for breakfast, 1 for lunch, 2 for dinner
    filtered_breakfast = filtered_meals[0]
    filtered_lunch = filtered_meals[1]
    filtered_dinner = filtered_meals[2]
    
    # Select one random recommendation from each filtered list
    recommendations = {
        'Breakfast': random.choice(filtered_breakfast),
        'Lunch': random.choice(filtered_lunch),
        'Dinner': random.choice(filtered_dinner)
    }
    
    return recommendations

# --- CSV Saving Logic (Same as before) ---
def save_to_csv(user_data, recommendations):
    """Saves user input and recommendations to a CSV file."""
    if not ENABLE_CSV_WRITING:
        return "CSV Saving DISABLED for testing."

    # Create a unified dictionary for CSV
    csv_data = user_data.copy()
    csv_data['reco_breakfast'] = recommendations.get('Breakfast', '')
    csv_data['reco_lunch'] = recommendations.get('Lunch', '')
    csv_data['reco_dinner'] = recommendations.get('Dinner', '')
    
    fieldnames = list(csv_data.keys())
    
    # File existence check and writing
    file_exists = os.path.isfile(CSV_FILE)
    
    try:
        with open(CSV_FILE, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
                
            writer.writerow(csv_data)
        return "Data successfully saved to user_inputs.csv."
    except Exception as e:
        return f"Error saving to CSV: [Errno 13] Permission denied: '{CSV_FILE}'. Check console for details."

# --- Flask Routes (Same as before) ---

@app.route('/')
def index():
    """Renders the main input form."""
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit_form():
    """Handles the form submission, generates recommendations, and saves data."""
    
    user_data = request.get_json()
    
    if not user_data:
        return jsonify({"message": "Invalid data received"}), 400

    # 1. GENERATE RECOMMENDATIONS
    recommendations = get_recommendations(user_data) 
    
    # Check if the recommendation function returned an error message 
    if "Error" in recommendations.get('Breakfast', ''):
        return jsonify({
            "message": "Processing failed.",
            "recommendations": recommendations,
            "save_status": "No data saved due to input error."
        }), 400
        
    # 2. SAVE DATA
    save_status = save_to_csv(user_data, recommendations)

    # 3. RETURN RESPONSE TO FRONTEND
    return jsonify({
        "message": "Recommendations generated!",
        "recommendations": recommendations,
        "save_status": save_status
    })

if __name__ == '__main__':
    if ENABLE_CSV_WRITING:
        print(f"CSV writing is ENABLED. Target file: {CSV_FILE}")
    else:
        print("CSV writing is DISABLED.")

    app.run(debug=True)
