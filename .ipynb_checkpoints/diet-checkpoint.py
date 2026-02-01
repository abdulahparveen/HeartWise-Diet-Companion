{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "098d7f98-0933-4e75-9b22-07ad4c155d02",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Dataset loaded and encoded!\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "\n",
    "# Load dataset\n",
    "df = pd.read_csv('Personalized_Diet_Recommendations.csv')  # Make sure this file is in the same folder\n",
    "\n",
    "# Drop unnecessary columns\n",
    "df = df.drop(columns=['Patient_ID', 'Unnamed: 30'])\n",
    "\n",
    "# Encode categorical columns\n",
    "categorical_cols = ['Gender', 'Chronic_Disease', 'Smoking_Habit', 'Dietary_Habits',\n",
    "                    'Preferred_Cuisine', 'Food_Aversions', 'Recommended_Meal_Plan']\n",
    "\n",
    "label_encoders = {}\n",
    "for col in categorical_cols:\n",
    "    le = LabelEncoder()\n",
    "    df[col] = le.fit_transform(df[col].astype(str))\n",
    "    label_encoders[col] = le\n",
    "\n",
    "print(\"✅ Dataset loaded and encoded!\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "6eb9c5fd-11b0-41aa-b33e-17dabf1abb46",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ All columns encoded to numeric!\n"
     ]
    }
   ],
   "source": [
    "from sklearn.preprocessing import LabelEncoder\n",
    "\n",
    "# Load dataset\n",
    "df = pd.read_csv('Personalized_Diet_Recommendations.csv')  # Make sure this file is correct\n",
    "df = df.drop(columns=['Patient_ID', 'Unnamed: 30'])\n",
    "\n",
    "# Encode all object-type columns\n",
    "label_encoders = {}\n",
    "for col in df.columns:\n",
    "    if df[col].dtype == 'object':\n",
    "        le = LabelEncoder()\n",
    "        df[col] = le.fit_transform(df[col].astype(str))\n",
    "        label_encoders[col] = le\n",
    "\n",
    "print(\"✅ All columns encoded to numeric!\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "25a954cc-fbb1-4173-ae62-47803692b135",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Data split into training and testing sets!\n"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "X = df.drop(columns=['Recommended_Meal_Plan'])\n",
    "y = df['Recommended_Meal_Plan']\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "print(\"✅ Data split into training and testing sets!\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "ec734a62-b983-41a0-a95b-a7270c45449e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Models defined!\n"
     ]
    }
   ],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "rf = RandomForestClassifier(n_estimators=100, random_state=42)\n",
    "gb = GradientBoostingClassifier(n_estimators=100, random_state=42)\n",
    "\n",
    "svm_pipeline = Pipeline([\n",
    "    ('scaler', StandardScaler()),\n",
    "    ('svm', SVC(probability=True, kernel='linear', random_state=42))\n",
    "])\n",
    "\n",
    "print(\"✅ Models defined!\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "0487e4e6-73ba-4a7e-a3c0-d90557d8b534",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ All string columns encoded to numeric!\n"
     ]
    }
   ],
   "source": [
    "from sklearn.preprocessing import LabelEncoder\n",
    "\n",
    "# Encode all object-type columns\n",
    "label_encoders = {}\n",
    "for col in df.columns:\n",
    "    if df[col].dtype == 'object':\n",
    "        le = LabelEncoder()\n",
    "        df[col] = le.fit_transform(df[col].astype(str))\n",
    "        label_encoders[col] = le\n",
    "\n",
    "print(\"✅ All string columns encoded to numeric!\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "83352ed6-ee0c-4932-87df-02c97f624086",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "🧪 Non-numeric columns in X_train: ['Genetic_Risk_Factor', 'Allergies', 'Alcohol_Consumption']\n",
      "\n",
      "🔍 Sample values in 'Genetic_Risk_Factor':\n",
      "['No' 'Yes']\n",
      "\n",
      "🔍 Sample values in 'Allergies':\n",
      "[nan 'Lactose Intolerance' 'Gluten Intolerance' 'Nut Allergy']\n",
      "\n",
      "🔍 Sample values in 'Alcohol_Consumption':\n",
      "['Yes' 'No']\n"
     ]
    }
   ],
   "source": [
    "# Check which columns are still non-numeric\n",
    "non_numeric_cols = X_train.select_dtypes(include=['object']).columns.tolist()\n",
    "print(\"🧪 Non-numeric columns in X_train:\", non_numeric_cols)\n",
    "\n",
    "# Check sample values from those columns\n",
    "for col in non_numeric_cols:\n",
    "    print(f\"\\n🔍 Sample values in '{col}':\")\n",
    "    print(X_train[col].unique())\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "6dba7713-e7cf-4c58-83e2-c54f6f58204f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Remaining columns encoded successfully!\n"
     ]
    }
   ],
   "source": [
    "from sklearn.preprocessing import LabelEncoder\n",
    "\n",
    "# Encode only the remaining string columns\n",
    "for col in ['Genetic_Risk_Factor', 'Allergies', 'Alcohol_Consumption']:\n",
    "    le = LabelEncoder()\n",
    "    X_train[col] = le.fit_transform(X_train[col].astype(str))\n",
    "    X_test[col] = le.transform(X_test[col].astype(str))  # use same encoder for test set\n",
    "\n",
    "print(\"✅ Remaining columns encoded successfully!\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "3348f043-7e56-4c52-b47e-ff52d7d0bcf2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Ensemble model trained!\n"
     ]
    }
   ],
   "source": [
    "ensemble.fit(X_train, y_train)\n",
    "print(\"✅ Ensemble model trained!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "0457a1cb-175c-4cee-bf87-2c667fe868f8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "💬 Model Performance:\n",
      "\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.29      0.25      0.27       259\n",
      "           1       0.26      0.31      0.28       248\n",
      "           2       0.20      0.18      0.19       211\n",
      "           3       0.25      0.27      0.26       282\n",
      "\n",
      "    accuracy                           0.25      1000\n",
      "   macro avg       0.25      0.25      0.25      1000\n",
      "weighted avg       0.25      0.25      0.25      1000\n",
      "\n",
      "✅ Model and encoders saved!\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import classification_report\n",
    "import joblib\n",
    "joblib.dump(X_train.columns.tolist(), 'model_features.pkl')\n",
    "\n",
    "# Evaluate model\n",
    "y_pred = ensemble.predict(X_test)\n",
    "print(\"💬 Model Performance:\\n\")\n",
    "print(classification_report(y_test, y_pred))\n",
    "\n",
    "# Save model and encoders\n",
    "joblib.dump(ensemble, 'heartwise_model.pkl')\n",
    "joblib.dump(label_encoders, 'label_encoders.pkl')\n",
    "\n",
    "print(\"✅ Model and encoders saved!\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "50d014e1-6672-4f24-a4fd-ad883a2c3a52",
   "metadata": {},
   "outputs": [],
   "source": [
    "sample_input = {\n",
    "    'Age': 58,\n",
    "    'Gender': 'Female',\n",
    "    'Height_cm': 155,\n",
    "    'Weight_kg': 62,\n",
    "    'BMI': 25.8,\n",
    "    'Chronic_Disease': 'Hypertension',\n",
    "    'Blood_Pressure_Systolic': 145,\n",
    "    'Blood_Pressure_Diastolic': 92,\n",
    "    'Cholesterol_Level': 210,\n",
    "    'Blood_Sugar_Level': 135,\n",
    "    'Genetic_Risk_Factor': 'Yes',\n",
    "    'Allergies': 'Nut Allergy',\n",
    "    'Daily_Steps': 2500,\n",
    "    'Exercise_Frequency': 1,\n",
    "    'Sleep_Hours': 5,\n",
    "    'Alcohol_Consumption': 'No',\n",
    "    'Smoking_Habit': 'No',\n",
    "    'Dietary_Habits': 'Vegetarian',\n",
    "    'Caloric_Intake': 1700,\n",
    "    'Protein_Intake': 45,\n",
    "    'Carbohydrate_Intake': 190,\n",
    "    'Fat_Intake': 55,\n",
    "    'Preferred_Cuisine': 'South_Indian',\n",
    "    'Food_Aversions': 'Spicy',\n",
    "    'Stress_Eating_Habit': 1,\n",
    "    'Sodium_Sensitivity': 1,\n",
    "    'Sugar_Sensitivity': 1,\n",
    "    # These will be filled with default values if missing:\n",
    "    # 'Recommended_Calories', 'Recommended_Carbs', 'Recommended_Fats', 'Recommended_Protein'\n",
    "}\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "d3fc7b4c-f68e-4d77-a279-a7c4c768fc70",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "🌟 Diet Recommendation:\n",
      "Based on your profile, we recommend the '🌸 Nourishing Simplicity' plan to support your heart and emotional well-being.\n"
     ]
    }
   ],
   "source": [
    "recommendation = recommend_diet(sample_input)\n",
    "print(\"\\n🌟 Diet Recommendation:\")\n",
    "print(recommendation['Message'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "be1c4b18-f7f9-4c9e-b553-6c77de2dfd4e",
   "metadata": {},
   "outputs": [],
   "source": [
    "def recommend_diet(user_input_dict):\n",
    "    # Load model, encoders, and feature list\n",
    "    model = joblib.load('heartwise_model.pkl')\n",
    "    encoders = joblib.load('label_encoders.pkl')\n",
    "    expected_features = joblib.load('model_features.pkl')\n",
    "\n",
    "    # Clean and encode input\n",
    "    input_df = clean_and_encode_input(user_input_dict, expected_features, encoders)\n",
    "\n",
    "    # Predict label\n",
    "    pred = model.predict(input_df)[0]\n",
    "    decoded_label = encoders['Recommended_Meal_Plan'].inverse_transform([pred])[0]\n",
    "\n",
    "    # Symbolic label\n",
    "    symbolic_meals = {\n",
    "        'Low_Sodium': '🌿 Gentle Greens & Grains',\n",
    "        'Low_Sugar': '🍠 Rooted Comfort & Cinnamon Calm',\n",
    "        'Balanced': '🥗 Heart Harmony Bowl',\n",
    "        'High_Protein': '🐟 Ocean Strength & Lentil Warmth',\n",
    "        'Ayurvedic': '🪷 Healing Spices & Warmth',\n",
    "        'Mediterranean': '🍅 Olive Calm & Sea Breeze',\n",
    "        'South_Indian': '🌾 Millets & Rasam Serenity'\n",
    "    }\n",
    "    symbolic = symbolic_meals.get(decoded_label, '🌸 Nourishing Simplicity')\n",
    "\n",
    "    # Food suggestions by label\n",
    "    food_options = {\n",
    "        'Low_Sodium': ['Rice with moong dal', 'Vegetable upma', 'Oats khichdi'],\n",
    "        'Low_Sugar': ['Sweet potato mash', 'Cinnamon oats', 'Fruit salad with yogurt'],\n",
    "        'Balanced': ['Chapati with spinach curry', 'Brown rice with lentils', 'Grilled tofu with veggies'],\n",
    "        'High_Protein': ['Grilled fish with quinoa', 'Lentil soup', 'Egg white scramble'],\n",
    "        'Ayurvedic': ['Khichdi with ghee and turmeric', 'Rasam with millets', 'Steamed veggies with cumin'],\n",
    "        'Mediterranean': ['Hummus with whole wheat pita', 'Grilled zucchini with couscous', 'Quinoa tabbouleh'],\n",
    "        'South_Indian': ['Idli with sambar', 'Ragi dosa', 'Vegetable kurma with chapati']\n",
    "    }\n",
    "\n",
    "    # Personalization filters\n",
    "    cholesterol = user_input_dict.get('Cholesterol', 'Normal')\n",
    "    bmi = user_input_dict.get('BMI', 'Normal')\n",
    "    stress_eating = user_input_dict.get('Stress_Eating', False)\n",
    "\n",
    "    meals = food_options.get(decoded_label, [])\n",
    "    if cholesterol == 'High':\n",
    "        meals = [m for m in meals if 'fried' not in m and 'cheese' not in m]\n",
    "    if bmi in ['Overweight', 'Obese']:\n",
    "        meals = [m for m in meals if 'rice' not in m or 'brown rice' in m]\n",
    "    if stress_eating:\n",
    "        meals += ['Warm oats with banana', 'Chamomile tea with roasted chana', 'Herbal soup with lentils']\n",
    "\n",
    "    return {\n",
    "        'Recommended_Meal_Plan': decoded_label,\n",
    "        'Symbolic_Label': symbolic,\n",
    "        'Suggested_Foods': meals,\n",
    "        'Message': f\"Based on your health profile, we recommend the '{symbolic}' plan with meals that support your heart and emotional well-being.\"\n",
    "    }\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "e11ff130-1178-4a9c-b96f-87a448f2961d",
   "metadata": {},
   "outputs": [],
   "source": [
    "user_input_dict = {\n",
    "    'Age': 58,\n",
    "    'Gender': 'Female',\n",
    "    'Height': 160,\n",
    "    'Weight': 70,\n",
    "    'BMI': 'Overweight',\n",
    "    'Chronic_Disease': 'Hypertension',\n",
    "    'Blood_Pressure': 'High',\n",
    "    'Cholesterol': 'High',\n",
    "    'Sugar': 'Normal',\n",
    "    'Sleep_Hours': 6,\n",
    "    'Alcohol': 'No',\n",
    "    'Smoking': 'No',\n",
    "    'Stress_Eating': True,\n",
    "    'Dietary_Habits': 'Vegetarian',\n",
    "    'Sodium_Sensitivity': True,\n",
    "    'Sugar_Sensitivity': False\n",
    "}\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "7c547515-10c7-4c0d-bde3-471548863590",
   "metadata": {},
   "outputs": [],
   "source": [
    "def recommend_food(user_input):\n",
    "    # Extract inputs\n",
    "    age = user_input.get('Age')\n",
    "    gender = user_input.get('Gender')\n",
    "    height = user_input.get('Height')\n",
    "    weight = user_input.get('Weight')\n",
    "    bmi = user_input.get('BMI')\n",
    "    chronic = user_input.get('Chronic_Disease')\n",
    "    bp = user_input.get('Blood_Pressure')\n",
    "    cholesterol = user_input.get('Cholesterol')\n",
    "    sugar = user_input.get('Sugar')\n",
    "    sleep = user_input.get('Sleep_Hours')\n",
    "    alcohol = user_input.get('Alcohol')\n",
    "    smoking = user_input.get('Smoking')\n",
    "    stress_eating = user_input.get('Stress_Eating')\n",
    "    diet = user_input.get('Dietary_Habits')\n",
    "    sodium_sensitive = user_input.get('Sodium_Sensitivity')\n",
    "    sugar_sensitive = user_input.get('Sugar_Sensitivity')\n",
    "\n",
    "    # Base food pool\n",
    "    foods = [\n",
    "        'Oats khichdi', 'Vegetable upma', 'Brown rice with dal',\n",
    "        'Grilled tofu with veggies', 'Chapati with spinach curry',\n",
    "        'Sweet potato mash', 'Fruit salad with yogurt',\n",
    "        'Khichdi with turmeric', 'Rasam with millets',\n",
    "        'Idli with sambar', 'Ragi dosa', 'Quinoa salad'\n",
    "    ]\n",
    "\n",
    "    # Filters\n",
    "    if cholesterol == 'High':\n",
    "        foods = [f for f in foods if 'cheese' not in f and 'fried' not in f]\n",
    "    if sugar == 'High' or sugar_sensitive:\n",
    "        foods = [f for f in foods if 'sweet' not in f and 'fruit' not in f]\n",
    "    if sodium_sensitive or bp == 'High':\n",
    "        foods = [f for f in foods if 'pickle' not in f and 'papad' not in f]\n",
    "    if bmi in ['Overweight', 'Obese']:\n",
    "        foods = [f for f in foods if 'rice' not in f or 'brown rice' in f]\n",
    "    if stress_eating:\n",
    "        foods += ['Warm oats with banana', 'Chamomile tea', 'Herbal soup with lentils']\n",
    "    if diet == 'Vegetarian':\n",
    "        foods = [f for f in foods if 'fish' not in f and 'egg' not in f]\n",
    "\n",
    "    return foods\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "71cf1c5f-00ab-4d8a-a219-5e18dfa147ef",
   "metadata": {},
   "outputs": [],
   "source": [
    "user_input = {\n",
    "    'Age': 58,\n",
    "    'Gender': 'Female',\n",
    "    'Height': 160,\n",
    "    'Weight': 70,\n",
    "    'BMI': 'Overweight',\n",
    "    'Chronic_Disease': 'Hypertension',\n",
    "    'Blood_Pressure': 'High',\n",
    "    'Cholesterol': 'High',\n",
    "    'Sugar': 'Normal',\n",
    "    'Sleep_Hours': 6,\n",
    "    'Alcohol': 'No',\n",
    "    'Smoking': 'No',\n",
    "    'Stress_Eating': True,\n",
    "    'Dietary_Habits': 'Vegetarian',\n",
    "    'Sodium_Sensitivity': True,\n",
    "    'Sugar_Sensitivity': False\n",
    "}\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "aaa11125-2f95-4fd1-b83c-8bf1a6a331d0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Recommended Foods: ['Oats khichdi', 'Vegetable upma', 'Chapati with spinach curry', 'Sweet potato mash', 'Fruit salad with yogurt', 'Khichdi with turmeric', 'Rasam with millets', 'Idli with sambar', 'Ragi dosa', 'Quinoa salad', 'Warm oats with banana', 'Chamomile tea', 'Herbal soup with lentils']\n"
     ]
    }
   ],
   "source": [
    "recommended = recommend_food(user_input)\n",
    "print(\"Recommended Foods:\", recommended)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "405cb9a3-728f-4890-a04f-20e9a37b2fb3",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
