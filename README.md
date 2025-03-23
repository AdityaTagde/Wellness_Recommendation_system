# 🌿 Wellness Recommendation System

## Overview
The **Wellness Recommendation System** is a Streamlit-based web application that provides personalized recommendations for **Exercise, Diet, and Meditation**. It leverages **Word2Vec, TF-IDF, and Sentence Transformers** with **Cosine Similarity** to offer relevant suggestions based on user input.

## Features
- **Exercise Recommendation System**: Suggests exercises based on a selected exercise type.
- **Diet Recommendation System**: Provides top 50 diet recommendations based on a selected cuisine and nutritional sorting (Protein, Carbs, or Fats).
- **Meditation Recommendation System**: Recommends meditation techniques based on user-provided descriptions.
- **Interactive UI**: Built with Streamlit, featuring an intuitive sidebar navigation and enhanced UI styling.

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Streamlit
- Pandas
- NumPy
- scikit-learn
- SentenceTransformers
- Pickle (for loading models)

### Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/wellness-recommendation-system.git
   cd wellness-recommendation-system
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place the pre-trained models in the root directory:
   - `exercise_recommender.pkl`
   - `diet_recommender.pkl`
   - `meditation_recommender_st.pkl`

4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## How It Works
### Exercise Recommendation
- Select an exercise from the dropdown menu.
- Click "Get Recommendations" to retrieve details such as **target muscle, calorie burn, difficulty level, sets, reps, benefits, and required equipment**.

### Diet Recommendation
- Choose a **Cuisine Type** from the dropdown.
- Select a sorting preference (**Protein, Carbs, or Fats**).
- Click "Get Recommendations" to see the **top 50 diet options** sorted accordingly.

### Meditation Recommendation
- Enter a description of the desired meditation type (e.g., "stress relief").
- Click "Get Recommendations" to receive **top meditation suggestions** based on Sentence Transformers.

## File Structure
```
|-- app.py                     # Main Streamlit application
|-- exercise_recommender.pkl   # Pickle file for Exercise recommendations
|-- diet_recommender.pkl       # Pickle file for Diet recommendations
|-- meditation_recommender_st.pkl  # Pickle file for Meditation recommendations
|-- requirements.txt           # List of dependencies
|-- README.md                  # Project documentation
```

## Dependencies
```
streamlit
pandas
numpy
scikit-learn
sentence-transformers
pickle
```

## Acknowledgments
- **Word2Vec & TF-IDF**: Used for Exercise & Diet Recommendations
- **Sentence Transformers**: Used for Meditation Recommendations
- **Streamlit**: For building an interactive UI

## Interface Screenshots
- 🏋️ **Exercise Recommendation System**:
-  ![App Screenshot](https://github.com/AdityaTagde/Wellness_Recommendation_system/blob/main/exercise_i.png )
- 🍽️ **Diet Recommendation System**:
- ![App Screenshot](https://github.com/AdityaTagde/Wellness_Recommendation_system/blob/main/diets_i.png)
- 🧘 **Meditation Recommendation System**:
- ![App Screenshot](https://github.com/AdityaTagde/Wellness_Recommendation_system/blob/main/meditation_i.png )

## Future Improvements
- Integrate exercise GIFs for better visualization.
- Expand dataset for more personalized recommendations.
- Add user authentication for personalized history tracking.

## Contact
For any queries or contributions, feel free to reach out at: **tagdeaditya5@gmail.com**

---
**Author**: Aditya Tagde 

