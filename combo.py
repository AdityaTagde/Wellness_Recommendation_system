import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ✅ Cache Model Loading to Improve Speed
@st.cache_resource
def load_exercise_model():
    with open("exercise_recommender.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_diet_model():
    with open("diet_recommender.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_meditation_model():
    with open("meditation_recommender_st.pkl", "rb") as f:
        med_df, med_embeddings = pickle.load(f)
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return med_df, med_embeddings, model

# ✅ Load Models Once
similarity_matrix, exercise_df, scaler, word2vec_model = load_exercise_model()
cosine_sim, tfidf, diet_df = load_diet_model()
med_df, med_embeddings, med_model = load_meditation_model()

# 🎨 **Improved Sidebar Styling**
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color:  #B0B0B0;  /* Cement Color Sidebar */
    }
    [data-testid="stSidebar"] h1 {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.title("💪🍽️ Wellness Recommendation System")
page = st.sidebar.radio("Select a Recommendation", ["🏋️ Exercise", "🍽️ Diet", "🧘 Meditation"])

# ==============================
# 🚀 EXERCISE RECOMMENDATION SYSTEM
# ==============================
if page == "🏋️ Exercise":
    st.title("🏋️ Exercise Recommendation System")

    exercise_type = st.selectbox("Select Exercise Type", exercise_df["Name of Exercise"].unique())

    def recommend_exercise(exercise_type):
        filtered_exercise = exercise_df[exercise_df["Name of Exercise"] == exercise_type]
        return filtered_exercise if not filtered_exercise.empty else pd.DataFrame()

    if st.button("Get Recommendations"):
        recommendations = recommend_exercise(exercise_type)
        if recommendations.empty:
            st.warning("⚠ No exercises found for the selected type.")
        else:
            for _, row in recommendations.iterrows():
                st.markdown(
                    f"""
                    <div style="border-radius:10px; padding:15px; background-color:#f0f2f6; border-left: 5px solid #4CAF50;">
                        <h3 style="color:#2E3B4E;">🏋️ <b>{row['Name of Exercise']}</b></h3>
                        <p><span style="color:#FF5733;">💪 <b>Target Muscle:</b></span> {row['Target Muscle Group']}</p>
                        <p><span style="color:#FFC300;">🔥 <b>Burns:</b></span> {row['Burns Calories (per 30 min)']} Calories</p>
                        <p><span style="color:#1E88E5;">📊 <b>Difficulty:</b></span> {row['Difficulty Level']}</p>
                        <p><span style="color:#4CAF50;">🔄 <b>Sets:</b></span> {row['Sets']} | <span style="color:#4CAF50;">🔢 <b>Reps:</b></span> {row['Reps']}</p>
                        <p><span style="color:#673AB7;">🎯 <b>Benefit:</b></span> {row['Benefit']}</p>
                        <p><span style="color:#009688;">🛠 <b>Equipment Needed:</b></span> {row['Equipment Needed'] if row['Equipment Needed'] else 'None'}</p>
                    </div>
                    """, unsafe_allow_html=True
                )
            st.markdown("---")

# ==============================
# 🚀 DIET RECOMMENDATION SYSTEM
# ==============================
elif page == "🍽️ Diet":
    st.title("🍽️ Diet Recommendation System")

    unique_cuisines = sorted(diet_df["Cuisine_type"].unique())
    selected_cuisine = st.selectbox("Choose a Cuisine Type", unique_cuisines)
    sort_option = st.selectbox("Sort by:", ["Protein(g)", "Carbs(g)", "Fat(g)"])

    def get_top_cuisine_recommendations(cuisine_type, sort_by):
        filtered_df = diet_df[diet_df["Cuisine_type"].str.lower() == cuisine_type.lower()].reset_index(drop=True)
        if filtered_df.empty:
            return None
        filtered_indices = filtered_df.index.to_numpy()
        filtered_cosine_sim = cosine_sim[filtered_indices][:, filtered_indices]
        similarity_scores = filtered_cosine_sim.sum(axis=1)
        filtered_df["similarity_score"] = similarity_scores
        return filtered_df.sort_values(by=[sort_by, "similarity_score"], ascending=[False, False]).head(50)

    if st.button("Get Recommendations"):
        recommendations = get_top_cuisine_recommendations(selected_cuisine, sort_option)
        if recommendations is None:
            st.warning("⚠ No recipes found for this cuisine type.")
        else:
            st.write(f"**Top 50 {selected_cuisine} Recipes (Sorted by {sort_option}):**")
            st.dataframe(recommendations[["Recipe_name", "Protein(g)", "Carbs(g)", "Fat(g)"]].reset_index(drop=True))

# ==============================
# 🚀 MEDITATION RECOMMENDATION SYSTEM
# ==============================
elif page == "🧘 Meditation":
    st.title("🧘 Meditation Recommendation System")

    user_input = st.text_area("Describe what kind of meditation you need (e.g., relaxation, stress relief, mindfulness)")

    if st.button("Get Recommendations"):
        if user_input.strip() == "":
            st.warning("⚠ Please enter a meditation description.")
        else:
            user_embedding = med_model.encode([user_input])
            similarities = cosine_similarity(user_embedding, med_embeddings).flatten()
            med_df["similarity_score"] = similarities
            recommendations = med_df.sort_values(by="similarity_score", ascending=False).head(5)

            if recommendations.empty:
                st.warning("⚠ No matching meditations found.")
            else:
                for _, row in recommendations.iterrows():
                    st.success(f"### 🧘 {row['Name']}")
                    st.write(f"💡 **Description:** {row['Description']}")
                    st.markdown("---")
