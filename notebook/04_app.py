import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from PIL import Image
import re

# Page setup
st.set_page_config(
    page_title="Budget Bites",
    page_icon="🍽️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS styling
st.markdown("""
<style>
.main {
    background-color: #fcf7f7;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 850px;
}

.hero-box {
    background: linear-gradient(135deg, #fff7f0 0%, #fdf2e9 100%);
    border: 2px solid #f3d2b4;
    padding: 1.5rem;
    border-radius: 22px;
    text-align: center;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    margin-bottom: 1.5rem;
}

.hero-title {
    font-size: 2.3rem;
    font-weight: 800;
    color: #7a3e16;
    margin-bottom: 0.3rem;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #5f5b57;
    margin-bottom: 0.2rem;
}

.section-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #8b4513;
    margin-bottom: 0.7rem;
}

.stButton > button {
    background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.7rem 1.2rem;
    font-weight: 700;
    width: 100%;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #b45309 0%, #d97706 100%);
    color: white;
}

.recipe-card {
    background: #fffdf9;
    border: 1px solid #ead7c3;
    padding: 1.2rem 1.4rem;
    border-radius: 18px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.05);
    margin-top: 1rem;
    margin-bottom: 1rem;
}

.recipe-label {
    font-weight: 700;
    color: #8b4513;
    margin-top: 0.8rem;
    margin-bottom: 0.2rem;
    font-size: 1.05rem;
}

.recipe-value {
    color: #3f3f46;
    line-height: 1.6;
    white-space: pre-wrap;
}

.logo-wrap {
    display: flex;
    justify-content: center;
    margin-bottom: 0.5rem;
}

div[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# Gemini API setup
GOOGLE_API_KEY = ""
genai.configure(api_key=GOOGLE_API_KEY)

# Load logo
logo = Image.open("logo.png")

# Load model and data
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data
def load_dataset():
    return pd.read_csv("recipe_dataset_with_text.csv")


@st.cache_resource
def load_faiss_index():
    return faiss.read_index("recipe_faiss.index")


model = load_embedding_model()
df = load_dataset()
index = load_faiss_index()

# Helper functions
def build_query(ingredients, cuisine=None, nutrition_goal=None, category=None):
    parts = []

    if ingredients:
        parts.append(f"Ingredients: {ingredients}")
    if cuisine and cuisine != "Any":
        parts.append(f"Cuisine: {cuisine}")
    if nutrition_goal and nutrition_goal != "Any":
        parts.append(f"Nutrition Goal: {nutrition_goal}")
    if category and category != "Any":
        parts.append(f"Category: {category}")

    return " | ".join(parts)


def retrieve_recipes(
    ingredients,
    max_cooking_time=None,
    cuisine=None,
    nutrition_goal=None,
    category=None,
    top_k=5,
    search_k=50
):
    query = build_query(
        ingredients=ingredients,
        cuisine=cuisine,
        nutrition_goal=nutrition_goal,
        category=category
    )

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, search_k)
    retrieved_df = df.iloc[indices[0]].copy()

    if max_cooking_time is not None:
        retrieved_df = retrieved_df[retrieved_df["Tottime"] <= max_cooking_time]

    if cuisine is not None and cuisine != "Any":
        retrieved_df = retrieved_df[
            retrieved_df["Cuisine"].str.strip().str.lower() == cuisine.strip().lower()
        ]

    if category is not None and category != "Any":
        retrieved_df = retrieved_df[
            retrieved_df["Category"].str.strip().str.lower() == category.strip().lower()
        ]

    if retrieved_df.empty:
        return retrieved_df

    if nutrition_goal == "High protein":
        retrieved_df = retrieved_df.sort_values(by="Protein_g", ascending=False)
    elif nutrition_goal == "Low calorie":
        retrieved_df = retrieved_df.sort_values(by="Calories", ascending=True)
    elif nutrition_goal == "Low fat":
        retrieved_df = retrieved_df.sort_values(by="Fats_g", ascending=True)

    return retrieved_df.head(top_k)


def format_retrieved_recipes_for_prompt(retrieved_df):
    blocks = []

    for i, (_, row) in enumerate(retrieved_df.iterrows(), start=1):
        block = f"""
Recipe {i}
Name: {row['Recipe_name']}
Category: {row['Category']}
Cuisine: {row['Cuisine']}
Description: {row['Description']}
Ingredients: {row['Ingredients']}
Steps: {row['Recipe_steps']}
Total Time: {row['Tottime']} minutes
Calories: {row['Calories']}
Protein: {row['Protein_g']} g
Fats: {row['Fats_g']} g
Carbs: {row['Carbs_g']} g
""".strip()
        blocks.append(block)

    return "\n\n".join(blocks)


def build_final_prompt(
    ingredients,
    max_cooking_time=None,
    cuisine=None,
    nutrition_goal=None,
    category=None,
    retrieved_df=None
):
    retrieved_text = format_retrieved_recipes_for_prompt(retrieved_df)

    prompt = f"""
You are Budget Bites, a recipe assistant for busy students with limited time and budget.

The user gave these inputs:

Ingredients available: {ingredients}
Maximum cooking time: {max_cooking_time if max_cooking_time is not None else "Not specified"}
Cuisine preference: {cuisine if cuisine else "Any"}
Nutrition goal: {nutrition_goal if nutrition_goal else "Any"}
Category: {category if category else "Any"}

Below are the most relevant retrieved recipes from the dataset:

{retrieved_text}

Your task:
Generate ONE final recipe recommendation based on the user's inputs and the retrieved recipes.

Rules:
- Keep the recipe practical and realistic.
- Match the user's ingredients as closely as possible.
- Respect the maximum cooking time if provided.
- Respect the cuisine, category, and nutrition goal if provided.
- Use the retrieved recipes as guidance, but you can improve or adapt them for eg exact portions of required ingredients.
- Keep the response clear and well-structured.
- Put every output field on a separate new line.

Return the answer in exactly this format:

Recipe Name:
Description:
Ingredients:
Steps:
Total Time:
Calories:
Protein:
Fats:
Carbs:
""".strip()

    return prompt


def call_llm(prompt):
    llm = genai.GenerativeModel("models/gemini-2.5-flash")
    response = llm.generate_content(prompt)
    return response.text


# If Gemini quota fails, use this temporary mock version instead:
# def call_llm(prompt):
#     return '''Recipe Name: Kerala Mutton Stew Recipe
# Description: Mild, creamy coconut-based mutton stew with subtle spices, comforting and aromatic.
# Ingredients: Mutton, coconut milk, onion, ginger, garlic, green chili, whole spices, curry leaves, oil, salt
# Steps: 1. Cook mutton with water and salt until tender.
# 2. Heat oil, sauté onion, ginger, garlic, chili, and curry leaves.
# 3. Add cooked mutton and mix well.
# 4. Pour coconut milk and simmer gently.
# 5. Cook briefly and serve hot.
# Total Time: 45 minutes
# Calories: 380
# Protein: 23 g
# Fats: 28 g
# Carbs: 10 g'''


def generate_recipe_response(
    ingredients,
    max_cooking_time=None,
    cuisine=None,
    nutrition_goal=None,
    category=None,
    top_k=3
):
    retrieved_df = retrieve_recipes(
        ingredients=ingredients,
        max_cooking_time=max_cooking_time,
        cuisine=cuisine,
        nutrition_goal=nutrition_goal,
        category=category,
        top_k=top_k
    )

    if retrieved_df.empty:
        return None, "Sorry, no matching recipes were found for those constraints."

    prompt = build_final_prompt(
        ingredients=ingredients,
        max_cooking_time=max_cooking_time,
        cuisine=cuisine,
        nutrition_goal=nutrition_goal,
        category=category,
        retrieved_df=retrieved_df
    )

    final_answer = call_llm(prompt)
    return retrieved_df, final_answer


def parse_recipe_output(text):
    fields = {
        "Recipe Name": "",
        "Description": "",
        "Ingredients": "",
        "Steps": "",
        "Total Time": "",
        "Calories": "",
        "Protein": "",
        "Fats": "",
        "Carbs": ""
    }

    current_field = None
    lines = text.splitlines()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        matched = False
        for field in fields.keys():
            if stripped.lower().startswith(field.lower() + ":"):
                fields[field] = stripped.split(":", 1)[1].strip()
                current_field = field
                matched = True
                break

        if not matched and current_field is not None:
            if fields[current_field]:
                fields[current_field] += "\n" + stripped
            else:
                fields[current_field] = stripped

    return fields

# Header / Hero section
col_left, col_mid, col_right = st.columns([1, 2, 1])
with col_mid:
    st.image(logo, use_container_width=True)

st.markdown("""
<div class="hero-box">
    <div class="hero-title">Budget Bites</div>
    <div class="hero-subtitle">
        Smart recipe planning for busy students using AI
    </div>
    <div class="hero-subtitle">
        Enter what you have at home and get a practical recipe recommendation.
    </div>
</div>
""", unsafe_allow_html=True)

# Input section
st.markdown('<div class="section-title">Tell us what you want to cook</div>', unsafe_allow_html=True)

ingredients = st.text_area(
    "Ingredients",
    placeholder="Example: chicken, rice, onion, garlic",
    height=120
)

col1, col2 = st.columns(2)

with col1:
    max_cooking_time = st.number_input(
        "Max cooking time (minutes)",
        min_value=0,
        value=0,
        step=5
    )

with col2:
    nutrition_goal = st.selectbox(
        "Nutrition goal",
        ["Any", "High protein", "Low calorie", "Low fat"]
    )

cuisine_options = ["Any"] + sorted(df["Cuisine"].dropna().astype(str).str.strip().unique().tolist())
category_options = ["Any"] + sorted(df["Category"].dropna().astype(str).str.strip().unique().tolist())

col3, col4 = st.columns(2)

with col3:
    cuisine = st.selectbox("Cuisine", cuisine_options)

with col4:
    category = st.selectbox("Category", category_options)

generate_button = st.button("Generate Recipe")

# Output section
if generate_button:
    if not ingredients.strip():
        st.warning("Please enter at least one ingredient.")
    else:
        with st.spinner("Cooking up your recipe idea..."):
            max_time_value = None if max_cooking_time == 0 else max_cooking_time

            retrieved_df, result = generate_recipe_response(
                ingredients=ingredients,
                max_cooking_time=max_time_value,
                cuisine=cuisine,
                nutrition_goal=nutrition_goal,
                category=category,
                top_k=3
            )

        st.markdown('<div class="section-title">Final Recipe Recommendation</div>', unsafe_allow_html=True)

        parsed = parse_recipe_output(result)


        for field in ["Recipe Name", "Description", "Ingredients", "Steps", "Total Time", "Calories", "Protein", "Fats", "Carbs"]:
            value = parsed.get(field, "").strip()
            if value:
                st.markdown(f'<div class="recipe-label">{field}</div>', unsafe_allow_html=True)

                if field == "Steps":
                    formatted_steps = re.sub(r'(?<!\n)(\d+\.)', r'\n\1', value).strip()
                    st.text(formatted_steps)
                else:
                    st.markdown(f'<div class="recipe-value">{value}</div>', unsafe_allow_html=True)


        if retrieved_df is not None and not retrieved_df.empty:
            st.markdown('<div class="section-title">Top Retrieved Recipes</div>', unsafe_allow_html=True)
            st.dataframe(
                retrieved_df[[
                    "Recipe_name",
                    "Cuisine",
                    "Category",
                    "Tottime",
                    "Calories",
                    "Protein_g",
                    "Fats_g",
                    "Carbs_g"
                ]].reset_index(drop=True),
                use_container_width=True
            )