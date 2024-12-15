import os
import pickle
import ast
import numpy as np
import pandas as pd
import requests
from flask import Flask, render_template, request

popular_df = pickle.load(open('model_pkl/books_model_pkl/popular.pkl', 'rb'))
books = pickle.load(open('model_pkl/books_model_pkl/books.pkl', 'rb'))
pt = pickle.load(open('model_pkl/books_model_pkl/pt.pkl', 'rb'))
similarity_score = pickle.load(open('model_pkl/books_model_pkl/similarity_score.pkl', 'rb'))

movies = pickle.load(open('model_pkl/movies_model_pkl/new_movies_df.pkl', 'rb'))
movies_similarity = pickle.load(open('model_pkl/movies_model_pkl/similarity.pkl', 'rb'))

music_dict = pickle.load(open("model_pkl/musics_model_pkl/music_recommendation_binary.pkl", "rb"))
music = pd.DataFrame(music_dict)
similarity = pickle.load(open("model_pkl/musics_model_pkl/similarities_binary.pkl", "rb"))

svm = pickle.load(open("model_pkl/medicines_model_pkl/medicine_df.pkl", "rb"))
description = pd.read_csv("object_datasets/medicine_datasets/description.csv")
diet = pd.read_csv("object_datasets/medicine_datasets/diets.csv")
medication = pd.read_csv("object_datasets/medicine_datasets/medications.csv")
precaution = pd.read_csv("object_datasets/medicine_datasets/precautions_df.csv")
workout = pd.read_csv("object_datasets/medicine_datasets/workout_df.csv")

def is_internet_connected():
    response = os.system("ping -c 1 google.com" if os.name != "nt" else "ping -n 1 google.com")
    return response == 0


def fetch_poster(movie_id):
    if not is_internet_connected():
        return "No internet connection. Please check your connection and try again."
    key = "132e1e1480a08eb037d0542e5274cc61"
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={key}&language=en-US"
    img_path = "https://image.tmdb.org/t/p/w185/"
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIxMzJlMWUxNDgwYTA4ZWIwMzdkMDU0MmU1Mjc0Y2M2MSIsIm5iZiI6MTcyOTUxNzE2Ny41NTExODcsInN1YiI6IjY3MTYyZjE1Y2VmMTQ2MjhmZWY2MzIyNyIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.xMmXke0j6uAd4WQg8K5OqQ1IdYqKJsofdc-OlIdjfzw"
    }

    response = requests.get(url, headers=headers)
    data = response.json()
    return img_path + data['poster_path']


def movie_data():
    indices = movies[movies['title'] == 'Avatar'].index[0]
    if indices > -1:
        distance = movies_similarity[indices]
        movies_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:15]
        data = []
        for i in movies_list:
            poster = []
            poster.append(movies.iloc[i[0]].title)
            poster.append(fetch_poster(movies.iloc[i[0]].movie_id))
            data.append(poster)
        return data


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html', book_name=list(popular_df['Book-Title'].values),
                           author=list(popular_df['Book-Author'].values),
                           image=list(popular_df['Image-URL-M'].values),
                           voting=list(popular_df['num_ratings'].values),
                           rating=list(popular_df['avg_ratings'].values),
                           movie_box=movie_data()
                           )


@app.route('/BookRecommender')
def book_recommender():
    return render_template('BookRecommender.html')


@app.route('/book_forms', methods=["post"])
def book_forms():
    entered_book_name = request.form.get("submitted_book_name")
    indices = np.where(pt.index == entered_book_name)[0]
    if len(indices) > 0:
        index = indices[0]
        similar_items = sorted(list(enumerate(similarity_score[index])), key=lambda x: x[1], reverse=True)[1:6]
        data = []
        for i in similar_items:
            item = []
            temp_df = books[books['Book-Title'] == pt.index[i[0]]]
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
            data.append(item)
        return render_template('BookRecommender.html', data=data)
    elif len(entered_book_name.strip()) == 0:
        return render_template('BookRecommender.html', error="Please enter the book name and try again.")
    else:
        return render_template('BookRecommender.html',
                               error="Book not found. Please check the book name and try again.")


@app.route('/MoviesRecommender')
def movie_recommender():
    return render_template('MoviesRecommender.html')


@app.route('/movie_forms', methods=["post"])
def movie_forms():
    entered_movie_name = request.form.get("submitted_movie_name")

    indices = np.where(movies['title'] == entered_movie_name)[0]
    if len(indices) > 0:
        index = movies[movies['title'] == entered_movie_name].index[0]
        distance = movies_similarity[index]
        movies_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:11]
        data = []
        for i in movies_list:
            poster = []
            poster.append(movies.iloc[i[0]].title)
            poster.append(fetch_poster(movies.iloc[i[0]].movie_id))
            data.append(poster)
        return render_template("MoviesRecommender.html", data=data)

    elif len(entered_movie_name.strip()) == 0:
        return render_template('BookRecommender.html', error="Please enter the book name and try again.")
    else:
        return render_template('BookRecommender.html',
                               error="Book not found. Please check the book name and try again.")


# MusicRecommender
@app.route('/MusicRecommender')
def music_recommender():
    selected_option = None
    return render_template('MusicRecommender.html')


def fetch_music_poster(music_title):
    response = requests.get(
        f"https://saavan-api-psi.vercel.app/api/search/songs?query={music_title}".format(music_title))
    data = response.json()
    return data['data']['results'][0]['artists']['primary'][0]['image'][2]['url']


# print(fetch_poster("Tere Bin"))


def recommend(musics):
    # Check if the song exists in the DataFrame
    if musics in music['title'].values:
        # Find the index of the music in the DataFrame
        music_indexval = music[music['title'] == musics].index[0]

        # Get the corresponding similarity distances for the song
        distance = similarity[music_indexval]

        # Enumerate the distances and sort them in descending order, then take the top 5
        music_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:6]

        recommend_music = []
        recommend_music_poster = []

        # Print the recommended songs
        try:
            for i in music_list:
                music_title = music.iloc[i[0]].title
                recommend_music.append(music.iloc[i[0]].title)
                recommend_music_poster.append(fetch_music_poster(music_title))
                # recommend_music_poster = [fetch_poster(item) for item in music_title]
            return recommend_music, recommend_music_poster
        except Exception as e:
            return "e1", "e2"
    else:
        return musics, "e3"


@app.route('/music_forms', methods=["GET", "POST"])
def music_forms():
    if request.method == 'POST':
        entered_music_name = request.form.get("submitted_music_name")
        if entered_music_name == 'SONG NAME':
            return render_template('MusicRecommender.html', error="Please select the value")

        name, poster = recommend(entered_music_name)
        data = []
        for i in range(len(name)):
            data.append([poster[i], name[i]])
        return render_template("MusicRecommender.html", data=data)


@app.route('/MedicineRecommender')
def medicine_recommender():
    return render_template('MedicineRecommender.html')


symptom_dict = {
    'itching': 0,
    'skin_rash': 1,
    'nodal_skin_eruptions': 2,
    'continuous_sneezing': 3,
    'shivering': 4,
    'chills': 5,
    'joint_pain': 6,
    'stomach_pain': 7,
    'acidity': 8,
    'ulcers_on_tongue': 9,
    'muscle_wasting': 10,
    'vomiting': 11,
    'burning_micturition': 12,
    'spotting_ urination': 13,
    'fatigue': 14,
    'weight_gain': 15,
    'anxiety': 16,
    'cold_hands_and_feets': 17,
    'mood_swings': 18,
    'weight_loss': 19,
    'restlessness': 20,
    'lethargy': 21,
    'patches_in_throat': 22,
    'irregular_sugar_level': 23,
    'cough': 24,
    'high_fever': 25,
    'sunken_eyes': 26,
    'breathlessness': 27,
    'sweating': 28,
    'dehydration': 29,
    'indigestion': 30,
    'headache': 31,
    'yellowish_skin': 32,
    'dark_urine': 33,
    'nausea': 34,
    'loss_of_appetite': 35,
    'pain_behind_the_eyes': 36,
    'back_pain': 37,
    'constipation': 38,
    'abdominal_pain': 39,
    'diarrhoea': 40,
    'mild_fever': 41,
    'yellow_urine': 42,
    'yellowing_of_eyes': 43,
    'acute_liver_failure': 44,
    'fluid_overload': 45,
    'swelling_of_stomach': 46,
    'swelled_lymph_nodes': 47,
    'malaise': 48,
    'blurred_and_distorted_vision': 49,
    'phlegm': 50,
    'throat_irritation': 51,
    'redness_of_eyes': 52,
    'sinus_pressure': 53,
    'runny_nose': 54,
    'congestion': 55,
    'chest_pain': 56,
    'weakness_in_limbs': 57,
    'fast_heart_rate': 58,
    'pain_during_bowel_movements': 59,
    'pain_in_anal_region': 60,
    'bloody_stool': 61,
    'irritation_in_anus': 62,
    'neck_pain': 63,
    'dizziness': 64,
    'cramps': 65,
    'bruising': 66,
    'obesity': 67,
    'swollen_legs': 68,
    'swollen_blood_vessels': 69,
    'puffy_face_and_eyes': 70,
    'enlarged_thyroid': 71,
    'brittle_nails': 72,
    'swollen_extremeties': 73,
    'excessive_hunger': 74,
    'extra_marital_contacts': 75,
    'drying_and_tingling_lips': 76,
    'slurred_speech': 77,
    'knee_pain': 78,
    'hip_joint_pain': 79,
    'muscle_weakness': 80,
    'stiff_neck': 81,
    'swelling_joints': 82,
    'movement_stiffness': 83,
    'spinning_movements': 84,
    'loss_of_balance': 85,
    'unsteadiness': 86,
    'weakness_of_one_body_side': 87,
    'loss_of_smell': 88,
    'bladder_discomfort': 89,
    'foul_smell_of urine': 90,
    'continuous_feel_of_urine': 91,
    'passage_of_gases': 92,
    'internal_itching': 93,
    'toxic_look_(typhos)': 94,
    'depression': 95,
    'irritability': 96,
    'muscle_pain': 97,
    'altered_sensorium': 98,
    'red_spots_over_body': 99,
    'belly_pain': 100,
    'abnormal_menstruation': 101,
    'dischromic _patches': 102,
    'watering_from_eyes': 103,
    'increased_appetite': 104,
    'polyuria': 105,
    'family_history': 106,
    'mucoid_sputum': 107,
    'rusty_sputum': 108,
    'lack_of_concentration': 109,
    'visual_disturbances': 110,
    'receiving_blood_transfusion': 111,
    'receiving_unsterile_injections': 112,
    'coma': 113,
    'stomach_bleeding': 114,
    'distention_of_abdomen': 115,
    'history_of_alcohol_consumption': 116,
    'fluid_overload.1': 117,
    'blood_in_sputum': 118,
    'prominent_veins_on_calf': 119,
    'palpitations': 120,
    'painful_walking': 121,
    'pus_filled_pimples': 122,
    'blackheads': 123,
    'scurring': 124,
    'skin_peeling': 125,
    'silver_like_dusting': 126,
    'small_dents_in_nails': 127,
    'inflammatory_nails': 128,
    'blister': 129,
    'red_sore_around_nose': 130,
    'yellow_crust_ooze': 131,
}
disease_list = {
    0: '(vertigo) Paroymsal  Positional Vertigo',
    1: 'AIDS',
    2: 'Acne',
    3: 'Alcoholic hepatitis',
    4: 'Allergy',
    5: 'Arthritis',
    6: 'Bronchial Asthma',
    7: 'Cervical spondylosis',
    8: 'Chicken pox',
    9: 'Chronic cholestasis',
    10: 'Common Cold',
    11: 'Dengue',
    12: 'Diabetes ',
    13: 'Dimorphic hemmorhoids(piles)',
    14: 'Drug Reaction',
    15: 'Fungal infection',
    16: 'GERD',
    17: 'Gastroenteritis',
    18: 'Heart attack',
    19: 'Hepatitis B',
    20: 'Hepatitis C',
    21: 'Hepatitis D',
    22: 'Hepatitis E',
    23: 'Hypertension ',
    24: 'Hyperthyroidism',
    25: 'Hypoglycemia',
    26: 'Hypothyroidism',
    27: 'Impetigo',
    28: 'Jaundice',
    29: 'Malaria',
    30: 'Migraine',
    31: 'Osteoarthristis',
    32: 'Paralysis (brain hemorrhage)',
    33: 'Peptic ulcer diseae',
    34: 'Pneumonia',
    35: 'Psoriasis',
    36: 'Tuberculosis',
    37: 'Typhoid',
    38: 'Urinary tract infection',
    39: 'Varicose veins',
    40: 'hepatitis A',
}


def predict_value(patient_ill_name):
    input_vector = np.zeros(len(symptom_dict))
    for i in patient_ill_name:
        input_vector[symptom_dict[i]] = 1
    input_df = pd.DataFrame([input_vector], columns=symptom_dict.keys())
    prediction = svm.predict(input_df)
    return disease_list[prediction[0]]


def get_all_solution(my_problem):
    # my_problem2 = ['itching', 'skin_rash', 'shivering']
    solution = predict_value(my_problem)
    des = list(description[description['Disease'] == solution]['Description'].values)[0]
    dt = list(diet[diet['Disease'] == solution]['Diet'].values)[0]
    med = list(medication[medication['Disease'] == solution]['Medication'].values)[0]
    val = precaution[precaution['Disease'] == solution][
        ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    val = [list(pre) for pre in val.values]
    val = [v for v in val[0]]
    work = list(workout[workout['disease'] == solution]['workout'].values)[0]
    return des, dt, med, val, work, solution


@app.route('/medicine_form', methods=['POST'])
def medicine_form():
    selected_items = request.form
    selected_items_list = list(selected_items.keys())
    description_list, diet_list, medication_list, precaution_list, workout_list, solution=get_all_solution(selected_items_list)
    return render_template('MedicineRecommender.html', description_list=description_list,diet_list=ast.literal_eval(diet_list),medication_list=ast.literal_eval(medication_list),precaution_list=precaution_list,workout_list=workout_list.split(' '),solution=solution)


@app.route('/AboutUs')
def about_us():
    return render_template('AboutUs.html')

@app.route('/SendMessages')
def send_mesages():
    return render_template('SendMessages.html')

@app.route('/Gallery')
def gallery():
    return render_template('Gallery.html',book_name=list(popular_df['Book-Title'].values),
                           author=list(popular_df['Book-Author'].values),
                           image=list(popular_df['Image-URL-M'].values),
                           voting=list(popular_df['num_ratings'].values),
                           rating=list(popular_df['avg_ratings'].values),movie_box=movie_data())

if __name__ == "__main__":
    app.run(debug=True)
