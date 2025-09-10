# sentiment_app.py

import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from pymorphy3 import MorphAnalyzer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import base64
import joblib 
import os 

# ----- Конфигурация страницы и загрузка ресурсов -----

st.set_page_config(
    page_title="Анализ тональности отзывов",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Загрузка необходимых пакетов NLTK
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
download_nltk_resources()

# Функция для кодирования изображения в base64
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Функция для установки фона
def set_background(jpg_file):
    bin_str = get_base64(jpg_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{bin_str}");
        background-size: cover;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

try:
    set_background('sakura.jpg')
except FileNotFoundError:
    st.warning("Файл 'sakura.jpg' не найден. Фон не будет установлен.")

# Кастомный CSS для темы "Ночная сакура"
st.markdown("""
<style>
    .stApp { color: #FFFFFF; }
    h1, h2, h3 { color: #FFC0CB; }
    .css-1d391kg {
        background-color: rgba(0, 0, 0, 0.6);
        border-right: 2px solid #FFC0CB;
    }
    .stButton>button {
        background-color: #FF69B4; color: #FFFFFF; border-radius: 8px; border: 1px solid #FFC0CB;
    }
    .stButton>button:hover {
        background-color: #FFC0CB; color: #000000; border: 1px solid #FF69B4;
    }
    .stTextArea textarea {
        background-color: rgba(30, 30, 30, 0.7); color: #FFFFFF; border: 1px solid #FFC0CB;
    }
    .stMetric {
        background-color: rgba(40, 40, 40, 0.7); border-radius: 10px; padding: 10px; border: 1px solid #FF69B4;
    }
    .stDataFrame { background-color: rgba(0, 0, 0, 0.5); }
</style>
""", unsafe_allow_html=True)

# Стиль для графиков Matplotlib
matplotlib.rc('axes', facecolor='black', edgecolor='pink', labelcolor='white', titlecolor='pink')
matplotlib.rc('xtick', color='white'); matplotlib.rc('ytick', color='white')
matplotlib.rc('figure', facecolor='black', edgecolor='pink')
matplotlib.rc('legend', facecolor='black', edgecolor='pink', labelcolor='white')

# ----- Функции приложения -----

# Кэшируем загрузку модели и векторизатора
@st.cache_resource
def load_model_and_vectorizer():
    """Загружает сохраненную модель и векторизатор из файлов."""
    model_path = 'linear_svc_model.joblib'
    vectorizer_path = 'tfidf_vectorizer.joblib'
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        st.error(
            "Файлы модели не найдены! Пожалуйста, сначала запустите `train_model.py`,"
            "чтобы обучить и сохранить модель."
        )
        st.stop()

    classifier = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return classifier, vectorizer

# Кэшируем NLP инструменты
@st.cache_resource
def get_nlp_tools():
    morph = MorphAnalyzer()
    russian_stopwords = stopwords.words("russian")
    return morph, russian_stopwords

morph, russian_stopwords = get_nlp_tools()

def preprocess_text(text):
    """Очищает и лемматизирует текст (функция должна быть идентична той, что при обучении)."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^а-яА-ЯёЁ\s]', ' ', text)
    words = text.split()
    lemmatized_words = [morph.parse(word)[0].normal_form for word in words if word not in russian_stopwords]
    return " ".join(lemmatized_words)

# --- Основная часть приложения ---

st.title('🌸 Анализ тональности отзывов на фильмы')

# Загружаем модель сразу, чтобы проверить наличие файлов
classifier, vectorizer = load_model_and_vectorizer()
st.session_state['model_loaded'] = True

st.sidebar.header("Панель управления")
app_mode = st.sidebar.selectbox(
    "Выберите раздел:",
    ["Прогнозирование тональности", "Оценка модели"]
)

if app_mode == "Прогнозирование тональности":
    st.header("Предсказать тональность отзыва")
    
    review_text = st.text_area("Введите текст отзыва здесь:", height=200, 
                               placeholder="Сегодня посмотрел новый фильм, и он оказался просто восхитительным!...")

    if st.button("Определить тональность"):
        if review_text.strip() == "":
            st.error("Пожалуйста, введите текст отзыва.")
        else:
            with st.spinner("Анализирую..."):
                processed_text = preprocess_text(review_text)
                vectorized_text = vectorizer.transform([processed_text])
                prediction = classifier.predict(vectorized_text)[0]
                
                st.subheader("Результат анализа:")
                if prediction == 'pos':
                    st.success(f"## Позитивный 👍")
                elif prediction == 'neu':
                    st.warning(f"## Нейтральный 😐")
                else:
                    st.error(f"## Негативный 👎")

elif app_mode == "Оценка модели":
    st.header("Оценка качества обученной модели")
    st.info("Здесь показана производительность модели на валидационном наборе данных `kp_valid.csv`, который не использовался при обучении.")

    try:
        df_valid = pd.read_csv('kp_valid.csv', sep='|').dropna().drop_duplicates()
        
        with st.spinner("Обработка валидационных данных и оценка..."):
            df_valid['processed_review'] = df_valid['review'].apply(preprocess_text)
            X_valid = vectorizer.transform(df_valid['processed_review'])
            y_valid = df_valid['sentiment']

            y_pred_valid = classifier.predict(X_valid)
            valid_accuracy = accuracy_score(y_valid, y_pred_valid)
            report = classification_report(y_valid, y_pred_valid, zero_division=0)
            cm = confusion_matrix(y_valid, y_pred_valid, labels=classifier.classes_)

        st.metric("Точность (Accuracy) на валидационном наборе", f"{valid_accuracy:.2%}")
        
        st.text("Детальный отчет (Classification Report):")
        st.code(report)
        
        st.subheader("Матрица ошибок (Confusion Matrix)")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', 
                    xticklabels=classifier.classes_, yticklabels=classifier.classes_, ax=ax)
        ax.set_xlabel('Предсказанный класс')
        ax.set_ylabel('Истинный класс')
        st.pyplot(fig)

    except FileNotFoundError:
        st.error("Файл 'kp_valid.csv' не найден. Он нужен для оценки модели.")