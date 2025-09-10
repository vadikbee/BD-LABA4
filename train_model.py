# train_model.py (v3 - High Accuracy)

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from pymorphy3 import MorphAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression # Изменено с LinearSVC
import joblib

print("--- Начало процесса обучения МОДЕЛИ С ПОВЫШЕННОЙ ТОЧНОСТЬЮ ---")

# --- 1. Загрузка и подготовка инструментов NLP ---
print("Шаг 1/6: Подготовка NLP-инструментов...")
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Загрузка стоп-слов NLTK...")
    nltk.download('stopwords')

morph = MorphAnalyzer()
russian_stopwords = stopwords.words("russian")
# РАСШИРЕННЫЙ СПИСОК СТОП-СЛОВ: Добавлены специфичные для домена слова
russian_stopwords.extend([
    'это', 'весь', 'который', 'свой', 'просто', 'еще', 'фильм', 'кино',
    'очень', 'также', 'этот', 'самый', 'когда', 'однако', 'только', 'сказать',
    'вообще', 'например', 'какой-то', 'сцена', 'сюжет', 'актер', 'роль',
    'картина', 'режиссер', 'просмотр', 'история', 'персонаж', 'герой'
])

def preprocess_text(text):
    """Очищает и лемматизирует текст."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^а-яА-ЯёЁ\s]', ' ', text)
    words = text.split()
    lemmatized_words = [morph.parse(word)[0].normal_form for word in words if word not in russian_stopwords]
    return " ".join(lemmatized_words)

# --- 2. Загрузка и обработка данных ---
try:
    print("Шаг 2/6: Загрузка тренировочных данных 'kp_train.csv'...")
    df_train = pd.read_csv('kp_train.csv', sep='|').dropna().drop_duplicates().reset_index(drop=True)
except FileNotFoundError:
    print("\n[ОШИБКА] Файл 'kp_train.csv' не найден.")
    exit()

print("Шаг 3/6: Предобработка текста... (Это может занять несколько минут)")
df_train['processed_review'] = df_train['review'].apply(preprocess_text)

# --- 3. Векторизация с улучшенными параметрами ---
print("Шаг 4/6: Векторизация текста с помощью настроенного TF-IDF...")
# Изменен max_features для лучшей генерализации
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.7,
    max_features=40000
)

X_train = vectorizer.fit_transform(df_train['processed_review'])
y_train = df_train['sentiment']

# --- 4. Обучение модели с улучшенными параметрами ---
print("Шаг 5/6: Обучение модели LogisticRegression...")

# ЗАМЕНА КЛАССИФИКАТОРА НА БОЛЕЕ ЭФФЕКТИВНЫЙ
classifier = LogisticRegression(
    random_state=42,
    class_weight='balanced',
    solver='liblinear',
    C=0.5,
    max_iter=1000
)

classifier.fit(X_train, y_train)

# --- 5. Сохранение модели и векторизатора ---
print("Шаг 6/6: Сохранение обученной модели и векторизатора в файлы...")
joblib.dump(classifier, 'linear_svc_model.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

print("\n--- Процесс обучения МОДЕЛИ С ПОВЫШЕННОЙ ТОЧНОСТЬЮ успешно завершен! ---")
print("Созданы обновленные файлы 'linear_svc_model.joblib' и 'tfidf_vectorizer.joblib'.")