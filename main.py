def getForecastByCity(city):
    import csv
    import urllib.request
    import json
    import datetime
    import pytz

    def get_12_hour_forecast(weather_data):
        ukraine_tz = pytz.timezone('Europe/Kiev')

        current_time = datetime.datetime.now(ukraine_tz).time()
        current_hour = current_time.hour
        result = []
        switched = False

        for day in weather_data['days']:
            if 'hours' in day:
                for hour in day['hours']:
                    hour_time = datetime.datetime.strptime(hour['datetime'], "%H:%M:%S").time()
                    if switched and len(result) < 12:
                        result.append(hour)
                    elif hour_time >= current_time and len(result) < 12:
                        result.append(hour)
                    elif len(result) >= 12:
                        break   
            if len(result) >= 12:
                break
            switched = True     

        return result

    #city = input("Введіть назву міста: ")
    url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/" + city + "/next12hours?key=Q9FH7HGQAXYQM4GGQFBHMEC3S&include=hours"
    with urllib.request.urlopen(url) as response:
        data = response.read().decode()
        data_dict = json.loads(data)

        # Отримуємо прогноз погоди на наступні 12 годин
        forecast_data = get_12_hour_forecast(data_dict)

        # Відкриваємо CSV-файл для запису даних
        with open('weather1.csv', mode='w', newline='') as file:
            writer = csv.writer(file)

            # Записуємо заголовки стовпців
            writer.writerow(['hour_datetime', 'hour_datetimeEpoch','hour_temp','hour_humidity','hour_dew', 'hour_precip', 'hour_precipprob', 'hour_snow', 'hour_windgust', 'hour_windspeed', 'hour_winddir', 'hour_pressure', 'hour_visibility','hour_cloudcover' ,'hour_severerisk', 'hour_conditions','city'])

            # Записуємо дані для кожного годинного прогнозу
            for hour in forecast_data:
                writer.writerow([
                    hour['datetime'], # Записуємо дату та час для годинного прогнозу
                    hour['datetimeEpoch'], # Записуємо Unix-час для годинного прогнозу
                    hour['temp'],
                    hour['humidity'],
                    hour['dew'], # Записуємо температуру точки роси
                    hour['precip'], # Записуємо кількість опадів
                    hour['precipprob'], # Записуємо ймовірність опадів
                    hour['snow'], # Записуємо кількість снігу
                    hour['windgust'], # Записуємо швидкість поривів вітру
                    hour['windspeed'], # Записуємо швидкість вітру
                    hour['winddir'], # Записуємо напрямок вітру
                    hour['pressure'], # Записуємо атмосферний тиск
                    hour['visibility'], # Записуємо видимість
                    hour['cloudcover'],
                    hour['severerisk'], # Записуємо рівень ризику небезпечної погоди
                    hour['conditions'], # Записуємо опис погодних умов
                    city
                ])

    print('CSV-файл створено успішно!')

    # %%
    import pandas as pd
    import glob
    import datetime
    import pytz
    import requests
    from bs4 import BeautifulSoup
    import os
    from IPython.display import display, HTML
    # Get the path of the current notebook
    notebook_path = os.getcwd()

    # Specify the folder name
    folder_name = "data/raw_html"

    # Create the folder path by joining the notebook path and folder name
    folder_path = notebook_path #os.path.join(notebook_path, folder_name)

    ukraine_tz = pytz.timezone('Europe/Kiev')
    most_recent_report = datetime.datetime.now(ukraine_tz).date()-datetime.timedelta(days=1)

    base_url = "https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-"

    date_str = most_recent_report.strftime('%B-%d-%Y').replace('-0', '-')
    url = base_url + date_str

    # Виконати запит до сайту та отримати HTML код сторінки
    response = requests.get(url)

    # Перевірити код статусу відповіді сервера
    if response.status_code == 200:
        html = response.content

        # Витягнути з HTML коду всі тексти
        soup = BeautifulSoup(html, "html.parser")
        texts = soup.get_text()

            # Зберегти HTML код у файл з назвою за датою
        file_path = os.path.join(folder_path, date_str + ".html")
        with open(file_path, "w") as f:
            f.write(str(soup))

        # Вивести всі тексти

    # %%
    files_by_days = glob.glob(f"{folder_path}/{date_str}.html")

    # %%
    all_data = []

    for file in files_by_days:
        d = {}

        file_name = file.split("/")[-1].split(".")
        date = datetime.datetime.strptime(file_name[0], '%B-%d-%Y')
        url = file_name[0].split(".")[0]

        with open(file, "r") as cfile:
            parsed_html = BeautifulSoup(cfile.read())
            title = parsed_html.head.find('title').text
            link = parsed_html.head.find('link', attrs={'rel':'canonical'}, href=True).attrs["href"]

            text_title = parsed_html.body.find('h1', attrs={'id':'page-title'}).text
            text_main = parsed_html.body.find('div', attrs={'class':'field field-name-body field-type-text-with-summary field-label-hidden'})

            d = {
                "date":date,
                "short_url":url,
                "title":title,
                "text_title":text_title,
                "full_url":link,
                "main_html":text_main
            }

            all_data.append(d)

    # %%
    df = pd.DataFrame.from_dict(all_data)

    # %%
    df = df.sort_values(by=['date'])

    # %%
    df.to_csv(f"{folder_path}/{date_str}.csv", sep=";", index=False)

    # %%
    df = pd.read_csv(f"{folder_path}/{date_str}.csv", sep=";")

    # %%
    test_row = df.iloc[0]
    page_html_text = test_row["main_html"]

    # %%
    def remove_names_and_dates(page_html_text):
        parsed_html = BeautifulSoup(page_html_text)
        p_lines = parsed_html.findAll('p')
        
        min_sentence_word_count = 13
        p_index = 0
        
        for p_line in p_lines:
            strong_lines = p_line.findAll('strong')
            if not strong_lines:
                p_index += 1
                continue 
                
            for s in strong_lines:
                if len(s.text.split(" ")) >= min_sentence_word_count:
                    break
                else:
                    p_index += 1
                    continue
                    
            for i in range(0, p_index):
                page_html_text = page_html_text.replace(str(p_lines[i]), "")
                
        return page_html_text


    # %%
    df['main_html_v2'] = df['main_html'].apply(lambda x: remove_names_and_dates(x))

    # %%
    test_row = df.iloc[0]
    page_html_text = test_row["main_html_v2"]

    # %%
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import PorterStemmer
    from nltk.stem import WordNetLemmatizer
    from collections import Counter
    from num2words import num2words


    import nltk
    import string
    import numpy as np
    import copy 
    import pickle
    import re
    import math

    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer

    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

    # %%
    pattern = "\[(\d+)\]"

    df['main_html_v3'] = df['main_html_v2'].apply(lambda x: re.sub(pattern,"",str(x)))

    # %%
    df['main_html_v4'] = df['main_html_v3'].apply(lambda x: BeautifulSoup(x).text)

    # %%
    df['main_html_v5'] = df['main_html_v4'].apply(lambda x: re.sub(r'http(\S+.*\s)', "", x))

    # %%
    df['main_html_v6'] = df['main_html_v5'].apply(lambda x: re.sub(r'(©2022|©2023|2022|2023)', "", x))

    # %%
    df['main_html_v7'] = df['main_html_v6'].apply(lambda x: re.sub(r'\n', "", x))

    # %%
    df['main_html_v8'] = df['main_html_v7'].apply(lambda x: re.sub(r'\xa0', "", x))

    # %%
    df2 = df.drop(['main_html_v2','main_html_v3','main_html_v4','main_html_v5','main_html_v6','main_html_v7'],axis=1)

    # %%
    df2.to_csv(f"{folder_path}/{date_str}.csv", sep=";", index=False)

    # %%
    data = pd.read_csv(f"{folder_path}/{date_str}.csv", sep=";")

    # %%
    def remove_one_letter_word(data):
        words = word_tokenize(str(data))
        
        new_text = ""
        for w in words:
            if (len(w) > 1):
                new_text = new_text + " " + w
                
        return new_text

    def convert_lower_case(data):
        return np.char.lower(data)

    stop_words = set(stopwords.words('english'))

    # %%
    def remove_stop_words(data):
        stop_words = set(stopwords.words('english'))
        stop_stop_words = {"no","not"}
        
        stop_words = stop_words - stop_stop_words
        
        words = word_tokenize(str(data))
        
        new_text = ""
        for w in words:
            if w not in stop_words and len(w) > 1:
                new_text = new_text + " " + w
                
        return new_text

    def remove_punctuations(data):
        symbols = " !()-[]{};:'\,<>./?@#$%^&*_~ '"""  
        
        for i in range(len(symbols)):
            data = np.char.replace(data,symbols[i]," ")
            data = np.char.replace(data,"  "," ")
        
        data = np.char.replace(data,",", " ")
        
        return data
                

    def stemming(data):
        stemmer = PorterStemmer()
        
        tokens = word_tokenize(str(data))
        new_text = ""
        for w in tokens:
            new_text = new_text + " " + stemmer.stem(w)
            
        return new_text

    def lemmatizing(data):
        lemmatizer = WordNetLemmatizer()
        
        tokens = word_tokenize(str(data))
        new_text = ""
        for w in tokens:
            new_text = new_text + " " + lemmatizer.lemmatize(w)
                
        return new_text

    def convert_numbers(data):
        
        tokens = word_tokenize(str(data))
        new_text = ""
        for w in tokens:
            if w.isdigit():
                if int(w) < 1000000000:
                    w = num2words(w)
                else:
                    w = " "
            new_text = new_text + " " + w
                
        new_text = np.char.replace(new_text,"-"," ")
            
        return new_text
                    
                    

    def remove_url_from_string(data):
        words = word_tokenize(str(data))
        new_text = ""

        
        for w in words:
            w = re.sub(r'https?:\/\/.*[\r\n]*',"",str(w),flags=re.MULTILINE)
            w = re.sub(r'http?:\/\/.*[\r\n]*',"",str(w),flags=re.MULTILINE)
            
            new_text = new_text + " " + w
            
        return new_text

    def preprocess(data, word_root_algo='lemm'):
        
        data = remove_one_letter_word(data)
        data = convert_lower_case(data)
        data = remove_stop_words(data)
        data = stemming(data)
        data = remove_punctuations(data)
        data = convert_numbers(data)
        data = remove_url_from_string(data)
        
        if word_root_algo =='lemm':
            data = lemmatizing(data)
        else: 
            data = stemming(data)
            
        
        data = remove_punctuations(data)
        data = remove_stop_words(data)
        
        return data

    # %%
    data['report_text_lemm'] = data['main_html_v8'].apply(lambda x: preprocess(x,"lemm"))

    # %%
    data['report_text_stemm'] = data['main_html_v8'].apply(lambda x: preprocess(x,"stemm"))

    # %%
    docs = data['report_text_lemm'].tolist()

    # %%
    cv = CountVectorizer()
    word_count_vector = cv.fit_transform(docs)

    word_count_vector.shape

    # %%
    with open("count_vectorizer_v1.pkl","wb") as handle:
        pickle.dump(cv, handle)

    # %%
    tfidf_transformer = TfidfTransformer(smooth_idf = True, use_idf = True)
    tfidf_transformer.fit(word_count_vector)

    # %%
    with open("tfidf_transformer_v1.pkl","wb") as handle:
        pickle.dump(tfidf_transformer, handle)

    # %%
    df_idf = pd.DataFrame(tfidf_transformer.idf_, index = cv.get_feature_names_out(),columns=["idf_weights"])
    df_idf.sort_values(by=['idf_weights'])

    # %%
    tf_idf_vector = tfidf_transformer.transform(word_count_vector)

    # %%
    tfidf = pickle.load(open("tfidf_transformer_v1.pkl","rb"))
    cv = pickle.load(open("count_vectorizer_v1.pkl","rb"))

    # %%
    def sort_coo(coo_matrix):
        
        tuples = zip(coo_matrix.col, coo_matrix.data)
        
        return sorted(tuples,key=lambda x: (x[1], x[0]), reverse=True)

    def extract_topn_from_vector(feature_names, sorted_items, topn=10):
        
        sorted_items = sorted_items[:topn]
        
        score_vals = []
        feature_vals = []
        
        for idx, score in sorted_items:
            
            score_vals.append(round(score,3))
            feature_vals.append(feature_names[idx])
            
        results = {}
        
        for idx in range(len(feature_vals)):
            results[feature_vals[idx]] = score_vals[idx]
        
        return results


    def convert_doc_to_vector(doc):
        feature_names = cv.get_feature_names_out()
        top_n = 10
        tf_idf_vector = tfidf.transform(cv.transform([doc]))
        
        sorted_items = sort_coo(tf_idf_vector.tocoo())
        
        keywords = extract_topn_from_vector(feature_names, sorted_items, top_n)
        
        return keywords

    # %%
    data['keywords'] = data['report_text_stemm'].apply(lambda x: convert_doc_to_vector(x))

    # %%
    data.to_csv(f"{folder_path}/{date_str}.csv", sep=";", index=False)

    # %%
    # Завантажуємо файли
    weather_df = pd.read_csv('weather1.csv', sep=",")
    keywords_df = pd.read_csv(f"{folder_path}/{date_str}.csv", sep=";")

    # %%
    # Об'єднуємо файли
    combined_df = pd.concat([weather_df]*len(keywords_df), ignore_index=True)

    # %%
    import numpy as np
    combined_df['keywords'] = np.tile(keywords_df['keywords'], len(weather_df))

    # %%
    # Зберігаємо дані в новому файлі
    combined_df.to_csv('predict.csv', index=False)

    # %%
    combined_df.head(12)

    # %%
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.linear_model import LogisticRegression
    import joblib

    # Відновлення моделі та OneHotEncoder з файлу
    model_logistic, ohe = joblib.load('logistic.pkl')

    # Приклад використання моделі та OneHotEncoder для передбачення на нових даних
    new_data = pd.read_csv('predict.csv')
    new_data_encoded = ohe.transform(new_data)
    y_pred = model_logistic.predict(new_data_encoded)
    y_pred = list(map(lambda x: True if x == 1.0 else False, y_pred))
    # Злиття передбачень та часу на який робиться прогноз
    result = list(zip(new_data['hour_datetime'], y_pred))

    return result



