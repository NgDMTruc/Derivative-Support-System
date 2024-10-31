import pandas as pd
import re

def clean_text(text):
    # Sử dụng regex để tìm chuỗi "Tin tức sự kiện về" cùng 8 ký tự trước đó và xóa tất cả phần phía sau, bao gồm xuống dòng
    cleaned_text = re.sub(r'.{8}Tin tức sự kiện về.*(\n.*)*', '', text)
    return cleaned_text.strip()

news = pd.read_csv('articles_data.csv')

data = pd.DataFrame()

data['Text'] = news['Tiêu đề'] + ' - ' + news['Nội dung H2']
data['Text'] = data['Text'].dropna().apply(clean_text)
data=data.dropna()

data.to_csv('sentiment.csv', index=False)