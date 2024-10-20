from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import requests

# Thiết lập trình điều khiển WebDriver
driver = webdriver.Chrome(executable_path='D:\Study file\Fall 2024 - Capstone\Capstone\chromedriver-win64\chromedriver.exe')  # Thay thế đường dẫn đến ChromeDriver của bạn
driver.get("https://cafef.vn/thi-truong-chung-khoan.chn")

# Scroll xuống hoặc ấn nút "tiếp theo" để tải thêm bài viết
while True:
    try:
        load_more_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//button[text()="Tiếp"]'))  # Thay đổi nếu button có text khác
        )
        load_more_button.click()
        time.sleep(2)  # Chờ một lát để trang web tải thêm nội dung
    except:
        # Khi không còn nút để tải thêm, thoát khỏi vòng lặp
        break

# Lấy nội dung trang đã tải
page_source = driver.page_source
soup = BeautifulSoup(page_source, 'html.parser')

# Lấy các tiêu đề bài viết và link bài viết
articles = soup.find_all('h3', class_='title')  # Thay đổi selector nếu cần thiết
article_links = []

for article in articles:
    title = article.get_text().strip()
    link = article.find('a')['href']
    if not link.startswith('http'):
        link = 'https://cafef.vn' + link  # Thêm URL đầy đủ nếu chỉ có đường dẫn tương đối
    article_links.append(link)
    print(f'Tiêu đề: {title}, Link: {link}')

# Vào từng bài viết để lấy nội dung h2
for link in article_links:
    response = requests.get(link)
    article_soup = BeautifulSoup(response.content, 'html.parser')
    h2_content = article_soup.find_all('h2')
    h2_texts = [h2.get_text().strip() for h2 in h2_content]
    print(f'Nội dung H2 của bài báo {link}: {h2_texts}')

# Đóng trình điều khiển
driver.quit()
