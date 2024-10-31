import requests
from bs4 import BeautifulSoup
import csv
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Thiết lập tên file CSV để lưu kết quả
output_file = "articles_data.csv"

# Thiết lập trình điều khiển WebDriver
driver = webdriver.Chrome()
driver.get("https://cafef.vn/thi-truong-chung-khoan.chn")

# Scroll xuống và nhấn nút "Xem thêm" để tải thêm bài viết
scroll_attempts = 0  # Đếm số lần cuộn trang
max_scroll_attempts = 5  # Số lần cuộn tối đa trước khi kiểm tra nút "Xem thêm"

while True:
    # Cuộn xuống cuối trang để tải thêm nội dung
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(3)  # Chờ một lát để trang web tải thêm nội dung
    scroll_attempts += 1

    # Sau khi cuộn một số lần nhất định, kiểm tra xem nút "Xem thêm" đã xuất hiện chưa
    if scroll_attempts >= max_scroll_attempts:
        try:
            # Tìm nút "Xem thêm" bằng cách tìm phần tử với class `btn-viewmore` và nhấp vào nó
            load_more_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//div[@class='btn-viewmore']"))
            )
            load_more_button.click()
            time.sleep(3)  # Chờ một lát để trang web tải thêm nội dung
            scroll_attempts = 0  # Đặt lại đếm số lần cuộn
        except:
            print('Không tìm thấy nút "Xem thêm" hoặc không còn nội dung để tải.')
            break
    # Lấy nội dung trang đã tải
    try:
        page_source = driver.page_source
        page_source_2 = page_source
    except:
        break

soup = BeautifulSoup(page_source_2, 'html.parser')

# Lấy các tiêu đề bài viết và link bài viết
articles = soup.find_all('h3', class_=False)
article_links = []

for article in articles:
    link_element = article.find('a')
    if link_element:
        title = link_element.get_text().strip()
        link = link_element['href']
        if not link.startswith('http'):
            link = 'https://cafef.vn' + link  # Thêm URL đầy đủ nếu chỉ có đường dẫn tương đối
        article_links.append((title, link))

# Đóng trình điều khiển
driver.quit()

# Mở file CSV để ghi dữ liệu
with open(output_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # Ghi header cho CSV
    writer.writerow(["Tiêu đề", "Link", "Nội dung H2"])

    # Vào từng bài viết để lấy nội dung h2 và ghi vào CSV
    for title, link in article_links:
        try:
            response = requests.get(link)
            if response.status_code == 200:
                article_soup = BeautifulSoup(response.content, 'html.parser')
                h2_content = article_soup.find_all('h2')

                # Lấy nội dung của tất cả thẻ <h2>
                h2_texts = [h2.get_text().strip() for h2 in h2_content if h2.get_text().strip()]
                h2_combined = "; ".join(h2_texts)  # Ghép nội dung <h2> thành một chuỗi, ngăn cách bởi dấu chấm phẩy

                # Ghi dữ liệu vào CSV
                writer.writerow([title, link, h2_combined])

            else:
                print(f"Lỗi khi truy cập link: {link} - Mã trạng thái HTTP: {response.status_code}")

        except Exception as e:
            print(f"Lỗi khi xử lý bài viết {link}: {e}")

print(f"Dữ liệu đã được lưu vào file {output_file}")
