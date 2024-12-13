import openai
import pandas as pd

# Đọc dữ liệu từ file CSV
input_file = "Data-finetune - eval.csv"  # Đổi tên file theo đúng file CSV của bạn
data = pd.read_csv(input_file)
# Hàm để đánh giá thông tin bằng GPT
api_key = "" #
openai.api_key = api_key
# Hàm gọi API GPT để tạo câu hỏi
def generate_questions(content):
    prompt = f"""
    Tạo ra hai câu hỏi liên quan đến nội dung sau để kiểm tra khả năng hiểu nội dung (embedding) và tìm kiếm thông tin (retrieval):
    Nội dung: "{content}"
    Trả về kết quả bằng tiếng Việt với format:
    - Question 1: ...
    - Question 2: ...
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are a knowledgeable assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        # Lấy kết quả từ API
        answer = response.choices[0].message.content.strip()
        questions = answer.split("\n")
        question_1 = questions[0].replace("- Question 1: ", "").strip()
        question_2 = questions[1].replace("- Question 2: ", "").strip()
        return question_1, question_2
    except Exception as e:
        print(f"Error occurred: {e}")
        return "Error generating question 1", "Error generating question 2"

# Tạo bộ dataset mới
results = []
for index, row in data.iterrows():
    content = row['Nội dung']
    question_1, question_2 = generate_questions(content)
    results.append([question_1, content])
    results.append([question_2, content])

# Chuyển kết quả thành DataFrame
new_dataset = pd.DataFrame(results, columns=['Question', 'Content'])

# Lưu bộ dataset mới ra file CSV
output_file = "dataset_with_questions_flat.csv"
new_dataset.to_csv(output_file, index=False)

print(f"Dataset mới đã được lưu tại: {output_file}")