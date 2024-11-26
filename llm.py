import torch
import json
import sys
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextStreamer
from sentence_transformers import SentenceTransformer
from utils.dbtool import append_to_postgresql

model = AutoModelForCausalLM.from_pretrained("MTruc/model-llama3.2-1b-instruct-orpo-v2")
tokenizer = AutoTokenizer.from_pretrained("MTruc/tokenizer-llama3.2-1b-instruct-orpo-v2")

# Move model to GPU
model = model.to("cuda")
text_streamer = TextStreamer(tokenizer)
alpaca_prompt = """Instruction: Bạn là một chuyên gia tài chính, bạn được cung cấp những chỉ số tài chính của một mô hình dự đoán giá và những tài liệu tài chính kèm theo. Người dùng có thể hỏi về những chỉ số này hoặc những câu hỏi khác, nếu người dùng hỏi về chỉ số hoặc thông tin liên quan đến mô hình, hãy dùng số liệu được cung cấp để trả lời, còn nếu người dùng hỏi những câu hỏi khác không liên quan đến mô hình thì không cần dùng những chỉ số tài chính của mô hình. Nếu gặp câu hỏi không biết hãy nói không, đừng bịa câu trả lời hay lặp lại câu hỏi và context.

### Context:
{}

### Input:
{}

### Response:
{}"""

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load FAISS index và metadata
# def load_faiss_index(faiss_path, metadata_path):
#     print("Đang tải FAISS index và metadata...")
#     index = faiss.read_index(faiss_path)
#     metadata = pd.read_csv(metadata_path).to_dict(orient='records')
#     return index, metadata

# Truy xuất tài liệu từ vector database
def retrieve_context(query, index, metadata, model, k=5):
    query_embedding = model.encode([query], show_progress_bar=False)[0]
    distances, indices = index.search(query_embedding.reshape(1, -1), k)
    results = [metadata[i]['chunk_content'] for i in indices[0]]
    return "\n".join(results)

# Pipeline trả lời câu hỏi
def rag_llm_pipeline(test_data, context, model, retrieval_model, tokenizer, text_streamer, index, metadata, alpaca_prompt, k=5):
    if type(test_data) is str:
      # Tạo input cho mô hình LLM
      rag_context = retrieve_context(test_data, index, metadata, retrieval_model, k)
      combine_context = context + 'Thông tin tài chính bổ sung:' + rag_context 
      inputs = tokenizer(
          [
              alpaca_prompt.format(
                  combine_context,
                  test_data,
                  ""  # Output để trống
              )
          ], return_tensors="pt").to("cuda")
    else:
      rag_context = retrieve_context(test_data['Text'], index, metadata, retrieval_model, k)
      combine_context = test_data['Context'] + 'Thông tin tài chính bổ sung:' + rag_context 
      inputs = tokenizer(
        [
            alpaca_prompt.format(
                combine_context, # context,
                test_data['Text'], # question,
                "", # output - leave this blank for generation
            )
        ], return_tensors = "pt").to("cuda")
    # Sinh phản hồi từ LLM

    outputs = model.generate(**inputs, streamer=text_streamer, max_new_tokens=256)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Lấy phần trả lời
    response_start = generated_text.find("### Response:") + len("### Response:")
    response = generated_text[response_start:].strip()

    return response

def llm_pipeline(test_data, context, model, tokenizer, text_streamer, alpaca_prompt):
    # Truy xuất context
    combine_context = context 

    if type(test_data) is str:
      # Tạo input cho mô hình LLM
      inputs = tokenizer(
          [
              alpaca_prompt.format(
                  combine_context,
                  test_data,
                  ""  # Output để trống
              )
          ], return_tensors="pt").to("cuda")
    else:
      inputs = tokenizer(
        [
            alpaca_prompt.format(
                test_data['Instruction'], # instruction
                test_data['Context']+ ' ' + context, # context,
                test_data['Text'], # question,
                "", # output - leave this blank for generation
            )
        ], return_tensors = "pt").to("cuda")
    # Sinh phản hồi từ LLM

    outputs = model.generate(**inputs, streamer=text_streamer, max_new_tokens=256)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Lấy phần trả lời
    response_start = generated_text.find("### Response:") + len("### Response:")
    response = generated_text[response_start:].strip()
    cleaned_response = response.replace("<|eot_id|>", "").strip()

    return cleaned_response

# Đường dẫn FAISS và metadata
faiss_path = "vector_database.faiss"
metadata_path = "metadata.csv"

if __name__ == "__main__":
    # Get the input question from command-line arguments
    question = sys.argv[1] if len(sys.argv) > 1 else "No question provided"
    context = sys.argv[2]
    if question:
        response = llm_pipeline(question, context, model, tokenizer, text_streamer, alpaca_prompt)
        print(response)
    