# File content

## crawl_sentiment.py
This file crawl data on cafef.vn, it will continuosly scroll down and press "Xem thêm" button if available.

## process_data.py
This file clean data after crawled from the web.

## prompt_price_movement.ipynb
This file use gpt4o to predict price movement base on web data after processed.

## return_price_movement_corr.ipynb
This file calculate the correlation between generated price movement and price (close value) percentage change, the result is low so we decided not to use this.

## prompt_QA_data.ipynb
This file use gpt4o to generate data for question answering system, it will base on some context, and there are refer answers and reject answers.

## prompt_embed_QA.ipynb
This file use gpt4o to generate data relate to context in RAG database, which will be used to evaluate embed models.