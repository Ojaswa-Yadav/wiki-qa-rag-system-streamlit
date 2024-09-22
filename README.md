# Wiki QA RAG System

This project implements a Multilingual Question Answering (QA) system using Retrieval-Augmented Generation (RAG) on the Wiki QA dataset. It provides an interactive interface built with Streamlit for exploring the dataset, asking questions, evaluating the system's performance, and fine-tuning the model.

## Features

- Semantic search using SBERT (Sentence-BERT) embeddings
- Question answering using a fine-tuned XLM-RoBERTa model
- Answer generation using a M2M100 multilingual language model
- Interactive exploration of the Wiki QA dataset
- Multilingual support for queries and answers
- System evaluation with ROUGE and BLEU scores
- Active learning capabilities for continuous improvement
- Model fine-tuning functionality
- Pinecone integration for efficient vector search

## Installation

1. Clone this repository:

   git clone https://github.com/Ojaswa-Yadav/wiki-qa-rag-system-streamlit
   
2. Install the required dependencies:

   pip install -r requirements.txt

3. Set up a Pinecone account and create an API key.

4. Set the Pinecone API key as an environment variable

## Usage

Run the Streamlit app locally:

streamlit run app.py

## Deployment

This app is designed to be deployed on Streamlit Cloud. To deploy:

1. Push your code to a GitHub repository.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud) and connect your GitHub account.
3. Select the repository and branch containing your Streamlit app.
4. Set the `PINECONE_API_KEY` as a secret in the Streamlit Cloud dashboard.
5. Deploy the app.

## How It Works

1. **Data Indexing**: The system loads the Wiki QA dataset and creates embeddings for each document using SBERT. These embeddings are then indexed in Pinecone for efficient similarity search.

2. **Question Answering**: 
   - When a query is received, the system performs a semantic search to find relevant documents.
   - It then uses an XLM-RoBERTa model to extract a specific answer from the retrieved documents.
   - Additionally, it uses an M2M100 model to generate a more comprehensive answer.

3. **Evaluation**: The system can evaluate its performance using ROUGE and BLEU scores on a subset of the Wiki QA dataset.

4. **Active Learning**: The system identifies challenging questions and stores them for potential model improvement.

5. **Fine-tuning**: Users can fine-tune the model on a subset of the Wiki QA dataset to improve its performance.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
   








   
