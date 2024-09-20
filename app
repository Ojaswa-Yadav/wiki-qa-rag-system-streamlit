import streamlit as st
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from tqdm import tqdm
from functools import lru_cache
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu
import logging

# Download NLTK data
nltk.download('punkt', quiet=True)

class EnhancedQARAGSystem:
    def __init__(self, sbert_model='all-MiniLM-L6-v2', qa_model='deepset/roberta-base-squad2', 
                 lm_model='google/flan-t5-base', max_length=512):
        self.sbert_model = SentenceTransformer(sbert_model)
        self.index = None
        self.documents = None
        self.questions = None
        
        self.qa_tokenizer = AutoTokenizer.from_pretrained(qa_model)
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model)
        
        self.lm_tokenizer = AutoTokenizer.from_pretrained(lm_model)
        self.lm_model = AutoModelForSeq2SeqLM.from_pretrained(lm_model)
        
        self.max_length = max_length
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def load_dataset(self, dataset_name='wiki_qa', split='train', question_column='question', 
                     context_column='answer', max_samples=None):
        try:
            dataset = load_dataset(dataset_name, split=split)
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            self.documents = dataset[context_column]
            self.questions = dataset[question_column]
            return len(self.documents)
        except Exception as e:
            logging.error(f"Error loading dataset: {str(e)}")
            return 0

    def encode_documents(self, batch_size=32):
        try:
            embeddings = []
            for i in tqdm(range(0, len(self.documents), batch_size), desc="Encoding documents"):
                batch = self.documents[i:i+batch_size]
                batch_embeddings = self.sbert_model.encode(batch, convert_to_tensor=True)
                embeddings.append(batch_embeddings)
            return torch.cat(embeddings).cpu().numpy()
        except Exception as e:
            logging.error(f"Error encoding documents: {str(e)}")
            return None

    def build_index(self, embeddings):
        try:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings)
        except Exception as e:
            logging.error(f"Error building index: {str(e)}")

    @lru_cache(maxsize=1000)
    def semantic_search(self, query, k=5):
        try:
            query_vector = self.sbert_model.encode([query])[0]
            distances, indices = self.index.search(np.array([query_vector]), k)
            results = [
                (self.documents[i], float(distances[0][j]))
                for j, i in enumerate(indices[0])
            ]
            return results
        except Exception as e:
            logging.error(f"Error in semantic search: {str(e)}")
            return []

    def extract_answer(self, question, context):
        try:
            inputs = self.qa_tokenizer.encode_plus(question, context, return_tensors="pt", max_length=self.max_length, truncation=True)
            with torch.no_grad():
                outputs = self.qa_model(**inputs)
            answer_start = torch.argmax(outputs.start_logits)
            answer_end = torch.argmax(outputs.end_logits) + 1
            answer = self.qa_tokenizer.convert_tokens_to_string(self.qa_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
            return answer
        except Exception as e:
            logging.error(f"Error extracting answer: {str(e)}")
            return "Unable to extract answer"

    def generate_answer(self, question, context):
        try:
            input_text = f"Question: {question}\nContext: {context}\nAnswer:"
            inputs = self.lm_tokenizer(input_text, return_tensors="pt", max_length=self.max_length, truncation=True)
            with torch.no_grad():
                outputs = self.lm_model.generate(**inputs, max_length=self.max_length, num_return_sequences=1, temperature=0.7)
            answer = self.lm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return answer
        except Exception as e:
            logging.error(f"Error generating answer: {str(e)}")
            return "Unable to generate answer"

    def process_query(self, query, num_relevant_docs=3):
        try:
            relevant_docs = self.semantic_search(query, k=num_relevant_docs)
            context = " ".join([doc for doc, _ in relevant_docs])
            
            extracted_answer = self.extract_answer(query, context)
            generated_answer = self.generate_answer(query, context)
            
            return {
                "query": query,
                "relevant_documents": relevant_docs,
                "extracted_answer": extracted_answer,
                "generated_answer": generated_answer
            }
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            return {
                "query": query,
                "error": "Unable to process query"
            }

    def evaluate_performance(self, test_set):
        rouge_scores = []
        bleu_scores = []
        
        for question, reference_answer in test_set:
            result = self.process_query(question)
            generated_answer = result['generated_answer']
            
            # Calculate ROUGE scores
            rouge_score = self.scorer.score(reference_answer, generated_answer)
            rouge_scores.append(rouge_score)
            
            # Calculate BLEU score
            reference_tokens = nltk.word_tokenize(reference_answer)
            generated_tokens = nltk.word_tokenize(generated_answer)
            bleu_score = sentence_bleu([reference_tokens], generated_tokens)
            bleu_scores.append(bleu_score)
        
        # Average scores
        avg_rouge = {key: np.mean([score[key].fmeasure for score in rouge_scores]) for key in rouge_scores[0].keys()}
        avg_bleu = np.mean(bleu_scores)
        
        return {
            "rouge_scores": avg_rouge,
            "bleu_score": avg_bleu
        }

    def fine_tune(self, train_dataset, eval_dataset, output_dir="./fine_tuned_model"):
        # Prepare the datasets
        train_encodings = self.lm_tokenizer(train_dataset["input_text"], truncation=True, padding=True)
        eval_encodings = self.lm_tokenizer(eval_dataset["input_text"], truncation=True, padding=True)

        train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_encodings["input_ids"]), torch.tensor(train_encodings["attention_mask"]))
        eval_dataset = torch.utils.data.TensorDataset(torch.tensor(eval_encodings["input_ids"]), torch.tensor(eval_encodings["attention_mask"]))

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
        )

        # Initialize Trainer
        trainer = Trainer(
            model=self.lm_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )

        # Fine-tune the model
        trainer.train()

        # Save the fine-tuned model
        self.lm_model.save_pretrained(output_dir)
        self.lm_tokenizer.save_pretrained(output_dir)

# Initialize the system
@st.cache_resource
def initialize_system():
    system = EnhancedQARAGSystem()
    num_docs = system.load_dataset(max_samples=10000)  # Limit to 10,000 documents for this example
    st.write(f"Loaded {num_docs} documents from Wiki QA dataset")
    
    embeddings = system.encode_documents()
    system.build_index(embeddings)
    st.write("Index built successfully")
    return system

# Streamlit UI
st.title("Wiki QA and RAG System")

# Initialize system
system = initialize_system()

# Dataset exploration
st.header("Explore Wiki QA Dataset")
if st.checkbox("Show sample questions and answers"):
    num_samples = st.slider("Number of samples to display", 1, 20, 5)
    samples = list(zip(system.questions, system.documents))[:num_samples]
    for i, (question, answer) in enumerate(samples, 1):
        st.write(f"{i}. Question: {question}")
        st.write(f"   Answer: {answer}")
        st.write("---")

# Query section
st.header("Ask a Question")
query = st.text_input("Enter your question:")
use_wiki_question = st.checkbox("Use a random question from Wiki QA dataset")

if use_wiki_question:
    random_index = np.random.randint(0, len(system.questions))
    query = system.questions[random_index]
    st.write(f"Selected question: {query}")

if st.button("Submit"):
    with st.spinner("Processing query..."):
        result = system.process_query(query)
    
    st.subheader("Results")
    st.write(f"Query: {result['query']}")
    st.write(f"Extracted Answer: {result['extracted_answer']}")
    st.write(f"Generated Answer: {result['generated_answer']}")
    
    st.subheader("Relevant Documents")
    for i, (doc, score) in enumerate(result['relevant_documents'], 1):
        st.write(f"{i}. {doc[:100]}... (Score: {score:.4f})")

# Evaluation section
st.header("System Evaluation")
st.write("Evaluate using Wiki QA dataset:")
num_eval_samples = st.number_input("Number of random samples for evaluation", min_value=1, max_value=100, value=10)

if st.button("Evaluate"):
    with st.spinner("Evaluating..."):
        eval_indices = np.random.choice(len(system.questions), num_eval_samples, replace=False)
        test_set = [(system.questions[i], system.documents[i]) for i in eval_indices]
        result = system.evaluate_performance(test_set)
    
    st.subheader("Evaluation Results")
    st.write(f"ROUGE Scores: {result['rouge_scores']}")
    st.write(f"BLEU Score: {result['bleu_score']}")

    st.subheader("Evaluated Samples")
    for i, (question, answer) in enumerate(test_set, 1):
        st.write(f"{i}. Question: {question}")
        st.write(f"   Reference Answer: {answer}")
        st.write("---")

# Fine-tuning section
st.header("Fine-tune the Model")
st.write("Fine-tune using Wiki QA dataset:")
num_train = st.number_input("Number of training samples", min_value=10, max_value=1000, value=100)
num_eval = st.number_input("Number of evaluation samples", min_value=10, max_value=100, value=20)

if st.button("Fine-tune"):
    with st.spinner("Fine-tuning the model..."):
        train_indices = np.random.choice(len(system.questions), num_train, replace=False)
        eval_indices = np.random.choice(len(system.questions), num_eval, replace=False)
        
        train_dataset = {
            "input_text": [f"Question: {system.questions[i]} Context: {system.documents[i]}" for i in train_indices]
        }
        eval_dataset = {
            "input_text": [f"Question: {system.questions[i]} Context: {system.documents[i]}" for i in eval_indices]
        }
        
        system.fine_tune(train_dataset, eval_dataset)
    st.success("Fine-tuning completed!")
