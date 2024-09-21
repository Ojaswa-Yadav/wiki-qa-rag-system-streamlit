import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm
from functools import lru_cache
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu
import logging
from langdetect import detect
from scipy.spatial.distance import cosine
import pinecone
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, M2M100ForConditionalGeneration, M2M100Tokenizer, TrainingArguments, Trainer

nltk.download('punkt', quiet=True)

class EnhancedQARAGSystem:
    def __init__(self, sbert_model='paraphrase-multilingual-mpnet-base-v2', 
                 qa_model='deepset/xlm-roberta-large-squad2', 
                 lm_model='facebook/m2m100_418M', max_length=512):
        self.sbert_model = SentenceTransformer(sbert_model)
        self.index = None
        self.documents = None
        self.questions = None
        
        self.qa_tokenizer = AutoTokenizer.from_pretrained(qa_model)
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model)
        
        self.lm_tokenizer = M2M100Tokenizer.from_pretrained(lm_model)
        self.lm_model = M2M100ForConditionalGeneration.from_pretrained(lm_model)
        
        self.max_length = max_length
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        self.active_learning_queue = []

        # Initialize Pinecone
        pinecone.init(api_key="YOUR_PINECONE_API_KEY", environment="YOUR_PINECONE_ENVIRONMENT")
        self.index_name = "wiki-qa-index"
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(self.index_name, dimension=768)  # 768 is the dimension for 'paraphrase-multilingual-mpnet-base-v2'
        self.pinecone_index = pinecone.Index(self.index_name)

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
            ids = []
            for i in tqdm(range(0, len(self.documents), batch_size), desc="Encoding documents"):
                batch = self.documents[i:i+batch_size]
                batch_embeddings = self.sbert_model.encode(batch, convert_to_tensor=True)
                embeddings.extend(batch_embeddings.tolist())
                ids.extend([str(j) for j in range(i, min(i+batch_size, len(self.documents)))])
            return ids, embeddings
        except Exception as e:
            logging.error(f"Error encoding documents: {str(e)}")
            return None, None

    def build_index(self, ids, embeddings):
        vectors = list(zip(ids, embeddings, [{"text": doc} for doc in self.documents]))
        self.pinecone_index.upsert(vectors=vectors)

    def detect_language(self, text):
        return detect(text)

    def translate(self, text, target_lang):
        src_lang = self.detect_language(text)
        if src_lang == target_lang:
            return text
        
        self.lm_tokenizer.src_lang = src_lang
        encoded = self.lm_tokenizer(text, return_tensors="pt")
        generated_tokens = self.lm_model.generate(**encoded, forced_bos_token_id=self.lm_tokenizer.get_lang_id(target_lang))
        return self.lm_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    def semantic_search(self, query, k=5):
        query_vector = self.sbert_model.encode([query])[0].tolist()
        results = self.pinecone_index.query(query_vector, top_k=k, include_metadata=True)
        return [(match['metadata']['text'], float(match['score'])) for match in results['matches']]

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
            query_lang = self.detect_language(query)
            english_query = self.translate(query, 'en')
            
            relevant_docs = self.semantic_search(english_query, k=num_relevant_docs)
            context = " ".join([doc for doc, _ in relevant_docs])
            
            extracted_answer = self.extract_answer(english_query, context)
            generated_answer = self.generate_answer(english_query, context)
            
            extracted_answer_translated = self.translate(extracted_answer, query_lang)
            generated_answer_translated = self.translate(generated_answer, query_lang)
            
            confidence = self.calculate_confidence(extracted_answer, generated_answer)
            
            if confidence < 0.5:  # Threshold for active learning
                self.active_learning_queue.append({
                    'query': query,
                    'extracted_answer': extracted_answer,
                    'generated_answer': generated_answer,
                    'confidence': confidence
                })
            
            return {
                "query": query,
                "relevant_documents": relevant_docs,
                "extracted_answer": extracted_answer_translated,
                "generated_answer": generated_answer_translated,
                "confidence": confidence
            }
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            return {
                "query": query,
                "error": "Unable to process query"
            }

    def calculate_confidence(self, extracted_answer, generated_answer):
        extracted_embedding = self.sbert_model.encode([extracted_answer])[0]
        generated_embedding = self.sbert_model.encode([generated_answer])[0]
        similarity = 1 - cosine(extracted_embedding, generated_embedding)
        return similarity

    def get_active_learning_samples(self, n=5):
        samples = sorted(self.active_learning_queue, key=lambda x: x['confidence'])[:n]
        self.active_learning_queue = self.active_learning_queue[n:]
        return samples

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

    def fine_tune(self, train_dataset, eval_dataset, output_dir="./fine_tuned_model", num_epochs=3):
        train_encodings = self.lm_tokenizer(train_dataset["input_text"], truncation=True, padding=True)
        train_labels = self.lm_tokenizer(train_dataset.get("labels", [""]), truncation=True, padding=True)
        eval_encodings = self.lm_tokenizer(eval_dataset["input_text"], truncation=True, padding=True)
        eval_labels = self.lm_tokenizer(eval_dataset.get("labels", [""]), truncation=True, padding=True)
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(train_encodings["input_ids"]),
            torch.tensor(train_encodings["attention_mask"]),
            torch.tensor(train_labels["input_ids"])
        )
        eval_dataset = torch.utils.data.TensorDataset(
            torch.tensor(eval_encodings["input_ids"]),
            torch.tensor(eval_encodings["attention_mask"]),
            torch.tensor(eval_labels["input_ids"])
        )
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            warmup_steps=0,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=1,
            evaluation_strategy="steps",
            eval_steps=1,
            save_steps=1,
            load_best_model_at_end=True,
        ) 
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            decoded_preds = self.lm_tokenizer.batch_decode(predictions, skip_special_tokens=True)
            decoded_labels = self.lm_tokenizer.batch_decode(labels, skip_special_tokens=True)
        
            # You can add more sophisticated metrics here
            return {"accuracy": int(decoded_preds[0] == decoded_labels[0])}
            # Initialize Trainer
        trainer = Trainer(
            model=self.lm_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )
        # Fine-tune the model
        trainer.train()
        # Save the fine-tuned model
        self.lm_model.save_pretrained(output_dir)
        self.lm_tokenizer.save_pretrained(output_dir)
        print(f"Model fine-tuned and saved to {output_dir}")
        # Load the fine-tuned model and tokenizer
        self.lm_model = M2M100ForConditionalGeneration.from_pretrained(output_dir)
        self.lm_tokenizer = M2M100Tokenizer.from_pretrained(output_dir)

        print("Fine-tuned model and tokenizer loaded and ready for use.")





    


