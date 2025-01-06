import os
from pinecone import Pinecone
import torch
import sentencepiece
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
from guardrails import Guardrails 
nltk.download('punkt', quiet=True)




class EnhancedQARAGSystemWithGuardrails(EnhancedQARAGSystem):
    def __init__(self, *args, **kwargs):
        self.guardrails = Guardrails()
        self.sbert_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
        self.qa_tokenizer = AutoTokenizer.from_pretrained("deepset/xlm-roberta-large-squad2")
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained("deepset/xlm-roberta-large-squad2")
        self.lm_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
        self.lm_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
        self.scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        self.active_learning_queue = []
        pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-west1-gcp")
        self.index = pinecone.Index("qa-index")

    def process_query(self, query, num_relevant_docs=3):
        try:
            validated_query = self.guardrails.validate_input(query)
            if validated_query.startswith("["):
                return {"query": query, "error": validated_query}

            query_lang = self.detect_language(validated_query)
            english_query = self.translate(validated_query, "en")

            relevant_docs = self.semantic_search(english_query, k=num_relevant_docs)
            context = " ".join([doc for doc, _ in relevant_docs])

            extracted_answer = self.extract_answer(english_query, context)
            generated_answer = self.generate_answer(english_query, context)

            extracted_answer = self.guardrails.validate_output(extracted_answer)
            generated_answer = self.guardrails.validate_output(generated_answer)

            extracted_answer_translated = self.translate(extracted_answer, query_lang)
            generated_answer_translated = self.translate(generated_answer, query_lang)

            confidence = self.calculate_confidence(extracted_answer, generated_answer)

            if confidence < 0.5:
                self.active_learning_queue.append({
                    "query": query,
                    "extracted_answer": extracted_answer,
                    "generated_answer": generated_answer,
                    "confidence": confidence,
                })

            return {
                "query": query,
                "relevant_documents": relevant_docs,
                "extracted_answer": extracted_answer_translated,
                "generated_answer": generated_answer_translated,
                "confidence": confidence,
            }
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            return {"query": query, "error": "Unable to process query"}

    def semantic_search(self, query, k=5):
        query_embedding = self.sbert_model.encode(query)
        results = self.index.query(query_embedding.tolist(), top_k=k, include_metadata=True)
        return [(match["metadata"]["text"], match["score"]) for match in results["matches"]]

    def extract_answer(self, question, context):
        inputs = self.qa_tokenizer.encode_plus(question, context, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.qa_model(**inputs)
        start = torch.argmax(outputs.start_logits)
        end = torch.argmax(outputs.end_logits) + 1
        return self.qa_tokenizer.decode(inputs["input_ids"][0][start:end], skip_special_tokens=True)

    def generate_answer(self, question, context):
        input_text = f"Question: {question}\nContext: {context}\nAnswer:"
        inputs = self.lm_tokenizer(input_text, return_tensors="pt")
        outputs = self.lm_model.generate(**inputs, max_length=512, num_beams=3, early_stopping=True)
        return self.lm_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def translate(self, text, target_lang):
        source_lang = self.detect_language(text)
        if source_lang == target_lang:
            return text
        self.lm_tokenizer.src_lang = source_lang
        encoded = self.lm_tokenizer(text, return_tensors="pt")
        generated_tokens = self.lm_model.generate(**encoded, forced_bos_token_id=self.lm_tokenizer.get_lang_id(target_lang))
        return self.lm_tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    def detect_language(self, text):
        return detect(text)



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
            try:
                result = self.process_query(question)
                generated_answer = result['generated_answer']
            except Exception as e:
                logging.error(f"Error processing query during evaluation: {e}")
                continue 
            
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
        # Tokenize the datasets
        train_encodings = self.lm_tokenizer(train_dataset["input_text"], truncation=True, padding=True, return_tensors="pt")
        train_labels = self.lm_tokenizer(train_dataset.get("labels", [""]), truncation=True, padding=True, return_tensors="pt")
        eval_encodings = self.lm_tokenizer(eval_dataset["input_text"], truncation=True, padding=True, return_tensors="pt")
        eval_labels = self.lm_tokenizer(eval_dataset.get("labels", [""]), truncation=True, padding=True, return_tensors="pt")
        # Convert datasets to PyTorch Tensors
        train_dataset = torch.utils.data.TensorDataset(
            train_encodings["input_ids"], train_encodings["attention_mask"], train_labels["input_ids"]
        )
        eval_dataset = torch.utils.data.TensorDataset(
            eval_encodings["input_ids"], eval_encodings["attention_mask"], eval_labels["input_ids"]
        )
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=50,
            save_steps=50,
            save_total_limit=2,
            load_best_model_at_end=True,
        )

    # Define the metric computation function
    def compute_metrics(eval_pred):
        """
        Computes evaluation metrics (BLEU and ROUGE).
        
        Args:
            eval_pred (tuple): Tuple of predictions and labels.

        Returns:
            dict: Dictionary containing BLEU and ROUGE scores.
        """
        predictions, labels = eval_pred
        decoded_preds = self.lm_tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.lm_tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Tokenize predictions and labels for BLEU and ROUGE computation
        bleu_scores = [
            sentence_bleu([nltk.word_tokenize(label)], nltk.word_tokenize(pred))
            for pred, label in zip(decoded_preds, decoded_labels)
        ]
        rouge_scores = [
            self.scorer.score(label, pred)
            for pred, label in zip(decoded_labels, decoded_preds)
        ]

        # Calculate average BLEU and ROUGE scores
        avg_bleu = np.mean(bleu_scores)
        avg_rouge = {
            key: np.mean([score[key].fmeasure for score in rouge_scores])
            for key in rouge_scores[0].keys()
        }

        return {
            "bleu": avg_bleu,
            "rouge1": avg_rouge["rouge1"],
            "rougeL": avg_rouge["rougeL"],
        }

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






    

