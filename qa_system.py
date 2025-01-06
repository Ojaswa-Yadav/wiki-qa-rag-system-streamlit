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

nltk.download('punkt', quiet=True)


class EnhancedQARAGSystemWithGuardrails(EnhancedQARAGSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Define offensive words and sensitive topics for validation
        self.offensive_words = ["badword1", "badword2", "hack", "bypass"]
        self.sensitive_topics = ["politics", "violence", "hate speech"]

    # Guardrail: Inappropriate Content Filter
    def inappropriate_content_filter(self, text):
        """Filters input for inappropriate or offensive content."""
        if any(word in text.lower() for word in self.offensive_words):
            return "[Content removed due to inappropriate language]"
        return text

    # Guardrail: Prompt Injection Shield
    def prompt_injection_shield(self, text):
        """Protects against malicious or harmful instructions."""
        forbidden_phrases = ["delete all data", "bypass security", "unauthorized access"]
        if any(phrase in text.lower() for phrase in forbidden_phrases):
            return "[Prompt rejected due to security risks]"
        return text

    # Guardrail: Offensive Language Filter
    def offensive_language_filter(self, text):
        """Filters offensive or disrespectful content in responses."""
        pattern = re.compile(r'\b(?:' + '|'.join(self.offensive_words) + r')\b', re.IGNORECASE)
        filtered_text = pattern.sub("[censored]", text)
        return filtered_text

    # Guardrail: Sensitive Content Scanner
    def sensitive_content_scanner(self, text):
        """Detects sensitive topics and flags them."""
        if any(topic in text.lower() for topic in self.sensitive_topics):
            return "[Content flagged due to sensitive topics]"
        return text

    # Unified Input Validation Guardrail
    def validate_input(self, text):
        """Validates input through all input guardrails."""
        text = self.inappropriate_content_filter(text)
        text = self.prompt_injection_shield(text)
        return text

    # Unified Output Validation Guardrail
    def validate_output(self, text):
        """Validates output through all output guardrails."""
        text = self.offensive_language_filter(text)
        text = self.sensitive_content_scanner(text)
        return text

    # Override process_query to integrate guardrails
    def process_query(self, query, num_relevant_docs=3):
        """
        Processes a user query, applies guardrails, and retrieves answers.
        """
        try:
            # Step 1: Validate the input query
            validated_query = self.validate_input(query)
            if validated_query.startswith("["):
                return {"query": query, "error": validated_query}

            # Step 2: Process the query with the original model logic
            query_lang = self.detect_language(validated_query)
            english_query = self.translate(validated_query, 'en')

            relevant_docs = self.semantic_search(english_query, k=num_relevant_docs)
            context = " ".join([doc for doc, _ in relevant_docs])

            extracted_answer = self.extract_answer(english_query, context)
            generated_answer = self.generate_answer(english_query, context)

            # Step 3: Validate the outputs
            extracted_answer = self.validate_output(extracted_answer)
            generated_answer = self.validate_output(generated_answer)

            # Step 4: Translate back to the original language
            extracted_answer_translated = self.translate(extracted_answer, query_lang)
            generated_answer_translated = self.translate(generated_answer, query_lang)

            # Step 5: Calculate confidence
            confidence = self.calculate_confidence(extracted_answer, generated_answer)

            # Step 6: Add low-confidence responses to active learning queue
            if confidence < 0.5:
                self.active_learning_queue.append({
                    'query': query,
                    'extracted_answer': extracted_answer,
                    'generated_answer': generated_answer,
                    'confidence': confidence
                })

            # Step 7: Return results
            return {
                "query": query,
                "relevant_documents": relevant_docs,
                "extracted_answer": extracted_answer_translated,
                "generated_answer": generated_answer_translated,
                "confidence": confidence
            }
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            return {"query": query, "error": "Unable to process query"}


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
    """
    Fine-tunes the language model using a training and evaluation dataset.
    
    Args:
        train_dataset (dict): Dictionary with 'input_text' and optionally 'labels' for training.
        eval_dataset (dict): Dictionary with 'input_text' and optionally 'labels' for evaluation.
        output_dir (str): Directory where the fine-tuned model will be saved.
        num_epochs (int): Number of training epochs.
    """
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






    

