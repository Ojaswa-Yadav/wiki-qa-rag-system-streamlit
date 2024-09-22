import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from qa_system import EnhancedQARAGSystem

# Streamlit page config
st.set_page_config(layout="wide", page_title="Multilingual Wiki QA System")

# Custom CSS for chat-like interface
st.markdown("""
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
</style>
""", unsafe_allow_html=True)

def display_chat_message(content, is_user=False):
    avatar = "🧑" if is_user else "🤖"
    st.markdown(f"""
    <div class="chat-message {"user" if is_user else "bot"}">
        <div class="avatar">
            {avatar}
        </div>
        <div class="message">{content}</div>
    </div>
    """, unsafe_allow_html=True)

@st.cache_resource
def initialize_system():
    try:
        system = EnhancedQARAGSystem()
        num_docs = system.load_dataset(max_samples=10000)  # Limit to 10,000 documents
        st.write(f"Loaded {num_docs} documents from Wiki QA dataset")
        return system
    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        return None

# Initialize system
system = initialize_system()

def main():
    st.title("Multilingual Wiki QA System with Active Learning")

    if system is None:
        st.error("System initialization failed. Please check your configuration and try again.")
        return

    # Query section
    st.header("Ask a Question")
    query = st.text_input("Enter your question:")
    if st.button("Submit"):
        if system:
            with st.spinner("Processing query..."):
                result = system.process_query(query)
            
            st.subheader("Results")
            st.write(f"Query: {result['query']}")
            st.write(f"Extracted Answer: {result['extracted_answer']}")
            st.write(f"Generated Answer: {result['generated_answer']}")
            
            st.subheader("Relevant Documents")
            for i, (doc, score) in enumerate(result['relevant_documents'], 1):
                st.write(f"{i}. {doc[:100]}... (Score: {score:.4f})")
        else:
            st.error("System is not initialized. Cannot process query.")

    # Evaluation section
    st.header("System Evaluation")
    st.write("Evaluate using Wiki QA dataset:")
    num_eval_samples = st.number_input("Number of random samples for evaluation", min_value=1, max_value=100, value=10)

    if st.button("Evaluate"):
        if system:
            with st.spinner("Evaluating..."):
                eval_indices = np.random.choice(len(system.questions), num_eval_samples, replace=False)
                test_set = [(system.questions[i], system.documents[i]) for i in eval_indices]
                result = system.evaluate_performance(test_set)
            
            st.subheader("Evaluation Results")
            st.write(f"ROUGE Scores: {result['rouge_scores']}")
            st.write(f"BLEU Score: {result['bleu_score']}")
        else:
            st.error("System is not initialized. Cannot perform evaluation.")

    # Active Learning Section
    st.header("Active Learning")
    if st.checkbox("Show Active Learning Samples"):
        if system:
            samples = system.get_active_learning_samples()
            if samples:
                for i, sample in enumerate(samples, 1):
                    st.subheader(f"Sample {i}")
                    st.write(f"Query: {sample['query']}")
                    st.write(f"Extracted Answer: {sample['extracted_answer']}")
                    st.write(f"Generated Answer: {sample['generated_answer']}")
                    st.write(f"Confidence: {sample['confidence']:.2f}")
                    if st.button(f"Improve Model for Sample {i}"):
                        st.write("Model improvement logic would be implemented here.")
            else:
                st.write("No active learning samples available at the moment.")
        else:
            st.error("System is not initialized. Cannot show active learning samples.")

    # Fine-tuning section
    st.header("Fine-tune the Model")
    st.write("Fine-tune using Wiki QA dataset:")
    num_train = st.number_input("Number of training samples", min_value=10, max_value=1000, value=100)
    num_eval = st.number_input("Number of evaluation samples", min_value=10, max_value=100, value=20)

    if st.button("Fine-tune"):
        if system:
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
        else:
            st.error("System is not initialized. Cannot perform fine-tuning.")

if __name__ == "__main__":
    main()
