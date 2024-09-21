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
    except ValueError as e:
        st.error(f"Error initializing system: {str(e)}")
        return None

# Initialize system
system = initialize_system()

def main():
    st.title("Multilingual Wiki QA System with Active Learning")

    if system is None:
        st.error("System initialization failed. Please check your Pinecone API key and try again.")
        return

    # Dataset exploration
    st.header("Explore Wiki QA Dataset")
    if st.checkbox("Show sample questions and answers"):
        num_samples = st.slider("Number of samples to display", 1, 20, 5)
        samples = list(zip(system.questions, system.documents))[:num_samples]
        for i, (question, answer) in enumerate(samples, 1):
            st.write(f"{i}. Question: {question}")
            st.write(f"   Answer: {answer}")
            st.write("---")

    # Language selection
    languages = ['en', 'es', 'fr', 'de', 'zh']  # Add more languages as needed
    selected_language = st.selectbox("Select your language:", languages)

    # Chat interface
    st.header("Chat with the QA System")
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        display_chat_message(message['content'], message['is_user'])

    query = st.text_input("Ask a question in any language:")
    use_wiki_question = st.checkbox("Use a random question from Wiki QA dataset")

    if use_wiki_question:
        random_index = np.random.randint(0, len(system.questions))
        query = system.questions[random_index]
        st.write(f"Selected question: {query}")

    if st.button("Submit"):
        display_chat_message(query, is_user=True)
        st.session_state.messages.append({"content": query, "is_user": True})
        
        with st.spinner("Processing query..."):
            result = system.process_query(query)
        
        answer = result['generated_answer']
        display_chat_message(answer)
        st.session_state.messages.append({"content": answer, "is_user": False})
        
        # Visualization for search relevance
        st.subheader("Search Relevance")
        relevance_data = pd.DataFrame({
            'Document': [doc[:50] + "..." for doc, _ in result['relevant_documents']],
            'Relevance Score': [score for _, score in result['relevant_documents']]
        })
        chart = alt.Chart(relevance_data).mark_bar().encode(
            x='Relevance Score',
            y=alt.Y('Document', sort='-x'),
            color=alt.Color('Relevance Score', scale=alt.Scale(scheme='viridis'))
        ).properties(
            width=600,
            height=300
        )
        st.altair_chart(chart)

        # Confidence visualization
        st.subheader("Answer Confidence")
        confidence_score = result['confidence']
        st.progress(confidence_score)
        st.write(f"Confidence: {confidence_score:.2f}")

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

    # Active Learning Section
st.header("Active Learning")
if st.checkbox("Show Active Learning Samples"):
    samples = system.get_active_learning_samples()
    if samples:
        for i, sample in enumerate(samples, 1):
            st.subheader(f"Sample {i}")
            st.write(f"Query: {sample['query']}")
            st.write(f"Extracted Answer: {sample['extracted_answer']}")
            st.write(f"Generated Answer: {sample['generated_answer']}")
            st.write(f"Confidence: {sample['confidence']:.2f}")
            if st.button(f"Improve Model for Sample {i}"):
                # Implement logic to fine-tune 
                with st.spinner("Improving model..."):
                    # Prepare the sample for fine-tuning
                    train_sample = {
                        "input_text": [f"Question: {sample['query']} Context: {sample['extracted_answer']}"],
                        "labels": [sample['generated_answer']]
                    }
                    eval_sample = {
                        "input_text": [f"Question: {sample['query']} Context: {sample['extracted_answer']}"],
                        "labels": [sample['generated_answer']]
                    }
                    
                    # Fine-tune the model with this sample
                    system.fine_tune(train_sample, eval_sample, num_epochs=1)
                    
                    st.success(f"Model improved using Sample {i}")
    else:
        st.write("No active learning samples available at the moment.")

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

if __name__ == "__main__":
    main()
