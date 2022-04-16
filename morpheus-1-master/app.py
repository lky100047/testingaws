import streamlit as st
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from morpheus_hf.morpheus import MorpheusHuggingfaceQA
from kitanaqa.augment.term_replacement import ReplaceTerms, DropTerms, RepeatTerms

model_name = st.selectbox(
     'Select a qa model',
     ('deepset/roberta-base-squad2', 'deepset/bert-base-cased-squad2'))

def load_qa_model():
    model = pipeline('question-answering', model=model_name, tokenizer=model_name)
    return model

qa = load_qa_model()

finishQA_flag = False
st.title("Ask Questions about your Text")
sentence = st.text_area('Please paste your article :', height=30)
question = st.text_input("Questions from this article?")
button = st.button("Get me Answers")
with st.spinner("Discovering Answers.."):
    if button and sentence:
        result = qa(question=question, context=sentence)
        st.write(result['answer'], result['score'], result['start'], result['end'])
        finishQA_flag = True

with st.spinner("Attacking.."):
    if finishQA_flag == True:

        test_morph_qa = MorpheusHuggingfaceQA(model_name)
        context = sentence
        q_dict = {"question": question , "id": "56ddde6b9a695914005b9628", "answers": [{"text": result['answer'], "answer_start": result['start']}], "is_impossible": False}

        text = test_morph_qa.morph(q_dict, context)
        st.write(text[0])
        
        answersEdited = qa(question=text[0], context=sentence)
        st.write(answersEdited['answer'])
        
        
        p = ReplaceTerms(rep_type="misspelling")
        num_terms = 3
        num_output_sents = 1
        question = p.replace_terms(question, num_replacements=num_terms, num_output_sents=num_output_sents)[0]
        st.write(question)

        p = DropTerms()
        question = p.drop_terms(question, num_terms, num_output_sents)[0]
        st.write(question)
        p = RepeatTerms()
        num_terms = 2
        question = p.repeat_terms(question, num_terms, num_output_sents)[0]
        st.write(question)
        result = qa(question=question, context=sentence)
        st.write(result['answer'])
        
