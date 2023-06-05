import streamlit as st
from transformers import T5TokenizerFast, T5ForConditionalGeneration
import torch

model_name = "abokbot/t5-end2end-questions-generation"

st.header("Question generation app")

st_model_load = st.text('Loading question generator model...')

@st.cache_resource
def load_model():
    print("Loading model...")
    tokenizer = T5TokenizerFast.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    print("Model loaded!")
    return tokenizer, model

tokenizer, model = load_model()
st.success('Model loaded!')
st_model_load.text("")

if 'text' not in st.session_state:
    st.session_state.text = ""
st_text_area = st.text_area('Enter text to generate the questions for (works best with short news/Wikipedia articles)', value=st.session_state.text, height=300)

def generate_questions():
    st.session_state.text = st_text_area

    generator_args = {
        "max_length": 256,
        "num_beams": 4,
        "length_penalty": 1.5,
        "no_repeat_ngram_size": 3,
        "early_stopping": True,
    }
    input_string = "generate questions: " + st_text_area + " </s>"
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    output = tokenizer.batch_decode(res, skip_special_tokens=True)
    output = [question.strip() + "?" for question in output[0].split("?") if question != ""]

    st.session_state.questions = output

# generate questions button
st_generate_button = st.button('Generate questions', on_click=generate_questions)

# question generation labels
if 'questions' not in st.session_state:
    st.session_state.questions = []

if len(st.session_state.questions) > 0:
    with st.container():
        st.subheader("Generated questions")
        for title in st.session_state.questions:
            st.markdown("__" + title + "__")