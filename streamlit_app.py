import os
import time
import json
import openai
import pandas as pd
import streamlit as st
from openai.embeddings_utils import cosine_similarity

st.set_page_config (layout="wide")
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

LAWS = {
    "公司法": "company", 
    "劳动法": "labor", 
    "婚姻法": "marriage"
}  


class Document:

    laws_path = {
        "labor": "data/中华人民共和国劳动法(2018-12).txt",
        "marriage": "data/中华人民共和国婚姻法(2005-05).txt",
        "company": "data/中华人民共和国公司法(2018-10).txt"
    }

    embedding_path = {
        "labor": "data/labor_embedding",
        "marriage": "data/marriage_embedding",
        "company": "data/company_embedding"
    }

    @property
    def length(self):
        return len(self.text)

    def __init__(self, doc_name):
        with open(self.laws_path[doc_name], "r", encoding="utf-8") as f:
            self.name = doc_name
            self.text = f.read().replace("\u3000", "")

    def load_embedding(self, chunk_overlap="300_100"):
        with open(self.embedding_path[self.name]+"_"+chunk_overlap+".json", "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

    def split(self, chunk_size, overlap_size):
        self.chunks = []
        step_size = chunk_size - overlap_size
        for i in range(0, ((len(self.text) - overlap_size) // step_size) + 1):
            self.chunks.append({
                "id": i, 
                "text": self.text[i*step_size: i*step_size + chunk_size]
                })
        print("Document has been splited into <obj.chunks[i]['text']>")

    def get_chunk_abstract(self):
        for i in range(0, len(self.chunks)):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                            {"role": "system", "content": "基于用户输入，给出总结文本。"},
                            {"role": "user", "content": self.chunks[i]["text"]}
                        ]
                    )
                self.chunks[i]["abstract"] = response['choices'][0]['message']['content']
            except:
                print("OpenAI Error!")
            time.sleep(15)
        print("Chunk abstract has been generated into <obj.chunks[i]['abstract']>")
    
    def get_chunk_text_embedding(self, model="text-embedding-ada-002"):
        for i in range(0, len(self.chunks)):
            text = self.chunks[i]["text"].replace("\n", " ")
            self.chunks[i]["embedding"] = openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']
            time.sleep(1)
        print("Chunk text embedding is generated in <obj.chunks[i]['embedding']>")

    def get_similar_chunk(self, input_text, num_chunks, model="text-embedding-ada-002"):
        input_embedding = openai.Embedding.create(input = [input_text], model=model)['data'][0]['embedding']
        for i in range(0, len(self.chunks)):
            self.chunks[i]["similarity_score"] = cosine_similarity(input_embedding, self.chunks[i]["embedding"])
        sorted_chunks = sorted(self.chunks, key=lambda x: x['similarity_score'], reverse=True)[:num_chunks]
        return sorted_chunks


st.markdown("<h2 style='text-align: center;'>法律问答机器人</h2>", unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    law = st.selectbox(
        "请选择法律法规：",
        ("公司法", "劳动法", "婚姻法")
    )
    chunk_overlap = st.radio(
        "请选择 Chunk_Overlap 大小：",
        ("300_100", "500_100"),
        index=1,
        horizontal=True
    )
    if st.button("加载法律法规"):
        st.session_state["doc"] = Document(LAWS[law])
        st.session_state["doc"].load_embedding(chunk_overlap=chunk_overlap)
    st.markdown("---")
    top_n_chunks = st.radio(
        "请选择参考Chunk的数量：",
        ("2", "3", "4", "5"),
        index=1,
        horizontal=True
    )
    st.markdown("---")
    if st.checkbox("使用自定义系统提示语"):
        system_prompt = st.text_area("系统提示语：", "请根据参考文档回答用户问题，并给出参考的条目在第几条。禁止提供不在参考文档内的内容。")
    else:
        system_prompt = "请根据参考文档回答用户问题，并给出参考的条目在第几条。禁止提供不在参考文档内的内容。"

col1, col2 = st.columns(2)

with col1:
    if "doc" in st.session_state:
        st.text_area(law+":", st.session_state["doc"].text, height=600)

with col2:
    if "doc" in st.session_state:
        input_text = st.text_input("请提问：", "")
        if st.button("提交", use_container_width=True):
            sorted_chunks = st.session_state["doc"].get_similar_chunk(input_text, int(top_n_chunks))
            reference = ""

            for chunk in sorted_chunks:
                reference += chunk["text"] + "\n\n"

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"### 参考文档 ###\n{reference}### 用户问题 ###\n{input_text}"}
                ])
            res_text = response['choices'][0]['message']['content']
            st.text_area("法律机器人：", res_text, height=460)
