import streamlit as st
from src.utils import *
from langchain_openai import AzureOpenAIEmbeddings

def main():
    st.title("PDFファイルをアップロード📁")
    
    openai_key = st.session_state.get("AZURE_OPENAI_API_KEY")
    openai_endpoint = st.session_state.get("AZURE_OPENAI_ENDPOINT")
            
    # emmbeddingsのモデルを取得
    embeddings = None
    if openai_key and openai_endpoint != "":
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment="embedding-3-small",
            openai_api_version="2024-12-01-preview",
            api_key=openai_key,
            azure_endpoint=openai_endpoint,
        )
    else:
        st.warning("EmmbedingモデルのAPI設定をしてください")
        st.stop()

    # FAISSの初期化
    faiss_db = None

    # アップロード pdf ファイル...
    uploaded_files = st.file_uploader("PDFファイル", type=["pdf"], accept_multiple_files=True)

    # ファイルの数だけ処理を行う
    for i, pdf in enumerate(uploaded_files):
        with st.spinner(f'{i+1}/{len(uploaded_files)} 処理中...'):
            # 1. PDFデータ読み込み
            text=read_pdf_data(pdf)
            st.write("1. PDFデータ読み込み")

            # 2. データをチャンクに小分けにする
            docs_chunks=split_data(text)
            st.write("2. データをチャンクに小分けにする")

            # 3. FAISSにベクトル化して格納
            faiss_db = add_to_faiss(
                faiss_db=faiss_db,               
                docs=docs_chunks,
                embeddings=embeddings
            )

    if faiss_db is not None:
        faiss_db.save_local("vector_store")
        st.write("3. ベクトルデータの保存")
        st.success("完了！")

if __name__ == '__main__':
    main()
