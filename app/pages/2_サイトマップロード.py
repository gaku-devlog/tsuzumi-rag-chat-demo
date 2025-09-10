import streamlit as st
from src.utils import *
from langchain_openai import AzureOpenAIEmbeddings

def main():
    st.title("サイトマップをロード🌐")
    
    st.warning("SitemapLoader は基本的に非常に料金がかかるため、現在利用不可の状態にしています。")
    # NOTE: デモでは無効化。将来有効化する場合は disabled=False に変更して処理を復帰。
    web_url = st.text_input('URL',
                            '',
                            key="web_url",
                            disabled=True)
    load_button = st.button("Load data to FAISS", key="load_button", disabled=True) 
    
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
    # 無効化のためにUIを考えて下記をコメントアウト
    # else:
    #     st.warning("EmmbedingモデルのAPI設定をしてください")
    #     st.stop()  
        
    # FAISSの初期化
    faiss_db = None

    # load_buttonクリックでウェブデータをFAISSに格納
    if load_button:
        
        # 1.ウェブデータ取得
        with st.spinner(f'ウェブデータ取得 処理中...'):
            site_data = get_website_data(web_url)
            st.write("1. ウェブデータ取得")

        # 2. データをチャンクに小分けにする
        docs_chunks = split_data(site_data)
        st.write("2. データをチャンクに小分けにする")

        # 3. FAISSにベクトル化して格納
        with st.spinner(f'ベクトルデータ 処理中...'):
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