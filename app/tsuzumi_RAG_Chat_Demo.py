import os
from dotenv import load_dotenv
import streamlit as st
from src.utils import *

from langchain_openai import AzureOpenAIEmbeddings
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel

from langchain_core.messages import (
    HumanMessage, 
    AIMessage,
)

# .env 読み込み
load_dotenv()

tsuzumi_model = os.getenv("AZURE_TSUZUMI_MODEL", "tsuzumi-7b-202509")
tsuzumi_version = os.getenv("AZURE_TSUZUMI_VERSION", "2024-05-01-preview")
embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
embedding_version = os.getenv("AZURE_OPENAI_EMBEDDING_VERSION", "2024-12-01-preview")

USER_NAME = "user"
ASSISTANT_NAME = "assistant"

def main():

    #　Title
    st.title('tsuzumi RAG Chat Demo')
    
    with st.sidebar:
        #API設定       
        st.subheader("🔑 API設定")
        # Chatモデル用
        tmp_inference_endpoint = st.text_input("tsuzumiモデル Endpoint",
                                           value=st.session_state.get("AZURE_INFERENCE_ENDPOINT", "")
                                           )
        
        tmp_inference_cred = st.text_input("tsuzumiモデル API Key",
                                       type="password",
                                       value=st.session_state.get("AZURE_INFERENCE_CREDENTIAL", "")
                                       )
        
        # Embedding用
        tmp_openai_endpoint = st.text_input("embedding-3-smallモデル Endpoint",
                                        value=st.session_state.get("AZURE_OPENAI_ENDPOINT", "")
                                        )
        
        tmp_openai_key = st.text_input("embedding-3-smallモデル API Key",
                                   type="password",
                                   value=st.session_state.get("AZURE_OPENAI_API_KEY", "")
                                   )

        if st.button("保存して反映"):
            try:
                with st.spinner("接続確認中..."):
                    # Chatモデルの接続テスト
                    test_model = AzureAIChatCompletionsModel(
                        azure_deployment=tsuzumi_model,
                        openai_api_version=tsuzumi_version,
                        endpoint=tmp_inference_endpoint,
                        credential=tmp_inference_cred,
                    )
                    _ = test_model.invoke("ping")
                    
                     # Embeddingの接続テスト
                    test_embeddings = AzureOpenAIEmbeddings(
                        azure_deployment=embedding_model ,
                        openai_api_version=embedding_version ,
                        azure_endpoint=tmp_openai_endpoint,
                        api_key=tmp_openai_key,
                    )
                    _ = test_embeddings.embed_query("ping")

                # 成功したら session_state に保存
                st.session_state["AZURE_INFERENCE_ENDPOINT"] = tmp_inference_endpoint
                st.session_state["AZURE_INFERENCE_CREDENTIAL"] = tmp_inference_cred
                st.session_state["AZURE_OPENAI_ENDPOINT"] = tmp_openai_endpoint
                st.session_state["AZURE_OPENAI_API_KEY"] = tmp_openai_key
                
                st.success("設定を反映しました ✅")

            except Exception as e:
                st.error(f"接続に失敗しました。API設定を確認してください。: {e}")
                st.stop()
                
        # 保存済みの値を取得
        inference_endpoint = st.session_state.get("AZURE_INFERENCE_ENDPOINT")
        inference_cred = st.session_state.get("AZURE_INFERENCE_CREDENTIAL")
        openai_endpoint = st.session_state.get("AZURE_OPENAI_ENDPOINT")
        openai_key = st.session_state.get("AZURE_OPENAI_API_KEY")        
            
        st.markdown("<div style='margin:30px 0;'></div>", unsafe_allow_html=True)
        
        st.subheader("🎚️ 類似度スコア閾値")
        score_threshold = st.slider(
            "スコアしきい値", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.3,   # デフォルト値
            step=0.05
        )
        
        st.subheader("🔎 検索件数 (k)")
        k_value = st.slider(
            "検索件数", 
            min_value=1, 
            max_value=10, 
            value=3,    # デフォルト値
            step=1
        )
        
        # チャットログの履歴を保持する数
        st.subheader("🔢 会話履歴数制限")
        message_num = st.slider('', min_value=5, max_value=50, value=10)
        
    # chatのモデルを取得
    model = None
    if inference_endpoint and inference_cred:
        model = AzureAIChatCompletionsModel(
            azure_deployment="tsuzumi-7b-202508",
            openai_api_version="2024-05-01-preview",
            endpoint=inference_endpoint,
            credential=inference_cred,
        )
    else:
        st.warning("tsuzumiモデルのAPI設定をしてください")
        
    # embeddingsのモデルを取得
    embeddings = None
    if openai_endpoint and openai_key:
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment="embedding-3-small",
            openai_api_version="2024-12-01-preview",
            azure_endpoint=openai_endpoint,
            api_key=openai_key,
        )
    else:
        st.warning("embedding-3-smallモデルのAPI設定をしてください")
        st.stop()

    # Chain取得
    contextualize_chain = get_contextualize_prompt_chain(model)
    chain = get_chain(model)

    # FAISSからretrieverを取得
    try:
        retriever = pull_from_faiss(embeddings, score_threshold=score_threshold, k=k_value)
    except FileNotFoundError:
        st.warning("ベクトルDBが存在しません。先にRAG用のファイルをアップロードしてください。")
        st.stop()
    except Exception as e:
        st.error(f"ベクトルDBの読み込みに失敗しました（詳細: {e}）")
        st.stop()

    # チャットログを保存したセッション情報を初期化
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []

    # チャットクリアボタン    
    if st.button("🗑️", help="チャットを削除"):
        st.session_state.chat_log = []
        
    # ユーザーのメッセージ入力   
    user_msg = st.chat_input("ここにメッセージを入力")

    if user_msg:
        
        # 以前のチャットログを表示
        for chat in st.session_state.chat_log:
            if isinstance(chat, AIMessage):
                with st.chat_message(ASSISTANT_NAME):
                    st.write(chat.content)
            else:
                with st.chat_message(USER_NAME):
                    st.write(chat.content)

        # ユーザーのメッセージを表示
        with st.chat_message(USER_NAME):
            st.write(user_msg)
            
        # 質問を修正する
        if st.session_state.chat_log:
            new_msg = contextualize_chain.invoke({"chat_history": st.session_state.chat_log, "input": user_msg})
        else:
            new_msg = user_msg
        print(user_msg, "=>", new_msg)

        # 類似ドキュメントを取得
        relavant_docs = retriever.invoke(new_msg, k=3)
        print("=== retriever raw docs ===")
        for i, d in enumerate(relavant_docs, 1):
            print(f"[{i}] type={type(d)}")
            if hasattr(d, "page_content"):
                print("content:", d.page_content[:200])
            else:
                print("no page_content:", d)

        # 質問の回答を表示 
        response = ""
        with st.chat_message(ASSISTANT_NAME):
            msg_placeholder = st.empty()
            
            for r in chain.stream({"chat_history": st.session_state.chat_log, "context": relavant_docs, "input": user_msg}):
                response += r.content
                msg_placeholder.markdown(response + "■")
            msg_placeholder.markdown(response)
            
            # ドキュメントのソースを表示
            if relavant_docs:
                sources = []
                seen = set()

                for doc in relavant_docs:
                    src = (doc.metadata or {}).get("source") 
                    if not src:
                        continue
                    if src in seen:
                        continue
                    seen.add(src)
                    sources.append(f'[[{len(sources)+1}]]({src})')

                if sources:  # ← 空でなければ
                    st.markdown(" ".join(sources))

                # ドキュメントのコンテンツを表示
                cols = st.columns(len(relavant_docs))
                for i, doc in enumerate(relavant_docs):
                    with cols[i]:
                        with st.popover(f"参照{i+1}"):
                            st.markdown(doc.page_content)    

        # セッションにチャットログ(会話履歴)を追加
        st.session_state.chat_log.extend([
            HumanMessage(content=user_msg),
            AIMessage(content=response)
        ])
        
        # チャットログを保持する数を超えた場合、古いログを削除
        if len(st.session_state.chat_log) > message_num:
            st.session_state.chat_log = st.session_state.chat_log[-message_num:]
        print(st.session_state.chat_log, len(st.session_state.chat_log))

if __name__ == '__main__':
    main()