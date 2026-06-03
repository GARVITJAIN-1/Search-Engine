import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType

st.set_page_config(page_title="LangChain Search Chatbot")
st.title("LangChain - Chat with Search")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I can search the web for you."}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask something..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not api_key:
        st.error("Please enter Groq API key.")
        st.stop()

    try:
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0
        )

        search = DuckDuckGoSearchRun()

        tools = [search]

        # ✅ WORKING AGENT (no tool-calling import needed)
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )

        response = agent.run(prompt)

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

        st.chat_message("assistant").write(response)

    except Exception as e:
        st.exception(e)
