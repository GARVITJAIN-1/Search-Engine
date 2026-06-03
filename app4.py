import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent

st.set_page_config(page_title="LangChain Search Chatbot")

st.title("LangChain - Chat with Search")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input(
    "Enter your Groq API Key:",
    type="password"
)
##
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi, I'm a chatbot that can search the web. How can I help you?"
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("What is machine learning?"):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    st.chat_message("user").write(prompt)

    if not api_key:
        st.error("Please enter your Groq API key.")
        st.stop()

    try:
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.3-70b-versatile",
            streaming=False
        )

        search = DuckDuckGoSearchRun()

        agent = create_agent(
            model=llm,
            tools=[search]
        )

        response = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
        )

        answer = response["messages"][-1].content

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        st.chat_message("assistant").write(answer)

    except Exception as e:
        st.exception(e)
