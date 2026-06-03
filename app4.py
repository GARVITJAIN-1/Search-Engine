import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="LangChain Search Chatbot")
st.title("LangChain - Chat with Search")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

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

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not api_key:
        st.error("Please enter your Groq API key.")
        st.stop()

    try:
        # ✅ LLM
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0
        )

        # ✅ Tool
        search = DuckDuckGoSearchRun()
        tools = [search]

        # ✅ Prompt (IMPORTANT for tool calling)
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant with web search access."),
            ("human", "{input}")
        ])

        # ✅ Correct agent
        agent = create_tool_calling_agent(llm, tools, prompt_template)

        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True
        )

        # ✅ Invoke
        response = agent_executor.invoke({"input": prompt})

        answer = response["output"]

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        st.chat_message("assistant").write(answer)

    except Exception as e:
        st.exception(e)
