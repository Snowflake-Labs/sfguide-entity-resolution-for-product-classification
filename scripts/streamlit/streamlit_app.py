import streamlit as st
from snowflake.snowpark.context import (
    get_active_session,
)
import os
import json
import requests
import pandas as pd
import _snowflake

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

session = get_active_session()

######################################
##### UPDATE VARIABLES IF NEEDED #####
######################################
num_chunks = 2
num_transcripts = 5
slide_window = 2
API_ENDPOINT = "/api/v2/cortex/analyst/message"
API_TIMEOUT = 50000
####################################
##### UPDATE CONFIGS IF NEEDED #####
####################################
def config_options():
    '''Set configs'''
    # Observed to work best with Llama 3.1
    #st.session_state.model_name = 'llama3.1-405b'
    st.session_state.model_name = 'llama3.1-70b'
    # Observed to work best without Chat History
    st.session_state.use_chat_history = False # DON'T CHANGE THIS - HAS NOT BEEN WELL TESTED
    st.session_state.cortex_search = True
    st.session_state.debug_prompt = False
    st.session_state.debug = False
    #show_sql = st.sidebar.checkbox('Show Analyst SQL Code')
    show_sql = 1

    return show_sql

################################################
##### YOU SHOULDN'T NEED TO TOUCH THE REST #####
################################################

def get_chat_history():
    """Gets chat history using a sliding window window"""
    chat_history = []

    if st.session_state.use_chat_history:
        start_index = max(0,len(st.session_state.messages)-slide_window)
        for i in range(start_index,len(st.session_state.messages)-1):
            chat_history.append(st.session_state.messages[i])

    return chat_history

def summarize_question_with_history(chat_history, question):
    """Creates prompt to summarize chat history"""
    prompt = f"""
        Based on the chat history below and the question, generate a query that extend the question
        with the chat history provided. The query should be in natual language. 
        
        Answer with only the query. Do not add any explanation.
    
        Chat History: {chat_history}
        Question : {question}
        
    """

    cmd = """
            select snowflake.cortex.complete(?,?) as response
    
    """

    df_response = session.sql(cmd,params = [st.session_state.model_name,prompt]).collect()

    summary = df_response[0].RESPONSE  

    if st.session_state.debug:
        st.text("Summary to be used to find similar chunks")
        st.caption(summary)

    summary = summary.replace("'","")

    return summary

def create_prompt_summarize_cortex_analyst_results(question, df, sql):
    """Creates prompt to summarize Cortex Analyst results in natural language."""
    prompt = f"""
          You are an expert Data analyst who translated the question contained between <question> and </question> tags
           <question>
           {question}
           </question>

           Into the SQL query contained between <SQL> and </SQL>  contained between <question> and </question> tags
            <SQL>
           {sql}
           </SQL>

            And retrieved the below resultset contained between <question> and </question> tags from this SQL Query:
            <df>
            {df}
            </df>
           
          
           Now share an answer to this question based on SQL query and resultset.
           be concise  and use mainly the CONTEXT provided and do not hallucinate. 
           If you don´t have the information just say so.

          Whenever possible arrange your response as bullet points.
          
           Do not mention the CONTEXT in your answer

           <df>
           {df}
           </df>
           <question>
           {question}
           </question>
           <SQL>
           {sql}
           </SQL>
           Answer:
    
    """


    if st.session_state.debug_prompt:
        st.text(f"Prompt being passed to {st.session_state.model_name}")
        st.caption(prompt)
    
    return prompt

def complete(myquestion, chat_history, prompt):
    """Sends final prompt to Cortex Complete."""
    cmd = f"""
        select snowflake.cortex.complete(?,?) as response

    """
    df_response = session.sql(cmd,params = [st.session_state.model_name,prompt]).collect()
    
    return df_response

def send_message(session, prompt):
    """Calls the Cortex REST API and returns the response."""
    request_body = {
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        "semantic_model_file": f"@PRODUCT_MATCHING_DB.MATCH.MODEL/matches_semantic_model.yml",
    }
    resp = _snowflake.send_snow_api_request(
        "POST",  # method
        API_ENDPOINT,  # path
        {},  # headers
        {},  # params
        request_body,  # body
        None,  # request_guid
        API_TIMEOUT,  # timeout in milliseconds
    )
    parsed_content = json.loads(resp["content"])
    # Check if the response is successful
    if resp["status"] < 400:
        # Return the content of the response as a JSON object
        return parsed_content, None
    else:
        raise Exception(
            f"Failed request with status {resp['status']} and content: {resp.get('content', 'No content')}"
        )


    # st.write(resp)
    # st.write(resp["message"]["content"])
    # st.write(resp["request_id"])
    # # request_id = resp.headers.get("X-Snowflake-Request-Id")
    # if resp.status < 400:
    #     return {**resp.json(), "request_id": request_id}  # type: ignore[arg-type]
    # else:
    #     raise Exception(
    #         f"Failed request (id: {request_id}) with status {resp.status_code}: {resp.text}"
    #     )
    # Content is a string with serialized JSON object
    # parsed_content = json.loads(resp["content"])

def process_message(session, prompt, show_sql, question_summary=None):
    """Processes a message and adds the response to the chat."""

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Cortex Analyst thinking..."):
            response = send_message(session=session, prompt=prompt)
            request_id = response[0]["request_id"]
            content = response[0]["message"]["content"]
            response_string = display_content(content=content, request_id=request_id, show_sql=show_sql, role='assistant')

    st.session_state.messages.append(
        {"role": "assistant", "content": response_string, "request_id": request_id}
    )

def suggestion_click(suggestion):
    """Sets session state if suggestion is clicked."""
    st.session_state.active_suggestion = suggestion

def display_content(
    content,
    request_id,
    message_index=None,
    show_sql=None,
    role=None
):
    """Displays a content item for a message."""
    message_index = message_index or len(st.session_state.messages)
    response_text = "Please refine that question."

    for item in content:
        if item["type"] == "text":
            question = item["text"]
            response_text = item["text"]
        elif item["type"] == "suggestions":
            st.write("This question is not valid. Please try one of the following questions:")
            response_text = ""
            with st.expander("Suggestions", expanded=True):
                for suggestion_index, suggestion in enumerate(item["suggestions"]):
                    if st.button(suggestion, key=f"{message_index}_{suggestion_index}", on_click=suggestion_click, args=[suggestion]):
                        st.session_state.active_suggestion = suggestion
        elif item["type"] == "sql":
            sql_statement = item["statement"]
            df = session.sql(sql_statement).collect()
            prompt = create_prompt_summarize_cortex_analyst_results(question=question, df=df, sql=sql_statement)
            response = complete(myquestion=prompt, chat_history=None, prompt=prompt)
            response_text = response[0].RESPONSE
            st.markdown(response_text)
        #else:
            with st.expander("SQL Query", expanded=False):
                st.code(item["statement"], language="sql")
            with st.expander("Results", expanded=False):
                with st.spinner("Running SQL..."):
                    df = pd.read_sql(item["statement"], session.connection)
                    if len(df.index) > 1:
                        data_tab, line_tab, bar_tab = st.tabs(
                            ["Data", "Line Chart", "Bar Chart"]
                        )
                        data_tab.dataframe(df)
                        if len(df.columns) > 1:
                            df = df.set_index(df.columns[0])
                        with line_tab:
                            st.line_chart(df)
                        with bar_tab:
                            st.bar_chart(df)
                    else:
                        st.dataframe(df)
            
    return response_text

def main():
    # with st.sidebar:
    #     st.title("Product Matching")
    #     st.button("Merchandising Analyst", key="analyst", help="A chatbot to provide insights in natural language.")
    col1, col2 = st.columns([14,2])
    col1.markdown(f"## Retail Merchandising Analyst :robot_face:")

    col2.button('Clear History', key='clear_conversation')
    
    # Set configs
    show_sql = config_options()
    
    # Initialize state
    if st.session_state.clear_conversation or "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "What question do you need assistance answering?",
            }
        ]
        st.session_state.suggestions = []
        st.session_state.active_suggestion = None

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Main q&a mechanism
    if question:= st.chat_input("What is your question?",):
         chat_history = get_chat_history()
         question_summary = question
         if chat_history != "":
            question_summary = summarize_question_with_history(chat_history,question)
            question_summary = question_summary.replace("or supporting documentation","")
        
         intent = "data"
         if intent == 'data':
            process_message(session=session, prompt=question, show_sql=show_sql, question_summary=question_summary)
    # If active suggestion is clicked
    if st.session_state.active_suggestion:
        process_message(session=session, prompt=st.session_state.active_suggestion, show_sql=show_sql)
        st.session_state.active_suggestion = None

main()
