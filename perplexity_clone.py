from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
from langchain_core.runnables import RunnableLambda,RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from tavily import TavilyClient
from pydantic import BaseModel,Field
import streamlit as st
from dotenv import load_dotenv

st.title("AI Powered Real-Time Chatbot 🤖")

user_input = st.text_input("Type your Query:", key="user_input")

send_button = st.button("Send")

st.markdown("""
    <style>
    button.st-emotion-cache-1vt4y43 {
        margin-left: 290px;
        width:100px
    }
    
    p {
        color: black !important;
    }
            
    a{
        color: blue !important;      
    }
            
    </style>
""", unsafe_allow_html=True)

if send_button==True:

    load_dotenv()


    def tavily_search_function(q):
        tavily_client=TavilyClient()
        search_results=tavily_client.search(q,max_results=7)
        return search_results['results']


    text_to_searchquery_prompt=ChatPromptTemplate.from_messages([
        ('system','''Write 3 google search queries to search online that form an "
            "objective opinion from the following: {question}\n".If the word 'current' or 'latest' is present in the given question then add '2024' to the search queries.
            "You must respond with a list of strings in the following format: "
            [query 1,query 2,query 3].''')
    ])

    # openai_llm=ChatOpenAI(model='gpt-4o-mini')
    groq_llm=ChatGroq(model='llama-3.1-70b-versatile',temperature=0)

    text_to_searchquery_chain=text_to_searchquery_prompt|groq_llm|StrOutputParser()|(lambda x:x[1:-1].replace('"',"").split(","))|(lambda x:[{'question':i} for i in x])

    web_page_qa_prompt1=ChatPromptTemplate.from_messages([
        ('system',"""{text} 
            -----------
            Using the above text, answer in short the following question: 
            > {question}
            -----------
            if the question cannot be answered using the text, imply summarize the text. Include all factual information, numbers, stats etc if available.""" )
    ])


    # web_page_qa_chain1=RunnablePassthrough.assign(summary=web_page_qa_prompt1|openai_llm|StrOutputParser())|(lambda x:f"Summary:{x['summary']}\nURL:{x['url']}")

    web_page_qa_chain1=RunnablePassthrough.assign(summary=lambda x:x['text'])|(lambda x:f"Summary:{x['summary']}\nURL:{x['url']}")

    multipage_qa_chain1=(RunnablePassthrough.assign(text=lambda x:tavily_search_function(x['question']))
    |(lambda x:[{'question':x['question'],'text':i['content'],'url':i['url']} for i in x['text']])
    |web_page_qa_chain1.map())

    def summary_list_exploder(l):
        rl=[]
        final_researched_content=""
        for i in l:
            rl.extend(i)
        for i in rl:
            final_researched_content=final_researched_content+i+" \n\n"
        return final_researched_content

    complete_summarizer_chain=(
        text_to_searchquery_chain
        |multipage_qa_chain1.map()
        |RunnableLambda(summary_list_exploder))

    WRITER_SYSTEM_PROMPT = "You are an AI critical thinker research assistant. Your sole purpose is to write well written, critically acclaimed, objective and structured answers for the questions asked based on the given text." 

    RESEARCH_REPORT_TEMPLATE = """Information:
    --------
    {research_summary}
    --------
    Using the above information, answer the following question or topic: "{question}" in a detailed answer -- \
    The report should focus on the answer to the question, be well-structured, informative, \
    in-depth, with facts and numbers if available.
    You should strive to write the answer as long as possible, using all relevant and necessary information provided.
    Write the answer with markdown syntax.
    You MUST determine your own concrete and valid opinion based on the given information. Avoid general or vague conclusions.You must always give more importance to the latest information that is from the year 2024 in your answer.

    For citations:
    1. Include APA in-text citations as clickable hyperlinks using this format: 
    [Author/Organization, Year](URL)
    Example: [Smith & Johnson, 2024](https://example.com)

    2. When referencing multiple pieces of information from the same source in one paragraph, include the hyperlinked citation at the end of the paragraph.

    3. At the end of the report, include a "References" section with all source URLs , with no duplicates.

    Please do your best, as this is very important to my career"""

    final_research_prompt=ChatPromptTemplate.from_messages([
        ('system',WRITER_SYSTEM_PROMPT),
        ('user',RESEARCH_REPORT_TEMPLATE)
    ])

    final_research_report_chain=(
        RunnablePassthrough.assign(research_summary=complete_summarizer_chain)
        |final_research_prompt
        |groq_llm
        |StrOutputParser())
   
    st.header('Response:-')
    
    class prompt_classifier(BaseModel):
        response:str=Field(description="'Yes' if the question is related to healthcare and 'No' if the question is not related to healthcare")

    json_parser=JsonOutputParser(pydantic_object=prompt_classifier)

    prompt_classifier_prompt=ChatPromptTemplate.from_messages([
        ('system','''Classify the provided question as either 'Yes' or 'No'. Respond 'Yes' if the question is related to healthcare and 'No' if it is unrelated to healthcare.\n{format_instructions}\nQuestion:{question}''')
    ])

    prompt_classifier_chain=(prompt_classifier_prompt|groq_llm|json_parser|(lambda x:x['response']))

    prompt_classifier_chain_response=prompt_classifier_chain.invoke({'question':user_input,'format_instructions':json_parser.get_format_instructions()})

    if prompt_classifier_chain_response=='Yes':
        info_container = st.empty()
        full_response = ""
        for chunk in final_research_report_chain.stream({'question': user_input}):
            full_response += chunk
            info_container.info(full_response)
    
    else:
        st.info("Hi,I'm A.R.I.S and I'm here to assist you with healthcare-related questions. Could you please rephrase or ask a question related to healthcare?")


