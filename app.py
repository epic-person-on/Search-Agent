import chainlit as cl
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import SearxSearchWrapper
from langchain_community.tools.searx_search.tool import SearxSearchResults
from langchain.agents import initialize_agent, AgentType
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Arxiv, Wikipedia, and Searxng Tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

searxng_wrapper = SearxSearchWrapper(searx_host="http://localhost:8080")
google = SearxSearchResults(name="Google", wrapper=searxng_wrapper, kwargs={"engines": ["google"]})
gitlab = SearxSearchResults(name="Gitlab", wrapper=searxng_wrapper, kwargs={"engines": ["gitlab"]})


# Initialize language model and tools outside the user input block
llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-specdec", streaming=True)
tools = [arxiv, wiki, google, gitlab]

# Initialize agent
search_agent = initialize_agent(
    tools, llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True
)

# Setup callback handler
@cl.on_chat_start
def setup_agent():
    # Store the agent in the user session for later use
    cl.user_session.set("search_agent", search_agent)


@cl.on_message
async def handle_message(message: cl.Message):
    user_input = message.content.lower()

    # Retrieve the agent from user session
    search_agent = cl.user_session.get("search_agent")

    # Process the message with LangChain's agent
    try:
        # Get the agent's response
        response = search_agent.run([{"role": "user", "content": user_input}])
        
        # Send the response to Chainlit chat interface
        await cl.Message(content=response).send()

    except ValueError as e:
        await cl.Message(content=f"An error occurred: {e}").send()
    except Exception as e:
        await cl.Message(content=f"Unexpected error occurred: {e}").send
