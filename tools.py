from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()
def web_search(query: str):
    return search.run(query)
