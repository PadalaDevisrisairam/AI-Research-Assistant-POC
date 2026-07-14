in this commit it update is rag is loaded from existing faiss database if it present if not then the database will be created it the rag data is changed then we need to delete the the faiss database

and one more thing is added the shortterm memory for the llm in a session so that it is easy for llm to maintain the context through the session