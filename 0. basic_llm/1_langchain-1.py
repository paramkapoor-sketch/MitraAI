# Note:
# Use something like this: conda create --name modernaipro python=3.11 --file requirements.txt
# conda activate modernaipro

from langchain_community.llms import Ollama
llm = Ollama(model="deepseek-r1:1.5b") # try qwen2 / llama3 if you have that model


for chunks in llm.stream("How to make a pizza?"):
    print(chunks, end='')
