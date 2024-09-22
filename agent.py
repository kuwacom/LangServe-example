from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "../text-generation-webui/models/Llama-3-ELYZA-JP-8B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    max_length=1024,
    device_map="auto",
    # torch_dtype=torch.float16,  # メモリ使用量を減らすために float16 を使用
    # low_cpu_mem_usage=True      # CPU メモリの使用を抑えるオプション
    )

from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

llama_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128,
    temperature=0.8,
    top_p=0.5,
    top_k=65,
    typical_p=1,
    min_p=0,
)

llm = HuggingFacePipeline(pipeline=llama_pipeline)

from langchain.prompts import PromptTemplate

from fastapi import FastAPI
from langserve import add_routes

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="LangchainのRunnableインターフェースを使ったシンプルなAPIサーバー",
)

add_routes(
    app,
    # chain,
    llm,
    path="/llama3"
)

from langchain.agents import create_react_agent, AgentExecutor, initialize_agent, AgentType
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.memory import ConversationBufferMemory

LLama3InstToken = {
    'begin_of_text': '<|begin_of_text|>',
    'start_header_id': '<|start_header_id|>',
    'end_header_id': '<|end_header_id|>',
    'eot_id': '<|eot_id|>',
}

tools = [
    DuckDuckGoSearchRun()
]

prompt = PromptTemplate(
    template=
    f"${LLama3InstToken['begin_of_text']}Answer the following questions as best you can. You have access to the following tools:\n"+
    "{tools}\n"+
    "Use the following format:\n"+
    "Question: the input question you must answer\n"+
    "Thought: you should always think about what to do\n"+
    "Action: the action to take, should be one of [{tool_names}]\n"+
    "Action Input: the input to the action\n"+
    "Observation: the result of the action\n"+
    "... (this Thought/Action/Action Input/Observation can repeat N times)\n"+
    "Thought: I now know the final answer\n"+
    "Final Answer: the final answer to the original input question\n"+
    "Begin!\n"+
    "Question: {input}\n"+
    "Thought:{agent_scratchpad}",
    input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
)

agent = initialize_agent(
    llm=llm,
    tools=tools,
    agent = AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    # output_key = "result",
    # handle_parsing_errors = True,
    # max_iterations=3,
    # early_stopping_method="generate",      
    memory = ConversationBufferMemory(memory_key = 'chat_history')               
)

print(agent.invoke(
    {
        "input": "東京の今日の天気は？"
    }
))

add_routes(
    app,
    agent,
    path="/llama3-agent"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8100)