import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# model_name = "meta-llama/Llama-2-7b-chat-hf"
# model_name = "models/ELYZA-japanese-Llama-2-7b"
model_name = "../../project/Character-BOT/models/Llama-3-ELYZA-JP-8B"

lora_model_name = ""
useLoRA = False

tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    max_length=1024,
    device_map="auto",
    torch_dtype=torch.float16,  # メモリ使用量を減らすために float16 を使用
    # low_cpu_mem_usage=True      # CPU メモリの使用を抑えるオプション
    )

# LoRA モデルの条件付きロード
if useLoRA:
    lora_model = PeftModel.from_pretrained(base_model, lora_model_name)
    model = lora_model
else:
    model = base_model

# if torch.cuda.device_count() > 1:
#     print(f"Using {torch.cuda.device_count()} GPUs!")
#     model = torch.nn.DataParallel(model)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

llama_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    # top_p=0.95,
    # repetition_penalty=1.15,
)

llm = HuggingFacePipeline(pipeline=llama_pipeline)


from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain_core.prompts import ChatPromptTemplate
import datetime

dt_now = str(datetime.datetime.now())
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "あなたはエージェント型チャットボットです。\n"
        "過去の会話を参照しながら対話者（僕）と会話することができます。\n"
        "発言は、100字以内で短く返してください。\n\n"
        # "{past_chats_context}"  # <--------- ここに過去の会話内容を挿入 
        f"現在の日時：{dt_now}\n\nそれでは、会話開始です。"
    ),
    MessagesPlaceholder(variable_name="today_history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "あなたの名前は 256大好き です。"),
#         ("system", "年齢は16歳で現在高校1年生です。"),
# #        ("system", "あなたは電子技術などに興味があるガジェットオタクです。電子工作やサーバー運営などを好んでいます。"),
# #        ("system", "あなたは俗にいう陰キャで、いつもDiscordでインターネット上の友達とチャットをしたり秋葉原に行ったり、ツイッターで毎日100件以上つぶやいています。"),
# #        ("system", "LINEでもネットと同じ名前を使っています。学校ではあまりなじめていないようです。"),
# #        ("system", "気分によっては陰湿な感じにもなります。いつもチャットでは端的に返信します。"),
# #        ("system", "ほかの人の	メッセージの連投チャットやネットミームに便乗したりしています。"),
# #        ("system", "利用している回線は 光回線 ソフトバンク光 モバイル ソフトバンク ドコモ POVO TONE です。"),
#         ("ai", "やあ"),
#         ("user", "{user_input}"),
#     ]
# )

# template = """{question}"""
# prompt = ChatPromptTemplate.from_messages(
#     [
#         HumanMessagePromptTemplate.from_template(template),
#         AIMessagePromptTemplate.from_template(""),
#     ]
# )

# prompt_template = """あなたは親切で知識豊富なアシスタントです。
# 以下は会話の履歴です。
# {history}
# ユーザー: {userInput}
# ボット:"""
# prompt_template = "ユーザー: {userInput}\nAI:"
# prompt = PromptTemplate(template=prompt_template, input_variables=["history", "userInput"])
# prompt = PromptTemplate(template=prompt_template, input_variables=["userInput"])

# chain = LLMChain(llm=llm, prompt=prompt)
chain = prompt | llm

from fastapi import FastAPI
from langserve import add_routes

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="LangchainのRunnableインターフェースを使ったシンプルなAPIサーバー",
)

add_routes(
    app,
    chain,
    path="/llama3"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8100)