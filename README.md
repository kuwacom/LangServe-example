# What Is This
LangChain の LangServe のテストコードです

ローカルの Transformers llama2 モデルを利用してLangServe API を起動することができます

事前に`venv`のセットアップと必要モジュールのインストールをしてください

## Pytorch
> CUDA !2.2
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Llama.cpp
> Win
```shell
set CMAKE_ARGS="-DLLAMA_CUDA=on"
set FORCE_CMAKE=1
pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
```
> Linux(Bash)
```bash
export CMAKE_ARGS="-DLLAMA_CUDA=on"
export FORCE_CMAKE=1
pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
```