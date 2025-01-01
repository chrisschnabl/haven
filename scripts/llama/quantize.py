#git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
# mdkir build && cd build && cmake .. && make

# add ./bin to PATH

# Download model ffrom h3 first
# pip install -U huggingface_hub
#huggingface-cli login
# huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct \
#--exclude "original/*" \
# --local-dir models/meta-llama/Meta-Llama-3.1-8B-Instruct

# python3 -m pip install -r requirements.txt
# python3 convert_hf_to_gguf.py models/meta-llama/Meta-Llama-3.1-8B-Instruct/


# git pull llama.cpp 
# make 

# ./quantize.sh
# ./llama-cli -m ./models/meta-llama/Meta-Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf--ftype -cnv -p "SYSTEM PROMPT"


# TODO:
#  for ec2 instance, try to copy and run already quantized model with llama cpp within the enclave