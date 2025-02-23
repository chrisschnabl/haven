from huggingface_hub import hf_hub_download

file_path = hf_hub_download(
    repo_id="TheBloke/Llama-2-7B-GGUF",
    filename="llama-2-7b.Q2_K.gguf"
)

print(f"File downloaded to: {file_path}")
# TODO CS: dowmload llama 3 at some point