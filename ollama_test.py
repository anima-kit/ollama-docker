### ollama_test
## Executes a simple test to check that an Ollama server in Docker can be properly invoked by a Python environment
## The script runs as follows:
#   - Initialize OllamaClient (setup client in Ollama Python library | https://github.com/ollama/ollama-python)
#   - Get LM response for given message and from given LM
#       - If specified LM is not in available models, pull it from Ollama
#       - Invoke client to get LM response
#       - Cleanup LM response (remove think tags and text within)
#       - Output final response

from ollama_utils import OllamaClient
from logger import logger

logger.info(f'⚙️ Starting Ollama test in `./ollama_test.py`')

## Initialize Ollama client
# Defaults to host on url 'http://localhost:11434'
client: OllamaClient = OllamaClient()

## Get response
# Defaults to use LM 'qwen3:0.6b' 
# Defaults to send message 'Why is the sky blue?'
client.get_response()

logger.info(f'✅ Finished Ollama test in `./ollama_test.py` \n\n')