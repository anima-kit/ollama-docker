### latency_test
## Executes a simple test of the latency of an Ollama server in Docker

import time
from pyfiles.ollama_utils import OllamaClient
from pyfiles.logger import logger

logger.info(f'⚙️ Starting latency test in `./scripts/latency_test.py`')

prompt = """
Respond with only the exact answer to: 'What is 2+2' and nothing else. 
Do not explain, do not add formatting, just the plain text response.
"""
lm_name = 'gemma2:2b'

def measure_latency(client, prompt=prompt, lm_name=lm_name):
    start = time.perf_counter()
    # Defaults to use model 'gemma2:2b'
    response = client.get_response(message=prompt, lm_name=lm_name)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(f"Latency: {elapsed_ms:.1f} ms")
    return elapsed_ms


n_tests = 100
latency_sum = 0
## Initialize Ollama client
# Defaults to host on url 'http://localhost:11434'
client: OllamaClient = OllamaClient()
# Warm up the client first
client.get_response(message='Hi', lm_name=lm_name)
for i in range(n_tests):
    latency = measure_latency(client)
    latency_sum += latency
    logger.info(f"Test {i}")

latency_avg = latency_sum/n_tests
logger.info(f"Latency average: {latency_avg:.1f} for LLM `{lm_name}`")
logger.info(f'✅ Finished latency test in `./scripts/latency_test.py` \n\n')