Created a google chrome browser extension that runs when prompting chat gpt or gemini:

runs local ollama or gpt2 model that shadows LLM production response and generates the mapping of each word to expected output with prospects/probabilities to determine if responses are straying away from initial prompt to indicate a potential hallucination

systematic algorithm works in theory but the local models used are too far behind gpt5+ and gemini3+ used in browser for accurate shadowing and prediction matching.
