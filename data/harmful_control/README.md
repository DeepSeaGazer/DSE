
## Overview

This subfolder provides a set of datasets and code designed to evaluate the balance between harmfulness and usefulness in AI models. As part of the larger project, it focuses on in-depth analysis of specific types of AI responses.

### Dataset and Code Description

- **`harmful_behaviors.csv`**: Sourced from [Zou et al.](https://github.com/llm-attacks/llm-attacks), this file contains instructions that could elicit harmful responses from AI models. It serves as a primary resource for evaluating model harmfulness.
- **`scripts/generate_dataset.py`**: A script to generate semantically similar harmless instructions, complementing the harmful dataset.
- **`combined_instructions_*.csv`**: These files consist of instruction data we generated for evaluation, showcasing a mix of harmful and harmless instructions.

### Generation Process

The generation of datasets leverages GPT-4 to create an equal number of harmless instructions, juxtaposed with the harmful instructions from `harmful_behaviors.csv`. However, due to the inherent randomness in GPT-4's outcomes, identical results in each generation are not guaranteed.

With the `generate_dataset.py` script, users have the capability to independently generate dataset files for their own evaluation purposes.

## Usage

You can generate your own evaluation dataset by running `generate_dataset.py`. Run the script with the appropriate arguments. For example:

```
python generate_dataset.py --api_key YOUR_API_KEY
```

**Script Parameters**:
- `--api_key`: Your OpenAI API key.
- `--base_url`: The base URL for the OpenAI API (default is 'https://api.openai.com/v1').
- `--gen_num`: The number of files to be generated (default is 1).
- `--model`: The model name (default is 'gpt-4-1106-preview').
