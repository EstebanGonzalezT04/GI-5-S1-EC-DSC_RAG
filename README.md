# GI-5-S1-EC-DSC - Retrieval Augmented Generation (RAG)

This repository contains the tutorial files for the 2025-2026 Data Science course of 5GI INSA Lyon.


In this lab, you will explore Retrieval Augmented Generation (RAG) techniques, which integrate information retrieval with generative large language models (LLMs) to improve the quality of generated text based on relevant external documents.


## Academic Integrity and Authorized Resources

To ensure fair evaluation and genuine learning, please follow these rules when completing the lab:

✅ You are allowed to:
* Search for information using search engines (Google, Bing, etc.)
* Consult official documentation for libraries and tools used in the lab
* Use educational or technical websites such as Stack Overflow, GeeksforGeeks, or academic blogs
* Discuss ideas and concepts with classmates at a high level (no code sharing)
* Ask the teacher questions during the lab sessions if you need clarification or guidance

🚫 You are not allowed to:
* Use generative AI tools (e.g., ChatGPT, Copilot, Gemini, Claude, etc.) to generate or modify code or written explanations
* Submit code, text, or analyses that were not written or understood by you
* Copy solutions directly from other students or online repositories

🧭 Important note

You are encouraged to understand, test, and experiment with the concepts presented in this lab.
External sources can help you learn — but all submitted work must reflect your own understanding and be authored by you.


## 1. Objective

This project explores the use of RAG to support decision-making and knowledge sharing in railway maintenance operations, using the SMRT Maintenance Logs dataset as a domain-specific knowledge base.

The dataset contains thousands of real-world maintenance records describing faults, diagnostics, and corrective actions across a wide range of train subsystems. By combining semantic retrieval with LLMs, the system must enable natural-language access to historical maintenance knowledge that would otherwise remain locked in unstructured logs.

The proposed RAG system will be designed to support three key user groups:
* Maintenance engineers, by retrieving similar past fault cases and summarizing proven corrective actions to accelerate troubleshooting and reduce downtime.
* Operations control centers, by providing rapid situational awareness and context when incidents occur, enabling informed operational decisions under time pressure.
* Knowledge transfer to junior staff, by transforming tacit, experience-based maintenance knowledge into explainable, searchable guidance derived from historical records.



## 2. Steps to Complete

![RAG Workflow](./assets/img/rag_workflow.png)

1. Preprocess the documents and split them into chunks.
2. Compute embeddings using `sentence-transformers`.
3. Index the data in a vector store (`ChromaDB`).
4. Implement retrieval and design the RAG prompt.
5. Use LLMs through a given API for the generation step.
6. Evaluate the system using a set of test questions.

> Sample code and notebooks are provided in the `notebooks/` directory to help you get started with each step:
> - [load dataset](./notebooks/1_load_dataset.ipynb)
> - [embedding computation](./notebooks/2_embeddings.ipynb)
> - [vector database setup](./notebooks/2_vector_db.ipynb)
> - [LLM generation with LMStudio](./notebooks/3_generation_llm.ipynb)
> - [LLM generation with Google's Gemini and Gemma models](./notebooks/4_generation_llm_gemini.ipynb)


## 3. Dataset

The [SMRT-Maintenance-Logs dataset](https://huggingface.co/datasets/parclytaxel/SMRT-Maintenance-Logs) is a text dataset consisting of 10,000 real maintenance log entries from SMRT, Singapore’s public transport operator. 
Each entry in the dataset includes two key fields:
* `text` – A string containing a maintenance log description. These logs typically describe operational faults or faults reported on trains (e.g., brake faults, lights not working, door issues), what checks were performed, and what rectifications were done. 
* `label` – A categorical class label associated with each log entry.


## 4. LLM Access

The use of LLMs is required for the generation step of your RAG system. Serveral options are available to you:


### 4.1. Provided API for LLM Access

You will be provided with API endpoints for sending queries to LLMs hosted on LM Studio instances on server machines managed by the DSI of INSA Lyon.

These instances are only accessible to queries from within the INSA virtual desktop: https://bv.insa-lyon.fr/.

> More information about INSA Virtual desktop here: https://dsi.insa-lyon.fr/content/bureau-virtuel

The following models from the [ministral](https://lmstudio.ai/models/ministral) family are available:
* `Ministral-3-3B-Instruct-2512`
* `Ministral-3-8B-Instruct-2512`
* `Ministral-3-14B-Instruct-2512`

The API endpoint URLs will be provided to you during the lab sessions.

An example notebook demonstrating how to query LLMs through LMStudio API is provided:
- [LLM generation with LMStudio](./notebooks/4_generation_llm_lmstudio.ipynb)

### 4.2. LM Studio API for Local LLMs

Another option is to run small local LLMs on your own machine using [LM Studio](https://lmstudio.ai/). LM Studio allows you to run various open-source LLMs locally and provides an OpenAI-compatible API for easy integration.

See the example notebook demonstrating how to query local LLMs through LMStudio API:
- [LLM generation with LMStudio](./notebooks/4_generation_llm_lmstudio.ipynb)

### 4.3. Google AI API for Gemini and Gemma Models

The third option is to use Google's Gemini and Gemma models through the [Google AI API](https://aistudio.google.com/). 

Most recent Gemini models have restricted limit rate (20 requests per day) while Gemma models have higher rate limits (14k requests per day).
To use the Google API you need a Google account and get a free API key from: https://aistudio.google.com/app/api-keys

An example notebook demonstrating how to query Gemini and Gemma models through the Google AI API is provided:

- [LLM generation with Google's Gemini and Gemma models](./notebooks/4_generation_llm_gemini.ipynb)



## 5. Experiments

You must experiment with different configurations to observe their impact on the performance of your RAG system. Below is a (non-ordered) list of experiments to be conducted as part of this project.

1. **Source Attribution**: Investigate methods to attribute generated answers back to the source documents.
    - Implement techniques to highlight which retrieved documents contributed to the final answer.
    - How does source attribution affect user trust in the system?

2. **Chunk Size**: Vary the size of the text chunks used for embedding. 
    - Test small (50 tokens), medium (200 tokens), and large (500 tokens) chunk sizes to see how they affect retrieval quality.
    - Overlap: Try different overlap sizes to see how context affects retrieval.
    - Which chunk size and overlap maximizes relevance?

3. **Top-K Retrieval**: Experiment with different values of K (e.g., 1, 3, 5, 10) for the number of top documents retrieved.
    - How does changing K impact the quality of the generated answers?

4. **Embedding Models**: Use different pre-trained models from `sentence-transformers` (e.g., `all-MiniLM-L6-v2`, `paraphrase-MiniLM-L12-v2`, etc.).
    - Compare their performance in terms of retrieval accuracy and answer quality.
    - Which model provides the best trade-off between speed and accuracy?

5. **Prompt Engineering**: Experiment with different prompt templates for the LLM.
    - Test various ways of framing the retrieved documents and questions to see how it affects the generated responses.
    - Which prompt structure yields the most accurate and coherent answers?

6. **Automatic Evaluation**: Implement automatic metrics to evaluate the quality of the generated answers.
    - Provide a set of reference answers for the evaluation questions.
    - Use metrics like BLEU, ROUGE, or cosine similarity with reference answers.

7. **Hallucination Testing**: Design specific questions that are likely to induce hallucinations in the LLM (out-of-scope questions).
    - Analyze how often the model generates incorrect or fabricated information.
    - What strategies can be employed to minimize hallucinations?

8. **Analyze Retrieval Quality**: Before passing the retrieved documents to the LLM, evaluate their relevance to the questions.
    - Experiment with: stop-word removal, text normalization, and other preprocessing techniques to improve retrieval.
    - How does retrieval quality correlate with the final answer quality?

9. **Test Multiple LLMs**: Test your RAG system with different local LLMs available in LM Studio.
    - Compare the performance of different models in terms of answer quality and response time.
    - Which LLM works best in conjunction with your retrieval system?



## 6. Deliverables

- A short technical report (4–6 pages)
- Code + notebook(s) with comments
- A summary table and figures comparing experiments