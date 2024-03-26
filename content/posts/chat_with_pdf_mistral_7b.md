+++
title = 'Chat with PDF Mistral-7B: Interacting with Documents Through AI'
date = 2024-02-02T18:30:33+05:30
draft = false
+++

In the age of information overload, efficiently extracting knowledge from documents is crucial. Chat with PDF’s Mistral-7B leverages AI to transform static PDF files into dynamic conversational partners. Imagine querying any PDF document as if you were chatting with an expert on its content. This blog will guide you through creating such an interactive experience, using the Mistral-7B model within a Streamlit web application.

## Setting Up the Environment
Before diving into the code, you need to set up your environment in Google Colab. Make sure to use an A100 GPU for optimal performance:

1. **Access Google Colab:** Start a new notebook in [Google Colab](https://colab.research.google.com/).
2. **Select A100 GPU:** Go to `Runtime > Change runtime type` and set the hardware accelerator to GPU. This ensures faster processing and better performance of the Mistral-7B model.

### Installing Necessary Python Packages
Install the following packages to get started with the project:

```bash
!pip install langchain torch sentence_transformers faiss-cpu huggingface-hub pypdf2 accelerate llama-cpp-python
```

- `langchain`: Utilized for chaining language tasks.
- `torch`: The backbone for deep learning models.
- `sentence_transformers`: For efficient sentence embeddings.
- `faiss-cpu`: For fast similarity searches and clustering large collections of vectors.
- `huggingface-hub`: To access and manage models and repositories.
- `pypdf2`: For extracting text from PDFs.
- `accelerate`: To speed up computations.
- `llama-cpp-python`: Python bindings for LLaMA models.

## Understanding the Code
The project relies on several Python libraries, each serving a unique purpose in the process of converting PDF documents into interactive chat experiences.

### Loading and Processing PDF Documents
To load and process PDF files, we use the `PyPDFDirectoryLoader` class from `langchain`:

```python
from langchain.document_loaders import PyPDFDirectoryLoader

# Load PDF files from a specified directory
loader = PyPDFDirectoryLoader("/content/sample_data/data")
data = loader.load()
print(data)
```

`PyPDFDirectoryLoader` loads all PDF files in the specified directory, making their content accessible for processing.

### Building the Core Functionality
#### Text Extraction and Chunking
We use `RecursiveCharacterTextSplitter` to divide the PDF text into manageable chunks:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
text_chunks = text_splitter.split_documents(data)
```

`RecursiveCharacterTextSplitter` breaks down the text into chunks of 10,000 characters, with a 20-character overlap between chunks to ensure continuity.

#### Generating Embeddings
To understand and retrieve information from the text, we convert the chunks into numerical representations called embeddings:

```python
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
```

`HuggingFaceEmbeddings` creates sentence embeddings, which `FAISS` then uses to index the text chunks for efficient retrieval.

### Setting Up the Mistral-7B Model for Q&A
The Mistral-7B model, a powerful language model, is integrated to provide accurate answers based on the text chunks:

```python
from langchain.llms import LlamaCpp

llm = LlamaCpp(
    streaming = True,
    model_path="/content/drive/MyDrive/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    temperature=0.65,
    top_p=1,
    verbose=True,
    n_ctx=4096
)
```

`LlamaCpp` loads the Mistral-7B model, with parameters set for temperature, top_p, and context length to control the generation's creativity and relevance.

## Creating the Streamlit Web App
The Streamlit web app is the user interface for interacting with the AI model. It enables users to upload PDFs, ask questions, and receive answers based on the document's content.

### Designing the Chat Interface
We use `streamlit_chat` to create a chat interface where users can interact with the Mistral-7B model:

```python
import streamlit as st
from streamlit_chat import message

st.title("Chat with PDF’s Mistral-7B")
st.sidebar.title("Upload PDF")
uploaded_files = st.sidebar.file_uploader("Choose a file", accept_multiple_files=True)
```

This sets up the basic UI, including a title and a sidebar for uploading PDF files.

### Integrating the Model with Streamlit
After uploading, the PDFs are processed, and the app communicates with the Mistral-7B model to answer queries based on the PDF content:

```python
if uploaded_files:
    for file in uploaded_files:
        text = extract_text(file)
        processed_text = process_text(text)
        answer = get_answer(processed_text, query)
        st.write(answer)
```

Here, `extract_text` reads the PDF, `process_text` prepares the text for the model, and `get_answer` queries the model to fetch the response.

### Handling User Queries and Displaying Responses
The user's questions are fed to the model, and responses are displayed in real-time:

```python
query = st.text_input("Ask a question:")
if st.button("Get Answer"):
    response = model.query(query)
    message(response, is_user=True)
```

The `query` function sends the user's question to the model, and `message` displays the model's response in the chat interface.

## Deploying the App
Deploying the app involves running the Streamlit application and using LocalTunnel to expose the local server to the public:

```bash
!streamlit run app.py &>/dev/null & curl ipv4.icanhazip.com
!npx localtunnel --port 8501
```

This makes the app accessible via a public URL, allowing anyone to interact with the AI model and the uploaded PDF documents.

## Conclusion
Chat with PDF’s Mistral-7B transforms static documents into dynamic conversational entities, leveraging AI to make information retrieval interactive and efficient. By following this guide, you can create a web-based chat interface to interact with the content of PDF files, powered by the advanced capabilities of the Mistral-7B model.