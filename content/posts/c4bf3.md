+++
title = 'Building an AI Chatbot with CodeLlama and Streamlit in Google Colab'
date = 2024-02-02T18:29:40+05:30
draft = false
+++

Let's dive into a detailed explanation of each step and the code, ensuring the readers understand what they are doing and why. Here's how we can structure and explain the blog:

---

In this comprehensive guide, we'll explore how to build an AI chatbot using the CodeLlama model and Streamlit framework, hosted in Google Colab with an A100 GPU. This setup ensures you have the computational power needed for real-time, responsive AI interactions.

---

### Setting Up Google Colab with an A100 GPU
To begin, ensure you're using Google Colab and have selected an A100 GPU as your hardware accelerator. This powerful GPU will significantly speed up model training and inference processes.

1. **Open Google Colab**: Navigate to [Google Colab](https://colab.research.google.com/).
2. **Select GPU**: Go to `Edit > Notebook settings` or `Runtime > Change runtime type` and choose `GPU` as the hardware accelerator.

### Installing Dependencies
The first step in our project is to install the necessary packages, including `localtunnel` for exposing our local server to the web, `streamlit` for creating the web app, and `transformers` for loading our AI model.

```bash
!npm install localtunnel
!pip install streamlit streamlit_chat git+https://github.com/huggingface/transformers.git@main accelerate
```

- **localtunnel**: Creates a public URL for our local web server.
- **streamlit**: A fast way to build custom web apps.
- **streamlit_chat**: Adds chat functionality to our Streamlit app.
- **transformers**: Provides access to pre-trained models like CodeLlama.
- **accelerate**: Optimizes PyTorch code for faster computation.

### Setting Up the Environment
Before diving into the code, it’s crucial to set the right environment variables for smooth execution:

```python
import os
os.environ['LC_ALL'] = 'C.UTF-8'
```

This ensures that the environment uses the 'C.UTF-8' locale, which can prevent common encoding errors during the execution.

### Initializing the AI Model
We use the CodeLlama model for our chatbot. This model is known for its ability to understand and generate human-like responses in code and natural language.

```python
from transformers import AutoTokenizer
import transformers
import torch

model = "codellama/CodeLlama-7b-Instruct-hf"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)
```

- **AutoTokenizer**: Loads the tokenizer that corresponds with the CodeLlama model.
- **pipeline**: Sets up a text generation pipeline with the model, optimized to run on the GPU.

### Generating Text with the Model
To interact with the model, we set up a system prompt and use the pipeline to generate responses:

```python
system = "You are a python Expert.Provide answers in python"
user_input1 = """def add_even_numbers(a, b):"""

prompt = f"<s><<SYS>>\n{system}\n<</SYS>>\n\n{user_input1}"

sequences = pipeline(
    prompt,
    do_sample=True,
    top_k=10,
    temperature=0.1,
    top_p=0.95,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
    truncation=True
)
```

This block of code demonstrates how to create a prompt for the model and request a generated response, with parameters controlling the diversity and length of the generation.

### Displaying the Result
To see the output of our model, we print the generated sequences:

```python
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
```

Sure, let's expand on the "Creating the Streamlit Web App" section to provide a full, detailed explanation:

---
### Creating the Streamlit Web App

To make our AI model accessible and interactive, we'll create a web application using Streamlit. This framework allows us to build a user-friendly interface where users can input questions and receive answers from the CodeLlama model. Here’s how to set it up:

1. **Writing the App Code:**
   First, we need to write the code for our Streamlit app. We'll save this code in a file named `app.py`. This file will contain the logic for our chat interface, model interaction, and response generation.

    ```python
    %%writefile app.py
    import streamlit as st
    from streamlit_chat import message
    from transformers import AutoTokenizer
    import transformers
    import torch

    # Function to generate responses
    def generate_response(prompt, model):
        sequences = model(
            prompt,
            do_sample=True,
            top_k=10,
            temperature=0.1,
            top_p=0.95,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=200,
            add_special_tokens=False
        )
        return sequences[0]['generated_text']

    # Set up model and tokenizer
    model_name = "codellama/CodeLlama-7b-Instruct-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Streamlit app layout
    st.title("CodeLlama Chatbot")
    st.write("Type your coding question below and get expert answers!")

    # Language selection in sidebar
    selected_language = st.sidebar.selectbox("Select Programming Language", ["Python", "Java", "C++"])

    # Language specific system message
    system_messages = {
        "Python": "You are a Python Expert. Provide answers in Python",
        "Java": "You are a Java Expert. Provide answers in Java",
        "C++": "You are a C++ Expert. Provide answers in C++"
    }
    system_message = system_messages[selected_language]

    # Handling chat
    if 'past' not in st.session_state:
        st.session_state['past'] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    with st.container():
        user_input = st.text_input("Query:", key='input')
        submit_button = st.button('Send')

        if submit_button and user_input:
            user_input_formatted = f"<s><<SYS>>\n{system_message}\n<</SYS>>\n\n{user_input}"
            response = generate_response(user_input_formatted, pipeline)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(response)

    if st.session_state['generated']:
        with st.container():
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=f"{i}_user")
                message(st.session_state["generated"][i], key=f"{i}")
    ```

    In this code:
    
    - We define a `generate_response` function that takes the user input and model to generate responses.
    - We set up the tokenizer and model pipeline similar to the previous setup.
    - The Streamlit app's interface is built using `st.title`, `st.write`, and `st.text_input` for displaying the title, instructions, and input box, respectively.
    - `st.sidebar.selectbox` allows the user to select the programming language, and based on this selection, the appropriate system message is chosen.
    - The chat history is managed using Streamlit’s session state to persist the conversation in the app.

2. **Running the App:**
   To run our Streamlit app, we execute it in the background and use `localtunnel` to expose it to the web:

    ```bash
    !streamlit run app.py &>/dev/null & curl ipv4.icanhazip.com
    !npx localtunnel --port 8501
    ```

    These commands start the Streamlit app and print the public URL where the app is accessible. The `localtunnel` command creates a public link to our local server running on port 8501, allowing external access to our Streamlit app.

## Conclusion
By following these steps, you have created an AI chatbot using the CodeLlama model and Streamlit in Google Colab with an A100 GPU. This setup not only provides a powerful backend for real-time AI interactions but also demonstrates how to integrate cutting-edge AI models into interactive web applications.
