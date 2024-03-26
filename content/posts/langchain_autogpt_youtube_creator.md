+++
title = 'LangChain AutoGPT Crafting a YouTube GPT Creator with Streamlit'
date = 2024-02-02T18:31:22+05:30
draft = false
+++

Welcome to the fascinating world of LangChain AutoGPT, where we merge the power of AI with the simplicity of Streamlit to create a YouTube GPT Creator. This blog is your all-access pass to understanding the nuts and bolts of each line of code that goes into making an automated content generation app.

Let's dive into the nitty-gritty of building an application that not only generates YouTube video titles and scripts but also learns and adapts over time.

## Installing the Required Libraries

Before we get our hands dirty with code, we need to gear up with the right tools:

**Streamlit**: This is our gateway to creating web applications swiftly. With Streamlit, we can turn data scripts into shareable web apps in a jiffy.

```bash
!pip install streamlit
```

**Localtunnel**: Since we're using Google Colab (a virtual machine), we need a way to expose our developing app to the outside world. Localtunnel serves this purpose, similar to how we've used ngrok in the past.

```bash
!npm install localtunnel
```

**LangChain**: This framework is the backbone of our app, enabling us to build applications powered by language models seamlessly.

```bash
!pip install langchain
```

**OpenAI**: By integrating OpenAI, we tap into the prowess of advanced natural language processing models, like GPT-3, which are crucial for generating content.

```bash
!pip install openai
```

**Wikipedia**: To enrich our generated content with accurate and reliable information, we employ the Wikipedia library, granting us easy access to a vast repository of knowledge.

```bash
!pip install wikipedia
```

## Setting Up the Environment

Now, with the stage set, let’s bring in the players — the libraries and modules needed to create our YouTube GPT Creator:

```python
import streamlit as st
import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
```

In this block, we import the necessary components:

- `streamlit`: for our web app interface.
- `os`: to interact with the operating system, especially for setting environment variables like the OpenAI API key.
- The `langchain` modules provide the structure and functionality needed to work with language models, handle user inputs, manage conversational memory, and integrate external knowledge sources like Wikipedia.

## API Key Configuration

To communicate with OpenAI's services, we need to authenticate our requests by setting the API key:

```python
os.environ['OPENAI_API_KEY'] = "your_openai_key_here"
```

This line instructs our app to use the specified API key for all interactions with OpenAI, ensuring secure and authorized access to its AI models.

## Building the Front End

Here's where we start shaping the user interface of our app:

```python
st.title('🦜🔗 YouTube GPT Creator')
prompt = st.text_input('Plug in your prompt here')
```

- `st.title` sets the title of our web application.
- `st.text_input` creates a text input field for users to enter their prompts, which serves as the seed for generating content.

## Defining the Prompt Templates

These templates are the blueprints for the content we want to generate:

```python
title_template = PromptTemplate(
    input_variables=['topic'],
    template='write me a youtube video title about {topic}'
)

script_template = PromptTemplate(
    input_variables=['title', 'wikipedia_research'],
    template='write me a youtube video script based on this title TITLE: {title} while leveraging this wikipedia research:{wikipedia_research}'
)
```

In the `title_template`, we define a structure for generating video titles based on a given topic. In the `script_template`, we expand on that title, incorporating Wikipedia research to create a comprehensive script.

## Managing Memory and Learning

To improve the app's understanding and contextual handling, we introduce memory elements:

```python
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')
```

These lines establish memory buffers for titles and scripts, allowing our application to reference previous interactions and learn from them, enhancing the relevance and quality of generated content.

## Configuring the Language Model

Setting up the language model involves specifying how it should process our prompts:

```python
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)
```

Here, `LLMChain` is used to create a pipeline for processing user prompts, generating titles and scripts, and managing memory. The `temperature` parameter controls the creativity of the responses.

## Integrating Wikipedia for Research

To enrich the content, we integrate Wikipedia data:

```python
wiki = WikipediaAPIWrapper()
```

This line initializes the Wikipedia API wrapper, allowing our app to fetch and incorporate Wikipedia content into the generated scripts.

## Displaying the Results and History

We then lay out the logic for displaying generated content and interaction history:

```python
if prompt:
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title=title, wikipedia_research=wiki_research)

    st.write(title)
    st.write(script)

    with st.expander('Title History'):
        st.info(title_memory.buffer)

    with st.expander('Script History'):
        st.info(script_memory.buffer)

    with st.expander('Wikipedia Research'):
        st.info(wiki_research)
```

This block checks if a prompt is entered and then uses the title and script chains to generate content, fetching related Wikipedia data. The results and historical data are displayed in expandable sections on the web interface.

## Launching the App

To make the app accessible via a web browser, we execute the following commands:

```bash
!streamlit run /content/app.py &>/content/logs.txt & curl ipv4.icanhazip.com
```

This runs the Streamlit app and logs its output while also displaying the public IP address of the Colab environment.

Then, to expose our app to the internet:

```bash
!npx localtunnel --port 8501
```

Localtunnel creates a public URL for accessing the app running on Colab's local server, bridging the gap between the development environment and potential users.
