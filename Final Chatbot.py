import os
from together import Together
import tiktoken
import json
from datetime import datetime
import streamlit as st


DEFAULT_API_KEY = "Your API key here" #paste your API key here
DEFAULT_BASE_URL = "https://api.together.xyz/v1"
DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOKEN_BUDGET = 4096

class ConversationManager:
    def __init__(self, api_key=None, base_url=None, model=None, history_file=None, temperature=None, max_tokens=None, token_budget=None):
        if not api_key:
            api_key = DEFAULT_API_KEY
        if not base_url:
            base_url = DEFAULT_BASE_URL
            
        self.client = Together(
            api_key=api_key,
            base_url=base_url
        )
        if history_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.history_file = f"conversation_history_{timestamp}.json"
        else:
            self.history_file = history_file

        self.model = model if model else DEFAULT_MODEL
        self.temperature = temperature if temperature else DEFAULT_TEMPERATURE
        self.max_tokens = max_tokens if max_tokens else DEFAULT_MAX_TOKENS
        self.token_budget = token_budget if token_budget else DEFAULT_TOKEN_BUDGET

        self.system_messages = {
            "sassy_assistant": "A sassy assistant who is fed up with answering questions.",
            "angry_assistant": "An angry assistant that likes yelling in all caps.",
            "thoughtful_assistant": "A thoughtful assistant, always ready to dig deeper. This assistant asks clarifying questions to ensure understanding and approaches problems with a step-by-step methodology.",
            "custom": "A placeholder for your custom system message."
        }
        self.system_message = self.system_messages["sassy_assistant"]  # Default persona

        self.load_conversation_history()

    def count_tokens(self, text):
        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")        
            

        tokens = encoding.encode(text)
        return len(tokens)

    def total_tokens_used(self):
        return sum(self.count_tokens(message['content']) for message in self.conversation_history)
    
    def enforce_token_budget(self):
        while self.total_tokens_used() > self.token_budget:
            if len(self.conversation_history) <= 1:
                break
            self.conversation_history.pop(1)

    def set_persona(self, persona):
        if persona in self.system_messages:
            self.system_message = self.system_messages[persona]
            self.update_system_message_in_history()
        else:
            raise ValueError(f"Unknown persona: {persona}. Available personas are: {list(self.system_messages.keys())}")

    def set_custom_system_message(self, custom_message):
        if not custom_message:
            raise ValueError("Custom message cannot be empty.")
        self.system_messages['custom'] = custom_message
        self.set_persona('custom')

    def update_system_message_in_history(self):
        if self.conversation_history and self.conversation_history[0]["role"] == "system":
            self.conversation_history[0]["content"] = self.system_message
        else:
            self.conversation_history.insert(0, {"role": "system", "content": self.system_message})

    def chat_completion(self, prompt, temperature=None, max_tokens=None):
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        self.conversation_history.append({"role": "user", "content": prompt})

        self.enforce_token_budget()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation_history,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        ai_response = response.choices[0].message.content
        self.conversation_history.append({"role": "assistant", "content": ai_response})
        
        self.save_conversation_history()

        return ai_response
    
    def load_conversation_history(self):
        try:
            with open(self.history_file, "r") as file:
                self.conversation_history = json.load(file)
        except FileNotFoundError:
            self.conversation_history = [{"role": "system", "content": self.system_message}]
        except json.JSONDecodeError:
            print("Error reading the conversation history file. Starting with an empty history.")
            self.conversation_history = [{"role": "system", "content": self.system_message}]

    def save_conversation_history(self):
        try:
            with open(self.history_file, "w") as file:
                json.dump(self.conversation_history, file, indent=4)
        except IOError as e:
            print(f"An I/O error occured while saving the conversation history:{e}")
        except Exception as e:
            print(f"An unexpected error occured while saving the conversation history:{e}")
            
    def reset_conversation_history(self):
        self.conversation_history = [{"role": "system", "content": self.system_message}]
        try:
            self.save_conversation_history()  # Attempt to save the reset history to the file
        except Exception as e:
            print(f"An unexpected error occurred while resetting the conversation history:{e}")

st.title("Sassy Chatbot :face_with_rolling_eyes:")

st.sidebar.header("Options")

if 'chat_manager' not in st.session_state:
    st.session_state['chat_manager'] = ConversationManager()
chat_manager = st.session_state['chat_manager']

max_tokens_per_message = st.sidebar.slider("Max Tokens Per Message", min_value=10, max_value=500, value=50)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
system_message = st.sidebar.selectbox("System message", ['Sassy', 'Angry', 'Thoughtful', 'Custom'])

if system_message == 'Sassy':
    chat_manager.set_persona('sassy_assistant')
elif system_message == 'Angry':
    chat_manager.set_persona('angry_assistant')
elif system_message == 'Thoughtful':
    chat_manager.set_persona('thoughtful_assistant')
# Open text area for custom system message if "Custom" is selected
elif system_message == 'Custom':
    custom_message = st.sidebar.text_area("Custom system message")
    if st.sidebar.button("Set custom system message"):
        chat_manager.set_custom_system_message(custom_message)
if st.sidebar.button("Reset conversation history", on_click=chat_manager.reset_conversation_history):
    st.session_state['conversation_history'] = chat_manager.conversation_history

if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = chat_manager.conversation_history

conversation_history = st.session_state['conversation_history']

# Chat input from the user
user_input = st.chat_input("Write a message")

# Call the chat manager to get a response from the AI. Uses settings from the sidebar.
if user_input:
    response = chat_manager.chat_completion(user_input, temperature=temperature, max_tokens=max_tokens_per_message)

# Display the conversation history
for message in conversation_history:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])