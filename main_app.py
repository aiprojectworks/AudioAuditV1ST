#https://medium.com/@leopoldwieser1/how-to-build-a-speech-transcription-app-with-streamlit-and-whisper-and-then-dockerize-it-88418fd4a90

import csv
from functools import partial
from groq import Groq
from io import BytesIO
import json
from operator import is_not
import glob
import pandas as pd
import re
import tkinter as tk
from tkinter import filedialog
import requests
import streamlit as st
# import sys
#WhisperX import
# import whisperx
# import gc 
import torch
import os
from datetime import datetime
#LLM import
# from langchain.embeddings import LlamaCppEmbeddings
# from langchain_community.llms import LlamaCpp
# from langchain_core.prompts import PromptTemplate
# from langchain.callbacks.manager import CallbackManager
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.tree import Tree
# from openpyxl import Workbook
# from openpyxl.styles import PatternFill
# from streamlit.components.v1 import html
# import openai
from openai import OpenAI
from mutagen.mp3 import MP3
from mutagen.id3 import ID3, ID3NoHeaderError
from pydub import AudioSegment
import zipfile
import io
from database import Session, User, seed_users
# from streamlit_cookies_manager import EncryptedCookieManager
# from streamlit_cookies_controller import CookieController
# from streamlit.web.server.websocket_headers import _get_websocket_headers 
# from urllib.parse import unquote
# import extra_streamlit_components as stx
# from streamlit.web.server.websocket_headers import _get_websocket_headers
import threading
import time
from sqlalchemy.exc import IntegrityError
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from streamlit.runtime import get_instance
from typing import Tuple
# import streamlit_js_eval
# from streamlit_js_eval import streamlit_js_eval
# from pydub.playback import play
from llmlingua import PromptCompressor
from transformers import AutoTokenizer

# import assemblyai as aai
# import httpx
# import threading
# from deepgram import (
# DeepgramClient,
# PrerecordedOptions,
# FileSource,
# DeepgramClientOptions,
# LiveTranscriptionEvents,
# LiveOptions,
# )


class KillableThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


st.set_page_config(page_title="IPPFA Trancribe & Audit",
                            page_icon=":books:")
groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
# deepgram = DeepgramClient(st.secrets["DEEPGRAM_API_KEY"])
client = OpenAI(api_key=st.secrets["API_KEY"])

try:
    llm_lingua = PromptCompressor(
        model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",  # Using base model
        model_config={"revision": "main"},
        use_llmlingua2=True,
        device_map="cpu"
    )
except Exception as e:
    print(f"Error initializing LLMLingua: {e}")


print("torch version:", torch.__version__)

def is_admin(username: str) -> bool:

    """Check if user has admin role"""
    try:
        session = Session()
        user = session.query(User).filter_by(username=username).first()
        return user is not None and user.role == "admin"
    except Exception as e:
        print(f"Database error checking admin status: {e}")
        return False
    finally:
        session.close()

def validate_password(password: str) -> Tuple[bool, str]:
    """
    Validate the password against specific requirements.
    Requirements:
    - Minimum 8 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one digit
    - At least one special character
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not any(char.isupper() for char in password):
        return False, "Password must contain at least one uppercase letter."
    if not any(char.islower() for char in password):
        return False, "Password must contain at least one lowercase letter."
    if not any(char.isdigit() for char in password):
        return False, "Password must contain at least one digit."
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least one special character."
    return True, ""

def add_user(username: str, password: str, role: str = "user") -> Tuple[bool, str]:
    """Add a new user to the database"""
    try:
        # Validate password
        session = Session()
        new_user = User(username=username, password=password, role=role)
        is_valid, message = validate_password(password)
        if not is_valid:
            return False, message
        session.add(new_user)
        session.commit()
        return True, "User added successfully"
    except IntegrityError:
        session.rollback()
        return False, "Username already exists"
    except Exception as e:
        session.rollback()
        return False, f"Error adding user: {e}"
    finally:
        session.close()

def delete_user(username: str) -> Tuple[bool, str]:
    """Delete a user from the database"""
    try:
        session = Session()
        user = session.query(User).filter_by(username=username).first()
        if user:
            if user.role == "admin":
                # Count number of admin users
                admin_count = session.query(User).filter_by(role="admin").count()
                if admin_count <= 1:
                    return False, "Cannot delete the last admin user"
                if st.session_state.get("username") == username:
                    return False, "Cannot delete the currently logged in user"
            cleanup_on_logout(username, refresh=False)
            session.delete(user)
            session.commit()
            return True, "User deleted successfully"
        return False, "User not found"
    except Exception as e:
        session.rollback()
        return False, f"Error deleting user: {e}"
    finally:
        session.close()

def get_all_users() -> list:
    """Get all users from the database"""
    try:
        session = Session()
        users = session.query(User).all()
        return [{"username": user.username, "role": user.role} for user in users]
    except Exception as e:
        print(f"Error fetching users: {e}")
        return []
    finally:
        session.close()


def change_password(username: str, new_password: str) -> Tuple[bool, str]:
    """Change a user's password"""
    try:
        session = Session()
        user = session.query(User).filter_by(username=username).first()
        if user:
            user.password = new_password
            session.commit()
            return True, "Password changed successfully"
        return False, "User not found"
    except Exception as e:
        session.rollback()
        return False, f"Error changing password: {e}"
    finally:
        session.close()

def change_role(username: str, new_role: str) -> Tuple[bool, str]:
    """Change a user's role"""
    try:
        session = Session()
        user = session.query(User).filter_by(username=username).first()
        if user:
            # Prevent removing the last admin
            if user.role == "admin" and new_role != "admin":
                admin_count = session.query(User).filter_by(role="admin").count()
                if admin_count <= 1:
                    return False, "Cannot remove the last admin user"
            user.role = new_role
            session.commit()
            return True, "Role changed successfully"
        return False, "User not found"
    except Exception as e:
        session.rollback()
        return False, f"Error changing role: {e}"
    finally:
        session.close()

def admin_interface():
    """Render the admin interface in Streamlit"""
    st.title("Admin Panel")
    
    # Add New User Section
    st.subheader("Add New User")
    col1, col2, col3 = st.columns(3)
    with col1:
        new_username = st.text_input("Username", key="new_username")
    with col2:
        new_password = st.text_input("Password", type="password", key="new_password")
    with col3:
        new_role = st.selectbox("Role", ["user", "admin"], key="new_role")
    
    if st.button("Add User"):
        if new_username and new_password:
            success, message = add_user(new_username, new_password, new_role)
            if success:
                st.success(message)
                create_log_entry(f"Admin Action: Added new user - {new_username}")
            else:
                st.error(message)
                create_log_entry(f"Admin Action Failed: Add user - {new_username} - {message}")
        else:
            st.warning("Please fill in all fields")

    # Manage Existing Users Section
    st.subheader("Manage Users")
    users = get_all_users()
    
    if users:
        for user in users:
            with st.expander(f"User: {user['username']} ({user['role']})"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    new_pass = st.text_input("New Password", type="password", key=f"pass_{user['username']}")
                    if st.button("Change Password", key=f"btn_pass_{user['username']}"):
                        if new_pass:
                            success, message = change_password(user['username'], new_pass)
                            if success:
                                st.success(message)
                                create_log_entry(f"Admin Action: Changed password for user - {user['username']}")
                            else:
                                st.error(message)
                
                with col2:
                    new_role = st.selectbox("New Role", ["user", "admin"], 
                                          index=0 if user['role']=="user" else 1,
                                          key=f"role_{user['username']}")
                    if st.button("Change Role", key=f"btn_role_{user['username']}"):
                        success, message = change_role(user['username'], new_role)
                        if success:
                            st.success(message)
                            create_log_entry(f"Admin Action: Changed role for user - {user['username']} to {new_role}")
                        else:
                            st.error(message)
                
                with col3:
                    if st.button("Delete User", key=f"btn_del_{user['username']}"):
                        success, message = delete_user(user['username'])
                        if success:
                            st.success(message)
                            create_log_entry(f"Admin Action: Deleted user - {user['username']}")
                            st.rerun()
                        else:
                            st.error(message)
    else:
        st.info("No users found")


def cleanup_on_logout(username = st.session_state.get("username"), refresh = True):
    """Handle cleanup when user logs out"""
    # username = st.session_state.get("username")
    if username:
        # Clear files
        directory = username
        if os.path.exists(directory):
            delete_mp3_files(directory)
            directory = "./" + directory
            os.rmdir(directory)

    if refresh:
        st.session_state.clear()   

def user_exists(username):
    """Check if user exists in database"""
    try:
        session = Session()
        user = session.query(User).filter_by(username=username).first()
        return user is not None
    except Exception as e:
        print(f"Database error checking user existence: {e}")
        return False
    finally:
        session.close()


def heartbeat(username):
    # print(f"Heartbeat for user: {username}")
    if not user_exists(username):
        print(f"User {username} not found. Stopping heartbeat.")
        # Use the instance to call stop_beating
        heartbeat_manager.active_threads[username].stop()
        cleanup_on_logout(username, refresh=False)  
        st.rerun()
        st.session_state["user_deleted"] = True

    

class HeartbeatManager:
    def __init__(self):
        self.active_threads = {}

    def start_beating(self, username):
        """Start heartbeat thread"""
        def heartbeat_loop():
            while not self.active_threads[username].stopped():
                # Get current session context
                ctx = get_script_run_ctx()
                runtime = get_instance()

                if ctx and runtime.is_active_session(session_id=ctx.session_id):
                    # Session is still active
                    heartbeat(username)
                    time.sleep(2)  # Wait for 2 seconds
                else:
                    # Session ended - clean up
                    print(f"Session ended for user: {username}")
                    cleanup_on_logout(username, refresh=False)
                    return

        # Create new killable thread
        thread = KillableThread(target=heartbeat_loop)
        
        # Add Streamlit context to thread
        add_script_run_ctx(thread)
        
        # Store thread reference
        self.active_threads[username] = thread
        
        # Start thread
        thread.start()

    def stop_beating(self, username):
        """Stop heartbeat thread for specific user"""
        if username in self.active_threads:
            self.active_threads[username].stop()
            # Remove join() if called from within the thread
            if threading.current_thread() != self.active_threads[username]:
                self.active_threads[username].join()  # Only join if called from a different thread
            del self.active_threads[username]

    def stop_all(self):
        for username in list(self.active_threads.keys()):
            self.stop_beating(username)

#!important
heartbeat_manager = HeartbeatManager()


def start_beating(username):
    """Start heartbeat thread"""
    thread = threading.Timer(interval=2, function=start_beating, args=(username,))
    
    # Add Streamlit context to thread
    add_script_run_ctx(thread)
    
    # Get current session context
    ctx = get_script_run_ctx()
    runtime = get_instance()
    
    
    if ctx and runtime.is_active_session(session_id=ctx.session_id):
        # Session is still active
        thread.start()
        #!no logs
        heartbeat(username)
    else:
        # Session ended - clean up
        print(f"Session ended for user: {username}")
        cleanup_on_logout(username, refresh=False)
        return

# Authenticate function
def authenticate(username, password):
    try:
        session = Session()
        user = session.query(User).filter_by(username=username, password=password).first()
        session.close()
        return user
    except Exception as e:
        st.error(f"Database error: {e}")
        return None

# Login Page
def login_page():
    """Display login form and authenticate users."""
    st.title("Login Portal")
    

    # Group inputs and button in a form for "Enter" support
    with st.form("login_form"):
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        # Visible Login button (inside the form for "Enter" key support)
        login_button = st.form_submit_button("Login")

    # Handle login logic when either the "Login" button is clicked or "Enter" is pressed
    if login_button:
        user = authenticate(username, password)  # Call authentication function
        if user:
            st.session_state["logged_in"] = True
            st.session_state["username"] = user.username
            st.rerun()
        else:
            st.error("Invalid username or password!")

def save_audio_file(audio_bytes, name):
    try:
        if name.lower().endswith(".wav") or name.lower().endswith(".mp3"):
            username = st.session_state["username"]

            user_folder = os.path.join(".", username)
            os.makedirs(user_folder, exist_ok=True)

            name = os.path.basename(name)
            file_name = os.path.join(user_folder, f"audio_{name}")


            with open(file_name, "wb") as f:
                f.write(audio_bytes)
            print(f"File saved successfully: {file_name}")
            return file_name  # Ensure you return the file name
    except Exception as e:
        print(f"Failed to save file: {e}")
        return None  # Explicitly return None on failure

def delete_mp3_files(directory):
    # Construct the search pattern for MP3 files
    mp3_files = glob.glob(os.path.join(directory, "*.mp3"))
    
    for mp3_file in mp3_files:
        try:
            os.remove(mp3_file)
            # print(f"Deleted: {mp3_file}")
        except FileNotFoundError:
            print(f"{mp3_file} does not exist.")
        except Exception as e:
            print(f"Error deleting file {mp3_file}: {e}")

def convert_audio_to_wav(audio_file):
    audio = AudioSegment.from_file(audio_file)
    wav_file = audio_file.name.split(".")[0] + ".wav"
    audio.export(wav_file, format="wav")
    return wav_file

def make_fetch_request(url, headers, method='GET', data=None):
    if method == 'POST':
        response = requests.post(url, headers=headers, json=data)
    else:
        response = requests.get(url, headers=headers)
    return response.json()

def speech_to_text_groq(audio_file):
    #print into dialog format
    dialog =""

    #Function to run Groq with user prompt
    #different model from Groq
    # Groq_model="llama3-8b-8192"
    # Groq_model="llama3-70b-8192"
    # Groq_model="mixtral-8x7b-32768"
    # Groq_model="gemma2-9b-it"

    # Transcribe the audio
    audio_model="whisper-large-v3-turbo"

    with open(audio_file, "rb") as file:
        # Create a transcription of the audio file
        transcription = groq_client.audio.transcriptions.create(
        file=(audio_file, file.read()), # Required audio file
        model= audio_model, # Required model to use for transcription
        prompt="Elena Pryor, Samir, Sahil, Mihir, IPP, IPPFA",
        temperature=0,
        response_format="verbose_json"
          
        )

    # Print the transcription text
    print(transcription.text)
    
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": """You are processing a cold-call conversation transcript between a telemarketer and a customer. Your task is to:

    1. Identify speakers based on these patterns:
    Telemarketer typically:
    - Asks to confirm identity ("Is this [name]?")
    - Introduces themselves and company
    - Explains services/products
    - Makes offers/proposals
    - Uses professional sales language
    
    Customer typically:
    - Responds to identity confirmation
    - Asks for clarification
    - Responds to offers
    - Makes decisions about proposals

    2. Format output as JSON:
    {
        "language_code": "original audio language (e.g., 'en', 'zh', 'ms')",
        "transcript": [
            {
                "speaker": "Telemarketer/Customer",
                "text": "exact speech content"
            }
        ]
    }

    3. Cold-call specific rules:
    - Initial identity confirmation is typically from telemarketer
    - Name repetition/clarification usually from customer
    - Service explanations always from telemarketer
    - Confusion/clarification requests typically from customer

    4. Accuracy requirements:
    - Maintain exact names as spoken
    - Preserve company names and terminology
    - Keep all hesitations and repetitions
    - Mark unclear segments with [unclear]"""},
            {"role": "user", "content": f"Process this audio transcript: {transcription.text}"}
        ],
        temperature=0,
        max_tokens=16384
    )


    output = response.choices[0].message.content
    print(output)
    dialog = output.replace("json", "").replace("```", "")
    formatted_transcript = ""
    dialog = json.loads(dialog)
    language_code = dialog["language_code"]
    print(language_code)
    for entry in dialog['transcript']:
        formatted_transcript += f"{entry['speaker']}: {entry['text']}  \n\n"
    print(formatted_transcript)

    # Joining the formatted transcript into a single string
    dialog = formatted_transcript

    
    return dialog, language_code



# def speech_to_text(audio_file):
#     dialog =""

#     # Transcribe the audio
#     transcription = client.audio.transcriptions.create(
#         model="whisper-1",
#         file=open(audio_file, "rb"),
#         prompt="Elena Pryor, Samir, Sahil, Mihir, IPP, IPPFA",
#         temperature=0

#     )
#     if not transcription or not transcription.text:
#         raise ValueError("No transcription text received from Whisper API")

#     dialog = transcription.text
#     # OPTIONAL: Uncomment the line below to print the transcription
#     # print("Transcript: ", dialog + "  \n\n")

#     response = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {"role": "system", "content": """You are analyzing a sales call transcript. Apply these EXACT rules in order:

#     1. FIRST: Identify Speech Patterns
#     Telemarketer MUST be speaker when:
#     - Using "Ms./Mrs. Hamid"
#     - Asking about schedule ("When is your off day?")
#     - Making confirmation statements ending with "right?"
#     - Explaining business/MAS
#     - Following up on customer's answers
    
#     Customer MUST be speaker when:
#     - Using "Ba" or broken English
#     - Showing resistance ("no no", "cannot")
#     - Expressing confusion
#     - Making excuses about time/schedule
#     - Using repeated words ("sorry sorry")

#     2. THEN: Check Question Types
#     Telemarketer Questions ONLY:
#     - Schedule inquiries
#     - Work-related questions
#     - Family-related questions
#     - Confirmation questions
#     - Location questions
    
#     Customer Questions ONLY:
#     - "What?"/"Pardon?"
#     - "Who are you?"
#     - Questions about documents
#     - Questions showing confusion

#     3. THEN: Check Response Patterns
#     Telemarketer Responses:
#     - Professional acknowledgments
#     - Schedule proposals
#     - Business explanations
#     - Family/work inquiries
    
#     Customer Responses:
#     - Short answers
#     - Time excuses
#     - Location excuses
#     - Confused responses

#     4. FINALLY: Verify Context
#     - Each confirmation question ("...right?") MUST be from Telemarketer
#     - Each statement of personal plans MUST be from Customer
#     - Each formal address ("Ms./Mrs. Hamid") MUST be from Telemarketer
#     - Each informal/broken English MUST be from Customer

#     Double-check that attributed speakers maintain these patterns consistently."""},
#             {"role": "user", "content": f"Process this transcript, applying rules strictly in order: {transcription.text}"}
#         ],
#         temperature=0,
#         max_tokens=16384
#     )
    
#     output = response.choices[0].message.content
#     # print(output)
#     dialog = output.replace("json", "").replace("```", "")
#     formatted_transcript = ""
#     dialog = json.loads(dialog)
#     language_code = dialog["language_code"]
#     print(language_code)
#     for entry in dialog['transcript']:
#         formatted_transcript += f"{entry['speaker']}: {entry['text']}  \n\n"
#     print(formatted_transcript)

#     # Joining the formatted transcript into a single string
#     dialog = formatted_transcript

#     return dialog, language_code


def speech_to_text(audio_file):
    dialog =""

    # Transcribe the audio
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=open(audio_file, "rb"),
        prompt="Elena Pryor, Samir, Sahil, Mihir, IPP, IPPFA",
        temperature=0

    )
    dialog = transcription.text
    # OPTIONAL: Uncomment the line below to print the transcription
    # print("Transcript: ", dialog + "  \n\n")

    total_tokens = 0  # Initialize total_tokens variable

    system_prompt = """Insert speaker labels for a telemarketer and a customer based on the dialogue as accurately as possible. Return in a JSON format together with the original language code. Translate the entire transcript accurately into English. Ensure the segmentation is logical and each line reflects a single speaker's statement. Maintain consistency in labeling (e.g., do not mix speaker roles). Preserve the original speaker intent and tone in the translation."""

    compressed_system = llm_lingua.compress_prompt(
            system_prompt,
            target_token=150,  # Adjust this value as needed
            force_tokens=["JSON", "English", "speaker", "telemarketer", "customer"],
            drop_consecutive=True,
        )

    compressed_transcript = llm_lingua.compress_prompt(
        dialog,
        rate=0.5,  # Adjust compression rate as needed
        force_tokens=["!", ".", "?", "\n"],
        drop_consecutive=True,
    )


    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
        {"role": "system", "content": compressed_system["compressed_prompt"]},
        {"role": "user", "content": f"The audio transcript is: {compressed_transcript['compressed_prompt']}"}
        ],
        temperature=0
    )

    
    
    total_tokens += response.usage.total_tokens
    print(f"Total tokens used for transcription: {total_tokens}")

    filename = os.path.basename(audio_file)
    if filename not in st.session_state.token_counts:
        st.session_state.token_counts[filename] = {}

    st.session_state.token_counts[filename]["transcription"] = total_tokens
    output = response.choices[0].message.content
    
    print("Raw output:", output)  # Let's see what we're getting
    # print(output)
    # dialog = output.replace("json", "").replace("```", "")
    try:
        # Clean up the output and parse JSON once
        cleaned_output = output.replace("json", "").replace("```", "").strip()
        print("After cleanup:", cleaned_output)
        parsed_json = json.loads(cleaned_output)

        # Extract what we need from the JSON
        language_code = parsed_json["language_code"]
        formatted_transcript = ""
        for entry in parsed_json['transcript']:
            formatted_transcript += f"{entry['speaker']}: {entry['text']}  \n\n"

        return formatted_transcript, language_code

    except json.JSONDecodeError as e:
        print("JSON Parse Error:", e)
        print("Failed to parse:", cleaned_output)
        raise




def groq_LLM_audit(dialog): #*not in use
    stage_1_prompt = """
    You are auditing a conversation between an IPP/IPPFA telemarketer and a potential customer.

    Evaluate these criteria with "Pass", "Fail", or "Not Applicable":
    1. Name introduction [Pass if telemarketer states their name]
    2. Company affiliation [Pass if mentions IPP/IPPFA/IPP Financial Advisors only]
    3. Contact source disclosure [N/A if not asked; Pass if source explained when asked]
    4. Financial services description [Pass if services are specified]
    5. Meeting/zoom scheduling [Pass if offered; include date/location if mentioned]
    6. Return claims [Pass if NO mentions of high/guaranteed returns; Fail if mentioned]
    7. Professional conduct [Pass if consistently polite/professional]

    Format: Return a JSON array of objects with:
    {
        "Criteria": "criterion being checked",
        "Reason": "specific evidence from conversation",
        "Result": "Pass/Fail/Not Applicable"
    }

    Conversation to evaluate:
    %s
    """

    stage_2_prompt = """
    Continuing the IPP/IPPFA telemarketing audit, evaluate:

    1. Service benefits [Pass if asked about customer's interest in IPPFA services]
    2. Uncertain customer response [If yes, Pass if meeting/zoom offered; N/A if customer certain]
    3. Pressure tactics [Pass if NO pressure for products/appointments; Fail if pressured]

    Format: Return a JSON array of objects with:
    {
        "Criteria": "criterion being checked",
        "Reason": "specific evidence from conversation",
        "Result": "Pass/Fail/Not Applicable"
    }

    Conversation to evaluate:
    %s
    """ % (dialog)

    chat_completion  = groq_client.chat.completions.create(
    model="llama3-groq-70b-8192-tool-use-preview",
    messages=[
        {
            "role": "system",
            "content": f"{stage_1_prompt}",
        },
        {
            "role": "user",
            "content": f"{dialog}",
        }
    ],
    temperature=0,
    max_tokens=4096,
    stream=False,
    stop=None,
    )
    stage_1_result = chat_completion.choices[0].message.content
    print(stage_1_result)
    

    stage_1_result = stage_1_result.replace("Audit Results:","")
    stage_1_result = stage_1_result.replace("### Input:","")
    stage_1_result = stage_1_result.replace("### Output:","")
    stage_1_result = stage_1_result.replace("### Response:","")
    stage_1_result = stage_1_result.replace("json","").replace("```","")
    stage_1_result = stage_1_result.strip()

    stage_1_result = json.loads(stage_1_result)

    print(stage_1_result)

    output_dict = {"Stage 1": stage_1_result}

    # for k,v in output_dict.items():
    #    person_names.append(get_person_entities(v[0]["Reason"]))

    #    if len(person_names) != 0:
            # print(person_names)
    #        v[0]["Result"] = "Pass"

    # print(output_dict)

    overall_result = "Pass"

    for i in range(len(stage_1_result)):
        if stage_1_result[i]["Result"] == "Fail":
            overall_result = "Fail"
            break  

    output_dict["Overall Result"] = overall_result

    if output_dict["Overall Result"] == "Pass":
        del output_dict["Overall Result"]

        chat_completion  = groq_client.chat.completions.create(
        model="llama3-groq-70b-8192-tool-use-preview",
        messages=[
            {
                "role": "system",
                "content": f"{stage_2_prompt}",
            },
            {
                "role": "user",
                "content": f"{dialog}",
            }
        ],
        temperature=0,
        max_tokens=4096,
        stream=False,
        stop=None,
        )
        stage_2_result = chat_completion.choices[0].message.content
        
        stage_2_result = stage_2_result.replace("Audit Results:","")
        stage_2_result = stage_2_result.replace("### Input:","")
        stage_2_result = stage_2_result.replace("### Output:","")
        stage_2_result = stage_2_result.replace("### Response:","")
        stage_2_result = stage_2_result.replace("json","").replace("```","")
        stage_2_result = stage_2_result.strip()

        # print(stage_2_result)

        stage_2_result = json.loads(stage_2_result)
        
        output_dict["Stage 2"] = stage_2_result

        overall_result = "Pass"

        for i in range(len(stage_2_result)):
            if stage_2_result[i]["Result"] == "Fail":
                overall_result = "Fail"
                break  
                
        output_dict["Overall Result"] = overall_result

    # print(output_dict)
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return output_dict
        

    



def LLM_audit(dialog):
    total_tokens = 0  # Initialize total_tokens variable
    stage_1_prompt = """
    You are an auditor for IPP or IPPFA. Return ONLY a valid JSON object.

    Required format:
    [
        {
            "Criteria": "<criterion being evaluated>",
            "Reason": "<specific evidence from conversation>",
            "Result": "Pass/Fail/Not Applicable"
        }
    ]

    You are tasked with auditing a conversation between a telemarketer from IPP or IPPFA and a customer. 
    The audit evaluates whether the telemarketer adhered to specific criteria during the dialogue.

    ### Instruction:
        - Review the transcript and evaluate the telemarketer's compliance with the criteria below.
        - Provide a detailed assessment for each criterion, quoting relevant parts of the conversation and assigning a result status.
        - Ensure all evaluations are based strictly on the content of the conversation. 
        - Only mark a criterion as "Pass" if there is clear evidence to support it.
        - Rename non-Singapore locations in the conversation to a similar location in Singapore.

        Audit Criteria:
            1. Did the telemarketer state their name (usually followed by "calling from". Pass if they have just said their name only.)?
            2. Did they specify they are calling from one of these: ['IPP', 'IPPFA', 'IPP Financial Advisors'] (without mentioning other insurers)?
            3. If asked, did they disclose who provided the customer's contact details? (NA if not asked.)
            4. Did they specify the financial services offered?
            5. Did they propose a meeting or Zoom session? (Provide the date and location if have.)
            6. Did they avoid claiming high/guaranteed returns or capital guarantee? (Fail if mentioned.)
            7. Were they polite and professional?

        ** End of Criteria**

    ### Response:
        Generate JSON objects for each criteria in a list that must include the following keys:
        - "Criteria": State the criterion being evaluated.
        - "Reason": Provide specific reasons based on the conversation.
        - "Result": Indicate whether the criterion was met with "Pass", "Fail", or "Not Applicable".

        For Example:
            [
                {
                    "Criteria": "Did the telemarketer asked about the age of the customer",
                    "Reason": "The telemarketer asked how old the customer was.",
                    "Result": "Pass"
                }
            ]
    IMPORTANT: 
    - Return ONLY the JSON array
    - No additional text before or after
    - No explanations or summaries
    - Ensure JSON is properly formatted
    ### Input:
        %s
    """ % (dialog)

    compressed_stage1 = llm_lingua.compress_prompt(
        stage_1_prompt,
        target_token=500,  # Adjust as needed
        force_tokens=["Pass", "Fail", "Not Applicable", "IPP", "IPPFA", "JSON"],
        drop_consecutive=True,
    )

    # Compress the dialog input
    compressed_dialog = llm_lingua.compress_prompt(
        dialog,
        rate=0.5,  # Adjust compression rate as needed
        force_tokens=["!", ".", "?", "\n"],
        drop_consecutive=True,
    )
    stage_2_prompt = """
    You are an auditor for IPP or IPPFA. Return ONLY  a valid JSON obect. 


    You are tasked with auditing a conversation between a telemarketer from IPP or IPPFA and a customer. 
    The audit evaluates whether the telemarketer adhered to specific criteria during the dialogue.

    ### Instruction:
        - Review the conversation transcript and evaluate compliance with the criteria below.
        - Provide detailed assessments for each criterion, quoting evidence from the conversation and assigning a result status.
        - Ensure all evaluations are based strictly on the content of the conversation. 
        - Only mark "Pass" if clear evidence supports it. Exclude words in brackets from the criteria when responding.

        Audit Criteria:
            1. Did the telemarketer ask if the customer is interested in IPPFA's services?
            2. If the customer showed uncertainty, did the telemarketer suggest a meeting or Zoom session with a consultant?
            3. Did the telemarketer avoid pressuring the customer (for product introduction or appointment setting)? (Fail if pressure was applied.)

        ** End of Criteria**

    ### Response:
        Generate JSON objects for each criteria in a list that must include the following keys:
        - "Criteria": State the criterion being evaluated.
        - "Reason": Provide specific reasons based on the conversation.
        - "Result": Indicate whether the criterion was met with "Pass", "Fail", or "Not Applicable".

        For Example (Required JSON format):
            [
                {
                    "Criteria": "Did the telemarketer asked about the age of the customer",
                    "Reason": "The telemarketer asked how old the customer was.",
                    "Result": "Pass"
                }
            ]
    IMPORTANT: 
    - Return ONLY the JSON object
    - No additional text before or after the JSON
    - No explanations or summaries
    - Ensure the JSON is properly formatted

    ### Input:
        %s
    """ % (dialog)


    # Set up the model and prompt
    # model_engine = "text-davinci-003"
    model_engine ="gpt-4o-mini"

    messages=[{'role':'user', 'content':f"{stage_1_prompt} {compressed_dialog['compressed_prompt']}"}]


    completion = client.chat.completions.create(
    model=model_engine,
    messages=messages,
    temperature=0,)

    # print(completion)
    total_tokens += completion.usage.total_tokens
    print(f"Total tokens used for audit: {total_tokens}")

    # extracting useful part of response
    stage_1_result = completion.choices[0].message.content
    stage_1_result = stage_1_result.replace("Audit Results:","")
    stage_1_result = stage_1_result.replace("### Input:","")
    stage_1_result = stage_1_result.replace("### Output:","")
    stage_1_result = stage_1_result.replace("### Response:","")
    stage_1_result = stage_1_result.replace("json","").replace("```","")
    stage_1_result = stage_1_result.strip()

    print(stage_1_result)

    def format_json_with_line_break(json_string):
        # Step 1: Add missing commas after "Criteria" and "Reason" key-value pairs
        corrected_json = re.sub(r'("Criteria":\s*".+?")(\s*")', r'\1,\2', json_string)
        corrected_json = re.sub(r'("Reason":\s*".+?")(\s*")', r'\1,\2', corrected_json)

        # Ensure there is a newline after the comma for "Criteria"
        corrected_json = re.sub(r'("Criteria":\s*".+?"),(\s*")', r'\1,\n\2', corrected_json)
        
        # Ensure there is a newline after the comma for "Reason"
        corrected_json = re.sub(r'("Reason":\s*".+?"),(\s*")', r'\1,\n\2', corrected_json)

        return corrected_json

    def get_person_entities(text):
        # Tokenize the text
        tokens = word_tokenize(text)
        
        # Apply part-of-speech tagging
        pos_tags = pos_tag(tokens)
        
        # Perform named entity recognition (NER)
        named_entities = ne_chunk(pos_tags)
        
        # Extract PERSON entities
        person_entities = []
        for chunk in named_entities:
            if isinstance(chunk, Tree) and chunk.label() == 'PERSON' or isinstance(chunk, Tree) and chunk.label() == 'FACILITY' or isinstance(chunk, Tree) and chunk.label() == 'GPE':
                person_name = ' '.join([token for token, pos in chunk.leaves()])
                person_entities.append(person_name)
        return person_entities

    # person_names = []


    stage_1_result = format_json_with_line_break(stage_1_result)
    stage_1_result = json.loads(stage_1_result)
    if "criteria" in stage_1_result:
        stage_1_result = stage_1_result["criteria"]

    output_dict = {"Stage 1": stage_1_result}

    # for k,v in output_dict.items():
    #    person_names.append(get_person_entities(v[0]["Reason"]))

    #    if len(person_names) != 0:
            # print(person_names)
    #        v[0]["Result"] = "Pass"

    # print(output_dict)

    overall_result = "Pass"

    for i in range(len(stage_1_result)):
        if stage_1_result[i]["Result"] == "Fail":
            overall_result = "Fail"
            break

    output_dict["Overall Result"] = overall_result

    if output_dict["Overall Result"] == "Pass":
        del output_dict["Overall Result"]

        compressed_stage2 = llm_lingua.compress_prompt(
                    stage_2_prompt,
                    target_token=500,
                    force_tokens=["Pass", "Fail", "Not Applicable", "IPP", "IPPFA", "JSON"],
                    drop_consecutive=True,
                )
        
        # messages=[{'role':'user', 'content':f"{compressed_stage2['compressed_prompt']} {compressed_dialog['compressed_prompt']}"}
        messages=[{'role':'user', 'content':f"{stage_2_prompt} {compressed_dialog['compressed_prompt']}"}]

        model_engine ="gpt-4o-mini"

        completion = client.chat.completions.create(
        model=model_engine,
        messages=messages,
        temperature=0,)

        
        # print(completion)

        # extracting useful part of response
        stage_2_result = completion.choices[0].message.content
        
        stage_2_result = stage_2_result.replace("Audit Results:","")
        stage_2_result = stage_2_result.replace("### Input:","")
        stage_2_result = stage_2_result.replace("### Output:","")
        stage_2_result = stage_2_result.replace("### Response:","")
        stage_2_result = stage_2_result.replace("json","").replace("```","")
        stage_2_result = stage_2_result.strip()

        print(stage_2_result)

        stage_2_result = format_json_with_line_break(stage_2_result)
        stage_2_result = json.loads(stage_2_result)
        
        output_dict["Stage 2"] = stage_2_result

        overall_result = "Pass"

        for i in range(len(stage_2_result)):
            if stage_2_result[i]["Result"] == "Fail":
                overall_result = "Fail"
                break  
                
        output_dict["Overall Result"] = overall_result

    print(output_dict)
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    current_file = list(st.session_state.token_counts.keys())[-1]
    st.session_state.token_counts[current_file]["audit"] = total_tokens

    return output_dict



def select_folder():
   root = tk.Tk()
   root.wm_attributes('-topmost', 1)
   root.withdraw()
   folder_path = filedialog.askdirectory(parent=root)
    
   root.destroy()
   return folder_path

def create_log_entry(event_description, log_file='logfile.txt', csv_file='logfile.csv'):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Log to text file
    with open(log_file, mode='a') as file:
        file.write(f"{timestamp} - {event_description}\n")
    
    # Log to CSV file
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write header if the file is new
        if file.tell() == 0:
            writer.writerow(["timestamp", "event_description"])
        writer.writerow([timestamp, event_description])

@st.fragment
def handle_download_json(count, data, file_name, mime, log_message):
    st.download_button(
        label=f"Download {file_name.split('.')[-1].upper()}", 
        data=data,
        file_name=file_name,
        mime=mime,
        on_click=create_log_entry,
        args=(log_message,),
        key=f"download_json_{count}"
    )

@st.fragment
def handle_download_csv(count, data, file_name, mime, log_message):
    st.download_button(
        label=f"Download {file_name.split('.')[-1].upper()}", 
        data=data,
        file_name=file_name,
        mime=mime,
        on_click=create_log_entry,
        args=(log_message,),
        key=f"download_csv_{count}"
    )

@st.fragment
def handle_download_log_file(data, file_name, mime, log_message):
    st.download_button(
        label="Download Logs", 
        data=data,
        file_name=file_name,
        mime=mime,
        on_click=create_log_entry,
        args=(log_message,),
    )

@st.fragment
def handle_download_text(count, data, file_name, mime, log_message):
    st.download_button(
        label="Download Transcript", 
        data=data,
        file_name=file_name,
        mime=mime,
        on_click=create_log_entry,
        args=(log_message,),
        key=f"download_text_{count}"
    )

@st.fragment
def zip_download(count, data, file_name, mime, log_message):
    st.download_button(
        label="Download All Files as ZIP",
        data=data,
        file_name=file_name,
        mime=mime,
        on_click=create_log_entry,
        args=(log_message,),
        key=f"download_zip_{count}"
    )

@st.fragment
def combined_audit_result_download(data, file_name, mime, log_message):
    st.download_button(
        label="Download Combined Audit Results As ZIP",
        data=data,
        file_name=file_name,
        mime=mime,
        on_click=create_log_entry,
        args=(log_message,),
    )


@st.fragment
def handle_combined_audit_result_download(data_text, data_csv, file_name_prefix):
    # Create an in-memory buffer for the ZIP file
    buffer = io.BytesIO()

    # Convert CSV data to a pandas DataFrame
    df = pd.read_csv(io.StringIO(data_csv))

    # Create an in-memory buffer for the Excel file (XLSX)
    xlsx_buffer = io.BytesIO()

    # Write DataFrame to XLSX format and add hyperlinks to the filenames
    with pd.ExcelWriter(xlsx_buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Results')

        # Access the xlsxwriter workbook and worksheet objects
        # workbook = writer.book
        worksheet = writer.sheets['Results']

        # Add hyperlinks to text files based on the filenames in the DataFrame
        for index, row in df.iterrows():
            filename = row['Filename']  # Assuming 'filename' column exists
            # print(filename)
            if pd.notna(filename):  # Check if filename is not NaN (valid string)
                # Replace file extensions and create the hyperlink
                hyperlink = f"./{filename.replace('.mp3', '.txt').replace('.wav', '.txt')}"
                # Add the hyperlink to the 'filename' column in the Excel file (adjust the column index)
                worksheet.write_url(f"F{index + 2}", hyperlink, string=filename)

    # Move the pointer to the beginning of the xlsx_buffer to prepare it for reading
    xlsx_buffer.seek(0)

    with zipfile.ZipFile(buffer, "w") as zip_file:
        # Add text files
        for k, v in data_text.items():
            zip_file.writestr(k.replace(".mp3", ".txt").replace(".wav", ".txt"), v)

        # Add the CSV file as plain text
        # zip_file.writestr(f"{file_name_prefix}_file.csv", data_csv)

        # Add the XLSX file to the ZIP archive
        zip_file.writestr(f"{file_name_prefix}_file.xlsx", xlsx_buffer.read())

    # Move buffer pointer to the beginning of the ZIP buffer
    buffer.seek(0)

    # Return the buffer containing the ZIP archive
    return buffer

@st.fragment
def handle_combined_download(data_text, data_json, data_csv, file_name_prefix):
    # Create an in-memory buffer for the ZIP file
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w") as zip_file:
        # Add text file
        zip_file.writestr(f"{file_name_prefix}_file.txt", data_text)

        # Add JSON file
        zip_file.writestr(f"{file_name_prefix}_file.json", data_json)

        # Add CSV file
        zip_file.writestr(f"{file_name_prefix}_file.csv", data_csv)

    # Move buffer pointer to the beginning
    buffer.seek(0)

    # Return the buffer containing the ZIP archive
    return buffer


def read_log_file(log_file='logfile.txt'):
    if os.path.exists(log_file):
        with open(log_file, 'r') as file:
            log_content = file.readlines()
        # Reverse the log entries
        log_content = log_content[::-1]
        if log_content:
            log_content[0] = f"<span style='color: yellow;'>{log_content[0].strip()}</span>\n"

        return ''.join(log_content)
    else:
        return "Log file does not exist."
    
    
def log_selection():
    method = st.session_state.upload_method
    if method == "Upload Files":
        create_log_entry("Method Chosen: File Upload")
    elif method == "Upload Folder":
        create_log_entry("Method Chosen: Folder Upload")

def is_valid_mp3(file_path):
    # Check if file exists
    if not os.path.isfile(file_path):
        print("File does not exist.")
        return False

    try:
        # Check the file using mutagen
        audio = MP3(file_path, ID3=ID3)
        
        # Check for basic properties
        if audio.info.length <= 0:  # Length should be greater than 0
            print("File is invalid: Length is zero or negative.")
            return False
        
        # You can check additional metadata if needed
        print("File is valid MP3 with duration:", audio.info.length)
        
        # Optional: Check if the file can be loaded with pydub
        AudioSegment.from_file(file_path)  # This will raise an exception if the file is not valid
        
        return True
    except (ID3NoHeaderError, Exception) as e:
        print(f"Invalid MP3 file: {e}")
        # create_log_entry(f"Error: Invalid MP3 file: {e}")
        return False
    

def main():
    try:
        if st.session_state.get("user_deleted", False):
            st.error("Your account has been deleted. Please contact the administrator.")
            st.session_state.clear()
            time.sleep(1)  # Give a moment for the message to display
            st.rerun()

        # cookies = controller.get("username")
        # if cookies:
        #     st.session_state["logged_in"] = True
        #     st.session_state["username"] = cookies
        # Add this with your other session state initializations
        if 'token_counts' not in st.session_state:
            st.session_state.token_counts = {}
        if "logged_in" not in st.session_state:
            st.session_state["logged_in"] = False
            st.session_state["username"] = None

        if not st.session_state["username"]:
            # Show the login page if not logged in
            login_page()
        else:
            if 'heartbeat_started' not in st.session_state:
                st.session_state.heartbeat_started = True
                heartbeat_manager.start_beating(st.session_state["username"])

            # After successful login, show the main dashboard
            st.sidebar.success(f"Logged in as: {st.session_state['username']}")

            if st.sidebar.button("Logout"):
                cleanup_on_logout()
                st.rerun()

            if is_admin(st.session_state["username"]):
                button_text = "Back to Transcription Service" if st.session_state.get('show_admin', False) else "Open Admin Panel"
                if st.sidebar.button(button_text, key="admin_toggle_button"):
                    st.session_state.show_admin = not st.session_state.get('show_admin', False)
                    st.rerun()



            if st.session_state.get('show_admin', False):
                cleanup_on_logout(username=st.session_state["username"], refresh=False)
                admin_interface()
                # if st.button("Back to Transcription Service"):
                #     st.session_state.show_admin = False
                #     st.rerun()
            else:
                with st.sidebar:
                    st.title("AI Model Selection")
                    transcribe_option = st.radio(
                        "Choose your transcription AI model:",
                        ("OpenAI (Recommended)", "Groq")
                    )
                    audit_option = st.radio(
                        "Choose your Audit AI model:",
                        ("OpenAI (Recommended)", "Groq")
                    )
                    st.write(f"Transcription Model:\n\n{transcribe_option.replace('(Recommended)','')}\n\nAudit Model:\n\n{audit_option.replace('(Recommended)','')}")
                    st.markdown('<p style="color:red;">Groq AI Models are not recommended for file sizes of more than 1MB. Model will start to hallucinate.</p>', unsafe_allow_html=True)

                st.title("AI Transcription & Audit Service")
                st.info("Upload an audio file or select a folder to convert audio to text.", icon=":material/info:")
                
                method = st.radio("Select Upload Method:", options=["Upload Files / Folder"], horizontal=True, key='upload_method', on_change=log_selection)


            

                audio_files = []
                status = ""

                # Initialize session state to track files and change detection
                if 'uploaded_files' not in st.session_state:
                    st.session_state.uploaded_files = {}
                if 'file_change_detected' not in st.session_state:
                    st.session_state.file_change_detected = False
                if 'audio_files' not in st.session_state:
                    st.session_state.audio_files = []

                save_to_path = st.session_state.get("save_to_path", None)

            # Choose upload method

            # with st.expander("Other Options"):
            #     save_audited_transcript = st.checkbox("Save Audited Results to Folder (CSV)")
            #     if save_audited_transcript:
            #         save_to_button = st.button("Save To Folder")
            #         if save_to_button:
            #             save_to_path = select_folder()  # Assume this is a function that handles folder selection
            #             if save_to_path:
            #                 st.session_state.save_to_path = save_to_path
            #                 create_log_entry(f"Action: Save To Folder - {save_to_path}")

            #     save_to_display = st.empty()

            #     if save_audited_transcript == False:
            #         st.session_state.save_to_path = None
            #         save_to_path = None
            #         save_to_display.empty()
            #     else:
            #         save_to_display.write(f"Save To Folder: {save_to_path}")

                if method == "Upload Files / Folder":
                    # File uploader
                    uploaded_files = st.file_uploader(
                        label="Choose audio files", 
                        label_visibility="collapsed", 
                        type=["wav", "mp3"], 
                        accept_multiple_files=True
                    )
                    # # Create a set to track unique filenames
                    # unique_filenames = set()

                    # # Check for duplicates and collect unique filenames
                    # for file in uploaded_files:
                    #     if file.name in unique_filenames:
                    #         del st.session_state['uploaded_files']
                    #         st.warning("File has already been added!")
                    #     else:
                    #         unique_filenames.add(file.name)

                    if uploaded_files is not None:
                        #!removing this because its broken
                        # # Track current files
                        # current_files = {file.name: file for file in uploaded_files}

                        # # Determine files that have been added
                        # added_files = [file_name for file_name in current_files if file_name not in st.session_state.uploaded_files]
                        # files_to_remove = [
                        #     file_name for file_name in st.session_state.uploaded_files 
                        #     if file_name not in added_files
                        # ]
                        # Track current files from the file uploader
                        current_files = {file.name: file for file in uploaded_files}

                        # Determine files that have been added (newly uploaded files)
                        added_files = [file_name for file_name in current_files if file_name not in st.session_state.uploaded_files]

                        # Determine files that have been removed (no longer in the uploader)
                        files_to_remove = [
                            file_name for file_name in st.session_state.uploaded_files 
                            if file_name not in current_files
                        ]

                        #* deletes the files that are not in the current_files
                        for file_name in files_to_remove:
                            # create_log_entry(f"Action: File Removed - {file_name}")

                            # Update `st.session_state.audio_files` to exclude the removed file
                            st.session_state.audio_files = [f for f in st.session_state.audio_files if not f.endswith(file_name)]
                            st.session_state.file_change_detected = True

                            username = st.session_state["username"]

                            # Delete the corresponding file from the directory
                            audio_file_name = "audio_" + file_name
                            full_path = os.path.join(username, audio_file_name)  # Root directory or adjust to your save directory
                            if os.path.exists(full_path):
                                try:
                                    # print(full_path)
                                    os.remove(full_path)  # Delete the file 
                                    
                                    create_log_entry(f"Action: File Deleted - {full_path}")
                                    del st.session_state.uploaded_files[file_name]

                                except Exception as e:
                                    st.error(f"Error deleting file {file_name}: {e}")
                                    create_log_entry(f"Error deleting file {file_name}: {e}")                    



                        for file_name in added_files:
                            create_log_entry(f"Action: File Uploaded - {file_name}")
                            file = current_files[file_name]
                            st.session_state.uploaded_files[file_name] = current_files[file_name]

                            try:
                                audio_content = file.read()
                                saved_path = save_audio_file(audio_content, file_name)
                                if is_valid_mp3(saved_path):
                                    st.session_state.audio_files.append(saved_path)
                                    st.session_state.file_change_detected = True
                                else:
                                    st.error(f"{saved_path[2:]} is an Invalid MP3 or WAV File")
                                    create_log_entry(f"Error: {saved_path[2:]} is an Invalid MP3 or WAV File")
                            except Exception as e:
                                st.error(f"Error loading audio file: {e}")
                                create_log_entry(f"Error loading audio file: {e}")  

                        if st.session_state.uploaded_files:
                            st.subheader("Uploaded Audio Files")
                            for file_name, file_obj in st.session_state.uploaded_files.items():
                                with st.expander(f"Audio: {file_name}"):
                                    st.audio(file_obj, format="audio/mp3", start_time=0)

                        # Determine files that have been removed
                        # removed_files = [file_name for file_name in st.session_state.uploaded_files if file_name not in current_files]
                        # for file_name in removed_files:
                        #     create_log_entry(f"Action: File Removed - {file_name}")
                        #     st.session_state.audio_files = [f for f in st.session_state.audio_files if not f.endswith(file_name)]
                        #     st.session_state.file_change_detected = True

                        # Update session state with the current file list if a change was detected
                        if st.session_state.file_change_detected:
                            st.session_state.uploaded_files = current_files
                            st.session_state.file_change_detected = False

                    
                    audio_files = list(st.session_state.audio_files)
                    # st.write(audio_files)
                    # print(st.session_state.audio_files)
                    # print(type(audio_files))

                elif method == "Upload Folder":
                    # create_log_entry("Method Chosen: Folder Upload")
                    # Initialize the session state for folder_path
                    selected_folder_path = st.session_state.get("folder_path", None)

                    # Create two columns for buttons
                    col1, col2 = st.columns(spec=[2, 8])

                    with col1:
                        # Button to trigger folder selection
                        folder_select_button = st.button("Upload Folder")
                        if folder_select_button:
                            selected_folder_path = select_folder()  # Assume this is a function that handles folder selection
                            if selected_folder_path:
                                st.session_state.folder_path = selected_folder_path
                                create_log_entry(f"Action: Folder Uploaded - {selected_folder_path}")

                    with col2:
                        # Option to remove the selected folder
                        if selected_folder_path:
                            remove_folder_button = st.button("Remove Uploaded Folder")
                            if remove_folder_button:
                                username = st.session_state["username"]
                                st.session_state.folder_path = None
                                selected_folder_path = None
                                directory = username
                                delete_mp3_files(directory)
                                create_log_entry("Action: Uploaded Folder Removed")
                                success_message = "Uploaded folder has been removed."

                    # Display the success message if it exists
                    if 'success_message' in locals():
                        st.success(success_message)

                    # Display the selected folder path
                    if selected_folder_path:
                        st.write("Uploaded folder path:", selected_folder_path)

                        # Get all files in the selected folder
                        files_in_folder = os.listdir(selected_folder_path)
                        st.write("Files in the folder:")

                        # Process each file
                        for file_name in files_in_folder:
                            try:
                                file_path = os.path.join(selected_folder_path, file_name)
                                with open(file_path, 'rb') as file:
                                    audio_content = file.read()
                                    just_file_name = os.path.basename(file_name)
                                    save_path = os.path.join(just_file_name)
                                    saved_file_path = save_audio_file(audio_content, save_path)
                                    if is_valid_mp3(saved_file_path):
                                        audio_files.append(saved_file_path)
                                    else:
                                        st.error(f"{saved_file_path[2:]} is an Invalid MP3 or WAV File")
                                        create_log_entry(f"Error: {saved_file_path[2:]} is an Invalid MP3 or WAV File")


                            except Exception as e:
                                st.warning(f"Error processing file '{file_name}': {e}")
                        
                        #Filter files that are not in MP3 or WAV extensions
                        audio_files = list(filter(partial(is_not, None), audio_files))

                        st.write(audio_files)
                        # print(audio_files)

                # Submit button
                submit = st.button("Submit", use_container_width=True)

                if submit and audio_files == []:
                    create_log_entry("Service Request: Fail (No Files Uploaded)")
                    st.error("No Files Uploaded, Please Try Again!")


                elif submit:
                    combined_results = []
                    all_text = {}
                    # if not save_audited_transcript or (save_audited_transcript and save_to_path != None):
                    current = 1
                    end = len(audio_files)
                    for audio_file in audio_files:
                        print(audio_file)
                        if not os.path.isfile(audio_file):
                            st.error(f"{audio_file[2:]} Not Found, Please Try Again!")
                            continue
                        else:
                            try:
                                with st.spinner("Transcribing & Auditing In Progress..."):
                                    if transcribe_option == "OpenAI (Recommended)":   
                                        text, language_code = speech_to_text(audio_file)
                                        if audit_option == "OpenAI (Recommended)":
                                            result = LLM_audit(text)
                                            if result["Overall Result"] == "Fail":
                                                status = "<span style='color: red;'> (FAIL)</span>"
                                            else:
                                                status = "<span style='color: green;'> (PASS)</span>"
                                        elif audit_option == "Groq":
                                            result = groq_LLM_audit(text)
                                            if result["Overall Result"] == "Fail":
                                                status = "<span style='color: red;'> (FAIL)</span>"
                                            else:
                                                status = "<span style='color: green;'> (PASS)</span>"
                                    elif transcribe_option == "Groq":
                                        text, language_code = speech_to_text_groq(audio_file)
                                        if audit_option == "OpenAI (Recommended)":
                                            result = LLM_audit(text)
                                            if result["Overall Result"] == "Fail":
                                                status = "<span style='color: red;'> (FAIL)</span>"
                                            else:
                                                status = "<span style='color: green;'> (PASS)</span>"
                                        elif audit_option == "Groq":
                                            result = groq_LLM_audit(text)
                                            if result["Overall Result"] == "Fail":
                                                status = "<span style='color: red;'> (FAIL)</span>"
                                            else:
                                                status = "<span style='color: green;'> (PASS)</span>"
                            except Exception as e:
                                error_message = f"Error processing file: {audio_file} - {e}"
                                create_log_entry(error_message)
                                st.error(error_message)
                                continue
                            col1, col2 = st.columns([0.9,0.1])
                            with col1:
                                with st.expander(audio_file[2:] + f" ({language_code})"):
                                # with st.expander(audio_file[2:]):
                                    st.write()
                                    tab1, tab2, tab3 = st.tabs(["Converted Text", "Audit Result", "Download Content"])
                            with col2:
                                st.write(f"({current} / {end})")
                                st.markdown(status, unsafe_allow_html=True)
                            with tab1:
                                st.write(text)
                                all_text[audio_file[2:]] = text
                                print(all_text)
                                handle_download_text(count=audio_files.index(audio_file) ,data=text, file_name=f'{audio_file[2:].replace(".mp3", ".txt").replace(".wav", ".txt")}', mime='text/plain', log_message="Action: Text File Downloaded")
                            with tab2:
                                st.write(result)
                                # Convert result to JSON string
                                json_data = json.dumps(result, indent=4)
                                filename = audio_file[2:]
                                if isinstance(result, dict) and "Stage 1" in result:
                                    cleaned_result_stage1 = result["Stage 1"]
                                    cleaned_result_stage2 = result.get("Stage 2", [])  # Default to an empty list if Stage 2 is not present
                                    overall_result = result.get("Overall Result", "Pass")
                                else:
                                    cleaned_result_stage1 = cleaned_result_stage2 = result
                                    overall_result = "Pass"

                                # Process Stage 1 results
                                if isinstance(cleaned_result_stage1, list) and all(isinstance(item, dict) for item in cleaned_result_stage1):
                                    df_stage1 = pd.json_normalize(cleaned_result_stage1)
                                    df_stage1['Stage'] = 'Stage 1'
                                else:
                                    df_stage1 = pd.DataFrame(columns=['Stage'])  # Create an empty DataFrame for Stage 1 if no valid results

                                # Process Stage 2 results
                                if isinstance(cleaned_result_stage2, list) and all(isinstance(item, dict) for item in cleaned_result_stage2):
                                    df_stage2 = pd.json_normalize(cleaned_result_stage2)
                                    df_stage2['Stage'] = 'Stage 2'
                                else:
                                    df_stage2 = pd.DataFrame(columns=['Stage'])  # Create an empty DataFrame for Stage 2 if no valid results

                                # Concatenate Stage 1 and Stage 2 results
                                df = pd.concat([df_stage1, df_stage2], ignore_index=True)

                                # Add the Overall Result as a new column (same value for all rows)
                                df['Overall Result'] = overall_result

                                # Add the filename as a new column (same value for all rows)
                                df['Filename'] = filename

                                # Save DataFrame to CSV
                                output = BytesIO()
                                df.to_csv(output, index=False)

                                # Get CSV data
                                csv_data = output.getvalue().decode('utf-8')

                                try:
                                    col1, col2 = st.columns([2, 6])
                                    with col1:
                                        handle_download_json(count=audio_files.index(audio_file) ,data=json_data, file_name=f'{audio_file[2:]}.json', mime='application/json', log_message="Action: JSON File Downloaded")

                                    with col2:
                                        handle_download_csv(count=audio_files.index(audio_file), data=csv_data, file_name=f'{audio_file[2:]}.csv', mime='text/csv', log_message="Action: CSV File Downloaded")
                                
                                except Exception as e:
                                    create_log_entry(f"{e}")
                                    st.error(f"Error processing data: {e}")
                            with tab3:
                                zip_buffer = handle_combined_download(
                                    data_text=text,
                                    data_json=json_data,
                                    data_csv=csv_data,
                                    file_name_prefix=audio_file[2:]
                                )
                                zip_download(count=audio_files.index(audio_file) ,data=zip_buffer, file_name=f'{audio_file[2:]}.zip', mime="application/zip", log_message="Action: Audited Results Zip File Downloaded")
                                
                        current += 1
                        create_log_entry(f"Successfully Audited: {audio_file[2:]}")
                        df.loc[len(df)] = pd.Series(dtype='float64')
                        combined_results.append(df)
                    #     if save_audited_transcript:
                    #         if save_to_path:
                    #             try:
                    #                 # Ensure the directory exists
                    #                 os.makedirs(os.path.dirname(save_to_path), exist_ok=True)
                    #                 file_name_without_extension, _ = os.path.splitext(audio_file[2:])
                    #                 full_path = os.path.join(save_to_path, file_name_without_extension + ".csv")
                    #                 # df = pd.json_normalize(csv_data)
                    #                 # Save the DataFrame as a CSV to the specified path
                    #                 df.to_csv(full_path, index=False)
                    #                 print(f"Saved audited results (CSV) to {save_to_path}")
                    #             except Exception as e:
                    #                 print(f"Failed to save file: {e}")
                    #         else:
                    #             print("Save path not specified.")
                    # if save_audited_transcript:
                    #     if save_to_path:
                    #         combined_df = pd.concat(combined_results, ignore_index=True)
                    #         os.makedirs(os.path.dirname(save_to_path), exist_ok=True)
                    #         full_path = os.path.join(save_to_path, "combined_results.csv")
                    #         combined_df.to_csv(full_path, index=False)

                    if combined_results != []:
                        # Concatenate all DataFrames
                        combined_df = pd.concat(combined_results, ignore_index=True)

                        # Create an in-memory CSV using BytesIO
                        output = BytesIO()
                        combined_df.to_csv(output, index=False)
                        output.seek(0)  # Reset buffer position to the start

                        # Get the CSV data as a string
                        combined_csv_data = output.getvalue().decode('utf-8')
                        with st.spinner("Preparing Consolidated Results..."):
                            zip_buffer = handle_combined_audit_result_download(
                                            data_text=all_text,
                                            data_csv=combined_csv_data,
                                            file_name_prefix="combined_audit_results"
                                        )
                            
                        combined_audit_result_download(data=zip_buffer, file_name='CombinedAuditResults.zip', mime="application/zip", log_message="Action: Audited Results Zip File Downloaded")  
                        username = st.session_state["username"]
                        directory = username
                        delete_mp3_files(directory)
                        
                    # else:
                    #     st.error("Please specify a destination folder to save audited transcript!")
                    if submit and audio_files and combined_results:
                        st.subheader("Token Usage Summary")
                        
                        # Create a DataFrame for token usage
                        token_data = []
                        for filename, counts in st.session_state.token_counts.items():
                            token_data.append({
                                'Filename': filename,
                                'Transcription Tokens': counts.get('transcription', 0),
                                'Audit Tokens': counts.get('audit', 0),
                                'Total Tokens': counts.get('transcription', 0) + counts.get('audit', 0)
                            })
                        
                        token_df = pd.DataFrame(token_data)
                        
                        # Display token usage in a nice table
                        st.dataframe(token_df.style.format({
                            'Transcription Tokens': '{:,.0f}',
                            'Audit Tokens': '{:,.0f}',
                            'Total Tokens': '{:,.0f}'
                        }))
                        
                        # Display grand total
                        total_tokens = token_df['Total Tokens'].sum()
                        st.markdown(f"**Grand Total Tokens Used: {total_tokens:,.0f}**")

                st.subheader("Event Log")
                log_container = st.container()
                with log_container:
                    # Read and display the log content
                    log_content = read_log_file()
                    log_content = log_content.replace('\n', '<br>')

                    # Display the log with custom styling
                    html_content = (
                        "<div style='height:200px; overflow-y:scroll; background-color:#2b2b2b; color:#f8f8f2; "
                        "padding:10px; border-radius:5px; border:1px solid #444;'>"
                        "<pre style='font-family: monospace; font-size: 13px; line-height: 1.5em;'>{}</pre>"
                        "</div>"
                    ).format(log_content)
                    
                    st.markdown(html_content, unsafe_allow_html=True)
                    
                csv_file = 'logfile.csv'
                st.markdown("<br>", unsafe_allow_html=True)
                if os.path.exists(csv_file):
                    with open(csv_file, 'rb') as file:
                        file_contents = file.read()
                        handle_download_log_file(data=file_contents, file_name='log.csv', mime='text/csv', log_message="Action: Event Log Downloaded")
    except Exception as e:
        # st.error(f"An error occurred: {e}")
        create_log_entry(f"Error: {e}")
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    seed_users()
    print(torch.cuda.is_available())  # Should return True if CUDA is set up

    main()
