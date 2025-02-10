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
from database import Session, User, seed_users, generate_totp_secret
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

from trulens.core import Feedback
from trulens.providers.openai import OpenAI as OpenAIProvider
from trulens.apps.custom import TruCustomApp

import smtplib
import traceback
import numpy as np
import pyotp
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import timedelta
import random

# from trulens_eval import feedback
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

openai_provider = OpenAIProvider(api_key=st.secrets["API_KEY"])


try:
    llm_lingua = PromptCompressor(
        model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",  # Using base model
        model_config={"revision": "main"},
        use_llmlingua2=True,
        device_map="cpu"
    )
except Exception as e:
    print(f"Error initializing LLMLingua: {e}")


# print("torch version:", torch.__version__)

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

def add_user(username: str, password: str, role: str = "user", email: str = None, totp_secret: str = None) -> Tuple[bool, str]:
    """Add a new user to the database"""
    try:
        # Validate password
        session = Session()
        new_user = User(
            username=username, 
            password=password, 
            role=role,
            email=email,
            totp_secret=totp_secret
            )
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
    print("Deleting user")
    """Delete a user from the database"""
    try:
        session = Session()
        user = session.query(User).filter_by(username=username).first()
        if user:
            print(f"Deleting user: {username}")
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
    col1, col2 = st.columns(2)
    with col1:
        new_username = st.text_input("Username", key="new_username")
        new_password = st.text_input("Password", type="password", key="new_password")
    with col2:
        new_email = st.text_input("Email", key="new_email")
        new_role = st.selectbox("Role", ["user", "admin"], key="new_role")

        # enable_2fa = st.checkbox("Enable 2FA", key="enable_2fa")
    
    if st.button("Add User"):
        if new_username and new_password and new_email:
            # Generate TOTP secret if 2FA is enabled
            totp_secret = generate_totp_secret()
            
            success, message = add_user(
                username=new_username,
                password=new_password,
                role=new_role,
                email=new_email,
                totp_secret=totp_secret
            )
            
            if success:
                st.success(message)
                # if enable_2fa:
                #     # Display TOTP QR code and secret for initial setup
                #     totp = pyotp.TOTP(totp_secret)
                #     qr_code = qrcode.make(totp.provisioning_uri(new_email, issuer_name="Your App Name"))
                #     st.image(qr_code, caption="Scan this QR code with Google Authenticator")
                #     st.code(totp_secret, language=None)
                create_log_entry(f"Admin Action: Added new user - {new_username}")
            else:
                st.error(message)
                create_log_entry(f"Admin Action Failed: Add user - {new_username} - {message}")
        else:
            st.warning("Please fill in all required fields")

    # Manage Existing Users Section
    st.subheader("Manage Users")
    users = get_all_users()
    
    if users:
        for user in users:
            with st.expander(f"User: {user['username']} ({user['role']})"):
                tab1, tab2 = st.tabs(["Account", "Security"])
                
                # Tab 1: Account Management
                with tab1:
                    col1, col2 = st.columns(2)
                    with col1:
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
                    
                    with col2:
                        new_email = st.text_input("New Email", 
                                                value=user.get('email', ''),
                                                key=f"email_{user['username']}")
                        if st.button("Update Email", key=f"btn_email_{user['username']}"):
                            success, message = update_user_email(user['username'], new_email)
                            if success:
                                st.success(message)
                                create_log_entry(f"Admin Action: Updated email for user - {user['username']}")
                            else:
                                st.error(message)
                
                # Tab 2: Security Management
                with tab2:
                    col1, col2 = st.columns(2)
                    with col1:
                        new_pass = st.text_input("New Password", 
                                               type="password",
                                               key=f"pass_{user['username']}")
                        if st.button("Change Password", key=f"btn_pass_{user['username']}"):
                            if new_pass:
                                success, message = change_password(user['username'], new_pass)
                                if success:
                                    st.success(message)
                                    create_log_entry(f"Admin Action: Changed password for user - {user['username']}")
                                else:
                                    st.error(message)
                    
                    # with col2:
                    #     if st.button("Reset 2FA", key=f"btn_reset_2fa_{user['username']}"):
                    #         success, message = reset_user_2fa(user['username'])
                    #         if success:
                    #             st.success("2FA has been reset. New QR code generated.")
                    #             create_log_entry(f"Admin Action: Reset 2FA for user - {user['username']}")
                    #             # Show new QR code
                    #             totp_secret = get_user_totp_secret(user['username'])
                    #             if totp_secret:
                    #                 totp = pyotp.TOTP(totp_secret)
                    #                 qr_code = qrcode.make(totp.provisioning_uri(user['email'], issuer_name="Your App Name"))
                    #                 st.image(qr_code, caption="New 2FA QR Code")
                    #                 st.code(totp_secret, language=None)
                    #         else:
                    #             st.error(message)
                
                # Tab 3: 2FA Status
                # with tab3:
                #     has_2fa = check_user_2fa_status(user['username'])
                #     st.write(f"2FA Status: {'Enabled' if has_2fa else 'Disabled'}")
                    
                #     if has_2fa:
                #         if st.button("Disable 2FA", key=f"btn_disable_2fa_{user['username']}"):
                #             success, message = disable_user_2fa(user['username'])
                #             if success:
                #                 st.success("2FA has been disabled")
                #                 create_log_entry(f"Admin Action: Disabled 2FA for user - {user['username']}")
                #                 st.rerun()
                #             else:
                #                 st.error(message)
                #     else:
                #         if st.button("Enable 2FA", key=f"btn_enable_2fa_{user['username']}"):
                #             success, message = enable_user_2fa(user['username'])
                #             if success:
                #                 st.success("2FA has been enabled. Scan the QR code below.")
                #                 create_log_entry(f"Admin Action: Enabled 2FA for user - {user['username']}")
                #                 # Show QR code
                #                 totp_secret = get_user_totp_secret(user['username'])
                #                 if totp_secret:
                #                     totp = pyotp.TOTP(totp_secret)
                #                     qr_code = qrcode.make(totp.provisioning_uri(user['email'], issuer_name="Your App Name"))
                #                     st.image(qr_code, caption="2FA QR Code")
                #                     st.code(totp_secret, language=None)
                #                 st.rerun()
                #             else:
                #                 st.error(message)
                
                # Delete User Button (outside tabs)
                if st.button("Delete User", key=f"btn_del_{user['username']}", type="secondary"):
                    success, message = delete_user(user['username'])
                    if success:
                        st.success(message)
                        create_log_entry(f"Admin Action: Deleted user - {user['username']}")
                        st.rerun()
                    else:
                        st.error(message)
    else:
        st.info("No users found")

def check_user_2fa_status(username):
    """Check if user has 2FA enabled"""
    try:
        session = Session()
        user = session.query(User).filter_by(username=username).first()
        return user is not None and user.totp_secret is not None
    except Exception as e:
        print(f"Error checking 2FA status: {e}")
        return False
    finally:
        session.close()

def enable_user_2fa(username):
    """Enable 2FA for a user"""
    try:
        session = Session()
        user = session.query(User).filter_by(username=username).first()
        if user:
            user.totp_secret = generate_totp_secret()
            session.commit()
            return True, "2FA enabled successfully"
        return False, "User not found"
    except Exception as e:
        session.rollback()
        return False, f"Error enabling 2FA: {e}"
    finally:
        session.close()

def disable_user_2fa(username):
    """Disable 2FA for a user"""
    try:
        session = Session()
        user = session.query(User).filter_by(username=username).first()
        if user:
            user.totp_secret = None
            session.commit()
            return True, "2FA disabled successfully"
        return False, "User not found"
    except Exception as e:
        session.rollback()
        return False, f"Error disabling 2FA: {e}"
    finally:
        session.close()

def reset_user_2fa(username):
    """Reset 2FA for a user"""
    try:
        session = Session()
        user = session.query(User).filter_by(username=username).first()
        if user:
            user.totp_secret = generate_totp_secret()
            session.commit()
            return True, "2FA reset successfully"
        return False, "User not found"
    except Exception as e:
        session.rollback()
        return False, f"Error resetting 2FA: {e}"
    finally:
        session.close()

def get_user_totp_secret(username):
    """Get user's TOTP secret"""
    try:
        session = Session()
        user = session.query(User).filter_by(username=username).first()
        return user.totp_secret if user else None
    except Exception as e:
        print(f"Error getting TOTP secret: {e}")
        return None
    finally:
        session.close()

def update_user_email(username, new_email):
    """Update user's email"""
    try:
        session = Session()
        user = session.query(User).filter_by(username=username).first()
        if user:
            user.email = new_email
            session.commit()
            return True, "Email updated successfully"
        return False, "User not found"
    except Exception as e:
        session.rollback()
        return False, f"Error updating email: {e}"
    finally:
        session.close()

def verify_seed():
    """
    Verify that users were seeded correctly with all required fields.
    """
    try:
        session = Session()
        users = session.query(User).all()
        
        for user in users:
            print(f"\nVerifying user: {user.username}")
            print(f"Role: {user.role}")
            print(f"Email set: {'Yes' if user.email else 'No'}")
            print(f"TOTP secret set: {'Yes' if user.totp_secret else 'No'}")
            
            # Verify password meets requirements
            is_valid, message = validate_password(user.password)
            print(f"Password valid: {'Yes' if is_valid else 'No - ' + message}")
            
            # Verify TOTP secret is valid
            if user.totp_secret:
                try:
                    totp = pyotp.TOTP(user.totp_secret)
                    totp.now()  # This will raise an exception if the secret is invalid
                    print("TOTP secret is valid")
                except Exception as e:
                    print(f"TOTP secret is invalid: {e}")
                    
    except Exception as e:
        print(f"Error verifying seed: {e}")
    finally:
        session.close()

def send_reset_email(email, reset_token):
    """Send password reset email with verification code"""
    sender_email = st.secrets["EMAIL_ADDRESS"]
    sender_password = st.secrets["EMAIL_PASSWORD"]
    
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = email
    msg['Subject'] = "Password Reset Request"
    
    body = f"""
    You have requested to reset your password.
    Your verification code is: {reset_token}
    
    This code will expire in 15 minutes.
    If you did not request this reset, please ignore this email.
    """
    
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

def verify_totp(secret, token):
    """Verify a TOTP token"""
    totp = pyotp.TOTP(secret)
    return totp.verify(token)

def initiate_password_reset(username_or_email):
    """Start the password reset process"""
    try:
        session = Session()
        # Check if input is username or email
        user = (session.query(User)
               .filter((User.username == username_or_email) | 
                      (User.email == username_or_email))
               .first())
        
        if not user:
            return False, "User not found"
            
        timestamp = int(time.time() * 1000)
        random_part = random.randint(100, 999)

        reset_token = (timestamp % 1000) * 1000 + random_part

        # reset_token = ''.join([str(random.randint(0, 9)) for _ in range(6)])
        
        # Set token expiry (15 minutes)
        user.reset_token = reset_token
        user.reset_token_expiry = datetime.now() + timedelta(minutes=15)
        
        # Send reset email
        if send_reset_email(user.email, reset_token):
            session.commit()
            return True, "Reset email sent"
        else:
            return False, "Failed to send reset email"
            
    except Exception as e:
        session.rollback()
        return False, f"Error initiating reset: {e}"
    finally:
        session.close()

def verify_reset_token(username_or_email, token):
    """Verify the reset token"""
    try:
        session = Session()
        user = (session.query(User)
               .filter((User.username == username_or_email) | 
                      (User.email == username_or_email))
               .first())
        
        if not user:
            return False, "User not found"
            
        if not user.reset_token or not user.reset_token_expiry:
            return False, "No reset token found"
            
        if datetime.now() > user.reset_token_expiry:
            return False, "Reset token expired"
            
        if user.reset_token != token:
            return False, "Invalid reset token"
            
        return True, "Token verified"
        
    except Exception as e:
        return False, f"Error verifying token: {e}"
    finally:
        session.close()

def reset_password_2fa(username_or_email, new_password, reset_token):
    """Reset password with 2FA verification"""
    try:
        # First verify the reset token
        is_valid, message = verify_reset_token(username_or_email, reset_token)
        if not is_valid:
            return False, message
            
        session = Session()
        user = (session.query(User)
               .filter((User.username == username_or_email) | 
                      (User.email == username_or_email))
               .first())
        
        # Validate new password
        is_valid, message = validate_password(new_password)
        if not is_valid:
            return False, message
        
        # Update password and clear reset token
        user.password = new_password
        user.reset_token = None
        user.reset_token_expiry = None
        
        # Generate new TOTP secret for 2FA
        user.totp_secret = generate_totp_secret()
        
        session.commit()
        return True, "Password reset successful"
        
    except Exception as e:
        session.rollback()
        return False, f"Error resetting password: {e}"
    finally:
        session.close()

def password_reset_interface():
    #!CANNOT USE SCHOOL WIFI TO SEND EMAIL!!!!!
    #!REMEMBER TO CHANGE THE EMAIL AND APP PASSWORD TO YOUR OWN ONE IN THE SECRETS.TOML FILE THANKS
    st.title("Password Reset")
    
    # Step 1: Request reset
    with st.form("reset_request_form"):
        username_or_email = st.text_input("Enter Username or Email")
        request_reset = st.form_submit_button("Request Reset")
        
        if request_reset:
            success, message = initiate_password_reset(username_or_email)
            if success:
                st.success(message)
                st.session_state['reset_requested'] = True
                st.session_state['reset_username'] = username_or_email
            else:
                st.error(message)
    
    # Step 2: Verify token and reset password
    if st.session_state.get('reset_requested'):
        with st.form("reset_password_form"):
            reset_token = st.text_input("Enter Reset Code", key="reset_token")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            
            reset_submit = st.form_submit_button("Reset Password")
            
            if reset_submit:
                if new_password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    success, message = reset_password_2fa(
                        st.session_state['reset_username'],
                        new_password,
                        reset_token
                    )
                    if success:
                        st.success(message)
                        # Clear reset state
                        st.session_state.pop('reset_requested', None)
                        st.session_state.pop('reset_username', None)
                        # Redirect to login
                        st.rerun()
                    else:
                        st.error(message)


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
        #!batch processing that isnt working
        # total_jobs = client.batches.list()
        # for job in total_jobs:
        #     print("jobs:", job.id, job.status, job) 
        #     if job.status != "completed" or job.status != "failed":
        #         try:
        #             client.batches.cancel(job.id)
        #         except Exception:
        #             pass
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
    if st.session_state['reset_mode'] is not True:
        st.title("Login Portal")

    if st.session_state.get('reset_mode'):
        password_reset_interface()
        if st.button("Back to Login"):
            st.session_state.pop('reset_mode', None)
            st.rerun()
        return

    # Group inputs and button in a form for "Enter" support
    with st.form("login_form"):
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        # Visible Login button (inside the form for "Enter" key support)
        col1, col2 = st.columns([2,1])
        with col1:
            login_button = st.form_submit_button("Login")
        
    # Add "Forgot Password?" link below the form
    if st.button("Forgot Password?"):
        st.session_state['reset_mode'] = True
        st.rerun()

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


def LLM_audit(dialog, audio_file):
    def audit_function(text):
        total_tokens = 0  # Initialize total_tokens variable
        filename = os.path.basename(audio_file)
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
        
        stage_2_prompt = """
        You are an auditor for IPP or IPPFA. Return ONLY  a valid JSON obect. 

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

        # Compress the dialog input
        compressed_dialog = llm_lingua.compress_prompt(
            dialog,
            rate=0.5,  # Adjust compression rate as needed
            force_tokens=["!", ".", "?", "\n"],
            drop_consecutive=True,
        )

        model_engine ="gpt-4o-mini"

        messages=[{'role':'user', 'content':f"{stage_1_prompt} {compressed_dialog['compressed_prompt']}"}]

        completion = client.chat.completions.create(
            model=model_engine,
            messages=messages,
            temperature=0,
        )

        total_tokens += completion.usage.total_tokens
        print(f"Total tokens used for audit: {total_tokens}")

        stage_1_result = completion.choices[0].message.content

        stage_1_result = process_stage_1(stage_1_result)

        print(stage_1_result)


        output_dict = {"Stage 1": stage_1_result}

        if determine_pass_fail(stage_1_result) == "Pass":
                        
            messages=[{'role':'user', 'content':f"{stage_2_prompt} {compressed_dialog}"}]

            model_engine ="gpt-4o-mini"

            completion = client.chat.completions.create(
                model=model_engine,
                messages=messages,
                temperature=0,
            )

            total_tokens += completion.usage.total_tokens

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
            # print("overall result = ", determine_overall_result(output_dict))
            output_dict["Overall Result"] = determine_overall_result(output_dict)

            # if output_dict["Overall Result"] == "Pass":
            #     del output_dict["Overall Result"]
        else: 
            output_dict["Overall Result"] = "Fail"

        output_dict["Total Tokens"] = total_tokens

        if filename not in st.session_state.token_counts:
            st.session_state.token_counts[filename] = {}

        st.session_state.token_counts[filename]["audit"] = total_tokens


        return output_dict



        # model_engine ="gpt-4o-mini"

        # messages=[{'role':'user', 'content':f"{stage_1_prompt} {compressed_dialog['compressed_prompt']}"}]

        #! batch processing that isnt working
        # requests = [
        #     {
        #         "custom_id": f"{filename}_stage1",
        #         "method": "POST",
        #         "url": "/v1/chat/completions", 
        #         "body": {
        #             "model": "gpt-4o-mini",
        #             "messages": [
        #                 {'role': 'user', 'content': f"{stage_1_prompt} {compressed_dialog['compressed_prompt']}"}
        #             ],
        #             "temperature": 0
        #         }
        #     },
        #     {
        #         "custom_id": f"{filename}_stage2",
        #         "method": "POST",
        #         "url": "/v1/chat/completions", 
        #         "body": {
        #             "model": "gpt-4o-mini",
        #             "messages": [
        #                 {'role': 'user', 'content': f"{stage_2_prompt} {compressed_dialog['compressed_prompt']}"}
        #             ],
        #             "temperature": 0
        #         }
        #     }
        # ]
        # username = st.session_state["username"]
        # user_folder = os.path.join(".", username)
        # os.makedirs(user_folder, exist_ok=True)
        # batch_file = os.path.join(user_folder, f"batch_requests_{filename}.jsonl")

        # with open(batch_file, "w") as f:
        #     for req in requests:
        #         f.write(json.dumps(req) + "\n")

        # try:
        #     # Upload JSONL file and create batch job
        #     with open(batch_file, "rb") as f:
        #         batch_file_upload = client.files.create(
        #             file=f,
        #             purpose="batch"
        #         )
            
        #     # Submit batch job - corrected parameters based on OpenAI docs
        #     batch_job = client.batches.create(
        #         input_file_id=batch_file_upload.id,
        #         endpoint="/v1/chat/completions",
        #         completion_window="24h"
        #     )
            
        #     # Wait for batch completion
        #     # During status checking
        #     total_jobs = client.batches.list()
        #     for job in total_jobs:
        #         print(f"\nJob ID: {job.id}")
        #         print(f"Status: {job.status}")
        #         print(f"Created at: {job.created_at}")
        #         print(f"Completed at: {job.completed_at}")
        #         print(f"Request counts: {job.request_counts}")
        #         if hasattr(job, 'output_file_id') and job.output_file_id is not None:
        #             print(f"Output file ID: {job.output_file_id}")
        #             try:
        #                 output_text = client.files.content(job.output_file_id).text
        #                 print(f"Output text: {output_text}")
        #             except Exception as e:
        #                 print(f"Error retrieving output text: {e}")
        #         if hasattr(job, 'failed_at'):
        #             print(f"Failed at: {job.failed_at}")
        #         print("----------------------")

        #             # print("jobs:", job.id, job.status, job)

                    
        #     while True:
        #         job_status = client.batches.retrieve(batch_job.id)
        #         # print(f"Current status: {job_status.status}")
        #         # print(f"Request counts: {job_status.request_counts}")
        #         # print(f"Detailed status: {job_status}")  # Add this to see all available information
                

        #         if job_status.status == "completed":
        #             break
        #         elif job_status.status == "failed":
        #             if hasattr(job_status, 'error'):
        #                 error_msg = job_status.error
        #             else:
        #                 error_msg = "Unknown error - batch job failed"
        #             raise Exception(f"Batch job failed: {error_msg}")
        #         elif job_status.status == "validating":
        #             print("Job is still validating...")
                
        #         time.sleep(10)


        #     # Get results
        #     if not job_status.output_file_id:
        #         raise Exception("No output file ID received")
                
        #     result_content = client.files.content(job_status.output_file_id).text
            
            # results = []
            # for line in result_content.split('\n'):
            #     if line:
            #         try:
            #             results.append(json.loads(line))
            #         except json.JSONDecodeError as e:
            #             print(f"Error parsing result line: {e}")
            #             continue

            # if not results:
            #     raise Exception("No valid results received from batch job")

            # # Initialize output dict
            # output_dict = {"Stage 1": [], "Overall Result": "Fail"}
            
            # try:
            #     # Process stage 1 result
            #     stage_1_response = next(r for r in results if r["custom_id"] == f"{filename}_stage1")
            #     if not stage_1_response:
            #         raise Exception("Stage 1 response not found in results")
                    
            #     stage_1_result = process_stage_1(stage_1_response["response"]["body"]["choices"][0]["message"]["content"])
            #     output_dict["Stage 1"] = stage_1_result

            #     # Only proceed to stage 2 if stage 1 passes
            #     if determine_pass_fail(stage_1_result) == "Pass":
            #         stage_2_response = next(r for r in results if r["custom_id"] == f"{filename}_stage2")
            #         if stage_2_response:
            #             stage_2_result = stage_2_response["response"]["body"]["choices"][0]["message"]["content"]
            #             stage_2_result = stage_2_result.replace("Audit Results:", "").replace("### Input:", "").replace("### Output:", "")
            #             stage_2_result = stage_2_result.replace("### Response:", "").replace("json", "").replace("```", "").strip()
            #             stage_2_result = format_json_with_line_break(stage_2_result)
            #             stage_2_result = json.loads(stage_2_result)
            #             output_dict["Stage 2"] = stage_2_result
            #             output_dict["Overall Result"] = determine_overall_result(output_dict)

            #     # Track tokens
            #     total_tokens = sum(r["response"].get("usage", {}).get("total_tokens", 0) for r in results)
            #     output_dict["Total Tokens"] = total_tokens

            #     if filename not in st.session_state.token_counts:
            #         st.session_state.token_counts[filename] = {}
            #     st.session_state.token_counts[filename]["audit"] = total_tokens

            # except Exception as e:
            #     print(f"Error processing results: {e}")
            #     # Don't return None, return a valid output_dict with error status
            #     output_dict["error"] = str(e)

            # # Clean up files
            # try:
            #     client.files.delete(batch_file_upload.id)
            #     client.files.delete(job_status.output_file_id)
            # except Exception as e:
            #     print(f"Error cleaning up OpenAI files: {e}")

            # return output_dict

        # except Exception as e:
        #     print(f"Error in batch processing: {e}")
        #     # Return a valid dict instead of None
        #     return {
        #         "Stage 1": [],
        #         "Overall Result": "Error",
        #         "error": str(e),
        #         "Total Tokens": 0
        #     }

        # finally:
        #     # Local file cleanup
        #     try:
        #         if os.path.exists(batch_file):
        #             os.remove(batch_file)
        #     except Exception as e:
        #         print(f"Error cleaning up local file: {e}")


    def determine_pass_fail(result):
        for item in result:
            if item["Result"] == "Fail":
                return "Fail"
        return "Pass"
    
    def determine_overall_result(output_dict):
        if "Stage 2" in output_dict:
            for item in output_dict["Stage 2"]:
                if item["Result"] == "Fail":
                    return "Fail"
        return "Pass"

    def process_stage_1(content):
    # Clean up the string
        content = content.replace("Audit Results:", "")
        content = content.replace("### Input:", "")
        content = content.replace("### Output:", "")
        content = content.replace("### Response:", "")
        content = content.replace("json", "").replace("```", "")
        content = content.strip()
        
        # Parse JSON once - this gives us the list of dictionaries
        content = format_json_with_line_break(content)
        content = json.loads(content)
        
        # Handle case where content might be wrapped in criteria key
        if isinstance(content, dict) and "criteria" in content:
            content = content["criteria"]
            
        # Return the full list of criteria
        return content


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
    
    def create_feedback_functions():
        def groundedness_wrapper(input_text, output_result):
            try:
                # Convert input/output to proper format
                input_str = str(input_text) if input_text else ""
                output_str = str(output_result) if output_result else ""
                
                # Get groundedness score and reasons
                result = openai_provider.groundedness_measure_with_cot_reasons(input_str, output_str)
                print("Groundedness Result (from openai):", result)

                # Extract score and explanation
                if isinstance(result, tuple):
                    score, reasoning = result
                    # Format the reasoning for display
                    detailed_feedback = reasoning.get('reasons', 'No detailed feedback available')
                    return {
                        'result': score,
                        'explanation': detailed_feedback
                    }
                return {
                    'result': result["result"],
                    'explanation': result["explanation"]
                }
            except Exception as e:
                print(f"Groundedness Error: {e}")
                return {
                    'result': 0.0,
                    'explanation': f"Error calculating groundedness: {str(e)}"
                }

        # For groundedness
        f_groundedness = (
        Feedback(groundedness_wrapper, name="Groundedness")
        .on_input_output()
        )

        # For answer relevance between question and response
        f_answer_relevance = (
        Feedback(openai_provider.relevance_with_cot_reasons, name="Answer Relevance")
        .on_input_output()
        )

        # For context relevance
        f_context_relevance = (
        Feedback(openai_provider.context_relevance_with_cot_reasons, name="Context Relevance")
        .on_input_output()
        .aggregate(np.mean)
        )

        return [f_groundedness, f_answer_relevance, f_context_relevance]



    feedback_functions = create_feedback_functions()
    print("Feedback Functions:", feedback_functions)
    

    recorder = TruCustomApp(
            app=audit_function,
            app_id="IPP_Audit_App",
            feedbacks=feedback_functions
        )
    try:
        with recorder as recording:
            result = audit_function(dialog)
            
            # Collect feedback results
            feedback_results = {}
            for feedback in feedback_functions:
                try:
                    result_value = feedback(dialog, result)
                    feedback_results[feedback.name] = {
                        'result': result_value,
                        # You can add more details if needed
                    }
                except Exception as e:
                    print(f"Error processing feedback {feedback.name}: {e}")
            
            print("Collected Feedback Results:", feedback_results)
            
            return result, feedback_results
    except Exception as e:
        print(f"Error in LLM_audit: {e}")
        return result, {}

    # with recorder as recording:
    #     result = audit_function(dialog)
    #     print("Recording Calls:", recording.calls)
    #     print("Recording Records:", recording.records)

    #     # current_file = list(st.session_state.token_counts.keys())[-1] #honestly dunno which one is the correct one lmao

    #     return result, recording
    
    # sample_input = "This is a test input."
    # sample_output = audit_function(sample_input)

    # for feedback in feedback_functions:
    #     feedback_result = feedback(sample_input, sample_output)
    #     print(f"Feedback Result for {feedback.name}: {feedback_result}")

    import gc
    gc.collect()
    torch.cuda.empty_cache()
    # current_file = list(st.session_state.token_counts.keys())[-1]



    return result, recording #!i dont think this is used

        



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
    
def parse_groundedness_details(groundedness_text: str):
    """
    Parses the groundedness feedback string into a structured table format.
    """
    statements = re.split(r'STATEMENT \d+:', groundedness_text)
    data = []

    for statement in statements:
        if statement.strip():
            criteria_match = re.search(r'Criteria:\s*(.*?)\s*Supporting Evidence:', statement)
            evidence_match = re.search(r'Supporting Evidence:\s*(.*?)\s*Score:', statement)
            score_match = re.search(r'Score:\s*([\d.]+)', statement)
            
            criteria = criteria_match.group(1).strip() if criteria_match else "N/A"
            evidence = evidence_match.group(1).strip() if evidence_match else "N/A"
            score = float(score_match.group(1)) if score_match else 0.0
            
            data.append({
                "Criteria": criteria,
                "Supporting Evidence": evidence,
                "Score": score
            })

    return pd.DataFrame(data)

@st.fragment
def display_trulens_feedback(feedback_results, unique_key):
    st.subheader("Audit Quality Metrics")

    try:
        # Print the feedback results to understand its structure
        print("Feedback Results:", feedback_results)

        # Check if feedback results are available
        if not feedback_results:
            st.warning("No feedback metrics available.")
            return

        col1, col2, col3 = st.columns(3)

        with col1:
            try:
                groundedness = feedback_results.get('Groundedness', {}).get('result', 0.0)
                # If groundedness is a tuple, extract the first value
                if isinstance(groundedness, tuple):
                    groundedness = groundedness[0]
                print("Groundedness Score:", groundedness)
                st.metric(
                    label="Groundedness",
                    value=f"{groundedness['result']:.2%}"
                )
            except Exception as e:
                st.metric(label="Groundedness", value="N/A")
                print(f"Error processing groundedness: {e}")

        with col2:
            try:
                answer_relevance = feedback_results.get('Answer Relevance', {}).get('result', 0.0)
                if isinstance(answer_relevance, tuple):
                    answer_relevance = answer_relevance[0]
                st.metric(
                    label="Answer Relevance",
                    value=f"{float(answer_relevance):.2%}"
                )
            except Exception as e:
                st.metric(label="Answer Relevance", value="N/A")
                print(f"Error processing answer relevance: {e}")

        with col3:
            try:
                context_relevance = feedback_results.get('Context Relevance', {}).get('result', 0.0)
                if isinstance(context_relevance, tuple):
                    context_relevance = context_relevance[0]
                st.metric(
                    label="Context Relevance",
                    value=f"{float(context_relevance):.2%}"
                )
            except Exception as e:
                st.metric(label="Context Relevance", value="N/A")
                print(f"Error processing context relevance: {e}")
        
        if st.checkbox("Show Detailed Feedback", key=f"detailed_feedback_toggle_{unique_key}"):
            for metric, data in feedback_results.items():
                if metric == 'Groundedness':
                    # Handle Groundedness separately for table display
                    groundedness_details = feedback_results.get('Groundedness', {}).get('result', {}).get('explanation', '')
                    if groundedness_details:
                        df = parse_groundedness_details(groundedness_details)
                        st.write("### Groundedness Details Table")
                        st.dataframe(df)
                    else:
                        st.info("No groundedness details available.")
                else:
                    if isinstance(data.get('result'), tuple) and len(data['result']) > 1:
                        reason = data['result'][1].get('reason', 'No detailed feedback available')
                    elif isinstance(data.get('result'), dict):
                        reason = data['result'].get('explanation', 'No detailed feedback available')
                    with st.expander(f"{metric} Details"):
                            st.write(reason)

    except Exception as e:
        st.error(f"Error displaying feedback metrics: {e}")
        print(f"Full error details: {traceback.format_exc()}")

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
        if 'reset_mode' not in st.session_state:
            st.session_state.reset_mode = False
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
                                            result, feedback_results = LLM_audit(text, audio_file)
                                            print("result of audit: ",result)

                                            display_trulens_feedback(feedback_results, unique_key=audio_file[2:])

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
                                    elif transcribe_option == "Groq": #!not in use
                                        text, language_code = speech_to_text_groq(audio_file)
                                        if audit_option == "OpenAI (Recommended)":
                                            result = LLM_audit(text, audio_file)
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
    # print(torch.cuda.is_available())  # Should return True if CUDA is set up

    main()
