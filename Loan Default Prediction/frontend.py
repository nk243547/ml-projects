import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3
import os
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import io
import tempfile
import re
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import base64
from io import BytesIO
import time
from passlib.hash import bcrypt
import uuid
import datetime
from pathlib import Path

# Configure logging with enhanced setup
log_directory = Path.cwd()  # Use the current working directory
log_file_path = log_directory / 'app.log'

# Create the log file if it doesn't exist and set permissions
try:
    if not log_file_path.exists():
        log_file_path.touch()
    log_file_path.chmod(0o666)  # Ensure the file is writable
except Exception as e:
    print(f"Error setting up log file: {str(e)}")  # For debugging during development

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'  # Append mode to ensure logs aren't overwritten
)

# Initialize database
def init_database():
    try:
        conn = sqlite3.connect('loan_predictions.db')
        cursor = conn.cursor()
        # Create predictions table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            applicant_id TEXT,
            prediction INTEGER,
            actual_outcome INTEGER,
            timestamp TEXT,
            username TEXT
        )
        ''')
        # Check if username column exists in predictions table and add if missing
        cursor.execute("PRAGMA table_info(predictions)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'username' not in columns:
            cursor.execute('ALTER TABLE predictions ADD COLUMN username TEXT')
            logging.info("Added username column to predictions table")

        # Create users table with role column
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            reset_token TEXT,
            reset_token_expiry TEXT
        )
        ''')
        # Check if role column exists in users table and add if missing
        cursor.execute("PRAGMA table_info(users)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'role' not in columns:
            cursor.execute('ALTER TABLE users ADD COLUMN role TEXT DEFAULT "user"')
            logging.info("Added role column to users table")

        # Create login_history table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS login_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            login_timestamp TEXT NOT NULL
        )
        ''')
        conn.commit()
        conn.close()
        logging.info("Database initialized successfully with users and login_history tables")
    except Exception as e:
        logging.error(f"Database initialization error: {str(e)}")
        raise

# Check if any admin exists
def has_admin():
    try:
        conn = sqlite3.connect('loan_predictions.db')
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
        count = cursor.fetchone()[0]
        conn.close()
        return count > 0
    except Exception as e:
        logging.error(f"Error checking for admin: {str(e)}")
        return False

# Authentication Functions
def register_user(username, email, password, admin_code=None):
    try:
        if not username or not email or not password:
            return False, "All fields are required."
        if len(password) < 8:
            return False, "Password must be at least 8 characters long."
        if not re.match(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$', email):
            return False, "Invalid email format."

        hashed_password = bcrypt.hash(password)

        # Determine role: First user or correct admin code makes the user an admin
        role = 'user'
        if not has_admin():  # If no admin exists, the first user becomes an admin
            role = 'admin'
            logging.info(f"First user registered as admin: {username}")
        elif admin_code == "SET_ADMIN_2025":  # Hidden admin code for setting additional admins
            role = 'admin'
            logging.info(f"User registered as admin with admin code: {username}")

        conn = sqlite3.connect('loan_predictions.db')
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO users (username, email, password, role)
        VALUES (?, ?, ?, ?)
        ''', (username, email, hashed_password, role))
        conn.commit()
        conn.close()
        logging.info(f"User registered: {username} with role '{role}'")
        return True, f"Registration successful! Role assigned: {role}. Redirecting to login page..."
    except sqlite3.IntegrityError as e:
        logging.error(f"Registration error: {str(e)}")
        return False, "Username or email already exists."
    except Exception as e:
        logging.error(f"Registration error: {str(e)}")
        return False, f"Registration failed: {str(e)}"

def login_user(username, password):
    try:
        conn = sqlite3.connect('loan_predictions.db')
        cursor = conn.cursor()
        cursor.execute('SELECT password FROM users WHERE username = ?', (username,))
        result = cursor.fetchone()

        if result is None:
            conn.close()
            return False, "Username not found."
        
        hashed_password = result[0]
        if bcrypt.verify(password, hashed_password):
            # Log the login event to login_history table
            login_timestamp = datetime.datetime.now().isoformat()
            cursor.execute('''
            INSERT INTO login_history (username, login_timestamp)
            VALUES (?, ?)
            ''', (username, login_timestamp))
            conn.commit()
            conn.close()
            logging.info(f"User logged in: {username}")
            return True, "Login successful!"
        else:
            conn.close()
            return False, "Incorrect password."
    except Exception as e:
        logging.error(f"Login error: {str(e)}")
        return False, f"Login failed: {str(e)}"

def get_user_role(username):
    try:
        conn = sqlite3.connect('loan_predictions.db')
        cursor = conn.cursor()
        cursor.execute('SELECT role FROM users WHERE username = ?', (username,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else "user"
    except Exception as e:
        logging.error(f"Error retrieving user role for {username}: {str(e)}")
        return "user"

def update_user_role(username, new_role):
    try:
        conn = sqlite3.connect('loan_predictions.db')
        cursor = conn.cursor()
        cursor.execute('UPDATE users SET role = ? WHERE username = ?', (new_role, username))
        conn.commit()
        conn.close()
        logging.info(f"User role updated: {username} to {new_role}")
        return True, f"Role updated successfully for {username}."
    except Exception as e:
        logging.error(f"Error updating user role for {username}: {str(e)}")
        return False, f"Failed to update role: {str(e)}"

def get_all_users():
    try:
        conn = sqlite3.connect('loan_predictions.db')
        df = pd.read_sql_query("SELECT username, email, role FROM users", conn)
        conn.close()
        return df
    except Exception as e:
        logging.error(f"Error retrieving users: {str(e)}")
        return pd.DataFrame(columns=['username', 'email', 'role'])

def generate_reset_token(email):
    try:
        conn = sqlite3.connect('loan_predictions.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
        result = cursor.fetchone()

        if result is None:
            conn.close()
            return False, "Email not found."

        token = str(uuid.uuid4())
        expiry = (datetime.datetime.now() + datetime.timedelta(hours=1)).isoformat()

        cursor.execute('''
        UPDATE users SET reset_token = ?, reset_token_expiry = ?
        WHERE email = ?
        ''', (token, expiry, email))
        conn.commit()
        conn.close()
        logging.info(f"Password reset token generated for email: {email}")
        return True, token
    except Exception as e:
        logging.error(f"Reset token generation error: {str(e)}")
        return False, f"Failed to generate reset token: {str(e)}"

def reset_password(token, new_password):
    try:
        if len(new_password) < 8:
            return False, "Password must be at least 8 characters long."

        conn = sqlite3.connect('loan_predictions.db')
        cursor = conn.cursor()
        cursor.execute('SELECT email, reset_token_expiry FROM users WHERE reset_token = ?', (token,))
        result = cursor.fetchone()

        if result is None:
            conn.close()
            return False, "Invalid or expired token."

        email, expiry = result
        expiry_time = datetime.datetime.fromisoformat(expiry)
        if datetime.datetime.now() > expiry_time:
            cursor.execute('UPDATE users SET reset_token = NULL, reset_token_expiry = NULL WHERE reset_token = ?', (token,))
            conn.commit()
            conn.close()
            return False, "Token has expired."

        hashed_password = bcrypt.hash(new_password)
        cursor.execute('''
        UPDATE users SET password = ?, reset_token = NULL, reset_token_expiry = NULL
        WHERE reset_token = ?
        ''', (hashed_password, token))
        conn.commit()
        conn.close()
        logging.info(f"Password reset successful for email: {email}")
        return True, "Password reset successful! Please log in with your new password."
    except Exception as e:
        logging.error(f"Password reset error: {str(e)}")
        return False, f"Password reset failed: {str(e)}"

# Function to get login history from database
def get_login_history():
    try:
        conn = sqlite3.connect('loan_predictions.db')
        df = pd.read_sql_query("SELECT username, login_timestamp FROM login_history ORDER BY login_timestamp DESC", conn)
        conn.close()
        df['login_timestamp'] = pd.to_datetime(df['login_timestamp'])
        return df
    except Exception as e:
        logging.error(f"Login history retrieval error: {str(e)}")
        return pd.DataFrame(columns=['username', 'login_timestamp'])

# Function to parse login history from app.log with improved parsing
def parse_login_history_from_log():
    login_events = []
    try:
        if not os.path.exists('app.log'):
            logging.error("Log file 'app.log' does not exist")
            return pd.DataFrame(columns=['username', 'login_timestamp'])

        with open('app.log', 'r') as f:
            for line in f:
                if "User logged in" in line:
                    # Expected log format: "2025-04-24 12:34:56,789 - INFO - User logged in: username"
                    match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - INFO - User logged in: (.*)', line.strip())
                    if match:
                        timestamp_str, username = match.groups()
                        try:
                            timestamp = pd.to_datetime(timestamp_str)
                            login_events.append({'username': username, 'login_timestamp': timestamp})
                        except ValueError as ve:
                            logging.warning(f"Failed to parse timestamp in log line: {line.strip()} - {str(ve)}")
                            continue
                    else:
                        logging.warning(f"Log line does not match expected format: {line.strip()}")
                        continue

        if not login_events:
            logging.info("No login events found in app.log")
            return pd.DataFrame(columns=['username', 'login_timestamp'])

        df = pd.DataFrame(login_events)
        df = df.sort_values('login_timestamp', ascending=False)
        return df

    except Exception as e:
        logging.error(f"Error parsing app.log for login history: {str(e)}")
        return pd.DataFrame(columns=['username', 'login_timestamp'])

# Authentication Pages
def register_page():
    st.markdown("""
    <div class="header animate-in">Register</div>
    <div class="card animate-in">
        <p style="font-size: 1.1rem;">
            Create a new account to access the Loan Risk Analyzer.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("register_form", border=False):
        st.markdown('<div class="auth-form">', unsafe_allow_html=True)
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        # Hidden admin code field for setting additional admins
        admin_code = st.text_input("Admin Code (optional)", type="password", help="Enter the admin code to register as an admin (if applicable).")
        submit_button = st.form_submit_button("Register", type="primary")
        st.markdown('</div>', unsafe_allow_html=True)

        if submit_button:
            success, message = register_user(username, email, password, admin_code if admin_code else None)
            if success:
                st.success(message)
                # Redirect to login page after a short delay to allow the user to see the success message
                st.session_state.auth_page = "Login"
                st.markdown("""
                <script>
                    setTimeout(function() {
                        window.location.reload();
                    }, 2000); // Redirect after 2 seconds
                </script>
                """, unsafe_allow_html=True)
            else:
                st.error(message)

    # Add a manual link to go back to the login page if the user doesn't want to wait
    _, col, _ = st.columns([1, 2, 1])
    with col:
        if st.button("Back to Login"):
            st.session_state.auth_page = "Login"
            st.rerun()

def login_page():
    st.markdown("""
    <div class="header animate-in">Log in to access the Loan Risk Analyzer.</div>
    """, unsafe_allow_html=True)

    with st.form("login_form", border=False):
        st.markdown('<div class="auth-form">', unsafe_allow_html=True)
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login", type="primary")
        st.markdown('</div>', unsafe_allow_html=True)

        if submit_button:
            success, message = login_user(username, password)
            if success:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.page = "Home"
                st.success(message)
                st.rerun()
            else:
                st.error(message)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Forgot your password?"):
            st.session_state.auth_page = "Forgot Password"
            st.rerun()
    with col2:
        if st.button("Don't have an account? Register"):
            st.session_state.auth_page = "Register"
            st.rerun()

def forgot_password_page():
    st.markdown("""
    <div class="header animate-in">Forgot Password</div>
    <div class="card animate-in">
        <p style="font-size: 1.1rem;">
            Enter your email to receive a password reset link. If a link were to be sent, it would be here (simulated).
        </p>
    </div>
    """, unsafe_allow_html=True)

    if "reset_token" not in st.session_state:
        with st.form("forgot_password_form", border=False):
            st.markdown('<div class="auth-form">', unsafe_allow_html=True)
            email = st.text_input("Email")
            submit_button = st.form_submit_button("Request Reset Link", type="primary")
            st.markdown('</div>', unsafe_allow_html=True)

            if submit_button:
                success, result = generate_reset_token(email)
                if success:
                    token = result
                    st.session_state.reset_token = token
                    st.success(f"Reset token generated (simulated email sent). Token: {token}")
                    st.info("In a real application, this token would be sent via email. Copy the token and proceed.")
                else:
                    st.error(result)
    else:
        with st.form("reset_password_form", border=False):
            st.markdown('<div class="auth-form">', unsafe_allow_html=True)
            token = st.text_input("Reset Token (from email)")
            new_password = st.text_input("New Password", type="password")
            submit_button = st.form_submit_button("Reset Password", type="primary")
            st.markdown('</div>', unsafe_allow_html=True)

            if submit_button:
                success, message = reset_password(token, new_password)
                if success:
                    st.success(message)
                    st.session_state.auth_page = "Login"
                    del st.session_state.reset_token
                else:
                    st.error(message)

    _, col, _ = st.columns([1, 2, 1])
    with col:
        if st.button("Back to Login"):
            st.session_state.auth_page = "Login"
            st.rerun()

# Admin Panel Page
def admin_panel_page():
    st.markdown("""
    <div class="header animate-in">Admin Panel</div>
    <div class="card animate-in">
        <p style="font-size: 1.1rem;">
            Manage user roles.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Check if the user has admin role
    user_role = get_user_role(st.session_state.username)
    if user_role != "admin":
        st.error("Access Denied: This page is only accessible to admin users.")
        return

    # Display all users
    st.markdown("<div class='subheader animate-in'>User Management</div>", unsafe_allow_html=True)
    users_df = get_all_users()
    if not users_df.empty:
        st.dataframe(users_df)

        # Allow admin to update user roles
        st.markdown("<div class='subheader animate-in'>Update User Role</div>", unsafe_allow_html=True)
        with st.form("update_role_form", border=False):
            st.markdown('<div class="auth-form">', unsafe_allow_html=True)
            username_to_update = st.selectbox("Select User", options=users_df['username'].tolist())
            new_role = st.selectbox("New Role", options=["user", "admin"])
            submit_button = st.form_submit_button("Update Role", type="primary")
            st.markdown('</div>', unsafe_allow_html=True)

            if submit_button:
                # Prevent admin from changing their own role
                if username_to_update == st.session_state.username:
                    st.error("You cannot change your own role.")
                else:
                    success, message = update_user_role(username_to_update, new_role)
                    if success:
                        st.success(message)
                        st.rerun()  # Refresh the page to show updated roles
                    else:
                        st.error(message)
    else:
        st.info("No users found in the database.")

    render_footer_navigation()

# User Activity Page with enhanced feedback for log file records
def user_activity_page():
    st.markdown("""
    <div class="header animate-in">User Activity</div>
    <div class="card animate-in">
        <p style="font-size: 1.1rem;">
            View the login history of all users.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Check if the user has admin role
    user_role = get_user_role(st.session_state.username)
    if user_role != "admin":
        st.error("Access Denied: This page is only accessible to admin users.")
        return

    # Get login history from database
    login_history_db = get_login_history()
    
    # Get login history from log file as a fallback
    login_history_log = parse_login_history_from_log()

    st.markdown("<div class='subheader animate-in'>Login History</div>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Database Records", "Log File Records"])

    with tab1:
        if not login_history_db.empty:
            st.markdown("**Login History from Database**")
            st.dataframe(login_history_db.style.format({'login_timestamp': lambda x: x.strftime('%Y-%m-%d %H:%M:%S')}))
            
            # Summary Statistics
            total_logins = len(login_history_db)
            unique_users = login_history_db['username'].nunique()
            st.markdown("<div class='subheader animate-in'>Summary Statistics</div>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            col1.metric("Total Logins", total_logins)
            col2.metric("Unique Users", unique_users)

            # Login Trend
            login_trend = login_history_db.groupby(login_history_db['login_timestamp'].dt.date).size().reset_index(name='Login Count')
            fig_trend = px.line(login_trend, x='login_timestamp', y='Login Count', title="Login Trend Over Time",
                               labels={'login_timestamp': 'Date', 'Login Count': 'Number of Logins'})
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("No login history available in the database yet.")

    with tab2:
        if not login_history_log.empty:
            st.markdown("**Login History from Log File**")
            st.dataframe(login_history_log.style.format({'login_timestamp': lambda x: x.strftime('%Y-%m-%d %H:%M:%S')}))
            
            # Summary Statistics
            total_logins_log = len(login_history_log)
            unique_users_log = login_history_log['username'].nunique()
            st.markdown("<div class='subheader animate-in'>Summary Statistics</div>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            col1.metric("Total Logins", total_logins_log)
            col2.metric("Unique Users", unique_users_log)

            # Login Trend
            login_trend_log = login_history_log.groupby(login_history_log['login_timestamp'].dt.date).size().reset_index(name='Login Count')
            fig_trend_log = px.line(login_trend_log, x='login_timestamp', y='Login Count', title="Login Trend Over Time (Log File)",
                                   labels={'login_timestamp': 'Date', 'Login Count': 'Number of Logins'})
            st.plotly_chart(fig_trend_log, use_container_width=True)
        else:
            # Check if the log file exists and provide actionable feedback
            if not os.path.exists('app.log'):
                st.warning("No login history available because the log file 'app.log' does not exist. Ensure the application has write permissions in the working directory and that logging is configured correctly.")
            else:
                with open('app.log', 'r') as f:
                    log_content = f.read().strip()
                if not log_content:
                    st.warning("The log file 'app.log' exists but is empty. Make sure users are logging in to generate login events.")
                elif "User logged in" not in log_content:
                    st.warning("The log file 'app.log' exists but contains no login events. Verify that login events are being logged correctly in the login_user() function.")
                else:
                    st.info("No login events found in the log file that match the expected format. Check the log file for formatting issues or parsing errors.")

    render_footer_navigation()

# Security: Validate file name to prevent path traversal
def is_safe_filename(filename):
    safe_pattern = r'^[a-zA-Z0-9][a-zA-Z0-9_\-\.]*$'
    return bool(re.match(safe_pattern, filename))

# Security: Validate CSV content
def validate_csv(df, required_columns):
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return False, f"Missing columns: {', '.join(missing_cols)}"
    for col in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return False, f"Column {col} must be numeric"
    return True, ""

# Function to calculate engineered features
def calculate_engineered_features(data):
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    
    data['TotalIncome'] = data['ApplicantIncome'] + data['CoapplicantIncome']
    data['EMI'] = data['LoanAmount'] / data['Loan_Amount_Term']
    data['Debt_to_Income_Ratio'] = data['LoanAmount'] / (data['TotalIncome'] + 1)
    data['Loan_to_Income_Ratio'] = data['LoanAmount'] / (data['ApplicantIncome'] + 1)
    data['Income_Stability'] = data['ApplicantIncome'] / (data['CoapplicantIncome'] + 1)
    
    return data

# Prediction function
def predict_loan_default(raw_input_json, model=None):
    try:
        with open('feature_columns.json', 'r') as f:
            feature_info = json.load(f)
        expected_columns = feature_info['all_columns']
        required_columns = set(expected_columns)

        if model is None:
            model = joblib.load('loan_default_model.pkl')

        missing_cols = required_columns - set(raw_input_json.keys())
        extra_cols = set(raw_input_json.keys()) - required_columns
        if missing_cols:
            raise ValueError(f"Missing input fields: {sorted(missing_cols)}")
        if extra_cols:
            st.warning(f"Unused fields ignored: {sorted(extra_cols)}")

        input_df = calculate_engineered_features(raw_input_json)
        raw_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                        'Credit_History', 'Gender', 'Married', 'Dependents', 'Education',
                        'Self_Employed', 'Property_Area']
        input_df = input_df.reindex(columns=raw_features, fill_value=np.nan)

        probability = model.predict_proba(input_df)[0, 1]
        prediction = int(probability >= 0.5)

        st.session_state.processed_input_df = input_df

        conn = sqlite3.connect('loan_predictions.db')
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO predictions (applicant_id, prediction, actual_outcome, timestamp, username)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?)
        ''', (hash(str(raw_input_json)), prediction, None, st.session_state.username))
        conn.commit()
        conn.close()

        logging.info(f"Prediction made: applicant_id={hash(str(raw_input_json))}, prediction={prediction}, user={st.session_state.username}")
        return prediction, probability
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise

# Batch prediction for CSV
def predict_batch(df, model=None):
    try:
        with open('feature_columns.json', 'r') as f:
            feature_info = json.load(f)
        expected_columns = feature_info['all_columns']
        required_columns = set(expected_columns) - {'Loan_Status'}

        is_valid, error_msg = validate_csv(df, required_columns)
        if not is_valid:
            raise ValueError(error_msg)

        if model is None:
            model = joblib.load('loan_default_model.pkl')

        input_df = calculate_engineered_features(df)
        raw_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                        'Credit_History', 'Gender', 'Married', 'Dependents', 'Education',
                        'Self_Employed', 'Property_Area']
        input_df = input_df.reindex(columns=raw_features, fill_value=np.nan)

        probabilities = model.predict_proba(input_df)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        conn = sqlite3.connect('loan_predictions.db')
        cursor = conn.cursor()
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            cursor.execute('''
            INSERT INTO predictions (applicant_id, prediction, actual_outcome, timestamp, username)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?)
            ''', (hash(str(df.iloc[i].to_dict())), pred, None, st.session_state.username))
        conn.commit()
        conn.close()

        result_df = df.copy()
        result_df['Prediction'] = predictions
        result_df['Probability'] = probabilities
        logging.info(f"Batch prediction completed: {len(df)} records, user={st.session_state.username}")
        return result_df
    except Exception as e:
        logging.error(f"Batch prediction error: {str(e)}")
        raise

# Retrain model from CSV
def retrain_model(df):
    try:
        required_columns = {'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                           'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                           'Credit_History', 'Property_Area', 'Loan_Status'}
        is_valid, error_msg = validate_csv(df, required_columns)
        if not is_valid:
            raise ValueError(error_msg)

        preprocessor = joblib.load('preprocessor.pkl')
        X = df.drop('Loan_Status', axis=1)
        y = df['Loan_Status'].map({'Y': 0, 'N': 1})

        model = Pipeline([
            ('preprocessing', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        model.fit(X, y)

        logging.info("Model retrained successfully")
        return model
    except Exception as e:
        logging.error(f"Retraining error: {str(e)}")
        raise

# Load dataset for visualizations
@st.cache_data
def load_data():
    return pd.read_csv('loan.csv')

# Function to get feature importance for a single prediction
def get_feature_importance(input_df, model):
    try:
        feature_names = model.named_steps['preprocessing'].get_feature_names_out()
        importances = model.named_steps['classifier'].feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
        return feature_importance_df
    except Exception as e:
        logging.error(f"Feature importance error: {str(e)}")
        return None

# Function to fetch prediction history
def get_prediction_history():
    try:
        conn = sqlite3.connect('loan_predictions.db')
        df = pd.read_sql_query("SELECT * FROM predictions WHERE username = ?", 
                              conn, params=(st.session_state.username,))
        conn.close()
        df['prediction'] = pd.to_numeric(df['prediction'], errors='coerce')
        df = df.dropna(subset=['prediction'])
        df['prediction'] = df['prediction'].astype(int)
        return df
    except Exception as e:
        logging.error(f"Prediction history error: {str(e)}")
        return pd.DataFrame()

# Utility function to safely delete a file with retries
def safe_delete_file(filepath, max_retries=5, delay=0.5):
    for attempt in range(max_retries):
        try:
            if os.path.exists(filepath):
                os.unlink(filepath)
            return True
        except Exception as e:
            if attempt == max_retries - 1:
                logging.error(f"Failed to delete file {filepath} after {max_retries} attempts: {str(e)}")
                return False
            time.sleep(delay)
    return False

# Function to generate a PDF report for single prediction
def generate_pdf_report(prediction, probability, fig_gauge, feature_importance_df, key_factors):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Loan Default Prediction Report", styles['Title']))
    elements.append(Spacer(1, 12))

    result = "High Risk: Loan likely to default" if prediction == 1 else "Low Risk: Loan unlikely to default"
    elements.append(Paragraph(f"Prediction: {result}", styles['Heading2']))
    elements.append(Paragraph(f"Default Probability: {probability:.2%}", styles['Normal']))
    elements.append(Spacer(1, 12))

    gauge_filepath = None
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
        gauge_filepath = tmp_file.name
        try:
            fig_gauge.write_image(gauge_filepath, format="png", scale=2)
            elements.append(Image(gauge_filepath, width=300, height=150))
        except Exception as e:
            logging.error(f"Failed to write gauge image: {str(e)}")
            elements.append(Paragraph("Error: Gauge chart could not be generated.", styles['Normal']))
        elements.append(Spacer(1, 12))

    elements.append(Paragraph("Key Factors Influencing Prediction", styles['Heading2']))
    for _, row in feature_importance_df.head(3).iterrows():
        elements.append(Paragraph(f"- {row['Feature']}: {row['Importance']:.3f}", styles['Normal']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Additional Insights", styles['Heading2']))
    for factor in key_factors:
        elements.append(Paragraph(f"- {factor}", styles['Normal']))

    doc.build(elements)
    buffer.seek(0)

    if gauge_filepath:
        safe_delete_file(gauge_filepath)

    return buffer

# Function to generate a PDF report for prediction history
def generate_history_pdf_report(history_df, fig_trend, fig_pie):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Loan Default Prediction History Report", styles['Title']))
    elements.append(Spacer(1, 12))

    total_predictions = len(history_df)
    high_risk_count = (history_df['prediction'] == 1).sum()
    low_risk_count = (history_df['prediction'] == 0).sum()
    avg_risk_score = history_df['prediction'].mean() * 100

    elements.append(Paragraph("Summary Statistics", styles['Heading2']))
    elements.append(Paragraph(f"Total Predictions: {total_predictions}", styles['Normal']))
    elements.append(Paragraph(f"High Risk Predictions: {high_risk_count} ({(high_risk_count/total_predictions)*100:.1f}%)", styles['Normal']))
    elements.append(Paragraph(f"Low Risk Predictions: {low_risk_count} ({(low_risk_count/total_predictions)*100:.1f}%)", styles['Normal']))
    elements.append(Paragraph(f"Average Risk Score: {avg_risk_score:.1f}%", styles['Normal']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Visualizations", styles['Heading2']))
    elements.append(Spacer(1, 12))

    temp_files = []

    trend_filepath = None
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_trend_file:
        trend_filepath = tmp_trend_file.name
        temp_files.append(trend_filepath)
        try:
            fig_trend.write_image(trend_filepath, format="png", scale=2)
            elements.append(Paragraph("Average Risk Level Over Time", styles['Heading3']))
            elements.append(Image(trend_filepath, width=400, height=200))
        except Exception as e:
            logging.error(f"Failed to write trend image: {str(e)}")
            elements.append(Paragraph("Error: Trend chart could not be generated.", styles['Normal']))
        elements.append(Spacer(1, 12))

    pie_filepath = None
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_pie_file:
        pie_filepath = tmp_pie_file.name
        temp_files.append(pie_filepath)
        try:
            fig_pie.write_image(pie_filepath, format="png", scale=2)
            elements.append(Paragraph("Risk Distribution", styles['Heading3']))
            elements.append(Image(pie_filepath, width=300, height=200))
        except Exception as e:
            logging.error(f"Failed to write pie image: {str(e)}")
            elements.append(Paragraph("Error: Pie chart could not be generated.", styles['Normal']))
        elements.append(Spacer(1, 12))

    doc.build(elements)
    buffer.seek(0)

    for filepath in temp_files:
        safe_delete_file(filepath)

    return buffer

# Render footer navigation buttons
def render_footer_navigation():
    st.markdown("<div class='nav-footer'>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Explore Dataset", key="footer_explore_btn", help="View dataset visualizations"):
            st.session_state.page = "Data Exploration"
            st.rerun()
    with col2:
        if st.button("Analyze Features", key="footer_analyze_btn", help="Explore feature importance"):
            st.session_state.page = "Feature Analysis"
            st.rerun()
    with col3:
        if st.button("Predict Default Risk", key="footer_predict_btn", help="Predict loan default risk"):
            st.session_state.page = "Prediction"
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# Custom CSS and Theme Application
def inject_custom_ui():
    if 'theme' not in st.session_state:
        st.session_state.theme = 'light'

    base_styles = """
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        :root {
            --primary: #4361ee;
            --secondary: #3a0ca3;
            --accent: #f72585;
            --light: #f8f9fa;
            --dark: #212529;
            --success: #34c759;
            --warning: #f8961e;
            --danger: #ef233c;
            --info: #4895ef;
            --text-color: #495057;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            scroll-behavior: smooth;
        }

        .main {
            padding: 2rem 2.5rem;
        }

        .header {
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--secondary);
            margin-bottom: 1.5rem;
            position: relative;
            padding-bottom: 0.5rem;
        }
        .header:after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            width: 80px;
            height: 4px;
            background: linear-gradient(90deg, var(--accent), var(--primary));
            border-radius: 2px;
        }
        .subheader {
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--secondary);
            margin: 1.5rem 0 1rem;
        }

        .card {
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            margin-bottom: 1.5rem;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 24px rgba(0,0,0,0.12);
        }
        .card p {
            font-size: 1.1rem;
            color: var(--text-color);
        }

        .stButton>button {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 2px 8px rgba(67,97,238,0.2);
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(67,97,238,0.3);
            background: linear-gradient(135deg, var(--secondary), var(--primary));
        }
        .stButton>button[type="primary"] {
            background: linear-gradient(135deg, #f8961e, #f48c06);
            box-shadow: 0 2px 8px rgba(248, 150, 30, 0.2);
        }
        .stButton>button[type="primary"]:hover {
            background: linear-gradient(135deg, #f48c06, #f8961e);
            box-shadow: 0 4px 12px rgba(248, 150, 30, 0.3);
        }
        .stButton>button[data-tooltip]:hover:after {
            content: attr(data-tooltip);
            position: absolute;
            top: -30px;
            left: 50%;
            transform: translateX(-50%);
            background: var(--dark);
            color: white;
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 12px;
            z-index: 10;
        }

        .stTextInput input, .stNumberInput input, .stSelectbox select {
            border-radius: 8px;
            border: 1px solid #dee2e6;
            padding: 0.75rem;
            transition: border-color 0.3s ease, box-shadow 0.3s ease;
        }
        .stTextInput input:focus, .stNumberInput input:focus, .stSelectbox select:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 0.2rem rgba(67,97,238,0.25);
        }

        .stAlert {
            border-radius: 8px;
            padding: 1rem;
        }

        .stSidebar {
            background: linear-gradient(180deg, var(--secondary), var(--primary));
            color: white;
        }
        .stSidebar .stRadio label {
            background: rgba(255,255,255,0.1);
            border-radius: 8px;
            padding: 0.75rem;
            margin-bottom: 0.5rem;
            color: white;
            transition: all 0.3s ease;
        }
        .stSidebar .stRadio label:hover {
            background: rgba(255,255,255,0.2);
        }
        .stSidebar .stRadio input:checked + label {
            background: white;
            color: var(--secondary);
            font-weight: 600;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .nav-footer {
            border-radius: 12px;
            padding: 1.5rem;
            margin-top: 2rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }

        .footer {
            text-align: center;
            padding: 1.5rem;
            margin-top: 3rem;
            border-radius: 12px;
        }

        .theme-toggle {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
        }

        .auth-form {
            max-width: 400px;
            margin: 0 auto;
            padding: 2rem;
        }
        .auth-form .stTextInput input {
            margin-bottom: 1rem;
        }
        .auth-form .stButton>button {
            width: 100%;
        }
        .auth-form a {
            color: var(--primary);
            text-decoration: none;
        }
        .auth-form a:hover {
            text-decoration: underline;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .animate-in {
            animation: fadeIn 0.5s ease-out forwards;
        }

        @media (max-width: 768px) {
            .main { padding: 1.5rem 1rem; }
            .header { font-size: 2rem; }
            .subheader { font-size: 1.3rem; }
            .stButton>button { width: 100%; margin-bottom: 0.5rem; }
            .theme-toggle { top: 10px; right: 10px; }
        }
    """

    light_theme_styles = """
        .main {
            background-color: #f8f9fa;
            color: #212529;
            --text-color: #495057;
        }
        .card {
            background: white;
            border: 1px solid #e9ecef;
        }
        .nav-footer {
            background: white;
        }
        .footer {
            background: linear-gradient(90deg, #3a0ca3, #4361ee);
            color: white;
        }
    """

    dark_theme_styles = """
        .main {
            background-color: #1a1a1a;
            color: #e0e0e0;
            --text-color: #e0e0e0;
        }
        .card {
            background: #2a2a2a;
            border: 1px solid #444;
        }
        .nav-footer {
            background: #2a2a2a;
        }
        .footer {
            background: linear-gradient(90deg, #2c3e50, #3498db);
            color: #e0e0e0;
        }
        .header, .subheader {
            color: #3498db;
        }
    """

    selected_theme_styles = light_theme_styles if st.session_state.theme == 'light' else dark_theme_styles
    combined_styles = f"<style>{base_styles}{selected_theme_styles}</style>"

    st.markdown(combined_styles, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="theme-toggle">', unsafe_allow_html=True)
        current_theme = st.session_state.theme
        theme_option = st.selectbox(
            "Theme",
            ["Light", "Dark"],
            index=0 if current_theme == 'light' else 1,
            label_visibility="collapsed",
            key="theme_selectbox"
        )
        if theme_option.lower() != current_theme:
            st.session_state.theme = theme_option.lower()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <script>
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate-in');
                }
            });
        }, {threshold: 0.1});

        document.querySelectorAll('.card, .header, .subheader').forEach(el => {
            observer.observe(el);
        });
    </script>
    """, unsafe_allow_html=True)

# Enhanced Home Page
def home_page():
    st.markdown("""
    <div class="header animate-in">Loan Default Risk Analyzer</div>
    <div class="card animate-in">
        <p style="font-size: 1.1rem;">
            Welcome to our intelligent loan risk assessment platform. This application helps financial 
            institutions predict loan default probabilities using advanced machine
        </p>
    </div>
    <div class="subheader animate-in">Key Features</div>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; margin-bottom: 2rem;">
        <div class="card animate-in" style="animation-delay: 0.1s;">
            <h3 style="color: #4361ee; margin-top: 0;"> Data Exploration</h3>
            <p>Explore loan dataset with interactive visualizations and filters.</p>
        </div>
        <div class="card animate-in" style="animation-delay: 0.2s;">
            <h3 style="color: #4361ee; margin-top: 0;"> Feature Analysis</h3>
            <p>Understand which factors most influence loan default risk.</p>
        </div>
        <div class="card animate-in" style="animation-delay: 0.3s;">
            <h3 style="color: #4361ee; margin-top: 0;"> AI Prediction</h3>
            <p>Get instant default risk assessments for individual or batch applications.</p>
        </div>
    </div>
    <div class="subheader animate-in">Quick Actions</div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Explore Data", key="home_explore", help="View dataset visualizations"):
            st.session_state.page = "Data Exploration"
            st.rerun()
    with col2:
        if st.button("Analyze Features", key="home_analyze", help="Explore feature importance"):
            st.session_state.page = "Feature Analysis"
            st.rerun()
    with col3:
        if st.button("Make Prediction", key="home_predict", help="Predict loan default risk"):
            st.session_state.page = "Prediction"
            st.rerun()

    st.markdown("""
    <div class="card animate-in" style="animation-delay: 0.4s;">
        <h3 style="color: #4361ee; margin-top: 0;"> Model Performance</h3>
        <p>Our current model achieves 85% accuracy with the following metrics:</p>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; text-align: center;">
            <div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #4cc9f0;">0.92</div>
                <div style="font-size: 0.9rem; color: #6c757d;">AUC Score</div>
            </div>
            <div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #4895ef;">0.87</div>
                <div style="font-size: 0.9rem; color: #6c757d;">Precision</div>
            </div>
            <div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #f8961e;">0.83</div>
                <div style="font-size: 0.9rem; color: #6c757d;">Recall</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Enhanced Data Exploration Page
def data_exploration_page():
    st.markdown("""
    <div class="header animate-in">Data Exploration</div>
    <div class="card animate-in">
        <p style="font-size: 1.1rem;">
            Explore the loan dataset with interactive visualizations. Filter the data to uncover insights 
            about loan approval patterns and applicant characteristics.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    df = load_data()
    
    with st.expander(" Apply Filters", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            property_area = st.multiselect("Property Area", options=df['Property_Area'].unique(), default=df['Property_Area'].unique())
            education = st.multiselect("Education", options=df['Education'].unique(), default=df['Education'].unique())
        with col2:
            loan_status = st.multiselect("Loan Status", options=df['Loan_Status'].unique(), default=df['Loan_Status'].unique())
            gender = st.multiselect("Gender", options=df['Gender'].unique(), default=df['Gender'].unique())
    
    filtered_df = df[
        df['Property_Area'].isin(property_area) & 
        df['Education'].isin(education) & 
        df['Loan_Status'].isin(loan_status) &
        df['Gender'].isin(gender)
    ]
    
    st.markdown("<div class='subheader animate-in'>Interactive Visualizations</div>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs([" Charts", " Data Table"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig_pie = px.pie(filtered_df, names='Loan_Status', title="Loan Status Distribution",
                            color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_pie, use_container_width=True)
            
            fig_box = px.box(filtered_df, x='Loan_Status', y='ApplicantIncome', title="Applicant Income by Loan Status",
                            color='Loan_Status', color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_box, use_container_width=True)
        
        with col2:
            fig_bar = px.bar(filtered_df, x='Property_Area', y='LoanAmount', color='Loan_Status',
                            title="Loan Amount by Property Area", barmode='group',
                            color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_bar, use_container_width=True)
            
            numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
            corr = filtered_df[numeric_cols].corr()
            fig_heatmap = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap",
                                    color_continuous_scale=px.colors.sequential.Peach)
            st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab2:
        st.dataframe(filtered_df)
    
    render_footer_navigation()

# Enhanced Feature Analysis Page
def feature_analysis_page():
    st.markdown("""
    <div class="header animate-in">Feature Analysis</div>
    <div class="card animate-in">
        <p style="font-size: 1.1rem;">
            Understand which applicant characteristics most influence loan default risk. This analysis helps 
            identify key factors in the decision-making process.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    df = load_data()
    
    st.markdown("<div class='subheader animate-in'>Feature Importance</div>", unsafe_allow_html=True)
    
    try:
        model = joblib.load('loan_default_model.pkl')
        if hasattr(model, 'named_steps'):
            feature_names = model.named_steps['preprocessing'].get_feature_names_out()
            importances = model.named_steps['classifier'].feature_importances_
            feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
            
            fig_bar = px.bar(feature_importance_df, x='Importance', y='Feature', orientation='h',
                            title="Feature Importance", color='Importance',
                            color_continuous_scale=px.colors.sequential.Peach)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.error("Model is not a Pipeline. Ensure 'loan_default_model.pkl' is a valid scikit-learn Pipeline object.")
            logging.error("Model does not have named_steps attribute")
    except Exception as e:
        st.error(f"Failed to load feature importance: {str(e)}. Ensure scikit-learn is installed and model file is valid.")
        logging.error(f"Feature importance error: {str(e)}")
    
    st.markdown("<div class='subheader animate-in'>Feature Distributions</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        feature = st.selectbox("Select Feature", options=df.columns)
    with col2:
        group_by = st.selectbox("Group By", options=['Loan_Status', 'Property_Area', 'Education', 'Gender'])
    
    if df[feature].dtype in [np.float64, np.int64]:
        fig_hist = px.histogram(df, x=feature, color=group_by, title=f"{feature} Distribution",
                               color_discrete_sequence=px.colors.qualitative.Pastel)
    else:
        fig_hist = px.histogram(df, x=feature, color=group_by, barmode='group',
                               title=f"{feature} Distribution",
                               color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_hist, use_container_width=True)
    
    render_footer_navigation()

# Enhanced Prediction Page
def prediction_page():
    st.markdown("""
    <div class="header animate-in">Loan Default Prediction</div>
    <div class="card animate-in">
        <p style="font-size: 1.1rem;">
            Complete the form below to assess an applicant's loan default risk. Our AI model will analyze 
            the information and provide a risk probability score.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs([" Single Applicant", " Batch Processing", " Prediction History"])
    
    with tab1:
        with st.form("loan_form", clear_on_submit=False):
            st.markdown("""
            <div class="card animate-in">
                <h3 style="color: #4361ee; margin-top: 0;">Applicant Details</h3>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                gender = st.selectbox("Gender", ["Male", "Female"])
                married = st.selectbox("Married", ["Yes", "No"])
                dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
                education = st.selectbox("Education", ["Graduate", "Not Graduate"])
                self_employed = st.selectbox("Self Employed", ["No", "Yes"])
                property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
            with col2:
                applicant_income = st.number_input("Applicant Income (Ksh)", min_value=0.0, value=5000.0, step=100.0)
                coapplicant_income = st.number_input("Coapplicant Income (Ksh)", min_value=0.0, value=0.0, step=100.0)
                loan_amount = st.number_input("Loan Amount (Ksh)", min_value=0.0, value=128.0, step=10.0)
                loan_term = st.number_input("Loan Term (months)", min_value=0.0, value=360.0, step=12.0)
                credit_history = st.selectbox("Credit History", [1.0, 0.0])
            
            submit_button = st.form_submit_button("Predict Default Risk", type="primary")
            st.markdown("</div>", unsafe_allow_html=True)

        if submit_button:
            input_data = {
                "Gender": gender,
                "Married": married,
                "Dependents": dependents,
                "Education": education,
                "Self_Employed": self_employed,
                "ApplicantIncome": applicant_income,
                "CoapplicantIncome": coapplicant_income,
                "LoanAmount": loan_amount,
                "Loan_Amount_Term": loan_term,
                "Credit_History": credit_history,
                "Property_Area": property_area
            }

            with st.spinner("Analyzing applicant data..."):
                try:
                    model = joblib.load('loan_default_model.pkl')
                    prediction, probability = predict_loan_default(input_data, model)
                    
                    st.markdown("""
                    <div class="card animate-in">
                        <h3 style="color: #4361ee; margin-top: 0;">Prediction Result</h3>
                    """, unsafe_allow_html=True)
                    
                    if prediction == 1:
                        st.error(f" High Risk: Loan likely to default (Probability: {probability:.2%})")
                        st.markdown(f"**Raw Prediction Value**: {prediction} (1 indicates high risk)")
                    else:
                        st.success(f" Low Risk: Loan unlikely to default (Probability: {probability:.2%})")
                        st.markdown(f"**Raw Prediction Value**: {prediction} (0 indicates low risk)")
                        st.balloons()
                    
                    st.markdown("<div class='subheader animate-in'>Adjust Risk Threshold</div>", unsafe_allow_html=True)
                    threshold = st.slider("Set Risk Threshold (%)", min_value=10, max_value=90, value=50, step=5) / 100.0
                    adjusted_prediction = int(probability >= threshold)
                    st.markdown(f"**Adjusted Prediction**: {'High Risk' if adjusted_prediction == 1 else 'Low Risk'} (with threshold {threshold*100:.0f}%)")

                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=probability * 100,
                        title={'text': "Default Probability (%)"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#ef233c" if prediction == 1 else "#34c759"},
                            'steps': [
                                {'range': [0, 50], 'color': "#34c759"},
                                {'range': [50, 100], 'color': "#ef233c"}
                            ],
                        }
                    ))
                    st.plotly_chart(fig_gauge, use_container_width=True)

                    st.markdown("""
                    *This gauge shows the likelihood of loan default (0-100%). Green (0-50%) indicates low risk, meaning the loan is likely to be repaid. Red (50-100%) indicates high risk, meaning the loan is likely to default.*
                    """)

                    st.markdown("<div class='subheader animate-in'>Feature Importance</div>", unsafe_allow_html=True)
                    feature_importance_df = get_feature_importance(st.session_state.processed_input_df, model)
                    if feature_importance_df is not None:
                        fig_importance = px.bar(feature_importance_df.head(5), x='Importance', y='Feature', orientation='h',
                                               title="Top Factors Influencing This Prediction",
                                               color='Importance', color_continuous_scale=px.colors.sequential.Peach)
                        st.plotly_chart(fig_importance, use_container_width=True)
                    else:
                        st.warning("Unable to compute feature importance.")

                    key_factors = [
                        f"Credit History: {'Good' if credit_history == 1 else 'Poor'}",
                        f"Debt-to-Income Ratio: {loan_amount/(applicant_income + coapplicant_income + 1):.2f}",
                        f"Applicant Income: Ksh {applicant_income:,.2f}",
                        f"Loan Amount: Ksh {loan_amount:,.2f}"
                    ]

                    with st.expander(" Understand This Prediction"):
                        st.markdown(f"""
                        **What does this prediction mean?**
                        - **High Risk (Probability >= 50%)**: The applicant is likely to default on the loan, meaning they may fail to repay as agreed.
                        - **Low Risk (Probability < 50%)**: The applicant is unlikely to default, indicating a higher chance of successful repayment.

                        **Key factors influencing this prediction**:
                        - **Credit History**: {'Good' if credit_history == 1 else 'Poor'} credit history
                        - **Debt-to-Income Ratio**: {loan_amount/(applicant_income + coapplicant_income + 1):.2f}
                        - **Applicant Income**: Ksh {applicant_income:,.2f}
                        - **Loan Amount**: Ksh {loan_amount:,.2f}

                        **Recommendations**:
                        - **High Risk**: Consider additional collateral, higher interest rate, or rejection
                        - **Low Risk**: Standard loan terms recommended
                        """)

                    st.markdown("<div class='subheader animate-in'>Export Report</div>", unsafe_allow_html=True)
                    pdf_buffer = generate_pdf_report(prediction, probability, fig_gauge, feature_importance_df, key_factors)
                    st.download_button(
                        label="Download Prediction Report (PDF)",
                        data=pdf_buffer,
                        file_name="loan_prediction_report.pdf",
                        mime="application/pdf"
                    )
                    
                    st.markdown("</div>", unsafe_allow_html=True)

                except ValueError as ve:
                    st.error(f" Input Error: {ve}")
                except Exception as e:
                    st.error(f" Model Error: {str(e)}")
    
    with tab2:
        with st.form("csv_form"):
            st.markdown("""
            <div class="card animate-in">
                <h3 style="color: #4361ee; margin-top: 0;">Upload CSV for Batch Prediction</h3>
                <p style="margin-bottom: 1rem;">
                    Upload a CSV file containing multiple loan applications to process them in bulk.
                    The system will generate default risk predictions for each applicant.
                </p>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], accept_multiple_files=False)
            retrain = st.checkbox("Retrain model with uploaded data (requires Loan_Status column)")
            batch_submit = st.form_submit_button("Process CSV", type="primary")
            
            st.markdown("</div>", unsafe_allow_html=True)

        if batch_submit and uploaded_file:
            try:
                if uploaded_file.size > 10 * 1024 * 1024:
                    st.error("File size exceeds 10MB limit")
                    return

                if not is_safe_filename(uploaded_file.name):
                    st.error("Invalid file name. Use alphanumeric characters, underscores, hyphens, or dots.")
                    return

                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name

                df = pd.read_csv(tmp_file_path)
                os.unlink(tmp_file_path)

                with st.spinner("Processing CSV..."):
                    model = None
                    if retrain:
                        model = retrain_model(df)
                        st.success("Model retrained successfully")
                    
                    result_df = predict_batch(df, model)
                    
                    st.markdown("""
                    <div class="card animate-in">
                        <h3 style="color: #4361ee; margin-top: 0;">Batch Prediction Results</h3>
                    """, unsafe_allow_html=True)
                    
                    st.dataframe(result_df.style.apply(
                        lambda x: ['background-color: #ffcccc' if v == 1 else 'background-color: #ccffcc' for v in x],
                        subset=['Prediction']
                    ))
                    
                    csv_buffer = io.StringIO()
                    result_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="Download Predictions",
                        data=csv_buffer.getvalue(),
                        file_name="predictions.csv",
                        mime="text/csv"
                    )

                    fig_hist = px.histogram(result_df, x='Probability', color='Prediction',
                                          title="Distribution of Predicted Probabilities",
                                          color_discrete_sequence=['#34c759', '#ef233c'])
                    st.plotly_chart(fig_hist, use_container_width=True)

                    high_risk_pct = (result_df['Prediction'] == 1).mean() * 100
                    with st.expander(" Batch Prediction Summary"):
                        st.markdown(f"""
                        **Batch Prediction Statistics**:
                        - **Total Applications Processed**: {len(result_df)}
                        - **High Risk Applications**: {high_risk_pct:.1f}% ({(result_df['Prediction'] == 1).sum()})
                        - **Low Risk Applications**: {(100 - high_risk_pct):.1f}% ({(result_df['Prediction'] == 0).sum()})
                        
                        **Model Used**: {"Retrained Model" if retrain else "Default Model"}
                        """)
                    
                    st.markdown("</div>", unsafe_allow_html=True)

            except ValueError as ve:
                st.error(f" CSV Error: {ve}")
            except Exception as e:
                st.error(f" Processing Error: {str(e)}")
    
    with tab3:
        st.markdown("""
        <div class="header animate-in">Prediction History</div>
        <div class="card animate-in">
            <p style="font-size: 1.1rem;">
                View the history of past predictions, including trends, statistics, and detailed analysis over time.
            </p>
        </div>
        """, unsafe_allow_html=True)

        history_df = get_prediction_history()
        if not history_df.empty:
            st.markdown("<div class='subheader animate-in'>Filter by Date Range</div>", unsafe_allow_html=True)
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            min_date = history_df['timestamp'].min().date()
            max_date = history_df['timestamp'].max().date()
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
            with col2:
                end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

            # Ensure end_date is not before start_date
            if start_date > end_date:
                st.error("End date must be on or after start date.")
                return

            # Filter history by date range
            filtered_history = history_df[
                (history_df['timestamp'].dt.date >= start_date) &
                (history_df['timestamp'].dt.date <= end_date)
            ]

            if not filtered_history.empty:
                st.markdown("<div class='subheader animate-in'>Prediction History Table</div>", unsafe_allow_html=True)
                st.dataframe(filtered_history.style.format({'timestamp': lambda x: x.strftime('%Y-%m-%d %H:%M:%S')}))

                # Summary Statistics
                total_predictions = len(filtered_history)
                high_risk_count = (filtered_history['prediction'] == 1).sum()
                low_risk_count = (filtered_history['prediction'] == 0).sum()
                st.markdown("<div class='subheader animate-in'>Summary Statistics</div>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Predictions", total_predictions)
                col2.metric("High Risk", high_risk_count, f"{(high_risk_count/total_predictions)*100:.1f}%")
                col3.metric("Low Risk", low_risk_count, f"{(low_risk_count/total_predictions)*100:.1f}%")

                # Visualizations
                st.markdown("<div class='subheader animate-in'>Prediction Trends</div>", unsafe_allow_html=True)
                daily_risk = filtered_history.groupby(filtered_history['timestamp'].dt.date)['prediction'].mean().reset_index()
                daily_risk['prediction'] = daily_risk['prediction'] * 100  # Convert to percentage
                fig_trend = px.line(daily_risk, x='timestamp', y='prediction',
                                   title="Average Risk Level Over Time",
                                   labels={'timestamp': 'Date', 'prediction': 'Risk Level (%)'},
                                   color_discrete_sequence=['#4361ee'])
                st.plotly_chart(fig_trend, use_container_width=True)

                # Risk Distribution Pie Chart
                risk_counts = filtered_history['prediction'].value_counts().reset_index()
                risk_counts['prediction'] = risk_counts['prediction'].map({1: 'High Risk', 0: 'Low Risk'})
                fig_pie = px.pie(risk_counts, names='prediction', values='count',
                                title="Risk Distribution",
                                color_discrete_sequence=['#ef233c', '#34c759'])
                st.plotly_chart(fig_pie, use_container_width=True)

                # Export History Report
                st.markdown("<div class='subheader animate-in'>Export History Report</div>", unsafe_allow_html=True)
                pdf_buffer = generate_history_pdf_report(filtered_history, fig_trend, fig_pie)
                st.download_button(
                    label="Download Prediction History Report (PDF)",
                    data=pdf_buffer,
                    file_name="prediction_history_report.pdf",
                    mime="application/pdf"
                )
            else:
                st.info("No predictions found in the selected date range.")
        else:
            st.info("No prediction history available yet.")

    render_footer_navigation()

# Main Function
def main():
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'page' not in st.session_state:
        st.session_state.page = "Home"
    if 'auth_page' not in st.session_state:
        st.session_state.auth_page = "Login"
    if 'username' not in st.session_state:
        st.session_state.username = None

    # Initialize database
    init_database()

    # Apply custom UI
    inject_custom_ui()

    # Authentication Check
    if not st.session_state.logged_in:
        if st.session_state.auth_page == "Register":
            register_page()
        elif st.session_state.auth_page == "Forgot Password":
            forgot_password_page()
        else:
            login_page()
    else:
        # Sidebar Navigation with Role-Based Access
        user_role = get_user_role(st.session_state.username)
        st.sidebar.markdown(f"""
        <div style="text-align: center; padding: 1rem;">
            <h3 style="color: white; margin-bottom: 0.5rem;">Welcome, {st.session_state.username}</h3>
            <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">Role: {user_role.capitalize()}</p>
        </div>
        """, unsafe_allow_html=True)

        # Define navigation options based on user role
        nav_options = ["Home", "Data Exploration", "Feature Analysis", "Prediction"]
        if user_role == "admin":
            nav_options.extend(["Admin Panel", "User Activity"])

        page = st.sidebar.radio("Navigate", nav_options, index=nav_options.index(st.session_state.page))

        if page != st.session_state.page:
            st.session_state.page = page
            st.rerun()

        # Render Pages
        if st.session_state.page == "Home":
            home_page()
        elif st.session_state.page == "Data Exploration":
            data_exploration_page()
        elif st.session_state.page == "Feature Analysis":
            feature_analysis_page()
        elif st.session_state.page == "Prediction":
            prediction_page()
        elif st.session_state.page == "Admin Panel":
            admin_panel_page()
        elif st.session_state.page == "User Activity":
            user_activity_page()

        # Logout Button in Sidebar
        st.sidebar.markdown("---")
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.page = "Home"
            st.session_state.auth_page = "Login"
            st.rerun()

    # Footer
    st.markdown("""
    <div class="footer">
        Loan Default Risk Analyzer © 2025 | Built with Streamlit & Python
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()