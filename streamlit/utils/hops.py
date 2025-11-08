"""
utils/hops.py

Cached Hopsworks connector helper for Streamlit app.

Usage:
    from utils.hops import connect_hopsworks
    project = connect_hopsworks()
    fs = project.get_feature_store()
"""

import os
from dotenv import load_dotenv
import streamlit as st

# Import hopsworks lazily so importing this module while offline doesn't crash the app immediately.
# Any errors during login will be surfaced when connect_hopsworks() is called.
try:
    import hopsworks
except Exception as e:
    hopsworks = None  # we'll handle this later


@st.cache_resource(show_spinner=False)
def connect_hopsworks():
    """
    Connect to Hopsworks and return the logged-in project object.
    This function is cached for the session lifetime — login only happens once per Streamlit session.
    """
    load_dotenv()  # ensure .env values are available

    # st.write("Attempting Hopsworks login...")


    project_name = os.getenv("PROJECT_NAME")
    api_key = os.getenv("HOPSWORKS_API_KEY")

    if not project_name or not api_key:
        st.error(
            "❌ Missing Hopsworks configuration. Make sure your `.env` contains PROJECT_NAME and HOPSWORKS_API_KEY."
        )
        st.stop()

    if hopsworks is None:
        st.error("❌ The `hopsworks` package is not available in your environment. Install it with `pip install hsfs`.")
        st.stop()

    try:
        # Attempt login. This can be slow on first call; caching ensures it's done only once.
        project = hopsworks.login(project=project_name, api_key_value=api_key)
        fs = project.get_feature_store()
        return project, fs
    except Exception as e:
        # Show a clear error and stop Streamlit execution on this page.
        st.error(f"❌ Failed to login to Hopsworks: {e}")
        st.stop()
