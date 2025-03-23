api_key = "AIzaSyAj0xbzFhkTB7uiCQ9cx8ajvZCrgSwciJk"
#paste your gemini api key from here https://aistudio.google.com/app/apikey
import pandas as pd
import sqlite3
import io
import pickle
import sqlalchemy
from sqlalchemy import create_engine, text, inspect
import json
import matplotlib.pyplot as plt
import re
import os
import time
import subprocess
import sys
import google.generativeai as genai
import gradio as gr
from PIL import Image
import glob

# Configure Gemini with the API key
genai.configure(api_key=api_key)

# Create necessary directories
os.makedirs("plots", exist_ok=True)

GUARDRAILS = {
    "blocked_entities": [],
    "blocked_combinations": [],
    "violation_message": "This query violates company policy as it requests information about a protected individual."
}

# Function to clean data
def clean_data(df):
    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Handle missing values
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    non_numeric_cols = df.select_dtypes(exclude=["number"]).columns
    df[non_numeric_cols] = df[non_numeric_cols].fillna("Unknown")

    # Remove duplicates
    df = df.drop_duplicates()

    return df

# Function to extract column info
def extract_column_info(df, sample_size=5):
    columns = df.columns.tolist()
    sample_data = {}
    for col in columns:
        # Get up to sample_size unique non-null values
        samples = df[col].dropna().unique().tolist()[:sample_size]
        # Convert non-serializable objects (like Timestamps) to string
        samples = [str(x) for x in samples]
        sample_data[col] = samples
    return columns, sample_data

# Function to extract JSON from response
def extract_json_from_response(response_text):
    match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
    return match.group(1) if match else response_text

# Function to get column insights
def get_column_insights(columns, sample_data, api_key):
    model = genai.GenerativeModel(model_name="gemini-2.0-flash")

    # Create a formatted prompt with columns and sample data
    prompt = f"""
You are provided with a list of column names and sample data from each column.
Columns: {columns}
Sample data: {json.dumps(sample_data, indent=2)}

For each column, please provide:
1. A refined column name (if needed).
2. A brief description of what the column might represent.

Return the results in **strict** JSON format as follows:
{{
    "column1": {{"refined_name": "new_column1", "description": "Description for column1"}},
    "column2": {{"refined_name": "new_column2", "description": "Description for column2"}}
}}
"""

    # Generate content using the Gemini model
    response = model.generate_content(prompt)

    # Extract the text from the response
    response_text = response.text

    # Remove markdown formatting
    response_text = extract_json_from_response(response_text)

    # Attempt to parse the JSON
    try:
        insights = json.loads(response_text)
    except Exception as e:
        print("Error parsing JSON:", e)
        insights = {}

    return insights

# Function to update dataframe columns
def update_dataframe_columns(df, insights):
    rename_mapping = {col: insights[col]['refined_name'] for col in df.columns if col in insights}
    updated_df = df.rename(columns=rename_mapping)
    return updated_df

# Function to generate table name
import os
import hashlib

def generate_table_name(file_name):
    base = os.path.basename(file_name)
    name, _ = os.path.splitext(base)
    # Option 1: Truncate the name to a maximum length (e.g., 30 characters)
    truncated_name = name[:30]
    # Option 2: Append a hash to ensure uniqueness if needed
    hash_suffix = hashlib.md5(name.encode()).hexdigest()[:8]
    return f"{truncated_name}_{hash_suffix}"


# Function to process uploaded data
def process_uploaded_data(file_obj):
    log_messages = []

    # Read uploaded file
    if file_obj is None:
        return "No file uploaded", None

    file_name = file_obj.name
    df = pd.read_csv(file_obj)
    log_messages.append(f"Uploaded: {file_name}, Shape: {df.shape}")

    # Clean data
    df = clean_data(df)
    log_messages.append(f"Cleaned: {file_name}, New Shape: {df.shape}")

    # Extract column info
    columns, sample_data = extract_column_info(df)
    log_messages.append(f"Extracted column info from {file_name}")

    # Get column insights
    insights = get_column_insights(columns, sample_data, api_key)
    log_messages.append("Generated column insights")

    # For each column that contains "date", append its first non-null value to the description
    for col in columns:
        if "date" in col.lower():
            # Get the first non-null value of the column
            non_null_series = df[col].dropna()
            if not non_null_series.empty:
                first_non_null = non_null_series.iloc[0]
                # Check if this column exists in the insights and has a description
                if col in insights and "description" in insights[col]:
                    insights[col]["description"] += f" FORMAT of date,note this during code generation : {first_non_null}"

    # Update dataframe columns
    updated_df = update_dataframe_columns(df, insights)
    log_messages.append("Updated dataframe columns with refined names")

    # Generate table name
    table_name = generate_table_name(file_name)
    log_messages.append(f"Generated table name: {table_name}")

    # Create metadata records
    metadata_records = []
    for original_col, meta in insights.items():
        refined_col = meta.get("refined_name", original_col)
        description = meta.get("description", "")
        if refined_col in updated_df.columns:
            data_type = str(updated_df[refined_col].dtype)
        else:
            data_type = "unknown"
        metadata_records.append({
            "table_name": table_name,
            "column": refined_col,
            "description": description,
            "data_type": data_type
        })

    # Convert records into a DataFrame
    metadata_df = pd.DataFrame(metadata_records)
    log_messages.append("Created metadata DataFrame")

    # Setup database connection
    log_messages.append("Setting up database connection...")
    try:
        # Connect to the database
        db_path = 'testdb.sqlite'
        engine = create_engine(f'sqlite:///{db_path}')

        # Upload data to database
        updated_df.to_sql(table_name, engine, if_exists="replace", index=False)
        log_messages.append(f"Data successfully uploaded to table '{table_name}'")

        # Upload metadata
        metadata_df.to_sql("metadata", engine, if_exists="replace", index=False)
        log_messages.append("Metadata successfully uploaded to table 'metadata'")

        return "\n".join(log_messages), updated_df
    except Exception as e:
        log_messages.append(f"Error setting up database: {str(e)}")
        return "\n".join(log_messages), None

def check_guardrails(query):
    query_lower = query.lower()
    
    # Print debugging information
    print(f"Checking query against guardrails: '{query_lower}'")
    print(f"Current blocked entities: {GUARDRAILS['blocked_entities']}")
    
    # Check for direct mentions of blocked entities
    for entity in GUARDRAILS["blocked_entities"]:
        print(f"Checking if '{entity}' is in query")
        if entity in query_lower:
            print(f"BLOCKED: Found '{entity}' in query")
            return False, GUARDRAILS["violation_message"]
    
    print("Query passed guardrail checks")
    return True, None
# Define routing logic for models
def score_query_complexity(query):
    """
    Compute a simple complexity score for a query.
    The score is based on the number of words and a boost for certain complexity-indicating keywords.
    """
    with open('query_complexity_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    # Use the model to make predictions
    query = "analyze sales trends by region"
    predicted_complexity = model.predict([query])[0]
    print(f"Predicted complexity: {predicted_complexity:.2f}") # Increased the boost for complexity

    return predicted_complexity

def route_query(query):
    """
    Route the query to an appropriate model based on keyword overrides and complexity.

    Overrides:
    - If the query contains "okay" (as a standalone word), always choose deepseek-r1.
    - If the query contains "ok" (as a standalone word) and not "okay", always choose gemma2b.

    Otherwise, use complexity:
    - For low complexity: choose deepseek-r1 (the best).
    - For medium complexity: choose llama3.
    - For high complexity: choose gemma2b.
    """
    # Define thresholds for complexity scoring
    LOW_THRESHOLD = 5  # Simple queries
    MEDIUM_THRESHOLD = 8  # Medium complexity

    # Lowercase word list for keyword matching
    words = query.lower().split()

    # Check override keywords first
    

    # Compute complexity score if no override applies
    score = score_query_complexity(query)

    # Route based on score and quality ranking
    if score <= LOW_THRESHOLD:
        selected_model = "gemma2b"
        reason = f"Query complexity score: {score} (Low) - routing to gemma2b"
    elif score <= MEDIUM_THRESHOLD:
        selected_model = "llama3"
        reason = f"Query complexity score: {score} (Medium) - routing to Llama 3"
    else:
        selected_model = "deepseek-r1"
        reason = f"Query complexity score: {score} (High) - routing to deepseek-r1"

    return selected_model, reason

# Function to run analysis
def run_analysis(user_query):
    output_buffer = []
    execution_output = []  # To capture printed output from code execution

    def log(message):
        output_buffer.append(message)
        return "\n".join(output_buffer)
    
    # Check guardrails - this was indented correctly now
    passes_guardrails, violation_message = check_guardrails(user_query)
    if not passes_guardrails:
        log("⚠️ Guardrail Violation")
        log(f"Message: {violation_message}")
        return "\n".join(output_buffer), None, "N/A", "Query blocked by guardrails", ""

    # Route the query to the appropriate model
    selected_model, routing_reason = route_query(user_query)
    log(f"🔄 Model Selection: {selected_model}")
    log(f"🧠 Routing Logic: {routing_reason}")
    log("-----------------------------------")

    # -----------------------------------------------------------------------------
    # Step 1: Connect to SQLite database and retrieve metadata
    # -----------------------------------------------------------------------------
    log("Connecting to SQLite database and retrieving metadata...")

    db_path = 'testdb.sqlite'
    engine = create_engine(f'sqlite:///{db_path}')

    inspector = inspect(engine)
    tables = inspector.get_table_names()
    metadata_table_name = "metadata" if "metadata" in tables else None

    if metadata_table_name is None:
        log("No metadata table found in the database!")
        metadata_json = "[]"
    else:
        query = f"SELECT * FROM `{metadata_table_name}`"
        metadata_df = pd.read_sql(query, engine)
        metadata_json = json.dumps(metadata_df.to_dict(orient="records"), indent=2)

    # -----------------------------------------------------------------------------
    # Step 2: Generate EDA Plan
    # -----------------------------------------------------------------------------
    def generate_eda_plan():
        log("Generating EDA plan...")
        model = genai.GenerativeModel(model_name='gemini-2.0-flash')
        prompt_plan = (
            f"User Query:\n{user_query}\n\n"
            f"Database Metadata (JSON format):\n{metadata_json}\n\n"
            "Generate a step-by-step EDA plan that uses the metadata to correctly access the database and perform the analysis."
        )
        response_plan = model.generate_content(prompt_plan)
        return response_plan.text.strip()

    eda_plan = generate_eda_plan()
    log("Generated EDA Plan:\n" + eda_plan)

    # -----------------------------------------------------------------------------
    # Step 3: Generate and Execute Code
    # -----------------------------------------------------------------------------
    def generate_code(eda_plan, previous_code=None, error_message=None):
        model = genai.GenerativeModel(model_name='gemini-2.0-flash')
        if previous_code and error_message:
            log("Previous attempt failed - regenerating code...")
            prompt_code = (
                f"\n\nError:\n{error_message}\n\nFailed Code:\n```python\n{previous_code}\n```\n"
                "Please correct the error and regenerate the code. Return ONLY the code in a code block."
            )
        else:
            prompt_code = (
                f"User Query:\n{user_query}\n\n"
                f"Database Metadata:\n{metadata_json}\n\n"
                f"EDA Plan:\n{eda_plan}\n\n"
                "Database connection:\n"
                "```python\n"
                f"engine = create_engine('sqlite:///testdb.sqlite')\n"
                "```\n\n"
                "Convert this to Python code that strictly follows the EDA plan. "
                "Important: Always use the refined_column names from the metadata, not the original column names.\n"
                "Important: When using table names in SQL queries with SQLite, always wrap table names in double quotes to handle spaces and special characters.\n"
                "For example, use: SELECT * FROM \"Table Name\" instead of SELECT * FROM Table Name.\n"
                "Use a unique filename for each plot with timestamp, e.g.: plt.savefig(f'plots/visualization_{int(time.time())}.png')\n"
                "Make sure to close any figures after saving them with plt.close()\n"
                "Save plots in 'plots/' folder. Return ONLY the code in a code block."
            )

        response_code = model.generate_content(prompt_code)
        generated_code_raw = response_code.text.strip()
        match = re.search(r"```(.*?)```", generated_code_raw, re.DOTALL)
        return match.group(1).strip() if match else generated_code_raw

    def execute_code(code):
        log("Executing generated code...")
        execution_output.clear()

        script_path = "temp_script.py"
        with open(script_path, "w") as f:
            f.write(code)

        try:
            result = subprocess.run([sys.executable, script_path],
                                  capture_output=True, text=True, check=True)
            if result.stdout:
                execution_output.append(result.stdout)
            return None  # No error
        except subprocess.CalledProcessError as e:
            return f"Error: {e.stderr}\nOutput before error: {e.stdout}"
        finally:
            if os.path.exists(script_path):
                os.remove(script_path)

    # Execution with retry logic
    max_attempts = 5
    attempt = 0
    generated_code = generate_code(eda_plan)
    log("\nGenerated Code:\n" + generated_code)

    while attempt < max_attempts:
        lines = generated_code.splitlines()
        cleaned_code = "\n".join(lines[1:]) if lines and lines[0].strip().lower() == "python" else generated_code

        error_message = execute_code(cleaned_code)
        if error_message is None:
            log("Execution succeeded!")
            break
        else:
            log(f"\nAttempt {attempt+1} failed. Error:\n{error_message}")
            time.sleep(2 ** attempt)
            generated_code = generate_code(eda_plan, previous_code=generated_code, error_message=error_message)
            log("\nRegenerated Code:\n" + generated_code)
            attempt += 1

    # Get results
    plot_image = None
    plot_files = glob.glob("plots/*.png")
    if plot_files:
        latest_plot = max(plot_files, key=os.path.getctime)
        plot_image = Image.open(latest_plot)

    # Combine all output
    full_output = "\n".join(output_buffer)
    execution_text = "\n".join(execution_output)

    return full_output, plot_image, selected_model, routing_reason, execution_text

# Function to add guardrails
def add_guardrail(guardrail_text):
    if not guardrail_text or guardrail_text.strip() == "":
        return "❌ Please enter entities to block."
    
    # Parse the input text
    entities = [entity.strip().lower() for entity in guardrail_text.split(",") if entity.strip()]
    
    if not entities:
        return "❌ No valid entities found."
    
    # Track which entities were newly added
    added_entities = []
    already_blocked = []
    
    # Add entities to the blocked list
    for entity in entities:
        if entity not in GUARDRAILS["blocked_entities"]:
            GUARDRAILS["blocked_entities"].append(entity)
            added_entities.append(entity)
        else:
            already_blocked.append(entity)
    
    # Generate status message
    if added_entities:
        if len(added_entities) == 1:
            status = f"✅ Added guardrail for: {added_entities[0]}"
        else:
            status = f"✅ Added {len(added_entities)} guardrails: {', '.join(added_entities)}"
    else:
        status = "⚠️ No new entities added."
    
    if already_blocked:
        if len(already_blocked) == 1:
            status += f"\n⚠️ Note: '{already_blocked[0]}' was already blocked."
        else:
            status += f"\n⚠️ Note: The following entities were already blocked: {', '.join(already_blocked)}"
    
    # Add debug info to show current blocklist
    status += f"\n\nCurrent blocklist: {', '.join(GUARDRAILS['blocked_entities'])}"
    
    return status

# Create Gradio interface
def create_interface():
    with gr.Blocks() as demo:
        # Hidden main interface components
        with gr.Column(visible=False) as main_interface:
            gr.Markdown("# Automated EDA with AI")

            with gr.Row():
                with gr.Column():
                    query_input = gr.Textbox(label="Analysis Query", lines=3,
                                           placeholder="e.g., 'Analyze sales trends by product category'")
                    submit_btn = gr.Button("Run Analysis", variant="primary")

                    # Model routing info (initially hidden)
                    with gr.Column(visible=False) as model_info:
                        gr.Markdown("## Query Routing")
                        model_name = gr.Textbox(label="Selected Model", interactive=False)
                        routing_reason = gr.Textbox(label="Routing Logic", interactive=False)

                    with gr.Accordion("Guardrail Management", open=False):
                        guardrails_input = gr.Textbox(
                            label="Input Guardrails (comma-separated list of restricted entities)", 
                            placeholder="e.g., Jehu Rudeforth, confidential sales, credit card",
                            lines=2
                        )
                        gr.Markdown("_Queries containing these terms will be blocked with a policy violation message_")
                        add_guardrail_btn = gr.Button("Add Guardrail", variant="secondary")
                        guardrails_status = gr.Markdown("", elem_id="guardrails-status")

                with gr.Column():
                    output_text = gr.Textbox(label="Analysis Log", lines=10, interactive=False)
                    execution_output_box = gr.Textbox(label="Execution Output", lines=4, interactive=False)
                    output_image = gr.Image(label="Generated Visualization", type="pil")

        # Intro screen components
        with gr.Column(visible=True) as intro_screen:
            gr.Markdown("# Welcome to Insight AI", elem_classes="big-header")
            gr.Markdown("### Insight AI turns your business data into a conversation—just ask, and it instantly analyzes, visualizes, and uncovers hidden insights for smarter decisions!")
            html_code = """
<div style="text-align: center;">
    <img src="https://i.etsystatic.com/7692591/r/il/15c311/5341239379/il_570xN.5341239379_jbqa.jpg" width="300" alt="Sample Image">
</div>
"""
            with gr.Row():
                with gr.Column(scale=2):
                    gr.HTML(html_code)
                with gr.Column(scale=3):
                    gr.Markdown("### Our AI Engine features multi-model intelligence")
                    gr.Markdown("- **Gemma2b**: Optimized for simple queries")
                    gr.Markdown("- **Llama 3**: Best for medium complexity analysis")
                    gr.Markdown("- **DeepSeek-R1**: Handles complex, detailed analysis")
                    file_upload = gr.File(label="Upload Your Data (CSV)")
                    initialize_btn = gr.Button("Initialize AI Engine", variant="primary")

        # Loading screen component
        with gr.Column(visible=False) as loading_screen:
            gr.Markdown("## Initializing Your AI Engine...")
            gr.Markdown("Downloading and installing model components (this may take a few moments)")
            loading_animation = gr.Image("https://i.gifer.com/ZZ5H.gif",
                                       label="Loading Animation")
            processing_log = gr.Textbox(label="Processing Log", lines=10, interactive=False)

        # Data processing and transition logic
        def process_data_and_transition(file_obj):
            # Show loading screen first
            yield {
                intro_screen: gr.Column(visible=False),
                loading_screen: gr.Column(visible=True),
                processing_log: "Starting data processing..."
            }

            # Process the data
            if file_obj is not None:
                log_output, df = process_uploaded_data(file_obj)
                yield {
                    processing_log: log_output
                }

                # Add a delay to simulate processing
                time.sleep(3)

                if df is not None:
                    yield {
                        loading_screen: gr.Column(visible=False),
                        main_interface: gr.Column(visible=True)
                    }
                else:
                    yield {
                        processing_log: log_output + "\n\nError processing data. Please check the log and try again."
                    }
            else:
                yield {
                    processing_log: "No file uploaded. Please upload a CSV file and try again."
                }
                # Wait a few seconds then go back to intro screen
                time.sleep(5)
                yield {
                    loading_screen: gr.Column(visible=False),
                    intro_screen: gr.Column(visible=True)
                }

        # Connect the button to the process_data_and_transition function
        initialize_btn.click(
            fn=process_data_and_transition,
            inputs=[file_upload],
            outputs=[intro_screen, loading_screen, processing_log, main_interface]
        )

        # Function to handle analysis and show model routing
        def run_analysis_and_show_routing(query):
            # First, make the model info visible
            yield {
                model_info: gr.Column(visible=True),
                model_name: "Analyzing query...",
                routing_reason: "Determining optimal model...",
                output_text: "Starting analysis...",
                execution_output_box: "",  # Clear previous execution output
                output_image: None
            }

            # Then run the actual analysis
            output, image, model, reason, execution_text = run_analysis(query)

            yield {
                model_name: model.upper(),
                routing_reason: reason,
                output_text: output,
                execution_output_box: execution_text,
                output_image: image
            }

        # Connect the analysis button to the new function
        submit_btn.click(
            fn=run_analysis_and_show_routing,
            inputs=query_input,
            outputs=[model_info, model_name, routing_reason, output_text, execution_output_box, output_image]
        )

        # Connect the guardrail button to the add_guardrail function
        add_guardrail_btn.click(
            fn=add_guardrail,
            inputs=[guardrails_input],
            outputs=[guardrails_status]
        )

    return demo

# Add custom CSS for better appearance
css = """


button {
    background-color: #4CAF50 !important; /* Material Green */
    color: white !important;
    border: none !important;
    border-radius: 12px !important; /* Rounded corners */
    padding: 10px 20px !important;
    font-size: 16px !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2) !important; /* Shadow effect */
    transition: all 0.3s ease-in-out !important;
}

button:hover {
    background-color: #388E3C !important; /* Darker green on hover */
    box-shadow: 0px 6px 10px rgba(0, 0, 0, 0.3) !important; /* Enhanced shadow on hover */
}

button:active {
    background-color: #2E7D32 !important; /* Even darker on click */
    box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.2) !important;
}



"""

# Launch the app
demo = create_interface()
demo.css = css
demo.launch(share=True, debug=True)