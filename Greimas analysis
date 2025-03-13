#Greimas analysis
import openai
import pandas as pd
import json
import re
import time
from google.colab import files
import os

# Include your OpenAI API key here
openai.api_key = 'sk-proj-'

# Step 1: Upload and load the Excel file
uploaded = files.upload()  # Upload the file
input_file_path = list(uploaded.keys())[0]  # Get the file name from the uploaded files

# Step 2: Load the file as a pandas DataFrame
file_extension = os.path.splitext(input_file_path)[1].lower()  # Get the file extension

try:
    if file_extension == '.csv':
        # Try to load as CSV with UTF-8 encoding first
        try:
            df = pd.read_csv(input_file_path, encoding='utf-8-sig')  # Default to UTF-8
        except UnicodeDecodeError:
            print("UTF-8 encoding failed, trying ISO-8859-1.")
            df = pd.read_csv(input_file_path, encoding='ISO-8859-1')  # Fallback to ISO-8859-1
    elif file_extension in ['.xls', '.xlsx']:
        # Load as Excel file
        df = pd.read_excel(input_file_path, engine='openpyxl')  # Use 'openpyxl' for .xlsx
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

    # Clean column names
    df.columns = df.columns.str.strip()
    print("Loaded DataFrame columns:", df.columns)
    print(df.head())

except Exception as e:
    print(f"An error occurred: {e}")

# Step 3: Define the updated system prompt for ChatGPT
coding_scheme_prompt = """
You are a political scientist specialized in analysing Slovenian national discourse. Analyse the provided text using Greimas's semiotic actantial model. For each actantial role—subject, object, sender, receiver, helper, and opponent—identify distinct examples of actants, actors, and characters. These should represent different instances or variants of each actantial role in the narrative. Ensure clear distinction between abstract functions (actants), entities (actors), and specific individuals (characters).
Follow the instructions below in a systematic way and in the order given to ensure a clear distinction and proper identification of each term. Always identify the actant first, followed by the actor, and finally the character.

A) Actantial Roles Definitions:
1. Subject: The abstract "X" striving to achieve the object.
2. Object: The abstract "X" that the subject is pursuing.
3. Sender: The abstract "X" that prompts the subject to pursue the object.
4. Receiver: The abstract "X" that benefits from the success or failure of the subject.
5. Helper: The abstract "X" that aids the subject in achieving the object.
6. Opponent: The abstract "X" that hinders the subject from achieving the object.

B) Instructions:
For each actantial role (subject, object, sender, receiver, helper, opponent) identify:
1. Actant: Identify abstract role or function (e.g., forces, ideas) that guides the narrative. Actants are abstract, syntactic functions without thematic content.
2. Actor: Identify the entity that manifests the actant’s role in the narrative. Actors represent the semantic content and individuality of the actant and can assume multiple actantial roles.
3. Character: Identify the specific, individualized entity (e.g., named characters in the story such as individual or collective subjects) that actualize the actors in the narrative. Characters emerge from cumulative actions that fill actantial roles throughout the narrative.

C) Author's stance: The relationship between these actants often reveals the author's implicit stance. For example:

    If the subject (hero) is framed positively and the opponent is demonized, the author is likely supportive of the narrative.
    If the opponent is framed as just or rational, and the subject is villainized, the author is likely critical of the narrative.

In summary:
- Actants are abstract roles in the deep structure of the narrative.
- Actors are semantic entities that represent the manifestation of these actants in the narrative's content.
- Characters are the specific, concrete manifestations of these actors within a given text, recognized as individuals by the audience.

Example for Subject:
•	Actant: The pursuit of historical truth and national unity.
•	Actor: Historians and critics
•	Character: Dr. Jože Možina


Always answer in English and strictly follow the following format:

1. Subject
Actant:
Actor:
Character:

2. Object
Actant:
Actor:
Character:

3. Sender
Actant:
Actor:
Character:

4. Receiver
Actant:
Actor:
Character:

5. Helper
Actant:
Actor:
Character:

6. Opponent
Actant:
Actor:
Character:

7. Author's stance
"Explanation"
"""

# Step 4: Function to parse the GPT response and separate it into actantial roles
def parse_chatgpt_response(response_text):
    parsed_result = {}

    # Initialize all actantial roles with empty values to avoid missing key errors
    actantial_roles = ['Subject', 'Object', 'Sender', 'Receiver', 'Helper', 'Opponent']
    for role in actantial_roles:
        parsed_result[f'{role}_Actant'] = ''
        parsed_result[f'{role}_Actor'] = ''
        parsed_result[f'{role}_Character'] = ''

    # Regular expression to capture each role's content
    sections = re.findall(r'(\d+\.\s*(\w+))\s*Actant:\s*(.*?)\s*Actor:\s*(.*?)\s*Character:\s*(.*?)(?=\n\d+\.|\Z)', response_text, re.DOTALL)

    # Log the matched sections for verification
    print(f"Parsed sections: {sections}\n")

    for section in sections:
        role = section[1]  # e.g., Subject, Object, etc.
        parsed_result[f'{role}_Actant'] = section[2].strip()
        parsed_result[f'{role}_Actor'] = section[3].strip()
        parsed_result[f'{role}_Character'] = section[4].strip()

    return parsed_result

# Step 5: Function to interact with OpenAI API and send the prompt with the text
def analyze_text_with_chatgpt(text):
    max_retries = 5
    retry_delay = 10  # Wait 10 seconds before retrying

    for attempt in range(max_retries):
        try:
            # Prepare the full prompt with the text from the Excel file
            full_prompt = f"{coding_scheme_prompt}\n\nHere is the text to analyze:\n{text}"

            # Call OpenAI's GPT-4 API
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": coding_scheme_prompt},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=1000,
                temperature=0.7,
                top_p=0.9
            )

            # Extract the response text
            response_text = response['choices'][0]['message']['content'].strip()

            # Print the raw response for debugging
            print(f"Raw GPT response: {response_text}\n")

            # Parse the response to break it into different roles
            return parse_chatgpt_response(response_text)

        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(retry_delay)

    return {}

# Step 6: Apply the function to each row and expand the results into separate columns
if 'Text' in df.columns and 'url' in df.columns:
    results = df['Text'].apply(analyze_text_with_chatgpt)

    # Create a DataFrame from the results (expanding each dictionary into separate columns)
    parsed_df = pd.DataFrame(results.tolist())

    # Combine the original DataFrame with the new parsed columns
    df = pd.concat([df, parsed_df], axis=1)

    # Now df has separate columns for each role's actant, actor, and character
else:
    print("Error: 'Text' or 'url' column not found in the dataframe.")

# Step 7: Save the results to a new CSV file
output_csv_file = 'output_file.csv'
df.to_csv(output_csv_file, index=False, encoding='utf-8-sig')

# Step 8: Save the JSON output
output_json_file = 'structured_output_file.json'
json_results = []

for index, row in df.iterrows():
    result = {
        "url": row['url'],
        "text": row['Text'],
        "analysis": {
            "Subject": {
                "Actant": row['Subject_Actant'],
                "Actor": row['Subject_Actor'],
                "Character": row['Subject_Character']
            },
            "Object": {
                "Actant": row['Object_Actant'],
                "Actor": row['Object_Actor'],
                "Character": row['Object_Character']
            },
            "Sender": {
                "Actant": row['Sender_Actant'],
                "Actor": row['Sender_Actor'],
                "Character": row['Sender_Character']
            },
            "Receiver": {
                "Actant": row['Receiver_Actant'],
                "Actor": row['Receiver_Actor'],
                "Character": row['Receiver_Character']
            },
            "Helper": {
                "Actant": row['Helper_Actant'],
                "Actor": row['Helper_Actor'],
                "Character": row['Helper_Character']
            },
            "Opponent": {
                "Actant": row['Opponent_Actant'],
                "Actor": row['Opponent_Actor'],
                "Character": row['Opponent_Character']
            }
        }
    }
    json_results.append(result)

# Save the JSON file
with open(output_json_file, 'w', encoding='utf-8') as json_file:
    json.dump(json_results, json_file, ensure_ascii=False, indent=4)

# Download the result files to the local machine
files.download(output_csv_file)
files.download(output_json_file)

print(f"Analysis complete. Results saved to {output_csv_file} and {output_json_file}.")
