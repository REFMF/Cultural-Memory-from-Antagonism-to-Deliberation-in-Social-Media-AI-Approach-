!pip install openai pandas

import openai
import pandas as pd
from google.colab import files
import os
import time  # For retry delay

# Include your OpenAI API key here
openai.api_key = 'sk-proj-'

# Step 4: Load the uploaded Excel file
uploaded = files.upload()  # This will use the file you uploaded (Test 1.csv)
input_file_path = list(uploaded.keys())[0]  # Get the file name from the uploaded files

# Step 2: Check the file extension and load the file accordingly
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

    # Additional operations on df can go here, like displaying or processing the data.
    print(df.head())  # Display the first few rows of the dataframe

except UnicodeDecodeError as e:
    print(f"Error loading the file due to encoding issues: {e}")
except ValueError as ve:
    print(f"File loading error: {ve}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# Step 5: Define the updated system prompt for ChatGPT
coding_scheme_prompt = """
You are an expert in Slovenian political and historical discourse. Your task is to analyze the levels of antagonism, agonism, and deliberation in political texts, focusing on rhetorical strategies, sentiment, and historical framing. This analysis quantifies the overall discourse by counting occurrences of each category and calculating a final score on a 1–10 scale based on the relative frequencies of antagonism, agonism, and deliberation.
Categories of Analysis:

   1. Antagonism:
        Antagonism is marked by a confrontational and polarizing tone. It typically involves:
•	Negative or aggressive language, often directed at political actors, ideologies, or historical events.
•	Personal attacks or accusations, often aimed at delegitimizing opposing figures or viewpoints.
•	Language that amplifies division, portraying the political landscape in terms of "us vs. them."
Indicators:
•	High negative sentiment, especially when associated with confrontational terms (e.g., "ponareja zgodovinska dejstva," "mehki totalitarizem").
•	Use of dismissive, derogatory, or inflammatory language toward political figures, institutions, or ideologies.
•	Emphasis on conflict, betrayal, or moral failure of political actors.

   2. Agonism:
      Agonism refers to a respectful but adversarial form of engagement, where disagreement is acknowledged but remains constructive. It includes:
•	Critical engagement with opposing views without resorting to personal attacks or inflammatory language.
•	A willingness to present reasoned arguments and counterarguments, even in the context of political conflict.
•	Efforts to challenge opposing ideas while maintaining respect for democratic principles and the legitimacy of other perspectives.
Indicators:
•	Balanced critique, where disagreement is paired with reasoned arguments and factual evidence (e.g., citing historical events such as the Molotov-Ribbentrop Pact).
•	Engagement with alternative perspectives without vilification or emotional manipulation.
•	Moderate language that critiques policies or actions without discrediting the individuals behind them.

   3. Deliberation:
        Deliberation is characterized by rational discourse aimed at fostering mutual understanding or problem-solving. It includes:
•	A neutral or balanced tone focused on presenting facts and evidence without emotional bias.
•	Structured argumentation that explores different sides of an issue thoughtfully and without hostility.
•	Respectful discussion that avoids personal attacks or polarizing language.
Indicators:
•	Use of neutral or factual language aimed at informing or educating the audience, rather than swaying them emotionally.
•	Clear, logical presentation of arguments supported by evidence or historical examples.
•	Absence of emotionally charged or inflammatory rhetoric, with a focus on constructive dialogue.
Steps:

    Identify Occurrences:
        Scan the text to identify and count the occurrences of antagonistic, agonistic, and deliberative discourse based on the indicators provided.

    Calculate Relative Frequencies:
        Compute the percentage of each discourse type relative to the total number of categorized occurrences:
            Relative Frequency of Antagonism = (Occurrences of Antagonism / Total Occurrences) × 100
            Relative Frequency of Agonism = (Occurrences of Agonism / Total Occurrences) × 100
            Relative Frequency of Deliberation = (Occurrences of Deliberation / Total Occurrences) × 100

    Generate Overall Score:
        Use the relative frequencies to calculate the overall score on a 1–10 scale, where:
            1 represents a highly antagonistic discourse.
            5 represents a balanced, agonistic discourse.
            10 represents a highly deliberative discourse.
        Formula for calculating the score:

        css

        Final Score = [(Relative Frequency of Deliberation × 10) + (Relative Frequency of Agonism × 5) + (Relative Frequency of Antagonism × 1)] / 100

    Provide Final Output:
        Occurrences Count: List the number of occurrences and relative percentages for each category.
        Overall Score: Display the overall score based on the calculated formula.
        Overall Classification: Based on the relative frequencies and score, classify the text as predominantly antagonistic, agonistic, or deliberative.
        Explanation: Explain, summarize and contextualize the substance of each category (Antagonism, Agonism, Deliberation)

Example Output that you strictly follow:

    Occurrences Count:
        Antagonism: 15 (50%)
        Agonism: 8 (26.67%)
        Deliberation: 7 (23.33%)
        Overall Score: 4.17/10

       Overall Classification: The text is primarily antagonistic, with moderate agonism and some elements of deliberation.

    Analysis:
        1. Antagonism
        Occurrences: "List and cite every Occurrence"
        Explanation: "Explain, summarize and contextualize occurrences of antagonism in the article"

        2. Agonism:
        Occurrences: "List and cite every Occurrence"
        Explanation: "EExplain, summarize and contextualize occurrences of agonism in the article"

        3. Deliberation:
        Occurrences: "List and cite every Occurrence"
        Explanation: "Explain, summarize and contextualize occurrences of deliberation in the article"
"""

# Step 6: Define a function to interact with the OpenAI API and send the prompt along with the text
def analyze_text_with_chatgpt(text):
    max_retries = 5
    retry_delay = 10  # Wait 10 seconds before retrying

    for attempt in range(max_retries):
        try:
            # Prepare the full prompt with the text from the Excel file
            full_prompt = f"{coding_scheme_prompt}\n\nHere is the text to analyze:\n{text}"

            # Call OpenAI's GPT-4 API (corrected model)
            response = openai.ChatCompletion.create(
                model="gpt-4o",  # Specify the correct chat model
                messages=[
                    {"role": "system", "content": coding_scheme_prompt},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=2000,
                temperature=0.7,
                top_p=0.9
            )

            # Extract the response text
            return response['choices'][0]['message']['content'].strip()

        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(retry_delay)

    return "Failed to get a response after multiple retries."

# Step 7: Apply the function to each row in the 'Text' column (assuming 'Text' is the name of the column to analyze)
if 'Text' in df.columns:
    df['P'] = df['Text'].apply(analyze_text_with_chatgpt)
else:
    print("Error: 'Text' column not found in the dataframe.")

# Step 8: Save the results to a new CSV file
output_file_path = 'output_file.csv'
df.to_csv(output_file_path, index=False, encoding='utf-8-sig')

# Step 9: Download the result CSV file to your local machine
files.download(output_file_path)

print(f"Analysis complete. Results saved to {output_file_path}.")
