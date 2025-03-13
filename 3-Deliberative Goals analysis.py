!pip install openai pandas

import openai
import pandas as pd
from google.colab import files
import os
import time
import re

# ---------------------------------------------------------------------------------
# Replace this with your own OpenAI API key
openai.api_key = 'sk-proj-'

# ---------------------------------------------------------------------------------
# Step 1: Upload your input file (CSV or Excel) from local machine
uploaded = files.upload()
input_file_path = list(uploaded.keys())[0]

# ---------------------------------------------------------------------------------
# Step 2: Determine file extension and load accordingly
file_extension = os.path.splitext(input_file_path)[1].lower()

try:
    if file_extension == '.csv':
        # Try reading as UTF-8 first, then fallback
        try:
            df = pd.read_csv(input_file_path, encoding='utf-8-sig')
        except UnicodeDecodeError:
            print("UTF-8 encoding failed, trying ISO-8859-1.")
            df = pd.read_csv(input_file_path, encoding='ISO-8859-1')
    elif file_extension in ['.xls', '.xlsx']:
        df = pd.read_excel(input_file_path, engine='openpyxl')
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

    print("File loaded successfully. Preview:")
    print(df.head())
except UnicodeDecodeError as e:
    print(f"Error loading the file due to encoding issues: {e}")
except ValueError as ve:
    print(f"File loading error: {ve}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# ---------------------------------------------------------------------------------
# Step 3: Define the new system prompt containing instructions for analysis
analysis_prompt = """
Instruction for Analysis: As an expert in Slovenian culture, politics, history, and current
events, analyze the text based on the following five goals. For each goal, assign a
quantitative rating from 0 to 100 based on its prominence in the text. The Additional
Parameter (Call to Action) should be rated separately based on the 0-100 scale.

Goals for Analysis:

1. Epistemic Goals (Knowledge Production and Dissemination) – KNOWLEDGE
Focuses on enhancing public understanding by making fact-based knowledge
accessible and clearly presented. Texts with an epistemic focus support informed
decision-making through reliable, straightforward information aimed at fostering
sound public reasoning.

2. Ethical Goals (Fairness, Justice, and Moral Accountability) – VALUES
Centers on promoting fairness, mutual respect, and moral responsibility in discourse.
Texts with an ethical purpose build relationships based on respect and justice,
demonstrate accountability, and encourage careful consideration of ethical standards
in social interactions.

3. Political or Ideological Goals (Influence and Social Change) – IDEOLOGY
Aims to actively support, question, or challenge political and social power structures
and norms. Texts with an ideological purpose seek to shape societal values by
advocating for, opposing, or transforming existing systems and beliefs.

4. Democratic Goals (Legitimacy, Inclusiveness, and Fair Representation) – DEMOCRACY
Emphasizes the importance of inclusive participation, transparency, and fair
representation in collective decision-making. Texts with a democratic purpose seek to
ensure all voices are heard, strengthen legitimacy through diverse representation,
and uphold accountability within democratic processes.

5. Additional Parameter: Call to Action (Influence and Engagement) – ACTION
The goal is to promote and appeal for direct action or participation in social change or
societal transformation by advocating for action – either by addressing the political, social or
cultural sphere or addressing the public (audience) to act.

Quantitative Scaling:

Prominence (0-100) Scale for Each Goal:
0 signifies no presence and 100 indicates an overarching, dominant focus.
Use the following descriptions as guidance for intermediate values:
 - 0: No Presence
 - 1-10: Minimal Presence
 - 11-20: Very Low Presence
 - 21-30: Low Presence
 - 31-40: Below Average Presence
 - 41-50: Moderate Presence
 - 51-60: Above Average Presence
 - 61-70: High Presence
 - 71-80: Very High Presence
 - 81-90: Dominant Presence
 - 91-100: Overarching Presence

Results Format:

1. Analysis and Explanation:
   Provide a brief summary explaining how each goal is reflected in the text,
   highlighting key points or themes that support the assigned scores.

2. Score List:
   Present the scores from highest to lowest, reordering the categories based on their scores.
   Always list ACTION at the bottom, separated by a line, and displayed in *italics*.
   Do not include any explanation; list scores only. Format as follows:

   KNOWLEDGE: (score)
   VALUES: (score)
   IDEOLOGY: (score)
   DEMOCRACY: (score)

   *ACTION: (score)*

Note: Each rating is independent. Scores for each goal should reflect its individual presence
without influencing or being influenced by the other ratings.

Please follow these instructions strictly for the final output.
"""

# ---------------------------------------------------------------------------------
# Helper function to parse the GPT output
#
# Expected final output from GPT looks like:
# "Analysis and Explanation: ...
#  ...
# Score List:
#  KNOWLEDGE: (NN)
#  VALUES: (NN)
#  IDEOLOGY: (NN)
#  DEMOCRACY: (NN)
#
#  *ACTION: (NN)*"
#
# We'll extract:
#   - The entire "Analysis and Explanation" text
#   - The numeric scores for each category
#   and return them.

def parse_gpt_output(gpt_text):
    # 1) Extract "Analysis and Explanation" part
    #    We'll look for the line: "Analysis and Explanation:" and read until "Score List:"
    analysis = ""
    scores = {
        "KNOWLEDGE": None,
        "VALUES": None,
        "IDEOLOGY": None,
        "DEMOCRACY": None,
        "ACTION": None
    }

    # Regex to capture the block after "Analysis and Explanation:" until "Score List:"
    analysis_pattern = re.compile(
        r"Analysis and Explanation:\s*(.*?)\s*Score List:",
        re.DOTALL | re.IGNORECASE
    )
    analysis_match = analysis_pattern.search(gpt_text)
    if analysis_match:
        analysis = analysis_match.group(1).strip()

    # 2) Extract numeric scores from "Score List"
    #    We expect lines like "KNOWLEDGE: (65)" or "*ACTION: (40)*"
    score_pattern = re.compile(
        r"(KNOWLEDGE|VALUES|IDEOLOGY|DEMOCRACY|ACTION):\s*\((\d+)\)",
        re.IGNORECASE
    )
    for match in score_pattern.finditer(gpt_text):
        category = match.group(1).upper()
        score_value = int(match.group(2))
        if category in scores:
            scores[category] = score_value

    return analysis, scores["KNOWLEDGE"], scores["VALUES"], scores["IDEOLOGY"], scores["DEMOCRACY"], scores["ACTION"]

# ---------------------------------------------------------------------------------
# Step 4: Define a function to interact with the OpenAI API and send the text
def analyze_text_with_chatgpt(text):
    max_retries = 5
    retry_delay = 10  # Wait 10 seconds before retrying

    for attempt in range(max_retries):
        try:
            # We provide the system prompt with the instructions,
            # then supply the user's text separately.
            response = openai.ChatCompletion.create(
                model="gpt-4o",  # Or 'gpt-4' if you have access to GPT-4 model
                messages=[
                    {"role": "system", "content": analysis_prompt},
                    {"role": "user", "content": f"Analyze the following text:\n\n{text}"}
                ],
                max_tokens=2000,
                temperature=0.7,
                top_p=0.9
            )

            gpt_output = response['choices'][0]['message']['content'].strip()
            # Parse the GPT output
            analysis, kn, val, ido, dem, act = parse_gpt_output(gpt_output)
            return analysis, kn, val, ido, dem, act

        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(retry_delay)

    # If all retries fail:
    return ("Failed to get a response after multiple retries.", None, None, None, None, None)

# ---------------------------------------------------------------------------------
# Step 5: Apply the function to each row of the DataFrame, assuming the column is named 'Text'.
if 'Text' in df.columns:
    analysis_list = []
    knowledge_list = []
    values_list = []
    ideology_list = []
    democracy_list = []
    action_list = []

    for idx, row in df.iterrows():
        text_input = str(row['Text'])
        analysis_result, kn, val, ido, dem, act = analyze_text_with_chatgpt(text_input)

        analysis_list.append(analysis_result)
        knowledge_list.append(kn if kn is not None else "")
        values_list.append(val if val is not None else "")
        ideology_list.append(ido if ido is not None else "")
        democracy_list.append(dem if dem is not None else "")
        action_list.append(act if act is not None else "")

    df['Analysis_and_Explanation'] = analysis_list
    df['Knowledge'] = knowledge_list
    df['Values'] = values_list
    df['Ideology'] = ideology_list
    df['Democracy'] = democracy_list
    df['Action'] = action_list

    # Step 6: Save and download the result CSV
    output_file_path = 'output_file.csv'
    df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
    files.download(output_file_path)
    print(f"Analysis complete. Results saved to {output_file_path}.")
else:
    print("Error: 'Text' column not found in the DataFrame. Please ensure your file has a 'Text' column.")
