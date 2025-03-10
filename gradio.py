import pandas as pd

df = pd.read_csv("./Collect_Dataset_Kidney_Disease/CKD.csv")
def create_mapping(choices_str):
    mapping = {}
    choices = choices_str.split(' | ')
    for choice in choices:
        key, value = choice.split(': ', 1)
        mapping[key.strip()] = value.strip()
    return mapping

# Apply create_mapping function to 'choices' column
df['map'] = df['Choices'].apply(create_mapping)

# Replace 'correct' column with corresponding values from 'map'
df['Correct'] = df.apply(lambda row: row['map'][row['Correct']], axis=1)

# Display the updated DataFrame
df


import gradio as gr
import pandas as pd

# Function to evaluate the quiz
def evaluate_quiz(*responses):
    score = 0
    for idx, response in enumerate(responses):
        if response == df.iloc[idx]["Correct"]:
            score += 1
    return f"Your score is {score}/{len(df)}"

# Create Gradio interface
def create_quiz_interface(df):
    question_elements = []
    for idx, row in df.iterrows():
        # Split the choices using '|' as the delimiter and map them to 'a', 'b', 'c', 'd'
        choices = row['Choices'].split(' | ')
        choice_dict = {choice.split(': ')[0]: choice.split(': ')[1] for choice in choices}
        question_elements.append(gr.Radio(label=row['MCQ'], choices=list(choice_dict.values())))

    quiz_interface = gr.Interface(
        fn=evaluate_quiz,
        inputs=question_elements,
        outputs="text",
        title="QuizCrafter",
        description="Select the correct answers and submit to see your score."
    )
    return quiz_interface

# Initialize and launch the Gradio app
quiz_interface = create_quiz_interface(df)
quiz_interface.launch()