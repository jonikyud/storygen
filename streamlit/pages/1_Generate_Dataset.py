import streamlit as st
import pandas as pd
import time
from datetime import datetime
import os
import sys

#import the parent directories so can import code
current_dir =  os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
root_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.append(root_dir)

from src.prompting import examples_generator


def getExamples(genre, parameters, instruction,style) -> pd.DataFrame:
    print("Inside getExamples: genre: ",genre," ,parameters: ",parameters," instruction: ",instruction)
    data = examples_generator.createDataset(5, parameters, genre,instruction,style)
    columns_to_keep = ['generated_key', 'generated_value']
    # Drop columns not in the list
    df = data[columns_to_keep]
    #rename the columns
    new_column_names = {'generated_key': 'properties', 'generated_value': 'story'}
    # Rename columns
    dataframe = df.rename(columns=new_column_names) 
    return dataframe
    

@st.cache_data
def get_cached_variable():
    # This function will be executed only once, and the result will be cached
    return None  # You can set the initial value as needed

@st.cache_data
def get_generated_file():
    # This function will be executed only once, and the result will be cached
    return None  # You can set the initial value as needed

@st.cache_data
def generate_file(numberRows,genre, parameters,instruction,style):
    # Your file generation logic goes here    
    print("Inside generate_file: genre: ",genre," ,parameters: ",parameters," instruction: ",instruction)
    data = examples_generator.createDataset(numberRows, parameters, genre,instruction,style)   
    return data.to_csv().encode('utf-8')

st.title('Lets generate a dataset!')
cached_variable = get_cached_variable()
generated_file = get_generated_file()


genre= st.text_input("Input story genre:", key="genre") 
if genre:            
    params = "magical character, location, adventure"
    parametersText = st.text_input("Please specify properties by which your story is defined?", key="parameters",placeholder=params)
    instructions ="Please write a fairytale story incorporating the specified magical creature and enchanting object."
    instructionsText = st.text_input("Please define instructions how to use the properties to generate the story:", key="instructions",placeholder=instructions)
    style ="Be very creative and diverse with the properties values, they should not be straightforward but imaginative and diverse"
    styleText = st.text_input("Please define the style of generation as fitting the chosen genre:", key="style",placeholder=style)    
    numberRows = st.text_input("Specify desired number of entries in the fine tuning dataset", key="numRows",placeholder="10")
        
    

submit_button1= st.button("Lets generate some examples!")
if submit_button1:
    if ('parametersText' not in locals() or 'instructionsText' not in locals()):
       st.error('Please enter a genre first...')
    else:        
       cached_variable = (parametersText,instructionsText, styleText)                  
with st.form("Generating examples"):
    if (cached_variable is not None):            
            if not cached_variable[0] or not cached_variable[1] or not cached_variable[2]:
                st.error('Please provide properties, instructions and style first...')
            else:
                text = 'Generating some examples...'
                another_load_state = st.text(text)
                examples = st.write(getExamples(genre,parametersText,instructionsText,styleText))
                another_load_state.text("Done!")    
    submit_button2 = st.form_submit_button("Generate me a dataset!")


    # Additional processing based on Form 2 input
    if submit_button2:                
        if ('genre' not in locals() or 'parametersText' not in locals() or 'instructionsText' not in locals()):
            st.error('Please enter a genre first...')
        else:
            if not parametersText or not instructionsText or not numberRows or not styleText:
                st.error('Please enter properties and instructions, style and desired number of entries first...And just a hint - its always better to generate some examples too!')
            else:                 
                if isinstance(numberRows, str): 
                    if (not numberRows.isdigit()):
                        numberRows = 10                
                    else:
                        numberRows = int(numberRows)
                current_datetime = datetime.now()
                formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")          
                filename = genre+"_"+formatted_datetime+".csv"
                
                if (numberRows and numberRows !=""):              
                    st.success("Generation of file "+filename+" started!")
                
                fine_tuning_file = generate_file(numberRows,genre,parametersText,instructionsText,styleText)
                print(fine_tuning_file)
                generated_file = True
                st.success("File generated successfully!")
if (generated_file is not None):    
    st.download_button(
        label="Download File",
        data=fine_tuning_file, 
        file_name=filename,
        key="download_button"
    )
         