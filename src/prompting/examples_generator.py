from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from langchain.llms import OpenAI
from langchain import HuggingFaceHub,LLMChain
from langchain.prompts import PromptTemplate
from langchain import FewShotPromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
import pandas as pd
from collections import Counter
from io import StringIO
import streamlit as st
import re
import os
import openai
import ast
import json
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field, validator
# Decorator for automatic retry requests

from pydantic import BaseModel, Field, conlist
from typing import List, Optional, Tuple
class OutputResult(BaseModel):
    key: conlist(str, min_length=3, max_length=5) = Field(description="The key with the story parameters. Must contain between 3 and 5 parameters")
    story:str = Field(description="The generated story for the given key")
    
@retry(
    retry = retry_if_exception_type((openai.APIError, openai.APIConnectionError,  openai.Timeout, ValueError, SyntaxError,KeyError)),
    # Function to add random exponential backoff to a request
    wait = wait_random_exponential(multiplier = 1, max = 60),
    stop = stop_after_attempt(10)
)
def run_llm_chain(hub_chain,user_input,parser):    
    output =hub_chain.run(input=user_input)        
    parsed_result = parser.parse(output)                   
    return parsed_result


def createDataset(iterations, key, story_type, instructions,style) -> pd.DataFrame:
    import os
    import openai
    import ast
    from langchain.chat_models import ChatOpenAI
# initialize the models
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai = ChatOpenAI(             
        model_name="gpt-3.5-turbo",
        openai_api_key=openai.api_key,
        temperature=1.5
    )  

    
    examples = [          
            {
                "input": """Generate a tuple with first part a key built like this [age,gender, superpower], 
                and the value in the tuple will be an entire story of maximum 100 words with detailed description for a super-hero with the given age, of the given gender and with the given superpower """,
                "output": OutputResult.model_validate({
                    "key": ["18", "man", "invisibility"],
                    "story": """A 18 year old man, tall with a strong yet athletic build. Noir eyes and light brown hair that seems to be a reflection of the warmth of his personality. 
                    His superpower of invisibility make him silent, 
                    introspective and observant. He knows when to be seen and when to remain invisible in the background; like a silent guardian protecting those around him. With a strong sense of justice and power,
                    he is an invaluable asset to those he holds near and dear. His kind and compassionate spirit give him an aura of protectiveness, making him a person of strength and courage in difficult moments."""            
                     }).model_dump_json().replace("{", "{{").replace("}", "}}"),
            },
            {
                "input": """Generate a tuple with first part a key built like this [product ,theme, details], 
                and the value in the tuple will be a gingle of maximum 100 words with commercial for the given product, in the given theme incorporating the provided details.""",
                "output": OutputResult.model_validate({
                    "key": ["Whiskers", "happy", "cat food-holiday season price reductions-great for your cat"],
                    "story": "We are so happy to announce holiday discounts for the best cat food outhere! For happy and healthy cat choose Whiskers! Meow!"            
                     }).model_dump_json().replace("{", "{{").replace("}", "}}"),                
            },
             {
                "input": """Generate a tuple with first part a key built like this [fictional character ,location, adventure], 
                and the value in the tuple will be a story of maximum 100 words describing an adventure of the given fictional character in the provided location.""",
                "output": OutputResult.model_validate({
                    "key": ["Baba Yaga", "Asia", "getting no respect"],
                    "story": """Once upon a time Baba Yaga wondered far far away from her home and ended up in remote Hokkaido island. 
                    She was used to locals showing her great respect out of fear and also because she was always one of the pillars of Slavic culture. 
                    But in Hokkaido the locals knew nothing about her, and she was very disappointed because they have shown her no respect. Eventually she decided there is no place like home and went back"""            
                     }).model_dump_json().replace("{", "{{").replace("}", "}}"),     
            },
        ]


    

    # create a example template
    example_template = """
        User: {input}
        AI: {output}
    """
    # create a prompt example from above template
    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template=example_template
    )

    parser = PydanticOutputParser(pydantic_object=OutputResult)

    # now break our previous prompt into a prefix and suffix
    # the prefix is our instructions    
    prefix = """You are a helpful assistant great in story telling. You follow the given instructions in a precise manner. 
    You need to generate a dataset where the key would be generated values string representing the story parameters according to the user given instructions, and the value will be a story written given this key. 
    Transform the output into structured object given those instructions: {format_instructions} Here are a few examples on how to generate the content of the dataset:
    """

    # and the suffix our user input and output indicator
    suffix = """
    User: {input}
    AI:"""


    # now create the few shot prompt template
    few_shot_prompt_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input"],
        example_separator="\n\n",
        partial_variables={"format_instructions": parser.get_format_instructions()},    
    )

    f_prompt = "Generate a tuple with first part a key built like this [{key}], and the value in the tuple will be an entire {type} of maximum 100 words .{instructions}. {style}"
    user_input = f_prompt.format(key=key, type=story_type,instructions = instructions, style=style)
    df = pd.DataFrame()
    for i in range(iterations):
            hub_chain = LLMChain(prompt=few_shot_prompt_template,llm=openai,verbose=True)              
            output  = run_llm_chain(hub_chain,user_input,parser)                                     
           
           
            generated_key = ", ".join(output.key) if output.key else 'Not specified'
            print('generated_key:',generated_key)

            generated_value = output.story if output.story else 'Not specified'
            print('generated_value:',generated_value)      
            
            

            # Access and print the key-value pairs
            
            new_row = {
            'keywords':key, 
            'story_type':story_type, 
            'instructions':instructions,         
            'generated_key':generated_key,
            'generated_value': generated_value
            }
            new_row = pd.DataFrame([new_row])
            df = pd.concat([df, new_row], axis=0, ignore_index=True)
    
    return df