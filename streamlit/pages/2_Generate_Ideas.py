import streamlit as st
import json
from huggingface_hub import HfApi, ModelCard, ModelFilter
from transformers import pipeline

@st.cache_resource()
def create_generator(model):
    # Long running operation to create the object based on the selected value
    return pipeline("text-generation", model=model, tokenizer=model)


def getModelParams(model):
    modelParams = models[model]
    return modelParams


def generateIdea(generator,parameters):    
    # Load the text generation pipeline   

    # Set the prompt for text generation
    prompt = f"{parameters}->"
    # Generate text
    print("loaded generator, beginning inference")
    generated_text = generator(prompt, max_length=500, num_return_sequences=1)[0]['generated_text']

    # Print the generated text
    print(generated_text)
    return generated_text.split("->")[1].strip() #return the part of the string after the SEPARATOR


def _getModelCards(models):
    dict={}  
    for model in models:
        print("getModelCards: Model:", model)
        card = None
        keyword = None
        instruction = None
        storytype = None
        id = model.id
        try:
            card = ModelCard.load(id).data.to_dict()
            print(card)
            if card:
                if 'keywords' in card:
                    keyword = card['keywords']
                if 'instruction' in card:
                    instruction =  card['instruction']
                if 'storytype' in card:
                    storytype =  card['storytype']
            if keyword and instruction and storytype:
                dict[id]=(keyword,instruction,storytype)
        except Exception as e:
            print(e)
    return dict

@st.cache_data
def getModels():
    api = HfApi()
    models = api.list_models(
        filter=ModelFilter(
            author="jonikyud"        
        )
    )    
    dictionary = _getModelCards(models)
    return dictionary


st.title('Lets generate ideas!')
models = getModels()

models_list  = list(models.keys())
print(models_list)

model = st.selectbox(
        'Select the fine-tuned model',
        models_list
    )

if model is not None:
    cached_generator = create_generator(model)  
    params = getModelParams(model)
    keywords = params[0]
    instruction= params[1]
    storytype = params[2]
    print(params)
    header = f"Will generate {storytype} based on the following properties, please enter values:"
    st.write(header)
    properties = st.text_input("Enter values for the following properties:", key="parameters",placeholder=keywords)

if (model is not None) and (properties is not None) and (properties !=""):
    submit_button = st.button("Generate my idea!")
    if submit_button:
        text = 'Generating idea...'
        another_load_state = st.text(text)
        idea = generateIdea(cached_generator,properties)
        another_load_state.text("Done!")  
        num_rows = len(idea) // 90 + 1  # Add 1 to ensure that even short strings occupy at least one row
        # Set the height of the text area based on the number of rows
        height = num_rows * 30  # Approximate height per row        
        #generate the idea        
        idea_text =st.text_area("An idea for you",value=idea,height=height)
