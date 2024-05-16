from langchain import PromptTemplate
from langchain import LLMChain
# from langchain.llms import CTransformers
from llama_cpp import Llama
from langchain_community.llms import LlamaCpp
from src.helper import *
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from langchain.memory import ConversationBufferMemory

class NMT:
    def __init__(self):
        pass

    def text_preparation(self,Instruction,language):
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS , E_SYS = "<<SYS>>\n","\n<</SYS>>\n\n"

        # Instruction = f"Convert the following text from Englist to {language} : \n {text}"
        # CUSTOM_SYSTEM_PROMPT="You are an advanced assistant that provides translation from English to {language}"
        CUSTOM_SYSTEM_PROMPT="You are an advanced assistant that provides financial assistance to user "

        System_Prompt = B_SYS+ CUSTOM_SYSTEM_PROMPT +E_SYS
        template = B_INST + System_Prompt + Instruction + E_INST

        prompt = PromptTemplate(template = template ,
                         input_variables = ['user_text'])
                        
        return prompt


      

    def text_generation(self, text, language):
        try:
            n_gpu_layers = 32  # Metal set to 1 is enough.
            n_batch = 216

            llm = LlamaCpp(
                            model_path=r"D:\GenAI_projects\Quantized_model\-unsloth.Q4_K_M.gguf",
                            n_gpu_layers = n_gpu_layers,
                            n_batch=n_batch,
                            n_ctx=2048,
                            f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
                            verbose=True,
                            generation_kwargs = {
                                                "temperature": 0.99,
                                                "max_tokens":128,
                                                "stop":["</s>"],
                                                "echo":False, # Echo the prompt in the output
                                                "top_k":1 # This is essentially greedy decoding, since the model will always return the highest-probability token. Set this value > 1 for sampling decoding
                        }
                )
            logging.info('Model calling sucssesfully')
            # x = f'Convert the following text from Englist to {language}'
            x = f"Give advice on  given financial text"
            Instruction = x + ": \n {user_text}"
            prompt = self.text_preparation(Instruction,language)
            logging.info('Prompt Prepared succesfully')

            LLM_chain = LLMChain(prompt= prompt , 
                    llm = llm)
            response = LLM_chain.invoke(text)  
            logging.info('Response generated succesfully')

            return response

        except Exception as e:
            logging.info('Exception occured during data ingestion stage.')
            raise CustomException(e,sys) 




