import re, string, os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "agent_prompts")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../agent_prompts")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "APIs")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../APIs")))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import importlib
from typing import List, Dict, Any
import tiktoken
from pandas import DataFrame
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

import sys
import json
import openai
import time
import pandas as pd
import datetime
from tqdm import tqdm
import argparse
import os
import io
import warnings

# to load prompt template
import importlib
from agent_prompts.prompt_extraction_direct import extraction_prompt

# to load the api implementation
import APIs.api_implementation as api
from APIs.api_implementation import (Date, DateRange, ISOCode, Country, CAMEOCode, Relation, Event, NewsArticle,
                        map_country_name_to_iso, map_iso_to_country_name, map_relation_description_to_cameo,
                        map_cameo_to_relation,
                        get_parent_relation, get_child_relations, get_sibling_relations, count_events, get_events,
                        get_entity_distribution, get_relation_distribution, count_news_articles, get_news_articles,
                        browse_news_article,
                        set_default_end_date, get_default_end_date, use_end_date)
print('loaded api_implementation')

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

pd.options.display.max_info_columns = 200

os.environ['TIKTOKEN_CACHE_DIR'] = './../tmp'

# Set the maximum allowed execution time in seconds
max_execution_time = 15 * 60  # 15 minutes

# Record the start time
code_start_time = time.time()


# catch timeout for each execution
import signal

# Define the exception to be raised on timeout
class TimeoutError(Exception):
    pass

# Define the signal handler
def handle_timeout(signum, frame):
    raise TimeoutError("Execution time exceeded 300 seconds")

# Set the signal alarm
signal.signal(signal.SIGALRM, handle_timeout)

# catch openai api error
def catch_openai_api_error():
    error = sys.exc_info()[0]
    if error == openai.error.APIConnectionError:
        print("APIConnectionError")
    elif error == openai.error.RateLimitError:
        print("RateLimitError")
        time.sleep(30)
    elif error == openai.error.APIError:
        print("APIError")
    elif error == openai.error.AuthenticationError:
        print("AuthenticationError")
    elif error == openai.error.InvalidRequestError:
        print("InvalidRequestError")
    else:
        print("API error:", error)

class DirectAgent:
    def __init__(self,
                 prompt_module,
                 direct_llm_name = 'gpt-3.5-turbo-1106',
                 temperature: float = 0.4
                 ) -> None:

        self.answer = ''
        self.scratchpad = ''
        self.finished = False
        self.end_state = ''

        self.step_n = 1

        self.direct_name = direct_llm_name

        self.prompt_module = prompt_module

        self.sys_prompt = prompt_module.sys_relation_prompt
        self.agent_prompt = prompt_module.relation_prompt

        self.json_log = []

        self.temp = temperature


        if 'gpt-3.5' in direct_llm_name:
            self.max_token_length = 15000
            self.llm = ChatOpenAI(temperature=self.temp,
                     max_tokens=2048,
                     model_name=direct_llm_name,
                     openai_api_key=OPENAI_API_KEY)
            
        elif 'gpt-4' in direct_llm_name:
            self.max_token_length = 128000
            self.llm = ChatOpenAI(temperature=self.temp,
                     max_tokens=2048,
                     model_name=direct_llm_name,
                     openai_api_key=OPENAI_API_KEY)

        self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

        self.answer_extractor = ChatOpenAI(temperature=0.2,
                     max_tokens=2048,
                     model_name="gpt-3.5-turbo-0125",
                     openai_api_key=OPENAI_API_KEY)

        self.__reset_agent()

    def run(self, query_info, reset=True):

        if reset:
            self.__reset_agent()

        self.query_info = query_info
        sys_prompt = self._build_sys_prompt()

        prompt, answer = self.prompt_agent()
        self.step_n += 1
        self.scratchpad += f'{answer}'
        if len(prompt) == 0: # openai error
            self.finished = True
            self.end_state = answer
            print(f'\n======\nAnswer with Error: {answer}')
        else:
            self.finished = True
            self.end_state = 'Final Answer'
            print(f'\n======\nFinal Answer: {answer}')

        ext_prompt, ext_request, self.answer = self.extract_answer(answer)

        return self.end_state, self.step_n-1, self.answer, self.scratchpad, self.json_log,  sys_prompt, ext_prompt, ext_request

    def extract_answer(self, final_info_str):
        print('\n==\nExtracting final answer...')

        curr_date_str = get_default_end_date()
        curr_date = datetime.datetime.strptime(curr_date_str, '%Y-%m-%d')
        curr_date_nlp = curr_date.strftime('%B %d, %Y')
        ext_prompt = extraction_prompt.format(
            current_date_nlp=curr_date_nlp,
            actor1_name=self.query_info['Actor1CountryName'],
            actor2_name=self.query_info['Actor2CountryName'],
            future_date_nlp=self.query_info['DateNLP'],
            future_date=self.query_info['DateStr'],
            actor1_code=self.query_info['Actor1CountryCode'],
            actor2_code=self.query_info['Actor2CountryCode'],
            info=final_info_str
            )
        ext_request = self.answer_extractor([HumanMessage(content=ext_prompt)]).content
        print('\nExtraction request:\n', ext_request)
        answer = self.extract_and_verify_dictionary(ext_request)
        print('\nFinal answer:\n', answer if len(answer) > 0 else 'No answer extracted.')
        return ext_prompt, ext_request, answer

    def extract_and_verify_dictionary(self, input_string):
        # Remove spaces, newlines, and any other characters that might cause issues
        cleaned_input = re.sub(r'\s+', '', input_string)

        # Regular expression to find content inside <answer> tags
        pattern = r'<answer>(.*?)</answer>'
        # Search for the pattern
        match = re.search(pattern, cleaned_input)

        # Check if a match was found
        if match:
            # Extract the content between the tags
            content = match.group(1)
            content.strip(' \n')
            try:
                # Try to parse the content as JSON
                parsed_dict = json.loads(content)

                # Check if the parsed content is a dictionary
                if isinstance(parsed_dict, dict):
                    return json.dumps(parsed_dict)  # Return the string representation of the dictionary
                else:
                    return ''  # Not a dictionary
            except json.JSONDecodeError:
                return ''  # Content was not valid JSON
        else:
            return ''  # No content found between tags

    def prompt_agent(self):
        trial = 0
        sys_prompt = self._build_sys_prompt()
        prompt = self._build_agent_prompt()
        messages = [SystemMessage(content=sys_prompt),
                    HumanMessage(content=prompt)]
        while trial < 3:
            try:
                request = self.llm(messages).content
                # print(request)
                return prompt, request.strip(' \n')
            except Exception as e:
                print(f"Error: {e}")
                print('prompt len:' + str(len(self.enc.encode(sys_prompt + prompt))))
                time.sleep(5)
                trial += 1
                err = str(e)
        return '', err


    def _build_sys_prompt(self) -> str:
        curr_date_str = get_default_end_date()
        curr_date = datetime.datetime.strptime(curr_date_str, '%Y-%m-%d')
        curr_date_nlp = curr_date.strftime('%B %d, %Y')
        return self.sys_prompt.format(current_date_nlp = curr_date_nlp)

    def _build_agent_prompt(self) -> str:
        curr_date_str = get_default_end_date()
        curr_date = datetime.datetime.strptime(curr_date_str, '%Y-%m-%d')
        curr_date_nlp = curr_date.strftime('%B %d, %Y')
        return self.agent_prompt.format(
            current_date_nlp = curr_date_nlp,
            actor1_name = self.query_info['Actor1CountryName'],
            actor2_name = self.query_info['Actor2CountryName'],
            future_date_nlp = self.query_info['DateNLP'],
            future_date = self.query_info['DateStr'],
            actor1_code = self.query_info['Actor1CountryCode'],
            actor2_code = self.query_info['Actor2CountryCode'])

    def is_finished(self) -> bool:
        return self.finished

    def __reset_agent(self) -> None:
        self.step_n = 1
        self.finished = False
        self.answer = ''
        self.scratchpad = ''
        self.json_log = []

    def extract_content(self, data):
        # Pattern matches optional ``` followed by optional language spec and newline, then captures all content until optional ```
        pattern = r'```(?:\w+\n)?(.*?)```|(.+)'
        match = re.search(pattern, data, re.DOTALL)
        if match:
            # Return the first non-None group
            return match.group(1) if match.group(1) is not None else match.group(2)
        return data  # Return data if no pattern matched


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="test_subset", choices=["test", "test_subset"])
    parser.add_argument("--timediff", type=int, default=1, help="date difference from the query date to the current date")

    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-0125",
                        choices=["gpt-3.5-turbo-0125", # latest GPT-3.5 turbo model (Sep 2021)
                                 "gpt-4-turbo-2024-04-09", # latest GPT-4 turbo model (Apr 2024)
                                 "gpt-4-1106-preview", # previous GPT-4 turbo preview model (Apr 2023)
                                 "gpt-4o-2024-05-13", # most advanced GPT-4o model (Oct 2023)
                                 ])
    parser.add_argument("--temperature", type=float, default=0.4, help="temperature of the model")
    parser.add_argument("--rounds", type=int, default=1, help="number of rounds")

    parser.add_argument("--plan", type=str, default="direct", choices=["direct", "cot"], help="planning strategy")
    parser.add_argument("--action", type=str, default="none", choices=["none"], help="action type")
    parser.add_argument("--api", type=str, default="none", choices=["none"], help="api type")
    parser.add_argument("--max_steps", type=int, default=0, help="maximum action steps")

    parser.add_argument("--output_dir", type=str, default="./../output")
    parser.add_argument("--data_dir", type=str, default="./../data/MIRAI")
    parser.add_argument("--api_dir", type=str, default="./../APIs/api_description_full.py")

    parser.add_argument("--alias", type=str, default="", help="alias for the output file")

    args = parser.parse_args()

    # make output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    setting_output_dir = os.path.join(args.output_dir, args.dataset, args.model_name, "timediff{}-maxsteps{}-{}-{}-{}-temp{}".format(args.timediff, args.max_steps, args.plan, args.action, args.api, args.temperature))
    if args.alias != "":
        setting_output_dir = setting_output_dir + '-' + args.alias
    if not os.path.exists(setting_output_dir):
        os.makedirs(setting_output_dir)

    # import prompt module
    prompt_module_name = f'prompts_{args.plan}'
    prompt_module = importlib.import_module(prompt_module_name)

    # load database
    data_kg = pd.read_csv(os.path.join(args.data_dir, "data_kg.csv"), sep='\t', dtype=str)
    data_news = pd.read_csv(os.path.join(args.data_dir, "data_news.csv"), sep='\t', dtype=str)

    # load api description
    api_dir = args.api_dir
    if args.api != 'full':
        api_dir = api_dir.replace('full', args.api)
    api_description = open(api_dir, 'r').read()

    # load query dataset
    data_query = pd.read_csv(os.path.join(args.data_dir, args.dataset, 'relation_query.csv'), sep='\t', dtype=str)

    query_ids = [i for i in range(1, len(data_query) + 1)]

    agent = DirectAgent(prompt_module=prompt_module,
                        direct_llm_name=args.model_name, temperature=args.temperature)
    with get_openai_callback() as cb:
        for curr_round in range(args.rounds):
            print(f"Round {curr_round + 1}")
            # make output directory
            curr_round_output_dir = os.path.join(setting_output_dir, f"round{curr_round + 1}")
            if not os.path.exists(curr_round_output_dir):
                os.makedirs(curr_round_output_dir)

            # run the agent
            for rowid, row in tqdm(data_query.iterrows(), total=len(data_query)):
                query_id = row['QueryId']
                query_date = row['DateStr']
                curr_date = datetime.datetime.strptime(query_date, '%Y-%m-%d') - datetime.timedelta(days=args.timediff)
                curr_date_str = curr_date.strftime('%Y-%m-%d')
                set_default_end_date(curr_date_str)
                use_end_date()

                # check if the output file directory exists
                output_file_dir = os.path.join(curr_round_output_dir, query_id + '.json')
                result = [{}]

                end_state, n_steps, answer, scratchpad, json_log, sys_prompt, ext_prompt, ext_request  = agent.run(row)

                result[-1]['query_id'] = query_id
                result[-1]['n_steps'] = n_steps
                result[-1]['end_state'] = end_state
                result[-1]['answer'] = answer
                result[-1]['gt_answer'] = row['AnswerDict']
                result[-1]['json_log'] = json_log
                result[-1]['sys_prompt'] = sys_prompt
                result[-1]['scratchpad'] = scratchpad
                result[-1]['ext_prompt'] = ext_prompt
                result[-1]['ext_request'] = ext_request

                # write to json file
                with open(output_file_dir, 'w') as f:
                    json.dump(result, f, indent=4)
        
    print(cb)

