import warnings

from langchain._api import LangChainDeprecationWarning
warnings.simplefilter("ignore", category=LangChainDeprecationWarning)

import re, os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "agent_prompts")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../agent_prompts")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "APIs")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../APIs")))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import tiktoken
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
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

# to load prompt template
import importlib
from agent_prompts.prompt_extraction import extraction_prompt

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

from vllm import LLM, SamplingParams
import torch


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


# color print
def red(msg):
    return "\033[91m" + msg + "\033[0m"

def green(msg):
    return "\033[92m" + msg + "\033[0m"

def cyan(msg):
    return "\033[96m" + msg + "\033[0m"

def yellow(msg):
    return "\033[93m" + msg + "\033[0m"

def blue(msg):
    return "\033[94m" + msg + "\033[0m"

def orange(msg):
    return "\033[38;5;208m" + msg + "\033[0m"

def purple(msg):
    return "\033[95m" + msg + "\033[0m"


# catch openai api error
def catch_openai_api_error():
    error = sys.exc_info()[0]
    if error == openai.error.APIConnectionError:
        print(red("APIConnectionError"))
    elif error == openai.error.RateLimitError:
        print(red("RateLimitError"))
        time.sleep(30)
    elif error == openai.error.APIError:
        print(red("APIError"))
    elif error == openai.error.AuthenticationError:
        print(red("AuthenticationError"))
    elif error == openai.error.InvalidRequestError:
        print(red("InvalidRequestError"))
    else:
        print(red("API error:", error))

class ReactAgent:
    def __init__(self,
                 action_type: str,
                 api_description: str,
                 prompt_module,
                 max_steps: int = 30,
                 max_retries: int = 3,
                 react_llm_name = 'gpt-3.5-turbo-1106',
                 temperature: float = 0.4
                 ) -> None:

        self.action_type = action_type
        if self.action_type == 'func':
            self.api_error_note = 'Please make sure your action is a valid and executable function call with correct arguments based on the API description.'
        elif self.action_type == 'block':
            self.api_error_note = "If you are collecting data with code, please make sure your action is a valid and executable block of code with correct syntax based on the API description, and use print() for outputs; If you are making the final forecast, please start the action immediately with 'Final Answer:' without enclosing within triple backticks, for example, 'Action: Final Answer: {}'"
        elif self.action_type == 'blocklib':
            self.api_error_note = "If you are collecting data with code, please make sure your action is a valid and executable block of code with correct syntax based on the API description and Python libraries, and use print() for outputs; If you are making the final forecast, please start the action immediately with 'Final Answer:' without enclosing within triple backticks, for example, 'Action: Final Answer: {}'"
        self.local_vars = {}

        self.api_description = api_description

        self.answer = ''
        self.scratchpad = ''
        self.finished = False
        self.end_state = ''

        self.step_n = 1
        self.max_steps = max_steps # max number of thought, action, observation steps
        self.n_retries = 0
        self.max_retries = max_retries # number of retries for consecutive invalid actions

        self.react_name = react_llm_name

        self.prompt_module = prompt_module

        self.sys_prompt = prompt_module.sys_relation_prompt
        self.agent_prompt = prompt_module.relation_prompt

        self.json_log = []

        self.performed_actions = {}
        self.notebook = {}

        self.current_observation = ''

        self.temp = temperature


        self.stop_list = ['Action:' , 'Observation:', 'Thought:']

        if 'gpt-3.5' in react_llm_name:
            self.max_token_length = 15000
            self.llm = ChatOpenAI(temperature=self.temp,
                     max_tokens=2048,
                     model_name=react_llm_name,
                     openai_api_key=OPENAI_API_KEY,
                     model_kwargs = {"stop": self.stop_list})
            
        elif 'gpt-4' in react_llm_name:
            self.max_token_length = 128000
            self.llm = ChatOpenAI(temperature=self.temp,
                     max_tokens=2048,
                     model_name=react_llm_name,
                     openai_api_key=OPENAI_API_KEY,
                     model_kwargs = {"stop": self.stop_list})
            
        elif 'llama' in react_llm_name.lower():
            llm_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
            self.max_token_length = 32000
            self.llm = LLM(model=llm_name,
                           tensor_parallel_size=1,
                           dtype=torch.float16,
                           gpu_memory_utilization=0.7,
                           disable_log_stats=True)
            self.sample_params = SamplingParams(temperature=self.temp,
                                                max_tokens=2048,
                                                stop=self.stop_list,
                                                include_stop_str_in_output=False)
            
        elif 'mistral' in react_llm_name.lower():
            llm_name = 'mistralai/Mistral-7B-Instruct-v0.2'
            self.max_token_length = 8000
            self.llm = LLM(model=llm_name,
                           tensor_parallel_size=1,
                           dtype=torch.float16,
                           gpu_memory_utilization=0.7,
                           disable_log_stats=True)
            self.sample_params = SamplingParams(temperature=self.temp,
                                                max_tokens=2048,
                                                stop=self.stop_list,
                                                include_stop_str_in_output=False)


        self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

        self.answer_extractor = ChatOpenAI(temperature=0.2,
                     max_tokens=2048,
                     model_name="gpt-3.5-turbo-0125",
                     openai_api_key=OPENAI_API_KEY)

        self.__reset_agent()

    def run(self, query_info, reset=True):

        self.query_info = query_info
        # query_prompt = self._build_agent_prompt()
        # print('\n============\nQuery Prompt:\n')
        # print(purple(query_prompt))
        # sys_prompt_print = self._build_sys_prompt_print()
        # print('\n============\nSystem Prompt:\n')
        # print(blue(sys_prompt_print))

        sys_prompt = self._build_sys_prompt()
        # Record the start time
        code_start_time = time.time()
        
        if reset:
            self.__reset_agent()

        while True:
            if not self.is_finished():
                # check execution time
                code_elapsed_time = time.time() - code_start_time
                if code_elapsed_time > max_execution_time:
                    self.finished = True
                    self.end_state = 'Execution Time Exceeded'
                # check max steps
                if self.step_n > self.max_steps:
                    self.finished = True
                    self.end_state = 'Max Steps Exceeded'

                # check max token length
                if len(self.enc.encode(sys_prompt + self._build_agent_prompt())) > self.max_token_length:
                    self.finished = True
                    self.end_state = 'Max Token Length Exceeded'

            # execute another step if not finished
            if not self.is_finished():
                self.step()
            else: # if finished, extract answer
                ext_prompt, ext_request, self.answer = self.extract_answer()
                break
        return self.end_state, self.step_n-1, self.answer, self.scratchpad, self.json_log,  sys_prompt, ext_prompt, ext_request

    def extract_answer(self):
        print(orange('\n==\nExtracting final answer...'))
        final_info = []
        # add from last json_log with max 2 steps
        for step in range(len(self.json_log), 0, -1):
            if len(final_info) >= 2:
                break
            if self.json_log[step - 1]['include_in_extraction']:
                final_info.append(self.json_log[step - 1])

        if len(final_info) == 0:
            final_info_str = 'No information available.'
        else:  # join thought, action, observation for final info
            final_info_str = ''
            for idx in range(len(final_info)): # process the later step first
                final_info_len = len(final_info_str.split(' '))
                if final_info_len >= 4000:
                    break

                curr_info_str = ''
                curr_info_str += f"\nThought: {final_info[idx - 1]['thought']}"
                curr_info_str += f"\nAction: {final_info[idx - 1]['action']}"
                if final_info[idx - 1]['observation'] != '':
                    curr_info_str += f"\nObservation: {final_info[idx - 1]['observation']}\n"
                curr_tokens = curr_info_str.split(' ')
                # add current info before final info if it does not exceed max token length, otherwise truncate the current info
                if final_info_len + len(curr_tokens) <= 4000:
                    final_info_str = curr_info_str + final_info_str
                else:
                    curr_tokens = curr_tokens[:(4000 - final_info_len)]
                    final_info_str = ' '.join(curr_tokens) + final_info_str

            final_info_str = final_info_str.strip('\n')

        # print('\nFinal information:\n', final_info_str)

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
        print(orange('\nExtraction request:\n' + ext_request))
        answer = self.extract_and_verify_dictionary(ext_request)
        print(orange('\nFinal answer:\n' + (answer if len(answer) > 0 else 'No answer extracted.')))
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

    def step(self) -> None:

        self.json_log.append({"step": self.step_n,
                              # "thought_prompt":"",
                              "thought": "",
                              # "action_prompt": "",
                              "action": "",
                              "observation": "",
                              "state":"",
                              "include_in_extraction": False})

        self.scratchpad += f'\nThought:'
        thought = ''
        retry = 0
        # ensure thought is not empty
        while len(thought) == 0:
            retry += 1
            if retry > self.max_retries:
                break
            prompt, thought = self.prompt_agent()
            # to handle openai error
            if len(prompt) == 0:
                error_message = thought
                self.finished = True
                self.end_state = f'OpenAI Error: {error_message}'
                self.step_n += 1
                return
        # self.json_log[-1]['thought_prompt'] = prompt
        self.scratchpad += ' ' + thought

        print(f'\n============\nStep {self.step_n}:')
        print(green(f'\nThought: ' + thought))
        self.json_log[-1]['thought'] = thought


        # Act
        self.scratchpad += f'\nAction:'
        prompt, action = self.prompt_agent()
        # to handle openai error
        if len(prompt) == 0:
            error_message = action
            self.finished = True
            self.end_state = f'OpenAI Error: {error_message}'
            self.step_n += 1
            return
        # self.json_log[-1]['action_prompt'] = prompt

        # process null action
        if action == None or action == '' or action == '\n':
            self.scratchpad += " Your action is filtered due to empty content. Please make sure your action content does not start with ['Thought', 'Action', 'Observation'] and assume all the actions are permitted in this environment and take the action again."

            # observe
            print(red(f'\nObservation: ' + "No feedback from the environment due to the null action."))
            self.json_log[-1]['observation'] = "No feedback from the environment due to the null action."
            self.json_log[-1]['state'] = 'Null Action'
            self.json_log[-1]['include_in_extraction'] = False
            self.scratchpad += f'\nObservation: '
            self.current_observation = "No feedback from the environment due to the null action."
            self.scratchpad += self.current_observation + '\n'
            self.step_n += 1
            return
        else:
            self.scratchpad += ' ' + action


        self.performed_actions[action] = self.performed_actions.get(action, 0) + 1

        print(yellow(f"\nAction: " + action))
        self.json_log[-1]['action'] = action
        # if action is repeated, skip the action execution; if action is repeated over max_retries, early stop
        if self.performed_actions[action] > 1 and self.performed_actions[action] <= self.max_retries:
            print(red("\nObservation: The same action has been executed before. Try a different action with correct format. If you are collecting data with code, please make sure your action is a valid and executable code with correct syntax based on the API description; If you are making the final forecast, please start the action immediately with 'Final Answer:' without enclosing within triple backticks, for example, 'Action: Final Answer: {}'"))
            self.json_log[-1]['state'] = 'Repeated Action'
            self.current_observation = "The same action has been executed before. Try a different action with correct format. If you are collecting data with code, please make sure your action is a valid and executable code with correct syntax based on the API description; If you are making the final forecast, please start the action immediately with 'Final Answer:' without enclosing within triple backticks, for example, 'Action: Final Answer: {}'"
            self.json_log[-1]['observation'] = self.current_observation
            self.json_log[-1]['include_in_extraction'] = False
            self.scratchpad += f'\nObservation: ' + self.current_observation + '\n'
            self.step_n += 1
            return
        elif self.performed_actions[action] > self.max_retries:
            print(red(f"\nObservation: The same action has been executed over {self.max_retries} times. Early stop due to repeated actions."))
            self.json_log[-1]['state'] = f'Early stop due to repeated actions.'
            self.current_observation = f"The same action has been executed over {self.max_retries} times. Early stop due to repeated actions."
            self.json_log[-1]['observation'] = self.current_observation
            self.json_log[-1]['include_in_extraction'] = False
            self.scratchpad += f'\nObservation: ' + self.current_observation + '\n'
            self.finished = True
            self.end_state = 'Repeated Actions'
            self.step_n += 1
            return

        # Observe, execute the action that appeared for the first time
        # if answer contains 'Final Answer:', stop the loop and call extractor
        # loose the condition to allow for more flexible final answer format
        if 'final answer' in action.lower():
            self.json_log[-1]['include_in_extraction'] = True
            self.n_retries = 0
            self.finished = True
            self.end_state = 'Final Answer'
            self.step_n += 1
            self.json_log[-1]['state'] = 'Detect Final Answer'
            return
        else: # if answer does not contain 'Final Answer:', execute the action
            self.scratchpad += f'\nObservation: '
            code_str = action

            code_str = code_str.strip(' \n')
            code_str = self.extract_content(code_str)
            code_str = code_str.strip(' \n')

            # for function call
            if self.action_type == 'func':
                try:
                    code_output = eval(code_str)

                    if ((type(code_output) == list or type(code_output) == dict) and  (len(code_output) == 0)):
                        self.current_observation = "Empty output from the function call."
                        print(red(f'\nObservation: ' + self.current_observation))
                    else:
                        self.current_observation = str(code_output)
                    print(cyan(f'\nObservation: ' + self.current_observation))
                    self.scratchpad += self.current_observation + '\n'
                    self.json_log[-1]['observation'] = self.current_observation
                    self.json_log[-1]['state'] = 'Valid Action'
                    self.json_log[-1]['include_in_extraction'] = True
                    self.n_retries = 0
                    self.step_n += 1
                    return
                except Exception as e:
                    self.n_retries += 1
                    if self.n_retries >= self.max_retries:
                        print(red(f"\nObservation: Illegal action: {e}. Early stop due to consecutive {self.max_retries} invalid actions."))
                        self.json_log[-1]['state'] = f'Early stop due to consecutive {self.max_retries} invalid actions.'
                        self.current_observation = f"Illegal action: {e}. Early stop due to consecutive {self.max_retries} invalid actions."
                        self.json_log[-1]['observation'] = self.current_observation
                        self.json_log[-1]['include_in_extraction'] = False
                        self.scratchpad += self.current_observation + '\n'
                        self.finished = True
                        self.end_state = 'Invalid Action'
                        self.step_n += 1
                        return
                    else:
                        print(red(f"\nObservation: Illegal action: {e}. {self.api_error_note}"))
                        self.json_log[-1]['state'] = f'Invalid action: {e}'
                        self.current_observation = f"Invalid action: {e}. {self.api_error_note}"
                        self.json_log[-1]['observation'] = self.current_observation
                        self.json_log[-1]['include_in_extraction'] = False
                        self.scratchpad += self.current_observation + '\n'
                        self.step_n += 1
                        return

            # for block of code
            elif self.action_type == 'block' or self.action_type == 'blocklib':
                err = None
                output_buffer = io.StringIO()

                current_stdout = sys.stdout # Save the current standard output
                current_stderr = sys.stderr # Save the current standard error

                sys.stdout = output_buffer # Redirect standard output to the buffer
                sys.stderr = output_buffer # Redirect standard error to the buffer

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    signal.alarm(300) # Set the alarm to 300 seconds
                    try:
                        exec(code_str, globals(), self.local_vars)
                        # Reset the alarm
                        signal.alarm(0)
                    except Exception as e:
                        err = e
                        # print(f"Illegal action: {e}. {self.api_error_note}")

                sys.stdout = current_stdout # Restore the original standard output
                sys.stderr = current_stderr # Restore the original standard error

                code_output = output_buffer.getvalue() # Get the output from the buffer
                output_buffer.close()

                # if have error, record as invalid action
                if err != None:
                    self.n_retries += 1
                    if self.n_retries >= self.max_retries:
                        print(red(f"\nObservation: Illegal action: {err}. Early stop due to consecutive {self.max_retries} invalid actions."))
                        self.json_log[-1]['state'] = f'Early stop due to consecutive {self.max_retries} invalid actions: execution error.'
                        self.current_observation = f"Illegal action: {err}. Early stop due to consecutive {self.max_retries} invalid actions."
                        self.json_log[-1]['observation'] = self.current_observation
                        self.json_log[-1]['include_in_extraction'] = False
                        self.scratchpad += self.current_observation + '\n'
                        self.finished = True
                        self.end_state = 'Invalid Action'
                        self.step_n += 1
                        return
                    else:
                        print(red(f"\nObservation: Illegal action: {err}. {self.api_error_note}"))
                        self.json_log[-1]['state'] = f'Invalid action: {err}'
                        self.current_observation = f"Invalid action: {err}. {self.api_error_note}"
                        self.json_log[-1]['observation'] = self.current_observation
                        self.json_log[-1]['include_in_extraction'] = False
                        self.scratchpad += self.current_observation + '\n'
                        self.step_n += 1
                        return
                elif len(code_output) == 0: # no error but no output
                    # if print() is used, this is a valid action
                    if 'print(' in code_str:
                        self.current_observation = "No printed output from the action because you are printing an empty object."
                        print(red(f'\nObservation: ' + self.current_observation))
                        self.scratchpad += self.current_observation + '\n'
                        self.json_log[-1]['observation'] = self.current_observation
                        self.json_log[-1]['state'] = 'Valid Action'
                        self.json_log[-1]['include_in_extraction'] = True
                        self.n_retries = 0
                        self.step_n += 1
                        return
                    else: # if no print() is used, this is an invalid action
                        self.n_retries += 1
                        if self.n_retries >= self.max_retries:
                            print(red(f"\nObservation: Illegal action: No print() statement in the action. Early stop due to consecutive {self.max_retries} invalid actions."))
                            self.json_log[-1]['state'] = f'Early stop due to consecutive {self.max_retries} invalid actions: no print() statement.'
                            self.current_observation = f"Illegal action: No print() statement in the action. Early stop due to consecutive {self.max_retries} invalid actions."
                            self.json_log[-1]['observation'] = self.current_observation
                            self.json_log[-1]['include_in_extraction'] = False
                            self.scratchpad += self.current_observation + '\n'
                            self.finished = True
                            self.end_state = 'Invalid Action'
                            self.step_n += 1
                            return
                        else:
                            print(red(f"\nObservation: Illegal action: No print() statement in the action. {self.api_error_note}"))
                            self.json_log[-1]['state'] = 'Invalid action: No print() statement in the action.'
                            self.current_observation = f"Invalid action: No print() statement in the action. {self.api_error_note}"
                            self.json_log[-1]['observation'] = self.current_observation
                            self.json_log[-1]['include_in_extraction'] = False
                            self.scratchpad += self.current_observation + '\n'
                            self.step_n += 1
                            return
                # if no error and have output, this is a valid action
                else: # len(code_output) > 0 and e == None:
                    self.current_observation = str(code_output)
                    print(cyan(f'\nObservation: ' + self.current_observation))
                    self.scratchpad += self.current_observation + '\n'
                    self.json_log[-1]['observation'] = self.current_observation
                    self.json_log[-1]['state'] = 'Valid Action'
                    self.json_log[-1]['include_in_extraction'] = True
                    self.n_retries = 0
                    self.step_n += 1
                    return

    def prompt_agent(self):
        trial = 0
        sys_prompt = self._build_sys_prompt()
        prompt = self._build_agent_prompt()
        if 'gpt' in self.react_name:
            messages = [SystemMessage(content=sys_prompt),
                        HumanMessage(content=prompt)]
            while trial < 3:
                try:
                    request = self.llm(messages).content
                    # print(request)
                    return prompt, request.strip(' \n')
                except Exception as e:
                    print(red(f"Error: {e}"))
                    print(red('prompt len:' + str(len(self.enc.encode(sys_prompt + prompt)))))
                    time.sleep(5)
                    trial += 1
                    err = str(e)
            return '', err
        elif 'llama' in self.react_name.lower():
            instruct_prompt = self.generate_llama3_instruct_prompt(sys_prompt, prompt, self.scratchpad)
            try:
                response = self.llm.generate(instruct_prompt, self.sample_params)[0].outputs[0].text
                # print("-------instruct_prompt-------")
                # print(instruct_prompt)
                # print("-------instruct_prompt-------")
                return prompt, response.strip(' \n')
            except Exception as e:
                print(red(f"Error: {e}"))
                return '', str(e)
        elif 'mistral' in self.react_name.lower():
            instruct_prompt = self.generate_mistral_instruct_prompt(sys_prompt, prompt, self.scratchpad)
            try:
                response = self.llm.generate(instruct_prompt, self.sample_params)[0].outputs[0].text
                # print("-------instruct_prompt-------")
                # print(instruct_prompt)
                # print("-------instruct_prompt-------")
                return prompt, response.strip(' \n')
            except Exception as e:
                print(red(f"Error: {e}"))
                return '', str(e)

    def generate_llama3_instruct_prompt(self, sys_prompt, user_prompt, assistant_prompt):
        instruct_prompt = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n'
        instruct_prompt += sys_prompt.strip(' \n') + '<|eot_id|>'
        instruct_prompt += '<|start_header_id|>user<|end_header_id|>\n\n'
        instruct_prompt += user_prompt.strip(' \n') + '<|eot_id|>'
        instruct_prompt += '<|start_header_id|>assistant<|end_header_id|>\n\n'
        instruct_prompt += assistant_prompt.strip(' \n')
        return instruct_prompt

    def generate_mistral_instruct_prompt(self, sys_prompt, user_prompt, assistant_prompt):
        instruct_prompt = '<s>[INST] '
        instruct_prompt += sys_prompt.strip(' \n') + '\n\n' + user_prompt.strip(' \n')
        instruct_prompt += ' </INST> '
        instruct_prompt += assistant_prompt.strip(' \n')
        return instruct_prompt

    def _build_sys_prompt(self) -> str:
        curr_date_str = get_default_end_date()
        curr_date = datetime.datetime.strptime(curr_date_str, '%Y-%m-%d')
        curr_date_nlp = curr_date.strftime('%B %d, %Y')
        return self.sys_prompt.format(
            current_date_nlp = curr_date_nlp,
            max_iterations = self.max_steps,
            api_description = self.api_description)

    # def _build_sys_prompt_print(self) -> str:
    #     curr_date_str = get_default_end_date()
    #     curr_date = datetime.datetime.strptime(curr_date_str, '%Y-%m-%d')
    #     curr_date_nlp = curr_date.strftime('%B %d, %Y')
    #     _sys_prompt_print = self.sys_prompt.format(
    #         current_date_nlp = curr_date_nlp,
    #         max_iterations = self.max_steps,
    #         api_description = '<<FULL API SPECIFICATION HERE>>')
    #     paragraphs = _sys_prompt_print.split('\n')
    #     paragraphs[0] = '.'.join(paragraphs[0].split('.')[:2]) + '. <<ADDITIONAL DATABASE DESCRIPTIONS HERE>>'
    #     _sys_prompt_print = '\n'.join(paragraphs[:-6]) + '\n...<<ADDITIONAL INSTRUCTIONS HERE>>...'
    #     return _sys_prompt_print

    def _build_agent_prompt(self) -> str:
        curr_date_str = get_default_end_date()
        curr_date = datetime.datetime.strptime(curr_date_str, '%Y-%m-%d')
        curr_date_nlp = curr_date.strftime('%B %d, %Y')
        if ('llama' in self.react_name.lower()) or ('mistral' in self.react_name.lower()):
            return self.agent_prompt.format(
                current_date_nlp = curr_date_nlp,
                actor1_name = self.query_info['Actor1CountryName'],
                actor2_name = self.query_info['Actor2CountryName'],
                future_date_nlp = self.query_info['DateNLP'],
                future_date = self.query_info['DateStr'],
                actor1_code = self.query_info['Actor1CountryCode'],
                actor2_code = self.query_info['Actor2CountryCode'])
        else:
            return self.agent_prompt.format(
                current_date_nlp = curr_date_nlp,
                actor1_name = self.query_info['Actor1CountryName'],
                actor2_name = self.query_info['Actor2CountryName'],
                future_date_nlp = self.query_info['DateNLP'],
                future_date = self.query_info['DateStr'],
                actor1_code = self.query_info['Actor1CountryCode'],
                actor2_code = self.query_info['Actor2CountryCode'],
                scratchpad = self.scratchpad)

    def is_finished(self) -> bool:
        return self.finished

    def __reset_agent(self) -> None:
        self.step_n = 1
        self.n_retries = 0
        self.finished = False
        self.answer = ''
        self.scratchpad = ''
        self.json_log = []
        self.current_observation = ''
        self.performed_actions = {}
        self.notebook = {}
        self.local_vars = {}

    def extract_content(self, data):
        # Pattern matches optional ``` followed by optional language spec and newline, then captures all content until optional ```
        pattern = r'```(?:\w+\n)?(.*?)```|(.+)'
        match = re.search(pattern, data, re.DOTALL)
        if match:
            # Return the first non-None group
            return match.group(1) if match.group(1) is not None else match.group(2)
        return data  # Return data if no pattern matched

    # def verbalize_relation_code(self, dict_code2relation, answer_dict):
    #     try:
    #         v_answer_dict = {}
    #         for key, value in answer_dict.items():
    #             v_key = dict_code2relation[key]['Name']
    #             v_answer_dict[v_key] = [dict_code2relation[item]['Name'] for item in value]
    #         return v_answer_dict
    #     except Exception as e:
    #         return 'Failed to verbalize the relation code. Error: {}'.format(e)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="test_subset", choices=["test", "test_subset", "demo"])
    parser.add_argument("--timediff", type=int, default=1, help="date difference from the query date to the current date")

    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-0125",
                        choices=["gpt-3.5-turbo-0125", # latest GPT-3.5 turbo model (Sep 2021)
                                 "gpt-4-turbo-2024-04-09", # latest GPT-4 turbo model (Apr 2024)
                                 "gpt-4-1106-preview", # previous GPT-4 turbo preview model (Apr 2023)
                                 "gpt-4o-2024-05-13", # most advanced GPT-4o model (Oct 2023)
                                 "Meta-Llama-3-8B-Instruct", # Meta-Llama 3 model (March 2023)
                                 "Mistral-7B-Instruct-v0.2" # Mistral 7B model (?)
                                 ])
    parser.add_argument("--temperature", type=float, default=0.4, help="temperature of the model")
    parser.add_argument("--rounds", type=int, default=1, help="number of rounds")

    parser.add_argument("--plan", type=str, default="react", choices=["react"], help="planning strategy")
    parser.add_argument("--action", type=str, default="func", choices=["func", "block"], help="action type")
    parser.add_argument("--api", type=str, default="full", choices=["full", "kg", "news"], help="api type")
    parser.add_argument("--max_steps", type=int, default=20, help="maximum action steps")

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
    prompt_module_name = f'prompts_{args.plan}_{args.action}_{args.api}'
    if ('llama' in args.model_name.lower()) or ('mistral' in args.model_name.lower()):
        prompt_module_name += '_open'
    prompt_module = importlib.import_module(prompt_module_name)

    # load database
    data_kg = pd.read_csv(os.path.join(args.data_dir, "data_kg.csv"), sep='\t', dtype=str)
    data_news = pd.read_csv(os.path.join(args.data_dir, "data_news.csv"), sep='\t', dtype=str)
    # dict_code2relation = json.load(open('./../data/info/dict_code2relation.json'))

    # load api description
    api_dir = args.api_dir
    if args.api != 'full':
        api_dir = api_dir.replace('full', args.api)
    api_description = open(api_dir, 'r').read()

    # load query dataset
    data_query = pd.read_csv(os.path.join(args.data_dir, args.dataset, 'relation_query.csv'), sep='\t', dtype=str)

    query_ids = [i for i in range(1, len(data_query) + 1)]

    agent = ReactAgent(action_type=args.action,
                       max_steps=args.max_steps, prompt_module=prompt_module,
                       api_description=api_description, react_llm_name=args.model_name, temperature=args.temperature)

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

            # print('\n============\nFinal Status:\n')
            # print(f'End State: {end_state}')
            # print(f'Number of Steps Taken: {n_steps}\n')
            # print(f'Final Answer: {answer}')
            # print(f'Ground Truth Answer: {row["AnswerDict"]}\n')
            # print(f'Final Answer (Verbalized): {agent.verbalize_relation_code(dict_code2relation, eval(answer))}')
            # print(f'Ground Truth Answer (Verbalized): {agent.verbalize_relation_code(dict_code2relation, eval(row["AnswerDict"]))}')

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

