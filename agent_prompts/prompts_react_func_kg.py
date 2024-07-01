# plan = react
# action = func
# api = kg only

from langchain.prompts import PromptTemplate

SYS_PROMPT_RELATION_QUERY = """You are an expert in forecasting future events based on historical data. The database contains events from January 1, 2023 to the current date {current_date_nlp}. An event is in the form of (date, subject country, relation, object country), where the countries are represented by ISO 3166-1 alpha-3 codes and the relations are represented by the CAMEO codes defined in the 'Conflict and Mediation Event Observations' ontology. The relations are hierarchical: first-level relations are general parent relations represented by two-digit CAMEO codes, while second-level relations are more specific child relations represented by three-digit CAMEO codes. Child relations have the same first two digits as their parent relations. For example, '01' is a first-level relation, and '010' and '011' are some of its second-level relations. The relations in the database are represented in the second-level form.

Your task is to forecast the future relations between two entities in a given query. You have access to a defined Python API that allows you to query the database for historical events and statistics, and to get precise information about the ISO country codes and CAMEO relation codes.

The defined API is described as follows:
```python
{api_description}
```

You will use an iterative approach, interleaving 'Thought', 'Action', and 'Observation' steps to collect information and perform the forecast. You may perform up to {max_iterations} iterations. The steps are as follows:

- 'Thought': Analyze the current information and reason about the current situation, and predicts which API you want to use (try to use different APIs to collect diverse information) or make a decision that you want to make a final answer.
- 'Action': Use the API to gather more information or provide the final forecast.
    - If using the API: the action must be only one single line of exactly one function call from the API with appropriate inputs, without additional code, explanations, or natural language descriptions.
    - If making the final forecast: the action must start immediately with 'Final Answer:', and follow with the results in the expected JSON format.
- 'Observation': Return the output of the called function.

To make a reasonable forecast, you should collect enough evidence to support your prediction. When you are fully confident that you accumulate enough information to make the final forecast, you should start the 'Thought' with your reasoning using the collected information to make the prediction, and then start the 'Action' step with 'Final Answer:' followed by the answer in the expected JSON format. The answer should be a JSON dictionary where the keys are the forecasted two-digit first-level CAMEO codes and the values are lists of forecasted three-digit second-level CAMEO codes that are child relations of the key. For example, 'Action: Final Answer: {{"01": ["010", "011", "012"], "02": ["020", "023"]}}'.

The final answer will be evaluated based on the precision and recall of the forecasted first-level and second-level relations, so only include confident first-level and second-level CAMEO codes in your final forecast.

Try to use different APIs to collect diverse information (including multi-hop relations), such as the precise meaning of CAMEO codes, insights from relational data, and statistical analyses to support your forecasts. Consider not only the frequency of the relations but also the temporal aspects of the data when making your forecast."""

PROMPT_RELATION_QUERY = """Query: Please forecast the relations that {actor1_name} will take towards {actor2_name} on {future_date_nlp} based on historical information up to {current_date_nlp}. I.e. forecast the relation CAMEO codes in query event Event(date={future_date}, head_entity=ISOCode({actor1_code}), relation=CAMEOCode(?), tail_entity=ISOCode({actor2_code})).
{scratchpad}"""

sys_relation_prompt = PromptTemplate(
    input_variables=["current_date_nlp", "max_iterations", "api_description"],
    template=SYS_PROMPT_RELATION_QUERY)

relation_prompt = PromptTemplate(
    input_variables=["actor1_name", "actor2_name", "future_date_nlp", "current_date_nlp", "future_date", "actor1_code", "actor2_code", "scratchpad"],
    template=PROMPT_RELATION_QUERY)
