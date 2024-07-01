# plan = direct

from langchain.prompts import PromptTemplate

SYS_PROMPT_RELATION_QUERY = """You are an expert in forecasting future events based on historical data. The events are in the form of (date, subject country, relation, object country), where the countries are represented by ISO 3166-1 alpha-3 codes and the relations are represented by the CAMEO codes defined in the 'Conflict and Mediation Event Observations' ontology. The relations are hierarchical: first-level relations are general parent relations represented by two-digit CAMEO codes, while second-level relations are more specific child relations represented by three-digit CAMEO codes. Child relations have the same first two digits as their parent relations. For example, '01' is a first-level relation, and '010' and '011' are some of its second-level relations. The relations in the database are represented in the second-level form.

Your task is to forecast the future relations between two entities in a given query. The answer should be a JSON dictionary where the keys are the forecasted two-digit first-level CAMEO codes and the values are lists of forecasted three-digit second-level CAMEO codes that are child relations of the key. For example, 'Final Answer: {{"01": ["010", "011", "012"], "02": ["020", "023"]}}'.

The final answer will be evaluated based on the precision and recall of the forecasted first-level and second-level relations, so only include confident first-level and second-level CAMEO codes in your final forecast."""

PROMPT_RELATION_QUERY = """Query: Please forecast the relations that {actor1_name} will take towards {actor2_name} on {future_date_nlp} based on your knowledge up to {current_date_nlp}. I.e. forecast the relation CAMEO codes in query event Event(date={future_date}, head_entity=ISOCode({actor1_code}), relation=CAMEOCode(?), tail_entity=ISOCode({actor2_code})).
Final Answer:"""

sys_relation_prompt = PromptTemplate(
    input_variables=["current_date_nlp"],
    template=SYS_PROMPT_RELATION_QUERY)

relation_prompt = PromptTemplate(
    input_variables=["actor1_name", "actor2_name", "future_date_nlp", "current_date_nlp", "future_date", "actor1_code", "actor2_code"],
    template=PROMPT_RELATION_QUERY)
