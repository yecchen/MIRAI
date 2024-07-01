# plan = react

from langchain.prompts import PromptTemplate

EXTRACTION_PROMPT = """Please help me extract final answer for forecasting the future relations between two entities in a given query: forecast the relations that {actor1_name} will take towards {actor2_name} on {future_date_nlp} based on historical information up to {current_date_nlp}. I.e. forecast the relation CAMEO codes in query event Event(date={future_date}, head_entity=ISOCode({actor1_code}), relation=CAMEOCode(?), tail_entity=ISOCode({actor2_code})).

I have used interleaving 'Thought', 'Action', and 'Observation' steps to collect information from the database and perform the forecast. The database contains news articles from January 1, 2023 to the current date {current_date_nlp} and the events extracted from these articles. The events are in the form of (date, subject country, relation, object country), where the countries are represented by ISO 3166-1 alpha-3 codes and the relations are represented by the CAMEO codes defined in the 'Conflict and Mediation Event Observations' ontology. The relations are hierarchical: first-level relations are general parent relations represented by two-digit CAMEO codes, while second-level relations are more specific child relations represented by three-digit CAMEO codes. Child relations have the same first two digits as their parent relations. For example, '01' is a first-level relation, and '010' and '011' are some of its second-level relations. The relations in the database are represented in the second-level form.

The final forecast answer need to forecast both first-level and second-level CAMEO codes, and will be evaluated based on the precision and recall of both levels of relations. The final answer content should be a JSON dictionary where the keys are the forecasted two-digit first-level CAMEO codes and the values are lists of forecasted three-digit second-level CAMEO codes that are child relations of the key. For example, {{"01": ["010", "011", "012"], "02": ["020", "023"]}}.

The latest information and forecast I have collected is as follows:
{info}

If final forecast answer has been made in the collected information indicated by "Final Answer:", you must only reformat the final forecast answer in the expected JSON dictionary format inside XML tags. For example: <answer>{{"01": ["010", "011", "012"], "02": ["020", "023"]}}</answer>.
Otherwise, if no final forecast is made, you must reason based on the information you have collected and generate a confident final forecast answer to the query, and then reformat your answer in the expected JSON dictionary format inside XML tags.
"""

extraction_prompt = PromptTemplate(
    input_variables=["current_date_nlp", "actor1_name", "actor2_name", "future_date_nlp", "future_date", "actor1_code", "actor2_code", "info"],
    template=EXTRACTION_PROMPT)
