from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import datetime
import json
import openai
import pandas as pd
import numpy as np
import os
from rank_bm25 import BM25Okapi

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# global variables
DEFAULT_START_DATE = "2023-01-01"
DEFAULT_END_DATE = None

# helper function to set and get the default end date
def set_default_end_date(date: str):
    global DEFAULT_END_DATE
    DEFAULT_END_DATE = date

def get_default_end_date():
    return DEFAULT_END_DATE

def use_end_date():
    if DEFAULT_END_DATE is None:
        print("DEFAULT_END_DATE is not set.")
    else:
        print(f"The DEFAULT_END_DATE is set to {DEFAULT_END_DATE}.")

# load kg data
data_kg = pd.read_csv('./../data/MIRAI/data_kg.csv', sep='\t', dtype=str)
# columns: DateStr, Actor1CountryCode, Actor2CountryCode, EventBaseCode, Actor1CountryName, Actor2CountryName, RelName, QuadEventCode, QuadEventName, Docid, Docids

# load news data
data_news = pd.read_csv('./../data/MIRAI/data_news.csv', sep='\t', dtype=str)
# columns: Docid, MD5, URL, Date, Title, Text, Abstract

# load country data
dict_iso2alternames = json.load(open('./../data/info/dict_iso2alternames_GeoNames.json'))
country_embeddings = np.load('./../data/info/country_embeddings.npy')

# create a dictionary mapping country names to ISO codes
dict_countryname2iso = {}
for iso, names in dict_iso2alternames.items():
    for name in names:
        dict_countryname2iso[name] = iso

# load relation data
dict_code2relation = json.load(open('./../data/info/dict_code2relation.json'))
relation_embeddings = np.load('./../data/info/relation_embeddings.npy')

# create a dictionary mapping relation names or descriptions to CAMEO codes
dict_relation2code = {}
for code, info in dict_code2relation.items():
    dict_relation2code[info['Name']] = code
    dict_relation2code[info['Description']] = code

# embedding functions
def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model).data[0].embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@dataclass
class Date:
    """Represents a date."""
    date: str # Date in the format 'YYYY-MM-DD'
    # Example: Date("2022-01-01")

    def __init__(self, date: str):
        # check type
        if not isinstance(date, str):
            raise ValueError(f"Attribute 'date' of class Date must be a string in the format 'YYYY-MM-DD', but received: {date} in type {type(date)}")

        # check if date is in the correct format by trying to convert it to a date object
        try:
            datetime.datetime.strptime(date, '%Y-%m-%d')
        except ValueError:
            raise ValueError(f"Date must be in the format 'YYYY-MM-DD', but received: {date}")
        if date < DEFAULT_START_DATE:
            raise ValueError(f"Date must be on or after {DEFAULT_START_DATE}, but received: {date}")
        if DEFAULT_END_DATE and date > DEFAULT_END_DATE:
            raise ValueError(f"Date must be on or before the current date {DEFAULT_END_DATE}, but received: {date}")

        self.date = date

    def __str__(self):
        return f"Date('{self.date}')"

    def __hash__(self):
        return hash(self.date)

    def __eq__(self, other):
        if isinstance(other, Date):
            return self.date == other.date
        return False


@dataclass
class DateRange:
    """Represents a range of dates (inclusive)."""
    start_date: Optional[Date] # If None, the earliest date is used
    end_date: Optional[Date] # If None, the current date is used
    # Example: DateRange(start_date=Date("2022-01-01"), end_date=Date("2022-01-31"))

    def __init__(self, start_date: Optional[Date] = None, end_date: Optional[Date] = None):
        # check type
        if start_date and not isinstance(start_date, Date):
            raise ValueError(f"Attribute 'start_date' of class DateRange must be a Date object, but received type {type(start_date)}")
        if end_date and not isinstance(end_date, Date):
            raise ValueError(f"Attribute 'end_date' of class DateRange must be a Date object, but received type {type(end_date)}")

        self.start_date = start_date if start_date else Date(DEFAULT_START_DATE)
        self.end_date = end_date if end_date else Date(DEFAULT_END_DATE)
        if start_date and end_date and start_date.date > end_date.date:
            raise ValueError("Start date must be before or equal to end date, but received: start_date={}, end_date={}".format(start_date.date, end_date.date))

    def __str__(self):
        return f"DateRange(start_date={self.start_date}, end_date={self.end_date})"

    def __hash__(self):
        return hash((self.start_date, self.end_date))

    def __eq__(self, other):
        if isinstance(other, DateRange):
            return self.start_date == other.start_date and self.end_date == other.end_date
        return False

@dataclass
class ISOCode:
    """Represents an ISO alpha-3 country code."""
    code: str # 3-letter ISO code
    # Example: ISOCode("USA")

    def __init__(self, code: str):
        # check type
        if not isinstance(code, str):
            raise ValueError(f"Attribute 'code' of class ISOCode must be a string, but received type {type(code)}")

        if len(code) != 3:
            raise ValueError(f"ISO code must be a 3-letter string, but received: {code}")
        if code not in dict_iso2alternames:
            raise ValueError(f"ISO code must be a valid ISO alpha-3 country code, but received: {code}")
        self.code = code

    def __str__(self):
        return f"ISOCode('{self.code}')"

    def __hash__(self):
        return hash(self.code)

    def __eq__(self, other):
        if isinstance(other, ISOCode):
            return self.code == other.code
        return False

@dataclass
class Country:
    """Represents a country entity."""
    iso_code: ISOCode
    name: str
    # Example: Country(iso_code=ISOCode("USA"), name="United States")

    def __init__(self, iso_code: ISOCode, name: str):
        # check type
        if not isinstance(iso_code, ISOCode):
            raise ValueError(f"Attribute 'iso_code' of class Country must be an ISOCode object, but received type {type(iso_code)}")
        if not isinstance(name, str):
            raise ValueError(f"Attribute 'name' of class Country must be a string, but received type {type(name)}")

        if dict_iso2alternames[iso_code.code][0] != name:
            raise ValueError(f"Country name must match the name corresponding to the ISO code, but received: {name} for ISO code: {iso_code.code}")
        self.iso_code = iso_code
        self.name = name

    def __str__(self):
        return f"Country(iso_code={self.iso_code}, name='{self.name}')"

    def __hash__(self):
        return hash((self.iso_code, self.name))

    def __eq__(self, other):
        if isinstance(other, Country):
            return self.iso_code == other.iso_code and self.name == other.name
        return False

@dataclass
class CAMEOCode:
    """Represents a CAMEO verb code."""
    code: str # 2-digit CAMEO code for first level relations, 3-digit CAMEO code for second level relations
    # Example: CAMEOCode("01"), CAMEOCode("010")

    def __init__(self, code: str):
        # check type
        if not isinstance(code, str):
            raise ValueError(f"Attribute 'code' of class CAMEOCode must be a string, but received type {type(code)}")

        if len(code) not in [2, 3]:
            raise ValueError(f"CAMEO code must be a valid 2 or 3-digit string defined in the 'Conflict and Mediation Event Observations' Codebook, but received: {code}")
        if code not in dict_code2relation:
            raise ValueError(f"CAMEO code must be a valid CAMEO code defined in the 'Conflict and Mediation Event Observations' Codebook, but received: {code}")
        self.code = code

    def __str__(self):
        return f"CAMEOCode('{self.code}')"

    def __hash__(self):
        return hash(self.code)

    def __eq__(self, other):
        if isinstance(other, CAMEOCode):
            return self.code == other.code
        return False

@dataclass
class Relation:
    """Represents a relation."""
    cameo_code: CAMEOCode
    name: str
    description: str # A brief description of what event the relation represents
    # Example: Relation(cameo_code=CAMEOCode("010"), name="Make statement, not specified", description="All public statements expressed verbally or in action, not otherwise specified."

    def __init__(self, cameo_code: CAMEOCode, name: str, description: str):
        # check type
        if not isinstance(cameo_code, CAMEOCode):
            raise ValueError(f"Attribute 'cameo_code' of class Relation must be a CAMEOCode object, but received type {type(cameo_code)}")
        if not isinstance(name, str):
            raise ValueError(f"Attribute 'name' of class Relation must be a string, but received type {type(name)}")
        if not isinstance(description, str):
            raise ValueError(f"Attribute 'description' of class Relation must be a string, but received type {type(description)}")

        if dict_code2relation[cameo_code.code]['Name'] != name:
            raise ValueError(f"Relation name must match the name corresponding to the CAMEO code, but received: {name} for CAMEO code {cameo_code.code}")
        if dict_code2relation[cameo_code.code]['Description'] != description:
            raise ValueError(f"Relation description must match the description corresponding to the CAMEO code, but received: {description} for CAMEO code {cameo_code.code}")
        self.cameo_code = cameo_code
        self.name = name
        self.description = description

    def __str__(self):
        return f"Relation(cameo_code={self.cameo_code}, name='{self.name}', description='{self.description}')"

    def __hash__(self):
        return hash((self.cameo_code, self.name, self.description))

    def __eq__(self, other):
        if isinstance(other, Relation):
            return self.cameo_code == other.cameo_code and self.name == other.name and self.description == other.description
        return False

@dataclass
class Event:
    """Represents an event characterized by date, head entity, relation, and tail entity."""
    date: Date
    head_entity: ISOCode
    relation: CAMEOCode
    tail_entity: ISOCode
    # Example: Event(date=Date("2022-01-01"), head_entity=ISOCode("USA"), relation=CAMEOCode("010"), tail_entity=ISOCode("CAN"))

    def __init__(self, date: Date, head_entity: ISOCode, relation: CAMEOCode, tail_entity: ISOCode):
        # check type
        if not isinstance(date, Date):
            raise ValueError(f"Attribute 'date' of class Event must be a Date object, but received type {type(date)}")
        if not isinstance(head_entity, ISOCode):
            raise ValueError(f"Attribute 'head_entity' of class Event must be an ISOCode object, but received type {type(head_entity)}")
        if not isinstance(relation, CAMEOCode):
            raise ValueError(f"Attribute 'relation' of class Event must be a CAMEOCode object, but received type {type(relation)}")
        if not isinstance(tail_entity, ISOCode):
            raise ValueError(f"Attribute 'tail_entity' of class Event must be an ISOCode object, but received type {type(tail_entity)}")

        self.date = date
        self.head_entity = head_entity
        self.relation = relation
        self.tail_entity = tail_entity

    def __str__(self):
        return f"Event(date={self.date}, head_entity={self.head_entity}, relation={self.relation}, tail_entity={self.tail_entity})"

    def __hash__(self):
        return hash((self.date, self.head_entity, self.relation, self.tail_entity))

    def __eq__(self, other):
        if isinstance(other, Event):
            return self.date == other.date and self.head_entity == other.head_entity and self.relation == other.relation and self.tail_entity == other.tail_entity
        return False

@dataclass
class NewsArticle:
    """Represents a news article, including metadata and content."""
    date: Date
    title: str
    content: str # Full text content of the news article
    events: List[Event] # List of events mentioned in the article
    # Example: NewsArticle(date=Date("2022-01-01"), title="Trade agreement between USA and China", content="On January 1, 2022, a trade agreement was signed between the USA and China...", events=[Event(date=Date("2022-01-01"), head_entity=ISOCode("USA"), relation=CAMEOCode("010"), tail_entity=ISOCode("CHN")])

    def __init__(self, date: Date, title: str, content: str, events: List[Event]):
        # check type
        if not isinstance(date, Date):
            raise ValueError(f"Attribute 'date' of class NewsArticle must be a Date object, but received type {type(date)}")
        if not isinstance(title, str):
            raise ValueError(f"Attribute 'title' of class NewsArticle must be a string, but received type {type(title)}")
        if not isinstance(content, str):
            raise ValueError(f"Attribute 'content' of class NewsArticle must be a string, but received type {type(content)}")
        if not isinstance(events, list):
            raise ValueError(f"Attribute 'events' of class NewsArticle must be a list, but received type {type(events)}")

        self.date = date
        self.title = title
        self.content = content
        self.events = events

    def __str__(self):
        return f"{self.date}:\n{self.title}\n{self.content}"

    def __hash__(self):
        return hash((self.date, self.title, self.content))

    def __eq__(self, other):
        if isinstance(other, NewsArticle):
            return self.date == other.date and self.title == other.title and self.content == other.content
        return False

def map_country_name_to_iso(name: str) -> List[Country]:
    """
    Maps a country name to the most likely corresponding Country objects with ISO codes.

    Parameters:
        name (str): The country name to map.

    Returns:
        List[Country]: A list of 5 most likely Country objects sorted by relevance.

    Example:
        >>> map_country_name_to_iso("Korea")
        [Country(iso_code=ISOCode("KOR"), name="Republic of Korea"), Country(iso_code=ISOCode("PRK"), name="Democratic People's Republic of Korea")]
    """
    # check type
    if not isinstance(name, str):
        raise ValueError(f"Input 'name' must be a string, but received type {type(name)}")

    if name in dict_countryname2iso:
        return [Country(iso_code=ISOCode(dict_countryname2iso[name]), name=name)]
    else:
        # get top 5 ISO codes with the highest cosine similarity
        name_embeddings = get_embedding(name)
        similarities = cosine_similarity(country_embeddings, name_embeddings)
        top_indices = np.argsort(similarities)[::-1][:5]
        countries = []
        for idx in top_indices:
            iso_code = list(dict_iso2alternames.keys())[idx]
            curr_name = dict_iso2alternames[iso_code][0]
            countries.append(Country(iso_code=ISOCode(iso_code), name=curr_name))
        return countries

def map_iso_to_country_name(iso_code: ISOCode) -> str:
    """
    Maps an ISO code to a country name.

    Parameters:
        iso_code (ISOCode): The ISO code to map.

    Returns:
        str: The corresponding country name.

    Example:
        >>> map_iso_to_country_name(ISOCode("CHN"))
        "China"
    """
    # check type
    if not isinstance(iso_code, ISOCode):
        raise ValueError(f"Input 'iso_code' must be an ISOCode object, but received type {type(iso_code)}")

    return dict_iso2alternames[iso_code.code][0]

def map_relation_description_to_cameo(description: str) -> List[Relation]:
    """
    Maps a relation description to the most likely Relation objects.

    Parameters:
        description (str): The relation description to map.

    Returns:
        List[Relation]: A list of 5 most likely Relations sorted by relevance.

    Example:
        >>> map_relation_description_to_cameo("Fight with guns")
        [Reltion(cameo_code=CAMEOCode("19"), name="Fight", description="All uses of conventional force and acts of war typically by organized armed groups."), Relation(cameo_code=CAMEOCode("193"), name="Fight with small arms and light weapons", description="Attack using small arms and light weapons such as rifles, machine-guns, and mortar shells."), Relation(cameo_code=CAMEOCode("190"), name="Use conventional military force, not specified", description="All uses of conventional force and acts of war typically by organized armed groups, not otherwise specified.")]
    """
    # check type
    if not isinstance(description, str):
        raise ValueError(f"Input 'description' must be a string, but received type {type(description)}")

    if description in dict_relation2code:
        code = dict_relation2code[description]
        return [Relation(cameo_code=CAMEOCode(code), name=dict_code2relation[code]['Name'], description=dict_code2relation[code]['Description'])]
    else:
        # get top 5 CAMEO codes with the highest cosine similarity
        description_embedding = get_embedding(description)
        similarities = cosine_similarity(relation_embeddings, description_embedding)
        top_indices = np.argsort(similarities)[::-1][:5]
        relations = []
        for idx in top_indices:
            code = list(dict_code2relation.keys())[idx]
            relations.append(Relation(cameo_code=CAMEOCode(code), name=dict_code2relation[code]['Name'], description=dict_code2relation[code]['Description']))
        return relations

def map_cameo_to_relation(cameo_code: CAMEOCode) -> Relation:
    """
    Maps a CAMEO code to a relation, including its name and description.

    Parameters:
        cameo_code (CAMEOCode): The CAMEO code to map.

    Returns:
        Relation: The corresponding relation.

    Example:
        >>> map_cameo_to_relation(CAMEOCode("190"))
        Relation(cameo_code=CAMEOCode("190"), name="Use conventional military force, not specified", description="All uses of conventional force and acts of war typically by organized armed groups, not otherwise specified.")
    """
    # check type
    if not isinstance(cameo_code, CAMEOCode):
        raise ValueError(f"Input 'cameo_code' must be a CAMEOCode object, but received type {type(cameo_code)}")

    info = dict_code2relation[cameo_code.code]
    return Relation(cameo_code=cameo_code, name=info['Name'], description=info['Description'])

def get_parent_relation(cameo_code: CAMEOCode) -> Relation:
    """
    Retrieves the parent relation of a given relation identified by CAMEO code.

    Parameters:
        cameo_code (CAMEOCode): The CAMEO code of the relation whose parent is sought. Only second level relations are accepted.

    Returns:
        Relation: The first level parent relation.

    Example:
        >>> get_parent_relation(CAMEOCode("193"))
        Relation(cameo_code=CAMEOCode("19"), name="Fight", description="All uses of conventional force and acts of war typically by organized armed groups.")
    """
    # check type
    if not isinstance(cameo_code, CAMEOCode):
        raise ValueError(f"Input 'cameo_code' must be a CAMEOCode object, but received type {type(cameo_code)}")

    if len(cameo_code.code) != 3:
        raise ValueError("Only second level relations are accepted, but received: {}".format(cameo_code.code))
    parent_code =cameo_code.code[:2]
    return map_cameo_to_relation(CAMEOCode(parent_code))

def get_child_relations(cameo_code: CAMEOCode) -> List[Relation]:
    """
    Retrieves child relations of a given relation identified by CAMEO code.

    Parameters:
        cameo_code (CAMEOCode): The CAMEO code of the relation whose children are sought. Only first level relations are accepted.

    Returns:
        List[Relation]: A list of second level child relations.

    Example:
        >>> get_child_relations(CAMEOCode("19"))
        [Relation(caemo_code=CAMEOCode("190"), name="Use conventional military force, not specified", description="All uses of conventional force and acts of war typically by organized armed groups, not otherwise specified."), Relation(cameo_code=CAMEOCode("191"), name="Impose blockade or restrict movement", description="Prevent entry into and/or exit from a territory using armed forces."), ...]
    """
    # check type
    if not isinstance(cameo_code, CAMEOCode):
        raise ValueError(f"Input 'cameo_code' must be a CAMEOCode object, but received type {type(cameo_code)}")

    if len(cameo_code.code) != 2:
        raise ValueError("Only first level relations are accepted, but received: {}".format(cameo_code.code))
    children = []
    for code, info in dict_code2relation.items():
        if code[:2] == cameo_code.code and len(code) == 3:
            children.append(Relation(cameo_code=CAMEOCode(code), name=info['Name'], description=info['Description']))
    return children

def get_sibling_relations(cameo_code: CAMEOCode) -> List[Relation]:
    """
    Retrieves sibling relations of a given relation identified by CAMEO code.

    Parameters:
        cameo_code (CAMEOCode): The CAMEO code of the relation whose siblings are sought. Both first and second level relations are accepted.

    Returns:
        List[Relation]: A list of sibling relations at the same level.

    Example:
        >>> get_sibling_relations(CAMEOCode("193"))
        [Relation(caemo_code=CAMEOCode("190"), name="Use conventional military force, not specified", description="All uses of conventional force and acts of war typically by organized armed groups, not otherwise specified."), Relation(cameo_code=CAMEOCode("191"), name="Impose blockade or restrict movement", description="Prevent entry into and/or exit from a territory using armed forces."), ...]
    """
    # check type
    if not isinstance(cameo_code, CAMEOCode):
        raise ValueError(f"Input 'cameo_code' must be a CAMEOCode object, but received type {type(cameo_code)}")

    if len(cameo_code.code) == 3:
        return get_child_relations(get_parent_relation(cameo_code).cameo_code)
    elif len(cameo_code.code) == 2:
        # get '01' to '20' relations
        first_level_codes = [str(i).zfill(2) for i in range(1, 21)]
        relations = []
        for code in first_level_codes:
            relations.append(map_cameo_to_relation(CAMEOCode(code)))
        return relations

def count_events(date_range: Optional[DateRange] = None, head_entities: Optional[List[ISOCode]] = None, tail_entities: Optional[List[ISOCode]] = None, relations: Optional[List[CAMEOCode]] = None) -> int:
    """
    Counts the number of events in the knowledge graph based on specified conditions.

    Parameters:
        date_range (Optional[DateRange]): Range of dates to filter the events. If None, all dates are included.
        head_entities (Optional[List[ISOCode]]): List of head entity ISO codes to be included. If None, all countries are included.
        tail_entities (Optional[List[ISOCode]]): List of tail entity ISO codes to be included. If None, all countries are included.
        relations (Optional[List[CAMEOCode]]): List of relation CAMEO codes to be included. If first level relations are listed, all second level relations under them are included. If None, all relations are included.

    Returns:
        int: Count of unique events matching the conditions.

    Example:
        >>> count_events(date_range=DateRange(start_date=Date("2022-01-01"), end_date=Date("2022-01-31")), head_entities=[ISOCode("USA"), ISOCode("CHN")], tail_entities=None, relations=[CAMEOCode("010")])
        4
    """
    # check type
    if date_range and not isinstance(date_range, DateRange):
        raise ValueError(f"Input 'date_range' must be a DateRange object, but received type {type(date_range)}")
    if head_entities and not isinstance(head_entities, list):
        raise ValueError(f"Input 'head_entities' must be a list, but received type {type(head_entities)}")
    if head_entities and not all(isinstance(iso, ISOCode) for iso in head_entities):
        raise ValueError(f"Elements in 'head_entities' must be ISOCode objects")
    if tail_entities and not isinstance(tail_entities, list):
        raise ValueError(f"Input 'tail_entities' must be a list, but received type {type(tail_entities)}")
    if tail_entities and not all(isinstance(iso, ISOCode) for iso in tail_entities):
        raise ValueError(f"Elements in 'tail_entities' must be ISOCode objects")
    if relations and not isinstance(relations, list):
        raise ValueError(f"Input 'relations' must be a list, but received type {type(relations)}")
    if relations and not all(isinstance(code, CAMEOCode) for code in relations):
        raise ValueError(f"Elements in 'relations' must be CAMEOCode objects")

    # process data_kg by filtering based on the specified conditions
    curr_data = data_kg.copy()
    curr_data.drop_duplicates(subset=['QuadEventCode'], inplace=True)
    curr_data = curr_data[curr_data['DateStr'] <= DEFAULT_END_DATE]
    if date_range:
        curr_data = curr_data[(curr_data['DateStr'] >= date_range.start_date.date) & (curr_data['DateStr'] <= date_range.end_date.date)]
    if head_entities:
        curr_data = curr_data[curr_data['Actor1CountryCode'].isin([iso.code for iso in head_entities])]
    if tail_entities:
        curr_data = curr_data[curr_data['Actor2CountryCode'].isin([iso.code for iso in tail_entities])]
    if relations:
        # if first level relations are listed, include all second level relations under them
        for code in relations:
            if len(code.code) == 2:
                relations.extend([CAMEOCode(c) for c in dict_code2relation if c[:2] == code.code and len(c) == 3])
        curr_data = curr_data[curr_data['EventBaseCode'].isin([code.code for code in relations])]
    return len(curr_data)

def get_events(date_range: Optional[DateRange] = None, head_entities: Optional[List[ISOCode]] = None, tail_entities: Optional[List[ISOCode]] = None, relations: Optional[List[CAMEOCode]] = None, text_description: Optional[str] = None) -> List[Event]:
    """
    Retrieves events from the knowledge graph based on specified conditions.
    Inherits common filter parameters from count_events. See count_events for more details on these parameters.

    Additional Parameters:
        text_description (Optional[str]): Textual description to match with the source news articles of events. If None, the returned events are sorted by date in descending order; otherwise, sorted by relevance of the source news article to the description.

    Returns:
        List[Event]: A list of maximum 30 events matching the specified conditions.

    Example:
        >>> get_events(date_range=DateRange(start_date=Date("2022-01-01"), end_date=Date("2022-01-31")), head_entities=[ISOCode("USA"), ISOCode("CHN")], tail_entities=None, relations=[CAMEOCode("010")], text_description="economic trade")
        [Event(date=Date("2022-01-15"), head_entity=ISOCode("USA"), relation=CAMEOCode("010"), tail_entity=ISOCode("CAN")), Event(date=Date("2022-01-10"), head_entity=ISOCode("CHN"), relation=CAMEOCode("010"), tail_entity=ISOCode("USA")), ...]
    """
    # check type
    if date_range and not isinstance(date_range, DateRange):
        raise ValueError(f"Input 'date_range' must be a DateRange object, but received type {type(date_range)}")
    if head_entities and not isinstance(head_entities, list):
        raise ValueError(f"Input 'head_entities' must be a list, but received type {type(head_entities)}")
    if head_entities and not all(isinstance(iso, ISOCode) for iso in head_entities):
        raise ValueError(f"Elements in 'head_entities' must be ISOCode objects")
    if tail_entities and not isinstance(tail_entities, list):
        raise ValueError(f"Input 'tail_entities' must be a list, but received type {type(tail_entities)}")
    if tail_entities and not all(isinstance(iso, ISOCode) for iso in tail_entities):
        raise ValueError(f"Elements in 'tail_entities' must be ISOCode objects")
    if relations and not isinstance(relations, list):
        raise ValueError(f"Input 'relations' must be a list, but received type {type(relations)}")
    if relations and not all(isinstance(code, CAMEOCode) for code in relations):
        raise ValueError(f"Elements in 'relations' must be CAMEOCode objects")
    if text_description and not isinstance(text_description, str):
        raise ValueError(f"Input 'text_description' must be a string, but received type {type(text_description)}")

    # process data_kg by filtering based on the specified conditions
    curr_data = data_kg.copy()
    curr_data = curr_data[curr_data['DateStr'] <= DEFAULT_END_DATE]
    if date_range:
        curr_data = curr_data[(curr_data['DateStr'] >= date_range.start_date.date) & (curr_data['DateStr'] <= date_range.end_date.date)]
    if head_entities:
        curr_data = curr_data[curr_data['Actor1CountryCode'].isin([iso.code for iso in head_entities])]
    if tail_entities:
        curr_data = curr_data[curr_data['Actor2CountryCode'].isin([iso.code for iso in tail_entities])]
    if relations:
        # if first level relations are listed, include all second level relations under them
        for code in relations:
            if len(code.code) == 2:
                relations.extend([CAMEOCode(c) for c in dict_code2relation if c[:2] == code.code and len(c) == 3])
        curr_data = curr_data[curr_data['EventBaseCode'].isin([code.code for code in relations])]
    if not text_description:
        # get max 30 events from the filtered data
        events = []
        curr_data.drop_duplicates(subset=['QuadEventCode'], inplace=True)
        # sorted by date in descending order
        curr_data.sort_values(by='DateStr', ascending=False, inplace=True)
        count = 0
        for _, row in curr_data.iterrows():
            if count >= 30:
                break
            events.append(Event(date=Date(row['DateStr']), head_entity=ISOCode(row['Actor1CountryCode']), relation=CAMEOCode(row['EventBaseCode']), tail_entity=ISOCode(row['Actor2CountryCode'])))
        return events[:30]
    else:
        # concat the Docids list of current data to get the news articles
        docids_list = [eval(docids) for docids in curr_data['Docids'].unique().tolist()]
        docids = list(set([item for sublist in docids_list for item in sublist]))
        docids = [str(docid) for docid in docids]
        news_articles = data_news[data_news['Docid'].isin(docids)]
        # get the max 30 docids with the highest BM25 score to the text_description
        corpus = news_articles['Title'] + ' ' + news_articles['Text']
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = text_description.split(" ")
        doc_scores = bm25.get_scores(tokenized_query)
        top_indices = np.argsort(doc_scores)[::-1][:30]
        news_articles = news_articles.iloc[top_indices]
        docids = news_articles['Docid'].tolist()
        # get max 30 events from the filtered data
        events = set()
        for docid in docids:
            if len(events) >= 30:
                break
            doc_curr_data = curr_data[curr_data['Docid'] == docid]
            # reverse the order of the events to get the latest events first
            doc_curr_data = doc_curr_data.sort_values(by='DateStr', ascending=False)
            for _, row in doc_curr_data.iterrows():
                events.add(Event(date=Date(row['DateStr']), head_entity=ISOCode(row['Actor1CountryCode']), relation=CAMEOCode(row['EventBaseCode']), tail_entity=ISOCode(row['Actor2CountryCode'])))
        return list(events)

def get_entity_distribution(date_range: Optional[DateRange] = None, involved_relations: Optional[List[CAMEOCode]] = None, interacted_entities: Optional[List[ISOCode]] = None, entity_role: Optional[str] = None) -> Dict[ISOCode, int]:
    """
    Gets the distribution of entities in the knowledge graph under specified conditions.

    Parameters:
        date_range (Optional[DateRange]): Range of dates to filter the events. If None, all dates are included.
        involved_relations (Optional[List[CAMEOCode]]): List of relations that the returned entities must be involved in any of. If first level relations are listed, all second level relations under them are included. If None, all relations are included.
        interacted_entities (Optional[List[ISOCode]]): List of entities that the returned entities must have interacted with any of. If None, all entities are included.
        entity_role (Optional[EntityRole]): Specifies the role of the returned entity in the events. Options are 'head', 'tail', or 'both'. If 'both' or None, the returned entity can be either head or tail.

    Returns:
        Dict[ISOCode, int]: A dictionary mapping returned entities' ISO codes to the number of events with the specified conditions in which they are involved, sorted by counts in descending order.

    Example:
        >>> get_entity_distribution(date_range=DateRange(start_date=Date("2022-01-01"), end_date=Date("2022-01-31")), involved_relations=[CAMEOCode("010")], interacted_entities=[ISOCode("USA"), ISOCode("CHN")], entity_role="tail")
        {ISOCode("USA"): 3, ISOCode("CHN"): 1}
    """
    # check type
    if date_range and not isinstance(date_range, DateRange):
        raise ValueError(f"Input 'date_range' must be a DateRange object, but received type {type(date_range)}")
    if involved_relations and not isinstance(involved_relations, list):
        raise ValueError(f"Input 'involved_relations' must be a list, but received type {type(involved_relations)}")
    if involved_relations and not all(isinstance(code, CAMEOCode) for code in involved_relations):
        raise ValueError(f"Elements in 'involved_relations' must be CAMEOCode objects")
    if interacted_entities and not isinstance(interacted_entities, list):
        raise ValueError(f"Input 'interacted_entities' must be a list, but received type {type(interacted_entities)}")
    if interacted_entities and not all(isinstance(iso, ISOCode) for iso in interacted_entities):
        raise ValueError(f"Elements in 'interacted_entities' must be ISOCode objects")
    if entity_role and entity_role not in ['head', 'tail', 'both']:
        raise ValueError(f"Input 'entity_role' must be a string 'head', 'tail', or 'both', but received: {entity_role}")

    # process data_kg by filtering based on the specified conditions
    curr_data = data_kg.copy()
    curr_data.drop_duplicates(subset=['QuadEventCode'], inplace=True)
    curr_data = curr_data[curr_data['DateStr'] <= DEFAULT_END_DATE]
    if date_range:
        curr_data = curr_data[(curr_data['DateStr'] >= date_range.start_date.date) & (curr_data['DateStr'] <= date_range.end_date.date)]
    if involved_relations:
        # if first level relations are listed, include all second level relations under them
        for code in involved_relations:
            if len(code.code) == 2:
                involved_relations.extend([CAMEOCode(c) for c in dict_code2relation if c[:2] == code.code and len(c) == 3])
        curr_data = curr_data[curr_data['EventBaseCode'].isin([code.code for code in involved_relations])]
    if interacted_entities:
        if entity_role=='head':
            curr_data = curr_data[curr_data['Actor2CountryCode'].isin([iso.code for iso in interacted_entities])]
        elif entity_role=='tail':
            curr_data = curr_data[curr_data['Actor1CountryCode'].isin([iso.code for iso in interacted_entities])]
        else:
            curr_data = curr_data[(curr_data['Actor1CountryCode'].isin([iso.code for iso in interacted_entities])) | (curr_data['Actor2CountryCode'].isin([iso.code for iso in interacted_entities]))]
    # count the number of events for each entity
    entity_counts = curr_data['Actor1CountryCode']._append(curr_data['Actor2CountryCode']).value_counts()
    entity_counts = entity_counts.to_dict()
    # sort the dictionary by values in descending order
    entity_counts = dict(sorted(entity_counts.items(), key=lambda item: item[1], reverse=True))
    entity_counts = {ISOCode(key): value for key, value in entity_counts.items()}
    return entity_counts

def get_relation_distribution(date_range: Optional[DateRange] = None, head_entities: Optional[List[ISOCode]] = None, tail_entities: Optional[List[ISOCode]] = None) -> Dict[CAMEOCode, int]:
    """
    Gets the distribution of second level relations in the knowledge graph under specified conditions.

    Parameters:
        date_range (Optional[DateRange]): Range of dates to filter the events. If None, all dates are included.
        head_entities (Optional[List[ISOCode]]): List of head entities that the events must involve any of. If None, all head entities are included.
        tail_entities (Optional[List[ISOCode]]): List of tail entities that the events must involve any of. If None, all tail entities are included.

    Returns:
        Dict[CAMEOCode, int]: A dictionary mapping second level relations' CAMEO codes to the number of events with the specified conditions in which they are involved, sorted by counts in descending order.

    Example:
        >>> get_relation_distribution(date_range=DateRange(start_date=Date("2022-01-01"), end_date=Date("2022-01-31")), head_entities=[ISOCode("USA"), ISOCode("CHN")], tail_entities=None)
        {CAMEOCode("010"): 3, CAMEOCode("011"): 1}
    """
    # check type
    if date_range and not isinstance(date_range, DateRange):
        raise ValueError(f"Input 'date_range' must be a DateRange object, but received type {type(date_range)}")
    if head_entities and not isinstance(head_entities, list):
        raise ValueError(f"Input 'head_entities' must be a list, but received type {type(head_entities)}")
    if head_entities and not all(isinstance(iso, ISOCode) for iso in head_entities):
        raise ValueError(f"Elements in 'head_entities' must be ISOCode objects")
    if tail_entities and not isinstance(tail_entities, list):
        raise ValueError(f"Input 'tail_entities' must be a list, but received type {type(tail_entities)}")
    if tail_entities and not all(isinstance(iso, ISOCode) for iso in tail_entities):
        raise ValueError(f"Elements in 'tail_entities' must be ISOCode objects")

    # process data_kg by filtering based on the specified conditions
    curr_data = data_kg.copy()
    curr_data.drop_duplicates(subset=['QuadEventCode'], inplace=True)
    curr_data = curr_data[curr_data['DateStr'] <= DEFAULT_END_DATE]
    if date_range:
        curr_data = curr_data[(curr_data['DateStr'] >= date_range.start_date.date) & (curr_data['DateStr'] <= date_range.end_date.date)]
    if head_entities:
        curr_data = curr_data[curr_data['Actor1CountryCode'].isin([iso.code for iso in head_entities])]
    if tail_entities:
        curr_data = curr_data[curr_data['Actor2CountryCode'].isin([iso.code for iso in tail_entities])]
    # count the number of events for each relation
    relation_counts = curr_data['EventBaseCode'].value_counts()
    relation_counts = relation_counts.to_dict()
    # sort the dictionary by values in descending order
    relation_counts = dict(sorted(relation_counts.items(), key=lambda item: item[1], reverse=True))
    relation_counts = {CAMEOCode(key): value for key, value in relation_counts.items()}
    return relation_counts

def count_news_articles(date_range: Optional[DateRange] = None, head_entities: Optional[List[ISOCode]] = None, tail_entities: Optional[List[ISOCode]] = None, relations: Optional[List[CAMEOCode]] = None, keywords: Optional[List[str]] = None) -> int:
    """
    Counts the number of news articles based on specified conditions.

    Parameters:
        date_range (Optional[DateRange]): Range of dates to filter the news articles. If None, all dates are included.
        head_entities (Optional[List[ISOCode]]): At least one of the entities must be mentioned in the articles and be the head entity in the events. If None, all entities are included.
        tail_entities (Optional[List[ISOCode]]): At least one of the entities must be mentioned in the articles and be the tail entity in the events. If None, all entities are included.
        relations (Optional[List[CAMEOCode]]): At least one of the relations must be mentioned in the articles. If first level relations are listed, all second level relations under them are included. If None, all relations are included.
        keywords (Optional[List[str]]): At least one of the keywords must be present in the articles. If None, all articles are included.

    Returns:
        int: The count of news articles matching the conditions.

    Example:
        >>> count_news_articles(date_range=DateRange(start_date=Date("2022-01-01"), end_date=Date("2022-01-31")), head_entities=[ISOCode("USA"), ISOCode("CHN")], tail_entities=[ISOCode("USA"), ISOCode("CHN")], relations=[CAMEOCode("010")], keywords=["trade"])
        2
    """
    # check type
    if date_range and not isinstance(date_range, DateRange):
        raise ValueError(f"Input 'date_range' must be a DateRange object, but received type {type(date_range)}")
    if head_entities and not isinstance(head_entities, list):
        raise ValueError(f"Input 'head_entities' must be a list, but received type {type(head_entities)}")
    if head_entities and not all(isinstance(iso, ISOCode) for iso in head_entities):
        raise ValueError(f"Elements in 'head_entities' must be ISOCode objects")
    if tail_entities and not isinstance(tail_entities, list):
        raise ValueError(f"Input 'tail_entities' must be a list, but received type {type(tail_entities)}")
    if tail_entities and not all(isinstance(iso, ISOCode) for iso in tail_entities):
        raise ValueError(f"Elements in 'tail_entities' must be ISOCode objects")
    if relations and not isinstance(relations, list):
        raise ValueError(f"Input 'relations' must be a list, but received type {type(relations)}")
    if relations and not all(isinstance(code, CAMEOCode) for code in relations):
        raise ValueError(f"Elements in 'relations' must be CAMEOCode objects")
    if keywords and not isinstance(keywords, list):
        raise ValueError(f"Input 'keywords' must be a list, but received type {type(keywords)}")
    if keywords and not all(isinstance(keyword, str) for keyword in keywords):
        raise ValueError(f"Elements in 'keywords' must be strings")

    # process data_kg by filtering based on the specified conditions
    curr_data = data_kg.copy()
    curr_data.drop_duplicates(subset=['QuadEventCode'], inplace=True)
    curr_data = curr_data[curr_data['DateStr'] <= DEFAULT_END_DATE]
    if date_range:
        curr_data = curr_data[(curr_data['DateStr'] >= date_range.start_date.date) & (curr_data['DateStr'] <= date_range.end_date.date)]
    if head_entities:
        curr_data = curr_data[curr_data['Actor1CountryCode'].isin([iso.code for iso in head_entities])]
    if tail_entities:
        curr_data = curr_data[curr_data['Actor2CountryCode'].isin([iso.code for iso in tail_entities])]
    if relations:
        # if first level relations are listed, include all second level relations under them
        for code in relations:
            if len(code.code) == 2:
                relations.extend([CAMEOCode(c) for c in dict_code2relation if c[:2] == code.code and len(c) == 3])
        curr_data = curr_data[curr_data['EventBaseCode'].isin([code.code for code in relations])]
    # concat the Docids list of current data to get the news articles
    docids_list = [eval(docids) for docids in curr_data['Docids'].unique().tolist()]
    docids = list(set([item for sublist in docids_list for item in sublist]))
    docids = [str(docid) for docid in docids]
    news_articles = data_news[data_news['Docid'].isin(docids)]
    if keywords:
        # filter the news articles that contain at least one of the keywords in the title or text string
        news_articles = news_articles[news_articles['Title'].str.contains('|'.join(keywords), case=False) | news_articles['Text'].str.contains('|'.join(keywords), case=False)]
    return len(news_articles)

def get_news_articles(date_range: Optional[DateRange] = None, head_entities: Optional[List[ISOCode]] = None, tail_entities: Optional[List[ISOCode]] = None, relations: Optional[List[CAMEOCode]] = None, keywords: Optional[List[str]] = None, text_description: Optional[str] = None) -> List[Tuple[Date, str]]:
    """
    Retrieves news articles based on specified conditions.
    Inherits common filter parameters from count_news_articles. See count_news_articles for more details on these parameters.

    Additional Parameters:
        text_description (Optional[str]): Textual description to match with the news articles. If None, the returned articles are sorted by date in descending order; otherwise, sorted by relevance to the description.

    Returns:
        List[Tuple[Date, str]]: A list of maximum 15 news articles matching the specified conditions, each represented by a tuple of date and title.

    Example:
        >>> get_news_articles(date_range=DateRange(start_date=Date("2022-01-01"), end_date=Date("2022-01-31")), head_entities=[ISOCode("USA"), ISOCode("CHN")], tail_entities=[ISOCode("USA"), ISOCode("CHN")], relations=[CAMEOCode("010")], keywords=["trade"], text_description="Economic trade is encouraged between USA and China.")
        [(NewsArticle.date=Date("2022-01-15"), NewsArticle.title="China and USA sign trade deal"), (NewsArticle.date=Date("2022-01-10"), NewsArticle.title="Trade agreement between USA and China")]
    """
    # check type
    if date_range and not isinstance(date_range, DateRange):
        raise ValueError(f"Input 'date_range' must be a DateRange object, but received type {type(date_range)}")
    if head_entities and not isinstance(head_entities, list):
        raise ValueError(f"Input 'head_entities' must be a list, but received type {type(head_entities)}")
    if head_entities and not all(isinstance(iso, ISOCode) for iso in head_entities):
        raise ValueError(f"Elements in 'head_entities' must be ISOCode objects")
    if tail_entities and not isinstance(tail_entities, list):
        raise ValueError(f"Input 'tail_entities' must be a list, but received type {type(tail_entities)}")
    if tail_entities and not all(isinstance(iso, ISOCode) for iso in tail_entities):
        raise ValueError(f"Elements in 'tail_entities' must be ISOCode objects")
    if relations and not isinstance(relations, list):
        raise ValueError(f"Input 'relations' must be a list, but received type {type(relations)}")
    if relations and not all(isinstance(code, CAMEOCode) for code in relations):
        raise ValueError(f"Elements in 'relations' must be CAMEOCode objects")
    if keywords and not isinstance(keywords, list):
        raise ValueError(f"Input 'keywords' must be a list, but received type {type(keywords)}")
    if keywords and not all(isinstance(keyword, str) for keyword in keywords):
        raise ValueError(f"Elements in 'keywords' must be strings")
    if text_description and not isinstance(text_description, str):
        raise ValueError(f"Input 'text_description' must be a string, but received type {type(text_description)}")

    # process data_kg by filtering based on the specified conditions
    curr_data = data_kg.copy()
    curr_data.drop_duplicates(subset=['QuadEventCode'], inplace=True)
    curr_data = curr_data[curr_data['DateStr'] <= DEFAULT_END_DATE]
    if date_range:
        curr_data = curr_data[
            (curr_data['DateStr'] >= date_range.start_date.date) & (curr_data['DateStr'] <= date_range.end_date.date)]
    if head_entities:
        curr_data = curr_data[curr_data['Actor1CountryCode'].isin([iso.code for iso in head_entities])]
    if tail_entities:
        curr_data = curr_data[curr_data['Actor2CountryCode'].isin([iso.code for iso in tail_entities])]
    if relations:
        # if first level relations are listed, include all second level relations under them
        for code in relations:
            if len(code.code) == 2:
                relations.extend([CAMEOCode(c) for c in dict_code2relation if c[:2] == code.code and len(c) == 3])
        curr_data = curr_data[curr_data['EventBaseCode'].isin([code.code for code in relations])]
    docids_list = [eval(docids) for docids in curr_data['Docids'].unique().tolist()]
    docids = list(set([item for sublist in docids_list for item in sublist]))
    docids = [str(docid) for docid in docids]
    news_articles = data_news[data_news['Docid'].isin(docids)]
    if keywords:
        # filter the news articles that contain at least one of the keywords in the title or text string
        news_articles = news_articles[news_articles['Title'].str.contains('|'.join(keywords), case=False) | news_articles['Text'].str.contains('|'.join(keywords), case=False)]
    if not text_description:
        # get max 15 news articles from the filtered data
        # sorted by date in descending order
        news_articles.sort_values(by='Date', ascending=False, inplace=True)
        news_articles = news_articles[['Date', 'Title']].head(15)
        return [(Date(row['Date']), row['Title']) for _, row in news_articles.iterrows()]
    else:
        # get the max 15 news articles with the highest BM25 score to the text_description
        corpus = news_articles['Title'] + ' ' + news_articles['Text']
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = text_description.split(" ")
        doc_scores = bm25.get_scores(tokenized_query)
        top_indices = np.argsort(doc_scores)[::-1][:15]
        news_articles = news_articles.iloc[top_indices]
        return [(Date(row['Date']), row['Title']) for _, row in news_articles.iterrows()]

def browse_news_article(date: Date, title: str) -> str:
    """
    Retrieves the full text of a news article by its title.

    Parameters:
        date (Date): The date of the news article to retrieve.
        title (str): The title of the news article to retrieve.

    Returns:
        str: The date, the title and full contents of the news article.

    Example:
        >>> browse_news_article(Date("2022-01-10"), "Trade agreement between USA and China")
        2022-01-10:
        Trade agreement between USA and China
        On January 10, 2022, a trade agreement was signed between the USA and China to promote economic cooperation...
    """
    # check type
    if not isinstance(date, Date):
        raise ValueError(f"Input 'date' must be a Date object, but received type {type(date)}")
    if not isinstance(title, str):
        raise ValueError(f"Input 'title' must be a string, but received type {type(title)}")

    # process data_news to find the news article with the specified date and title
    curr_data = data_news[(data_news['Date'] == date.date) & (data_news['Title'] == title)]
    if len(curr_data) == 0:
        raise ValueError(f"No news article found with the specified date {date.date} and title {title}")
    return f"{date}:\n{title}\n{curr_data['Text'].values[0]}"