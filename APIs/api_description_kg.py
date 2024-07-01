from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

@dataclass
class Date:
    """Represents a date."""
    date: str # Date in the format 'YYYY-MM-DD'
    # Example: Date("2022-01-01")

@dataclass
class DateRange:
    """Represents a range of dates (inclusive)."""
    start_date: Optional[Date] # If None, the earliest date is used
    end_date: Optional[Date] # If None, the current date is used
    # Example: DateRange(start_date=Date("2022-01-01"), end_date=Date("2022-01-31"))

@dataclass
class ISOCode:
    """Represents an ISO alpha-3 country code."""
    code: str # 3-letter ISO code
    # Example: ISOCode("USA")

@dataclass
class Country:
    """Represents a country entity."""
    iso_code: ISOCode
    name: str
    # Example: Country(iso_code=ISOCode("USA"), name="United States")

@dataclass
class CAMEOCode:
    """Represents a CAMEO verb code."""
    code: str # 2-digit CAMEO code for first level relations, 3-digit CAMEO code for second level relations
    # Example: CAMEOCode("01"), CAMEOCode("010")

@dataclass
class Relation:
    """Represents a relation."""
    cameo_code: CAMEOCode
    name: str
    description: str # A brief description of what event the relation represents
    # Example: Relation(cameo_code=CAMEOCode("010"), name="Make statement, not specified", description="All public statements expressed verbally or in action, not otherwise specified."

@dataclass
class Event:
    """Represents an event characterized by date, head entity, relation, and tail entity."""
    date: Date
    head_entity: ISOCode
    relation: CAMEOCode
    tail_entity: ISOCode
    # Example: Event(date=Date("2022-01-01"), head_entity=ISOCode("USA"), relation=CAMEOCode("010"), tail_entity=ISOCode("CAN"))

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
    pass

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
    pass

def map_relation_description_to_cameo(description: str) -> List[Relation]:
    """
    Maps a relation description to the most likely Relation objects.

    Parameters:
        description (str): The relation description to map.

    Returns:
        List[Relation]: A list of 5 most likely Relations sorted by relevance.

    Example:
        >>> map_relation_description_to_cameo("Fight with guns")
        [Reltion(cameo_code=CAMEOCode("19"）， name="Fight", description="All uses of conventional force and acts of war typically by organized armed groups."), Relation(cameo_code=CAMEOCode("193"), name="Fight with small arms and light weapons", description="Attack using small arms and light weapons such as rifles, machine-guns, and mortar shells."), Relation(cameo_code=CAMEOCode("190"), name="Use conventional military force, not specified", description="All uses of conventional force and acts of war typically by organized armed groups, not otherwise specified.")]
    """
    pass

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
    pass

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
    pass

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
    pass

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
    pass

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
    pass

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
    pass

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
    pass

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
    pass