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

@dataclass
class NewsArticle:
    """Represents a news article, including metadata and content."""
    date: Date
    title: str
    content: str # Full text content of the news article
    events: List[Event] # List of events mentioned in the article
    # Example: NewsArticle(date=Date("2022-01-01"), title="Trade agreement between USA and China", content="On January 1, 2022, a trade agreement was signed between the USA and China...", events=[Event(date=Date("2022-01-01"), head_entity=ISOCode("USA"), relation=CAMEOCode("010"), tail_entity=ISOCode("CHN")])

    def __str__(self):
        return f"{self.date}:\n{self.title}\n{self.content}"

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
        [Reltion(cameo_code=CAMEOCode("19"), name="Fight", description="All uses of conventional force and acts of war typically by organized armed groups."), Relation(cameo_code=CAMEOCode("193"), name="Fight with small arms and light weapons", description="Attack using small arms and light weapons such as rifles, machine-guns, and mortar shells."), Relation(cameo_code=CAMEOCode("190"), name="Use conventional military force, not specified", description="All uses of conventional force and acts of war typically by organized armed groups, not otherwise specified.")]
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
    pass

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
    pass

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
    pass