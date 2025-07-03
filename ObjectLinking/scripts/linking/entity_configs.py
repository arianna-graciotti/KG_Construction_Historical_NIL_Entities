#!/usr/bin/env python3
"""
Entity Configuration Module for Unified Entity Linking

This module defines entity-specific configurations for the unified entity linking system.
Each entity type has its own configuration class that defines paths, normalization rules,
and other specific behaviors.
"""

import os
import re
from typing import Dict, List, Optional, Tuple, Any, Type
from entity_linker import EntityConfig


class CountryConfig(EntityConfig):
    """Configuration for Country of Citizenship entity linking"""
    
    @property
    def entity_name(self) -> str:
        return "Country of Citizenship"
    
    @property
    def entity_type(self) -> str:
        return "country"
    
    @property
    def csv_file_path(self) -> str:
        return "https://github.com/arianna-graciotti/KG_Construction_Historical_NIL_Entities/blob/8ada448e0862a92a037abfea1455dffe44a261d3/ObjectLinking/lookup_tables/extracted_country_of_citizenship.csv"
    
    @property
    def instance_qids(self) -> List[str]:
        return [
            "Q6256",    # country
            "Q3624078", # sovereign state
            "Q3336843"  # microstate
        ]
    
    @property
    def extraction_patterns(self) -> List[Tuple[str, int]]:
        return [
            (r'answer\s*(?:is|:)?\s*[\'""]?([a-zA-Z\s\-]+)[\'""]?', 1),  # "answer is USA" or "answer: France"
            (r'country\s*(?:is|:)?\s*[\'""]?([a-zA-Z\s\-]+)[\'""]?', 1),  # "country is Germany" or "country: Italy"
            (r'citizenship\s*(?:is|:)?\s*[\'""]?([a-zA-Z\s\-]+)[\'""]?', 1),  # "citizenship is Canadian" or "citizenship: Italian"
            (r'citizen of\s*[\'""]?([a-zA-Z\s\-]+)[\'""]?', 1),  # "citizen of Spain"
        ]
    
    def normalize_entity(self, text: str) -> str:
        """
        Country-specific normalization has been deactivated.
        Simply returns the lowercased and trimmed text.
        """
        # Only lowercase and trim, no country-specific processing
        return text.lower().strip()


class FamilyNameConfig(EntityConfig):
    """Configuration for Family Name entity linking"""
    
    @property
    def entity_name(self) -> str:
        return "Family Name"
    
    @property
    def entity_type(self) -> str:
        return "family_name"
    
    @property
    def csv_file_path(self) -> str:
        return "https://github.com/arianna-graciotti/KG_Construction_Historical_NIL_Entities/blob/8ada448e0862a92a037abfea1455dffe44a261d3/ObjectLinking/lookup_tables/extracted_family_names.csv"
    
    @property
    def instance_qids(self) -> List[str]:
        return [
            "Q101352",   # family name
            "Q18972245", # surname
            "Q18972207", # family name variation
            "Q11455398", # onomastic suffix
            "Q98775491", # family name/surname
            "Q829026",   # double-barrelled name
            "Q27951364"  # maiden name
        ]
    
    @property
    def extraction_patterns(self) -> List[Tuple[str, int]]:
        return [
            (r'answer\s*(?:is|:)?\s*[\'""]?([a-zA-Z\s\-]+)[\'""]?', 1),  # "answer is Smith" or "answer: Smith"
            (r'name\s*(?:is|:)?\s*[\'""]?([a-zA-Z\s\-]+)[\'""]?', 1),  # "name is Smith" or "name: Smith"
            (r'surname\s*(?:is|:)?\s*[\'""]?([a-zA-Z\s\-]+)[\'""]?', 1),  # "surname is Smith"
            (r'family\s*name\s*(?:is|:)?\s*[\'""]?([a-zA-Z\s\-]+)[\'""]?', 1),  # "family name is Smith"
        ]
    
    def normalize_entity(self, text: str) -> str:
        """
        Family name-specific normalization has been deactivated.
        Simply returns the lowercased and trimmed text.
        """
        # Only lowercase and trim, no family name-specific processing
        return text.lower().strip()


class GivenNameConfig(EntityConfig):
    """Configuration for Given Name entity linking"""
    
    @property
    def entity_name(self) -> str:
        return "Given Name"
    
    @property
    def entity_type(self) -> str:
        return "given_name"
    
    @property
    def csv_file_path(self) -> str:
        return "https://github.com/arianna-graciotti/KG_Construction_Historical_NIL_Entities/blob/8ada448e0862a92a037abfea1455dffe44a261d3/ObjectLinking/lookup_tables/extracted_given_names.csv"
    
    @property
    def instance_qids(self) -> List[str]:
        return [
            "Q202444",   # given name
            "Q3409032",  # unisex given name
            "Q11879590", # female given name
            "Q12308941", # male given name
            "Q18131152", # feminine form of a given name
            "Q18123532", # masculine form of a given name
            "Q2358163",  # diminutive form of a given name
            "Q12766737"  # diminutive
        ]
    
    @property
    def extraction_patterns(self) -> List[Tuple[str, int]]:
        return [
            (r'answer\s*(?:is|:)?\s*[\'""]?([a-zA-Z\s\-]+)[\'""]?', 1),  # "answer is John" or "answer: John"
            (r'name\s*(?:is|:)?\s*[\'""]?([a-zA-Z\s\-]+)[\'""]?', 1),  # "name is John" or "name: John"
            (r'first\s*name\s*(?:is|:)?\s*[\'""]?([a-zA-Z\s\-]+)[\'""]?', 1),  # "first name is John"
            (r'given\s*name\s*(?:is|:)?\s*[\'""]?([a-zA-Z\s\-]+)[\'""]?', 1),  # "given name is John"
        ]
    
    def normalize_entity(self, text: str) -> str:
        """
        Given name-specific normalization has been deactivated.
        Simply returns the lowercased and trimmed text.
        """
        # Only lowercase and trim, no given name-specific processing
        return text.lower().strip()


class OccupationConfig(EntityConfig):
    """Configuration for Occupation entity linking"""
    
    @property
    def entity_name(self) -> str:
        return "Occupation"
    
    @property
    def entity_type(self) -> str:
        return "occupation"
    
    @property
    def csv_file_path(self) -> str:
        return "/https://github.com/arianna-graciotti/KG_Construction_Historical_NIL_Entities/blob/8ada448e0862a92a037abfea1455dffe44a261d3/ObjectLinking/lookup_tables/extracted_occupations.csv"
    
    @property
    def instance_qids(self) -> List[str]:
        return [
            "Q12737077",  # occupation
            "Q28640",     # profession
            "Q35120",     # job
            "Q96457344",  # role
            "Q96350374",  # career
            "Q106112097", # occupation role
            "Q96375753"   # activity
        ]
    
    @property
    def extraction_patterns(self) -> List[Tuple[str, int]]:
        return [
            (r'answer\s*(?:is|:)?\s*[\'""]?([a-zA-Z\s\-]+)[\'""]?', 1),  # "answer is doctor" or "answer: doctor"
            (r'occupation\s*(?:is|:)?\s*[\'""]?([a-zA-Z\s\-]+)[\'""]?', 1),  # "occupation is doctor"
            (r'profession\s*(?:is|:)?\s*[\'""]?([a-zA-Z\s\-]+)[\'""]?', 1),  # "profession is doctor"
            (r'job\s*(?:is|:)?\s*[\'""]?([a-zA-Z\s\-]+)[\'""]?', 1),  # "job is doctor"
            (r'career\s*(?:is|:)?\s*[\'""]?([a-zA-Z\s\-]+)[\'""]?', 1),  # "career is doctor"
            (r'works as\s*(?:a|an)?\s*[\'""]?([a-zA-Z\s\-]+)[\'""]?', 1),  # "works as a doctor"
        ]
    
    def normalize_entity(self, text: str) -> str:
        """
        Occupation-specific normalization has been deactivated.
        Simply returns the lowercased and trimmed text.
        """
        # Only lowercase and trim, no occupation-specific processing
        return text.lower().strip()


class SexGenderConfig(EntityConfig):
    """Configuration for Sex or Gender entity linking"""
    
    @property
    def entity_name(self) -> str:
        return "Sex or Gender"
    
    @property
    def entity_type(self) -> str:
        return "sex_gender"
    
    @property
    def csv_file_path(self) -> str:
        # Use the actual path to the extracted gender lookup_tables
        return "/home/arianna/PycharmProjects/KG_Construction_Historical_NIL_Entities/ObjectLinking/lookup_tables/extracted_gender.csv"
    
    @property
    def instance_qids(self) -> List[str]:
        return [
            "Q48264",    # gender
            "Q290",      # biological sex
            "Q4369513",  # gender identity
            "Q48277"     # sex/gender
        ]
    
    @property
    def extraction_patterns(self) -> List[Tuple[str, int]]:
        return [
            (r'answer\s*(?:is|:)?\s*[\'""]?([a-zA-Z\s\-]+)[\'""]?', 1),  # "answer is male" or "answer: female"
            (r'gender\s*(?:is|:)?\s*[\'""]?([a-zA-Z\s\-]+)[\'""]?', 1),  # "gender is male"
            (r'sex\s*(?:is|:)?\s*[\'""]?([a-zA-Z\s\-]+)[\'""]?', 1),  # "sex is female"
        ]
    
    def normalize_entity(self, text: str) -> str:
        """
        Sex/gender-specific normalization has been deactivated.
        Simply returns the lowercased and trimmed text.
        """
        # Only lowercase and trim, no gender-specific processing
        return text.lower().strip()
    
    @property
    def hardcoded_qid_mapping(self) -> Dict[str, str]:
        """
        Hardcoded mapping has been deactivated.
        Returns an empty dictionary.
        """
        # Empty dictionary - no hardcoded mappings
        return {}


def get_entity_config(entity_type: str) -> EntityConfig:
    """
    Factory function to get the appropriate entity config based on entity type.
    
    Args:
        entity_type: String identifier for the entity type
    
    Returns:
        An instance of the appropriate EntityConfig subclass
    
    Raises:
        ValueError: If the entity type is not recognized
    """
    config_map = {
        "country": CountryConfig,
        "family_name": FamilyNameConfig,
        "given_name": GivenNameConfig,
        "occupation": OccupationConfig,
        "sex_gender": SexGenderConfig
    }
    
    if entity_type not in config_map:
        valid_types = ", ".join(config_map.keys())
        raise ValueError(f"Unknown entity type: '{entity_type}'. Valid types are: {valid_types}")
    
    return config_map[entity_type]()