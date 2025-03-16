from typing import Optional, Dict, Any, List, TypedDict
from pydantic import BaseModel


class State(TypedDict):
    image_data: str
    image_path: str
    doc_type: Optional[str]
    detected_state: Optional[str]
    doc_type_confidence: Optional[float]
    extraction_attempts: int
    extracted_data: Optional[Dict]
    validation_status: bool
    validation_confidence: Optional[float]
    validation_errors: Optional[Dict[str, str]]
    suggested_corrections: Optional[Dict[str, Any]]
    error_message: Optional[str]
    total_tokens: int


class PersonName(BaseModel):
    first_name: str
    middle_name: Optional[str]
    last_name: str


class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str


class IdentificationResult(BaseModel):
    document_type: str
    document_number: str
    name: PersonName
    date_of_birth: str
    issue_date: str
    expiry_date: str
    address: Optional[Address]
    nationality: Optional[str]
    state: str


class DocTypeResponse(BaseModel):
    document_type: str
    confidence: float
    detected_state: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None


class ValidationResponse(BaseModel):
    """Schema for validation response"""
    is_valid: bool
    confidence: float
    error_details: Optional[Dict[str, str]] = None
    suggested_corrections: Optional[Dict[str, Any]] = None


class EvaluationMetrics(BaseModel):
    """Schema for evaluation metrics"""
    precision: float
    recall: float
    f1_score: float
    accuracy: float = 0.0
    error_rate: float = 0.0
    per_field_stats: Dict[str, Dict[str, float]]
    processing_stats: Dict[str, Any]
    token_usage: Dict[str, int]
    document_type: str
    validation_status: bool
    confidence_scores: Dict[str, float]
    error_messages: Optional[List[str]] = None
