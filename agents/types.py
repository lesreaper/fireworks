from typing import Optional, Dict, Any, List, TypedDict
from pydantic import BaseModel


class State(TypedDict, total=False):
    image_path: str
    image_data: Optional[str]
    ocr_text: str
    ocr_doc_text: str
    ocr_confidence: float
    ocr_raw_result: Dict[str, Any]
    doc_type: Optional[str]
    detected_state: Optional[str]
    doc_type_confidence: Optional[float]
    extraction_attempts: int
    extracted_data: Optional[Dict]
    validation_status: bool
    validation_confidence: float
    previous_attempt_score: float
    validation_errors: Optional[Dict[str, Any]]
    suggested_corrections: Optional[Dict[str, Any]]
    error_message: Optional[str]
    total_tokens: int
    doc_type_tokens: Optional[int]
    extraction_tokens: Optional[int]
    validation_tokens: Optional[int]


class PersonName(BaseModel):
    first_name: str
    middle_name: Optional[str]
    last_name: str


class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str


class Name(BaseModel):
    first_name: str
    middle_name: str = ""
    last_name: str


class IdentificationResult(BaseModel):
    document_type: str
    document_number: str
    name: Name
    date_of_birth: str
    issue_date: str
    expiry_date: str
    address: Optional[Address] = None
    nationality: Optional[str] = None
    place_of_birth: Optional[str] = None
    sex: Optional[str] = None
    state: Optional[str] = None


class DocTypeResponse(BaseModel):
    document_type: str
    confidence: float
    detected_state: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None


class ValidationResponse(BaseModel):
    """Schema for validation response"""
    is_valid: bool
    confidence: float
    error_details: Optional[Dict[str, Dict[str, str]]] = None
    suggested_corrections: Optional[Dict[str, str]] = None


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
