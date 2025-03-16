from datetime import date
from typing import Literal, Optional
from pydantic import BaseModel, Field


class IdentificationResult(BaseModel):
    document_type: Literal["State License", "State ID", "Passport"] = Field(
        ..., description="The type of document. Must be one of 'State License', 'State ID', or 'Passport'."
    )
    document_number: str = Field(..., description="The document's unique number.")
    first_name: str = Field(..., description="The first name as printed on the document.")
    last_name: str = Field(..., description="The last name as printed on the document.")
    state: str = Field(..., description="The state or region of issuance.")
    sex: str = Field(..., description="The gender or sex as indicated on the document.")
    date_of_birth: date = Field(..., description="The date of birth (YYYY-MM-DD).")
    issue_date: date = Field(..., description="The date the document was issued (MM/DD/YYYY).")
    expiration_date: date = Field(..., description="The expiration date of the document (MM/DD/YYYY).")

    height: Optional[str] = Field(None, description="The height as noted on the document.")
    address: Optional[str] = Field(None, description="The address found on the document.")
    hair: Optional[str] = Field(None, description="The hair color indicated on the document.")
    eyes: Optional[str] = Field(None, description="The eye color indicated on the document.")
    weight: Optional[str] = Field(None, description="The weight (optional).")
    class_field: Optional[str] = Field(None, alias="class", description="The class (optional).")
