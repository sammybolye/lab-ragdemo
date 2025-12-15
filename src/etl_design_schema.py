from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class SystemInfo(BaseModel):
    """Details about a source or target system."""
    system_name: str = Field(..., description="Name of the system (e.g., 'Salesforce', 'Postgres_DW')")
    object_name: str = Field(..., description="Name of the table, file, or API endpoint")
    description: Optional[str] = Field(None, description="Context about what this system/object represents")

class FieldMapping(BaseModel):
    """Defines how a single field moves and transforms from source to target."""
    source_field: str = Field(..., description="Column/Key name in the source")
    target_field: str = Field(..., description="Column/Key name in the target")
    transformation_logic: Optional[str] = Field(None, description="Logic applied (e.g., 'UPPER(x)', 'CAST to INT', 'Lookup ID')")
    business_rule: Optional[str] = Field(None, description="Human-readable explanation of why this mapping exists")

class ValidationRule(BaseModel):
    """A quality check to be applied to the data."""
    target_field: str = Field(..., description="Field in the target to validate")
    rule_type: str = Field(..., description="Type of validation (e.g., 'not_null', 'unique', 'reference_check', 'regex')")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Parameters for the rule (e.g., {'min': 0, 'max': 100})")
    severity: str = Field("error", description="Severity if failed: 'error' (stop pipeline) or 'warning' (log and continue)")

class EtlDesignDoc(BaseModel):
    """Complete specification for a single ETL pipeline."""
    pipeline_name: str = Field(..., description="Unique name for this integration process")
    summary: str = Field(..., description="Executive summary of the integration's purpose")
    source: SystemInfo = Field(..., description="Where data comes from")
    target: SystemInfo = Field(..., description="Where data goes to")
    field_mappings: List[FieldMapping] = Field(default_factory=list, description="List of all field-level maps")
    data_quality_rules: List[ValidationRule] = Field(default_factory=list, description="List of validation expectations")
