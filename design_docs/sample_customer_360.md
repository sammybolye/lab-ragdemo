# Integration Specification: Customer 360 (Salesforce -> Snowflake)

## Overview
This pipeline migrates customer data from the Salesforce `Account` object to the Snowflake `DIM_CUSTOMER` table. The goal is to provide a unified view of the customer for the analytics team.

## Source System
- **Name**: Salesforce CRM
- **Object**: Account Table

## Target System
- **Name**: Snowflake Data Warehouse
- **Object**: DIM_CUSTOMER

## Mappings

| Salesforce Field | Snowflake Column | Transformation | Notes |
|---|---|---|---|
| `AccountId` | `CUST_ID` | Direct Map | Primary Key, immutable. |
| `Name` | `CUST_NAME` | `UPPER(Name)` | Standardize to uppercase for reporting. |
| `BillingState` | `STATE_CODE` | `Lookup(StateMap)` | Standardize state names to 2-letter codes. |
| `AnnualRevenue` | `REVENUE` | `CAST(AnnualRevenue AS FLOAT)` | Ensure numeric type. |

## Data Quality Rules
1. **CUST_ID**: Must be **Unique** and **Not Null**. Critical for identity.
2. **REVENUE**: Must be `>= 0`. Negative revenue is data error.
3. **STATE_CODE**: Must be in the reference set of valid US State Codes.
