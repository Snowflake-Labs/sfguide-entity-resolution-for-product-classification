USE ROLE SYSADMIN;
CREATE OR REPLACE WAREHOUSE PRODUCT_MATCHING_WH; --by default, this creates an XS Standard Warehouse
CREATE OR REPLACE DATABASE PRODUCT_MATCHING_DB;
CREATE OR REPLACE SCHEMA MATCH;
USE WAREHOUSE PRODUCT_MATCHING_WH;
USE DATABASE PRODUCT_MATCHING_DB;
USE SCHEMA MATCH;
CREATE OR REPLACE STAGE MODEL DIRECTORY=(ENABLE=true); --to store semantic model for the chatbot
CREATE OR REPLACE STAGE STREAMLIT DIRECTORY=(ENABLE=true); --to store streamlit script