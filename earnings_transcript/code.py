import pyodbc
import pandas as pd
import numpy as np
import datetime

# Hive and Impala do not support transactional processing,
# so you must use autocommit=True

# Impala
cnxn_impala = pyodbc.connect('DSN=bdp-prod', autocommit=True)

# Get the company information
company_id = 321778         # Bed Bath and Beyond
keydeveventtypeid = 48      # Earnings Calls

query = f"""SELECT * FROM 
    bdp_transcripts.event_to_object_to_event_type as e
    JOIN bdp_transcripts.transcript as t
    JOIN bdp_transcripts.transcript_component as c
    JOIN bdp_transcripts.transcript_person as person
    JOIN bdp_transcripts.transcript_speaker_type as speaker_type
    JOIN bdp_transcripts.foundation_company as company
    ON  e.keydevid = t.keydevid AND 
        c.transcriptid = t.transcriptid AND 
        c.transcriptpersonid = person.transcriptpersonid AND
        person.speakertypeid =  speaker_type.speakertypeid AND
        e.objectid = company.companyid 
    WHERE e.objectid = {company_id} AND e.keydeveventtypeid = {keydeveventtypeid} AND substr(t.transcriptcreationdateutc, 1, 4) = '2020'
    ORDER BY c.transcriptid, c.componentorder
    LIMIT 1000"""
df_transcripts = pd.read_sql_query(query, cnxn_impala)

df_transcripts = df_transcripts.loc[:,~df_transcripts.columns.duplicated()].copy()
df_transcripts.to_csv('test.csv', index=None)
df_transcripts.to_hdf('test.h5', key='df_transcripts')

