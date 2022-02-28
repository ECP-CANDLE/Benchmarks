import pandas as pd
import numpy as np
import re

#read csvs
df_notes = pd.read_csv("/home/ueq/data/mimic/NOTEEVENTS.csv")
df_diag = pd.read_csv("/home/ueq/data/mimic/DIAGNOSES_ICD.csv")
df_proc = pd.read_csv("/home/ueq/data/mimic/PROCEDURES_ICD.csv")

#filter to discharge summary
df_notes = df_notes.loc[df_notes.CATEGORY == 'Discharge summary']

#retain useful columns
df_notes = df_notes[['ROW_ID','SUBJECT_ID','HADM_ID','TEXT']]
df_diag = df_diag[['HADM_ID','ICD9_CODE']]
df_proc = df_proc[['HADM_ID','ICD9_CODE']]
df_diag['ICD9_CODE'] = df_diag['ICD9_CODE'].astype(str)
df_proc['ICD9_CODE'] = df_proc['ICD9_CODE'].astype(str)

#drop last 2 characters in diagnoses to get category
df_diag_cat = df_diag.copy()
df_diag_cat['ICD9_CODE'] = df_diag_cat['ICD9_CODE'].apply(lambda x: x[:3])

#combine all labels belonging to same patient/ham_id
df_diag['ICD9_CODE'] = df_diag.groupby(['HADM_ID'])['ICD9_CODE'].transform(lambda x: ','.join(x))
df_diag = df_diag.drop_duplicates()
df_proc['ICD9_CODE'] = df_proc.groupby(['HADM_ID'])['ICD9_CODE'].transform(lambda x: ','.join(x))
df_proc = df_proc.drop_duplicates()
df_diag_cat['ICD9_CODE'] = df_diag_cat.groupby(['HADM_ID'])['ICD9_CODE'].transform(lambda x: ','.join(x))
df_diag_cat = df_diag_cat.drop_duplicates()
df_diag.rename(columns={'ICD9_CODE':'DIAG_CODES'},inplace=True)
df_proc.rename(columns={'ICD9_CODE':'PROC_CODES'},inplace=True)
df_diag_cat.rename(columns={'ICD9_CODE':'DIAG_CAT'},inplace=True)

#merge reports and addendums from same patient/ham_id
df_notes['TEXT'] = df_notes.groupby(['HADM_ID'])['TEXT'].transform(lambda x: '\n '.join(x))
df_notes = df_notes.drop_duplicates(['HADM_ID'])

#join text with labels
df_notes = df_notes.merge(df_diag,how='left',on=['HADM_ID'])
df_notes = df_notes.merge(df_proc,how='left',on=['HADM_ID'])
df_notes = df_notes.merge(df_diag_cat,how='left',on=['HADM_ID'])

#clean text
punc = ['.','?','!',',','#',':',';','(',')','%','/','-','+','=']

'''
Patterns to remove
[**Last Name **]
[**First Name **]
[**Name **]
[**Date **]
[**int-int**]
[**Location **]
[**MD **]
[**Job **]
[**Doctor **]
[**Hospital **]
[**Telephone **]
[** **]
'''

def re_clean_text(text):
    text = text.lower()
    text = re.sub('\[\*\*.*?\*\*\]', 'deidentified', text)
    text = re.sub('[0-9]+\.[0-9]+',' floattoken ', text)
    text = re.sub('[0-9][0-9][0-9]+',' largeint ', text)
    text = re.sub("['|\"]", '', text)
    text = re.sub("_", ' ', text)
    punc_str = '|\\'.join(punc)
    text = re.sub('[^\w_|\%s]+' % punc_str, ' ', text)
    for p in punc:
        text = re.sub("\%s{2,}" % p, '%s' % p, text)
        text = re.sub('\%s' % p, ' %s ' % p, text)
    text = (' ').join(text.split())
    return text

df_notes['TEXT'] = df_notes['TEXT'].apply(re_clean_text)
print(df_notes)
print(df_notes['TEXT'])

#save output
df_notes.to_csv('notes_diagnoses_procedures.csv')

df_notes_text = df_notes['TEXT']

df_notes_text.to_csv('notes_diagnoses_procedures_text.csv',header=False,index=False)
