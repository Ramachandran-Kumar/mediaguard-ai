import pandas as pd
df = pd.read_csv('output/fhir_converted_claims.csv')
df_clean = df[df['fraud_type'] == 'NONE'][['claim_id','cpt_code','icd_code']]
print(df_clean.to_string())