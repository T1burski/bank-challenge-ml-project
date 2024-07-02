from google.oauth2 import service_account
import pandas as pd
import json
import pandas_gbq
import warnings
warnings.filterwarnings('ignore')
from concurrent.futures import ThreadPoolExecutor


class BigQuery_DataOps:

    def __init__(self, sub_db):
        self.sub_db = sub_db

    
    def extract_data(self, tb_id, prod=False, n_clients=1):

        service_account_info = json.load(open('GBQ.json'))
        credentials = service_account.Credentials.from_service_account_info(service_account_info)
        
        def get_bq_data_chunk(query, credentials):
                return pandas_gbq.read_gbq(query, credentials=credentials, dialect="standard")

        if prod:
            query = f"""
            SELECT
            *
            FROM db_bank_{self.sub_db}.{tb_id}
            ORDER BY CAST(SUBSTR(ID_code, STRPOS(ID_code, '_') + 1) AS INT64) DESC
            LIMIT {n_clients}
            """
            return get_bq_data_chunk(query, credentials)

        else:    
            try:
                
                # Define your base query with a subquery to compute row numbers first
                base_query = f"""
                SELECT * FROM (
                    SELECT *, ROW_NUMBER() OVER() as row_num 
                    FROM db_bank_{self.sub_db}.{tb_id}
                ) WHERE MOD(row_num, {4}) = {{}}
                """
                
                # Create a list of queries for each chunk
                queries = [base_query.format(i) for i in range(4)]
                
                # Use ThreadPoolExecutor to run queries in parallel
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [executor.submit(get_bq_data_chunk, query, credentials) for query in queries]
                
                # Combine the results from all chunks
                results = [future.result() for future in futures]
                df = pd.concat(results, ignore_index=True)

                df.drop(columns=["row_num"], inplace=True)

                return df
            
            except FileNotFoundError:
                print("Error: File not found.")
                return None

    
    def load_data(self, tb_data, tb_name):

        db_id = 'db_bank_' + self.sub_db
        service_account_info = json.load(open('GBQ.json'))
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info)
        
        tb_data.to_gbq(credentials = credentials, destination_table = db_id + "." + tb_name, if_exists = 'replace')