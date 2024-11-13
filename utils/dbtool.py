from sqlalchemy import create_engine, inspect
import pandas as pd

def append_to_postgresql(df, table_name, user, password, host, port, dbname):
    engine = create_engine(f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}')
    for col in df.select_dtypes(include=['uint64']).columns:
        df[col] = df[col].astype('int64')
    df.to_sql(table_name, engine, if_exists='append', index=False)

def save_to_postgresql(df, table_name, user, password, host, port, dbname):
    engine = create_engine(f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}')
    for col in df.select_dtypes(include=['uint64']).columns:
        df[col] = df[col].astype('int64')
    df.to_sql(table_name, engine, if_exists='replace', index=False)

def read_from_postgresql_limit(table_name, user, password, host, port, dbname, schema, limit=30):
    engine = create_engine(f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}')
    # Construct the SQL query with a LIMIT clause
    query = f'SELECT * FROM {table_name} order by "Date" desc LIMIT {limit}'
    # Read the query results into a DataFrame
    df = pd.read_sql_query(query, engine)

    df = df.iloc[::-1].reset_index(drop=True)
    return df