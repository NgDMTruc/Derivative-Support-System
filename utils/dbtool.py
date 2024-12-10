
from sqlalchemy import create_engine, inspect, text
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

def read_feat_from_postgresql_limit(table_name, user, password, host, port, dbname, schema, limit=30):
    engine = create_engine(f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}')
    # Construct the SQL query with a LIMIT clause
    query = f'SELECT * FROM {table_name} order by "Date" desc, "time" desc LIMIT {limit}'
    # Read the query results into a DataFrame
    df = pd.read_sql_query(query, engine)

    df = df.iloc[::-1].reset_index(drop=True)
    return df
def read_feat_from_postgresql_date(table_name, user, password, host, port, dbname, schema, date_filter):
    engine = create_engine(f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}')
    
    query = f"""
    SELECT * FROM {schema}.{table_name}
    WHERE "Date" > '{date_filter}'

    """
   
    df = pd.read_sql_query(query, engine)
    
    return df
def read_feat_from_postgresql_date_time(table_name, user, password, host, port, dbname, schema, date_filter,time_filter):
    engine = create_engine(f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}')
    
    query = f"""
    SELECT * FROM {schema}.{table_name}
    WHERE "Date" >= '{date_filter}'AND 'time'>='{time_filter}'

    """
   
    df = pd.read_sql_query(query, engine)
    
    return df

def get_row_difference(last_time_feat, user, password, host, port, dbname, schema, table_name):
    last_time_str = str(last_time_feat)[:19]  # 'YYYY-MM-DD HH:MM:SS'
    
    # Create a connection to the database
    engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")
    
    try:
        with engine.connect() as connection:
            # Enable row mapping to access by column name
            connection = connection.execution_options(stream_results=True)
            query = text(f"""
                SELECT COUNT(*) AS row_count
                FROM {schema}.{table_name}
                WHERE "Date" > :last_time
            """)
            result = connection.execute(query, {"last_time": last_time_str}).mappings().fetchone()
            # Access the result using a key
            return result['row_count'] if result else 0
    except Exception as e:
        print(f"Error: {e}")
        return 0
    finally:
        engine.dispose()

def check_database_exists(table_name, user, password, host, port, dbname):
    try:
        engine = create_engine(f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}')
        query = f'SELECT * FROM {table_name} limit 1'
        df = pd.read_sql_query(query, engine)
        return df is not None
    except Exception as e:
        print(e)
        return False
    finally:
        engine.dispose()