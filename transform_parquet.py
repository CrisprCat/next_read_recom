import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


df = pd.read_csv("goodreads_interactions.csv")

table = pa.Table.from_pandas(df)

pq.write_table(table, "interaction.parquet")