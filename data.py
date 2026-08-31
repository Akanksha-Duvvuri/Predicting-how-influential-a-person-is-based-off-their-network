import pandas as pd 
import igraph as ig

file_path = "facebook_combined.txt"

print("loading data")

df = pd.read_csv(
    file_path,
    sep=r"\s+",
    comment="#",
    header=None,
    names=["source", "target"]
)