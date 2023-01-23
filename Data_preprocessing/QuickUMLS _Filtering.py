#QuickUMLS filtering
import pandas as pd
from quickumls import QuickUMLS
from constants import ACCEPTED_SEMTYPES #this is part of the QuickUMLS data.
from collections import defaultdict
import numpy as np

"""
It defines a function called "filter_text" that takes in a text and uses the QuickUMLS object to match it 
and extract certain terms. The function returns this new text after 
removing any duplicate words and cleaning it up by removing certain characters.
"""


# Initialize QuickUMLS matcher
matcher = QuickUMLS(quickumls_fp = "QuickUmlsData",accepted_semtypes=ACCEPTED_SEMTYPES)

# Read CSV file and create dataframe
df = pd.read_csv("Datasets/Mimic_test.csv")

# Define filter_text function
def filter_text(text):
    matches = matcher.match(text, best_match=True, ignore_syntax=False)
    filtered_words = [match[0]["term"] for match in matches]
    new_text = ' '.join(pd.unique(filtered_words).tolist())
    new_text = " ".join(new_text.split())
    new_text = new_text.replace('/',' ').replace('\\',' ')
    return new_text

# Filter the "text" column of dataframe and save result in "filtered_text" column
df["filtered_text"] = df["text"].map(filter_text)

# Save filtered text to Queries.txt for the CliniqIR model.
np.savetxt("Queries.txt", df["filtered_text"].values, fmt="%s")
