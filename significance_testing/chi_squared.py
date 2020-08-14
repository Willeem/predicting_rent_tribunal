from scipy.stats import chisquare, chi2_contingency
import pandas as pd
"""
Took this from http://hamelg.blogspot.com/2015/11/python-for-data-analysis-part-25-chi.html
"""

#Frequencies are hardcoded from results of the classifiers.
tables = {
    'afrekening_servicekosten': {
       'random': pd.DataFrame(['Correct']*(424+208) + ['Incorrect']*(164+50)),
       'future': pd.DataFrame(['Correct']*(159+5)+['Incorrect']*(34+19))
    },
    'gebreken': {
       'random': pd.DataFrame(['Correct']*(466+627) + ['Incorrect']*(104+66)),
       'future': pd.DataFrame(['Correct']*(113+118)+['Incorrect']*(18+17))
    },
    'punten': {
       'random': pd.DataFrame(['Correct']*(141+25) + ['Incorrect']*(38+6)),
       'future': pd.DataFrame(['Correct']*(56+2)+['Incorrect']*(10+2))
    },
    'toetsing': {
       'random': pd.DataFrame(['Correct']*(353+107) + ['Incorrect']*(60+26)),
       'future': pd.DataFrame(['Correct']*(100+77)+['Incorrect']*(19+11))
    },
}

for category in ['afrekening_servicekosten', 'gebreken', 'punten', 'toetsing']:
    future_table = pd.crosstab(index=tables[category]['future'][0], columns='count')
    random_table = pd.crosstab(index=tables[category]['random'][0], columns='count')
    ratios = random_table/len(tables[category]['random'])
    observed = future_table
    expected = ratios * len(tables[category]['future'])
    print(category)
    print(chisquare(f_obs=observed, f_exp=expected))
    print(chi2_contingency())