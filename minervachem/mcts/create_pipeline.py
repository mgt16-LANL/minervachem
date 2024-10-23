from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from minervachem.fingerprinters import GraphletFingerprinter
from minervachem.transformers import FingerprintFeaturizer
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import os
from rdkit import Chem


current = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(os.path.dirname(current))

df = pd.read_csv(os.path.join(root, 'demos/qm9_processed.csv'))
df['mol'] = df['smiles'].map(lambda s: Chem.AddHs(Chem.MolFromSmiles(s)))

train, test = train_test_split(
    df, 
    train_size=0.8, 
    random_state=42,
)

y_train, y_test = [sub_df['E_at'] for sub_df in [train, test]]


pipeline = Pipeline(
    [
        ("featurizer", FingerprintFeaturizer(fingerprinter=GraphletFingerprinter(max_len=3),
                                             verbose=0,
                                             n_jobs=-3,
                                             chunk_size='auto',)
                                             ),
        ("ridge", Ridge(fit_intercept=False, 
                        alpha=1e-5, 
                        solver='sparse_cg', 
                        tol=1e-5)
                        )
    ]
)

pipeline.fit(train['mol'], y_train)

y_pred = pipeline.predict(test['mol'])

# Save the trained pipeline
with open(os.path.join(current, 'ridge_pipeline.pkl'), 'wb') as f:
    pickle.dump(pipeline, f)

print("Pipeline object generated.")