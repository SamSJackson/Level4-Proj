
import pandas as pd

from data.code.implementation.wieting.models import load_model
from data.code.implementation.wieting.measure import make_sim_object, evaluate
from sacremoses import MosesTokenizer
from datetime import datetime

date = datetime.now().strftime("%d_%m_%Y")

base_path = "../../processed/train/"
file_location = base_path + f"evaluated/paraphrase_dipper_samples_mistralai_51_EVALUATED_16_01_2024.csv"

wieting_path = "../implementation/wieting"
para_model_name = f"{wieting_path}/paraphrase-at-scale-english/model.para.lc.100.pt"
sp_model = f"{wieting_path}/paraphrase-at-scale-english/paranmt.model"

para_model, _ = load_model(model_name=para_model_name,
                           sp_model=sp_model,
                           gpu=True)

entok = MosesTokenizer(lang='en')
S = make_sim_object(batch_size=32, entok=entok, model=para_model)

df = pd.read_csv(file_location)
no_paraphrases = 3
paraphrased_columns = [f"pp-kgw-{i}" for i in range(1, no_paraphrases+1)]
required_columns = ["kgw-watermarked",
                    *paraphrased_columns]

para_df = df[required_columns]

created_columns = [f"pp-sim-{i}" for i in range(1, no_paraphrases+1)]
# Need to apply
df[created_columns] = para_df.apply(lambda row: evaluate(S, row['kgw-watermarked'],
                                                         [row[f"pp-kgw-{i}"] for i in range(1, no_paraphrases+1)]),
                                    axis='columns',
                                    result_type='expand')

output_path = base_path + f"evaluated/paraphrase_dipper_mistralai_{df.shape[0]}_similarity_{date}.csv"
df.to_csv(output_path, index=False)

