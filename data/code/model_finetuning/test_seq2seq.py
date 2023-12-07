import logging

import pandas as pd
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_data = [
    [
        "Perseus “Percy” Jackson is the main protagonist and the narrator of the Percy Jackson and the Olympians series.",
        "Percy is the protagonist of Percy Jackson and the Olympians",
    ],
    [
        "Annabeth Chase is one of the main protagonists in Percy Jackson and the Olympians.",
        "Annabeth is a protagonist in Percy Jackson and the Olympians.",
    ],
]

train_df = pd.DataFrame(train_data, columns=["input_text", "target_text"])

eval_data = [
    [
        "Grover Underwood is a satyr and the Lord of the Wild. He is the satyr who found the demigods Thalia Grace, Nico and Bianca di Angelo, Percy Jackson, Annabeth Chase, and Luke Castellan.",
        "Grover is a satyr who found many important demigods.",
    ],
    [
        "Thalia Grace is the daughter of Zeus, sister of Jason Grace. After several years as a pine tree on Half-Blood Hill, she got a new job leading the Hunters of Artemis.",
        "Thalia is the daughter of Zeus and leader of the Hunters of Artemis.",
    ],
]

eval_df = pd.DataFrame(eval_data, columns=["input_text", "target_text"])

# Configure the model
model_args = Seq2SeqArgs()
model_args.num_train_epochs = 200
model_args.evaluate_generated_text = True
model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True

model = Seq2SeqModel(
    "roberta",
    "roberta-base",
    "bert-base-cased",
    args=model_args,
)

# Train the model
model.train_model(train_df, eval_data=eval_df)

# Evaluate the model
result = model.eval_model(eval_df)

# Use the model for prediction
print(
    model.predict(
        [
            "Tyson is a Cyclops, a son of Poseidon, and Percy Jackson’s half brother. He is the current general of the Cyclopes army."
        ]
    )
)

# Loading a saved model
model_reloaded = Seq2SeqModel(
    "roberta",
    encoder_decoder_name="outputs",
    args=model_args,
)

# Use the model for prediction
print(
    model_reloaded.predict(
        [
            "Tyson is a Cyclops, a son of Poseidon, and Percy Jackson’s half brother. He is the current general of the Cyclopes army."
        ]
    )
)

