from data.code.generation.watermarking import generate_documents_and_get_path
from data.code.generation.dipper_paraphrase import dipper_paraphrase_and_get_path
from data.code.generation.sentence_paraphrase import sentence_paraphrase_and_get_path
from data.code.generation.word_replacement import percentage_word_replace_and_get_path, noun_word_replace_and_get_path
from data.code.generation.clean_paraphrases import clean_paraphrases_and_get_path
from data.code.generation.paraphrase_similarity import calculate_similarity_and_get_path
from data.code.generation.z_scoring import evaluate_z_scores_and_get_path
from data.code.generation.perplexity_measure import evaluate_perplexity_on_dataframe

def cmd_print(text):
    print(f"{'-'*50}")
    print(text)
    print(f"{'-'*50}")

'''
    Tasking process:
        - Generate watermarked documents
        - Paraphrase/Word Replacement:
            - DIPPER paraphrase
            - Sentence-based paraphrase
            - Word Replacement
        - Clean Paraphrases
        - Paraphrase Similarity
        - Evaluate Z-Scores
'''

no_documents = 500
no_paraphrases = 3
target_dir = "../../processed/"

z_threshold = 4.0
gamma = 0.25
delta = float(5)

lexical = 40
order = 40

word_replacement = 25

watermarked_df_path = generate_documents_and_get_path(
    gamma=gamma,
    delta=delta,
    no_documents=no_documents,
    target_dir=target_dir
)

cmd_print("Finished watermarking...")

dipper_paraphrase_df_path = dipper_paraphrase_and_get_path(
    lexical=lexical,
    order=order,
    input_dir=watermarked_df_path,
    no_paraphrases=no_paraphrases,
    target_dir=target_dir
)

cmd_print("Finished paragraph paraphrasing...")

sentence_paraphrase_df_path = sentence_paraphrase_and_get_path(
    input_dir=dipper_paraphrase_df_path,
    no_paraphrases=no_paraphrases,
    target_dir=target_dir
)

cmd_print("Finished sentence paraphrasing...")

percentage_word_replacement_df_path = percentage_word_replace_and_get_path(
    replacement=word_replacement,
    no_paraphrases=no_paraphrases,
    input_dir=sentence_paraphrase_df_path,
    target_dir=target_dir,
)

cmd_print(f"Finished percentage ({word_replacement}%) word replacement paraphrasing...")

noun_word_replacement_df_path = noun_word_replace_and_get_path(
    no_paraphrases=no_paraphrases,
    input_dir=percentage_word_replacement_df_path,
    target_dir=target_dir,
)

cmd_print("Finished noun word replacement paraphrasing...")

cleaned_paraphrases_df_path = clean_paraphrases_and_get_path(
    input_dir=noun_word_replacement_df_path
)

cmd_print("Finished cleaning paraphrases...")

paraphrase_similarity_df_path = calculate_similarity_and_get_path(
    input_dir=cleaned_paraphrases_df_path,
    no_paraphrases=no_paraphrases
)

cmd_print("Finished similarity...")

evaluate_scores_df_path = evaluate_z_scores_and_get_path(
    gamma=gamma,
    z_threshold=z_threshold,
    input_dir=paraphrase_similarity_df_path,
    no_paraphrases=no_paraphrases
)

cmd_print("Finished Z-Scores")

perplexity_scores_df_path = evaluate_perplexity_on_dataframe(
        input_dir=evaluate_scores_df_path,
        target_dir="../../processed/",
        model_name="facebook/opt-2.7b"
)

cmd_print("Finished perplexity | FINISHED")

