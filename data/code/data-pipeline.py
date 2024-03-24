from generation import (
    watermarking,
    dipper_paraphrase,
    sentence_paraphrase,
    word_replacement,
    clean_paraphrases,
    paraphrase_similarity,
    z_scoring,
    perplexity_measure
)

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

no_documents = 2
no_paraphrases = 3
target_dir = "../processed/"

z_threshold = 4.0
gamma = 0.25
delta = float(5)

lexical = 40
order = 40

percentage_replacement = 25

watermarked_df_path = watermarking.generate_documents_and_get_path(
    gamma=gamma,
    delta=delta,
    no_documents=no_documents,
    target_dir=target_dir,
    input_dir="../prepared/train/daigt/daigt_prompts.csv"
)

cmd_print("Finished watermarking...")

dipper_paraphrase_df_path = dipper_paraphrase.dipper_paraphrase_and_get_path(
    lexical=lexical,
    order=order,
    input_dir=watermarked_df_path,
    no_paraphrases=no_paraphrases,
    target_dir=target_dir
)

cmd_print("Finished paragraph paraphrasing...")

sentence_paraphrase_df_path = sentence_paraphrase.sentence_paraphrase_and_get_path(
    input_dir=dipper_paraphrase_df_path,
    no_paraphrases=no_paraphrases,
    target_dir=target_dir
)

cmd_print("Finished sentence paraphrasing...")

percentage_word_replacement_df_path = word_replacement.percentage_word_replace_and_get_path(
    replacement=percentage_replacement,
    no_paraphrases=no_paraphrases,
    input_dir=sentence_paraphrase_df_path,
    target_dir=target_dir,
)

cmd_print(f"Finished percentage ({word_replacement}%) word replacement paraphrasing...")

noun_word_replacement_df_path = word_replacement.noun_word_replace_and_get_path(
    no_paraphrases=no_paraphrases,
    input_dir=percentage_word_replacement_df_path,
    target_dir=target_dir,
)

cmd_print("Finished noun word replacement paraphrasing...")

cleaned_paraphrases_df_path = clean_paraphrases.clean_paraphrases_and_get_path(
    input_dir=noun_word_replacement_df_path,
    target_dir=target_dir
)

cmd_print("Finished cleaning paraphrases...")

paraphrase_similarity_df_path = paraphrase_similarity.calculate_similarity_and_get_path(
    input_dir=cleaned_paraphrases_df_path,
    no_paraphrases=no_paraphrases,
    target_dir=target_dir
)

cmd_print("Finished similarity...")

evaluate_scores_df_path = z_scoring.evaluate_z_scores_and_get_path(
    gamma=gamma,
    z_threshold=z_threshold,
    input_dir=paraphrase_similarity_df_path,
    no_paraphrases=no_paraphrases,
    target_dir=target_dir
)

cmd_print("Finished Z-Scores")

perplexity_scores_df_path = perplexity_measure.evaluate_perplexity_on_dataframe(
        input_dir=evaluate_scores_df_path,
        target_dir=target_dir,
        model_name="facebook/opt-2.7b"
)

cmd_print("Finished perplexity | FINISHED")

