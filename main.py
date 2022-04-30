import sys
import math
import random
from transformers import pipeline
from tqdm.notebook import tqdm_notebook


def insert_new_word(compression):
    unmasker = pipeline('fill-mask', model='bert-base-cased')

    input_text = compression

    orig_text_list = input_text.split()
    len_input = len(orig_text_list)

    rand_idx = random.randint(1, len_input-2)

    new_text_list = orig_text_list[:rand_idx] + ['[MASK]'] + orig_text_list[rand_idx:]
    new_mask_sent = ' '.join(new_text_list)

    augmented_text_list = unmasker(new_mask_sent)
    augmented_text = augmented_text_list[0]['sequence']

    return new_mask_sent, augmented_text


def inset_many_words(compression, number_of_words):

    for _ in range(0, number_of_words):
        _, compression = insert_new_word(compression)

    return compression


def generate_new_sentence_compression_dataset(compressions, word_generation_aggressiveness='low'):

    new_sentences = []
    new_compressions = []
    compression_with_problems = 0

    for compression in tqdm_notebook(compressions, desc='Compressions analyzed'):

        if word_generation_aggressiveness == 'low':
            percentage = 0.1
        elif word_generation_aggressiveness == 'medium':
            percentage = 0.3
        else:
            percentage = 0.5

        for _ in range(0, 5):
            number_of_words = math.ceil(len(compression.split()) * percentage)
            try:
                augmented_text = inset_many_words(compression, number_of_words)
            except:
                compression_with_problems += 1
                break

            new_sentences.append(augmented_text)
            new_compressions.append(compression)

    return new_sentences, new_compressions, compression_with_problems


def main():

    data = 'data/google'
    sys.path.append(data)

    compressions_file_path = '%s/compression_en.txt' % data

    file_compressions = open(compressions_file_path, "r")

    compressions = []

    for line in file_compressions:
        compressions.append(line[:-1])

    new_sentences, new_compressions, compression_with_problems = \
        generate_new_sentence_compression_dataset(compressions, 'high')
    print(f'Number of wrong compressions: {compression_with_problems}.')


if __name__ == "__main__":
    main()
