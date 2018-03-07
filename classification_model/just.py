from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy


# new entity label
LABEL = 'Prescience_Entity'


TRAIN_DATA = [('How is Bitcoin doing today?', {'entities': [(15, 20, 'Prescience_Entity'), (21, 26, 'Prescience_Entity'), (7, 14, 'Prescience_Entity')]}), ('What People are talking about Bitcoin?', {'entities': [(5, 11, 'Prescience_Entity'), (16, 23, 'Prescience_Entity'), (30, 37, 'Prescience_Entity')]}), ('How Bitcoin will be doing tomorrow?', {'entities': [(4, 11, 'Prescience_Entity'), (26, 34, 'Prescience_Entity'), (20, 25, 'Prescience_Entity')]}), ('How Bitcoin will be doing one week?', {'entities': [(4, 11, 'Prescience_Entity'), (26, 34, 'Prescience_Entity'), (30, 34, 'Prescience_Entity')]}), ('How Bitcoin will be doing one month ahead?', {'entities': [(4, 11, 'Prescience_Entity'), (20, 25, 'Prescience_Entity'), (30, 35, 'Prescience_Entity'), (26, 35, 'Prescience_Entity'), (36, 41, 'Prescience_Entity')]}), ('What are the factors influencing Bitcoin prices?', {'entities': [(13, 20, 'Prescience_Entity'), (33, 40, 'Prescience_Entity'), (21, 32, 'Prescience_Entity')]})]



@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int))
def main(model='en', new_model_name='animal', output_dir='/Users/exepaul/', n_iter=20):
    
    if model is not None:
        nlp = spacy.load(model)  
        print("Loaded model '%s'" % model)

    nlp = spacy.load('en')


 
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe('ner')

    ner.add_label(LABEL)   # add new entity label to entity recognizer

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update([text], [annotations], sgd=optimizer, drop=0.35,
                           losses=losses)
            print(losses)

    # test the trained model
    test_text = 'What People are talking about Bitcoin?'
    doc = nlp(test_text)
    print("Entities in '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta['name'] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        doc2 = nlp2(test_text)
        for ent in doc2.ents:
            print(ent.label_, ent.text)


if __name__ == '__main__':
    plac.call(main)