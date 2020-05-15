from hw2_corpus_tool import *
import sys
from pycrfsuite import Trainer,Tagger
import random

def create_features(conversations) :

    convers_vecs = []
    for conversation in conversations :
        vectors = []
        labels = []
        previous_speaker = conversation[0][1]
        feat_vec = ['FIRST_UTTER']
        for utterance in conversation :
            if utterance[1] != previous_speaker :
                feat_vec.append('SPEAKER_CHANGED')
                previous_speaker = utterance[1]
            if utterance[2] :
                for posTag in utterance[2] :
                    feat_vec.append('TOKEN_'+posTag[0].lower())
                    feat_vec.append('POS_'+posTag[1])
            else :
                feat_vec.append('NO_WORDS')
            vectors.append(feat_vec)
            if utterance[0] :
                labels.append(utterance[0])
            feat_vec = []
        convers_vecs.append((vectors,labels))
    return convers_vecs


def main(argv) :

    inputDir = argv[0]
    testDir = argv[1]
    outputFPath = argv[2]


    trainData = list(get_data(inputDir))
    testData = list(get_data(testDir))

    random.shuffle(trainData)



    # create baseline features
    trainFeatures = create_features(trainData)
    testFeatures = create_features(testData)

    trainer = Trainer()
    for dialogue in trainFeatures :
        trainer.append(dialogue[0],dialogue[1])

    trainer.set_params({
        'c1': 1.0,  # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 50,  # stop earlier
        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })

    trainer.train('./model.pkl')

    outputFile = open(outputFPath,'w')
    tagger = Tagger()
    tagger.open('./model.pkl')

    totalUtter=correctUtter=0
    for dialogue in testFeatures :
        preds = tagger.tag(dialogue[0])
        labels = dialogue[1]
        for i,pred in enumerate(preds) :
            outputFile.write(pred+'\n')
            if len(labels)>0 :
                totalUtter += 1
                if labels[i]==pred :
                    correctUtter += 1
        outputFile.write('\n')

    if totalUtter > 0 :
        accuracy = correctUtter/totalUtter
        print('Accuracy: '+str(accuracy))
    outputFile.close()

if __name__ == "__main__":
    main(sys.argv[1:])