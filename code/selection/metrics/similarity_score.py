from sentence_transformers import SentenceTransformer, util


def get_score(model, sentence_1, sentence_2):
    """
    here, sentence_2 is not necessarily a single sentence, it can be a bunch of sentences
    """
    emb_1 = model.encode(sentence_1,  convert_to_tensor = True)
    emb_2 = model.encode(sentence_2,  convert_to_tensor = True)
    cosine_scores = util.pytorch_cos_sim(emb_1, emb_2)
    return cosine_scores


def similarity(sim_model1, sim_model2, sentence_1, sentence_2, select=True):
    if select:
        scores = get_score(sim_model1, sentence_1, sentence_2)[0]
        scores = [score.item() for score in scores]
    else:
        scores_1 = get_score(sim_model1, sentence_1, sentence_2)[0]
        scores_2 = get_score(sim_model2, sentence_1, sentence_2)[0]
        scores_1 = [score.item() for score in scores_1]
        scores_2 = [score.item() for score in scores_2]

        scores = []
        for score1, score2 in zip(scores_1, scores_2):
            if min(score1, score2) < 0.7: # Reducing score for nonsensical text
                score = min(score1, score2)
            else:
                score = score1
            score = 0.5*(score+1) # Scaling to 0 and 1
            scores.append(score)
            
    return scores
