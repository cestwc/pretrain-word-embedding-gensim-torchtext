from six import iteritems
from web.datasets.similarity import fetch_MEN, fetch_WS353, fetch_SimLex999
from web.embeddings import fetch_GloVe, load_embedding
from web.evaluate import evaluate_similarity

# Define tasks
tasks = {
    "MEN": fetch_MEN(),
    "WS353": fetch_WS353(),
    "SIMLEX999": fetch_SimLex999()
}

def benchmark(emb_name, size = 128):
	emb = load_embedding(emb_name, format="glove", load_kwargs={"vocab_size": len(open(emb_name, 'r').readlines()), "dim": size})
	scores = {}
	for name, data in iteritems(tasks):
		scores[name] = evaluate_similarity(emb, data.X, data.y)
	return scores
