from sense2vec.vectors import VectorMap
vector_map = VectorMap(128)
vector_map.load(".")
u = unicode("tennis|NOUN", "utf-8")
freq, query_vector = vector_map[u]
print vector_map.most_similar(query_vector, n=10)
