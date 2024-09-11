# Retrival Augumented QA Chain

1. PymuPDFLoader
2. RecursiveCharacterTextSPlitter(Chunk=200,overlap=50)
3. Embedding Model - `text-embedding-ada-002`
4. Vector Database (Qdrant inmemory)
   - Dimension = 1536
   - Embedding = `text-embedding-ada-002`
   - Distance = Cosine
  
# Synthetic test set generation

1. PymuPDFLoader
2. RecursiveCharacterTextSPlitter(Chunk=600,overlap=50)
3. Generator LLM - `gpt-3.5-turbo`
   Critic LLM - `gpt-4o-mini`
   Embedding - `text-embedding-3-large` 
   Distribution - Simple: 0.5, MultiContext: 0.4, Reasoning: 0.1
   Test Size: 20

# Generate Answers to Questions
- We use retrival augumented qa chain to answer the questions from the synthtic test set.
- Now we have dataset with question, anwser, context and ground truth.
- Run Ragas Evalution

# New chain with newer embedding model.

1. PymuPDFLoader
2. RecursiveCharacterTextSPlitter(Chunk=200,overlap=50)
3. Embedding Model - `text-embedding-3-small`
4. Vector Database (Qdrant inmemory)
   - Dimension = 1536
   - Embedding = `text-embedding-3-small`
   - Distance = Cosine
5. Generate answer for questions in dataset with this new chain
6. Run Ragas evalution on the result and compare the improvements.