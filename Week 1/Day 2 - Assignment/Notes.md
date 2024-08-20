# Week 1 - Day 2 Assignment

## Questions
### Question 1

#### 1a. Is there any way to modify the default embedding dimension of  `text-embedding-3-small` which is 1536?

As per docs "The number of dimensions the resulting output embeddings should have. Only supported in text-embedding-3 and later models." 

Yes we are allowed to configure the dimension but should be less than or equal to the Max dimension provided by the model.

`text-embedding-3-small` - Max Dimension 1536
`text-embedding-3-large` - Max Dimension 3072

Tested this by modifying the dimension for above models and failes when we execeed the default max dimension provided by the model.

#### 1b. What technique does OpenAI use to achieve this?

These embedding models are trained by using a technique called `Matryoshka Representation Learning`
Matryoshka Representation Learning (MRL) is a technique designed to create flexible and adaptable machine learning models, especially when dealing with different tasks that have varying requirements in terms of computational resources and accuracy.

Think of the famous Russian Matryoshka dolls—those (pronounced as Matroshka) wooden dolls that nest inside each other, with each smaller doll fitting perfectly inside the larger one. MRL works similarly by creating representations (or embeddings) of data that contain layers of information. These layers can be "peeled away" or "added on," depending on what the downstream task needs.

### Question 2 - What are the benefits of using an `async` approach to collecting our embeddings?

- **Concurrency** Allows to execute multiple I/O bound tasks concurrently without waiting for one task to complete before starting another.
- **Non-blocking** Allows the program to continue executing other tass while waiting for a response.
- **Scalability - Multiple Requests** When generating embeddings for a large dataset, using synchronous code would involve waiting for request to complete before moving to the next one efficiently managing multiple tasks by scheduling them as soon as their I/O operations are ready, rather than waiting to finish sequentially.
  

### Question 3: When calling the OpenAI API - are there any ways we can achieve more reproducible outputs?

- **Seed Parameter** Setting a fixed seed value, you ensure that the model produces the same output for the same input across different runs.
- **Temperature Parameter** A lower temperature (closer to 0) makes the model more deterministic, meaning it is more likely to pick the highest-probability tokens and produce consistent outputs. Ranges between 0 to 1. temperature=0  - Fully deterministic
- **top_k & top_p** `top_p` (nucleus sampling) and top_k parameters control the diversity of the model's output by limiting the pool of possible next tokens. Setting `top_p` to 1 or `top_k` to a large value reduces randomness and increases the likelihood of generating the same output for the same input.
- **Prompting Technique** Clear and concise prompts, Breaking the complex task into multiple simple tasks.


### Question 4: What prompting strategies could you use to make the LLM have a more thoughtful, detailed response?



## Improvements & Experimentation

### Chunking Strategy 
#### Understanding the Data
Title - First Line is always Title
Post Details:
Contents
    Chapters
       Parts
          Pages
            Chunks & Attributes

line 1 - Blog title
line 2  - description
line 3  - Author
line 4-6 - Blog attributes
line 7 Table of content
Line 8 Chapter
Line 8-25 part numbers
each part starts like this "Part <part number>:" example for part 1 it will start by "part 1:" 
Part title will be above part number line

#### Chunking



### Other Distance Measure

#### 1. **Cosine Similarity**
   - **Description**: Measures the cosine of the angle between two vectors, which indicates their directional similarity.
   - **Range**: [-1, 1]
   - **Use Case**: Often used in text analysis and information retrieval where the magnitude of the vectors (e.g., term frequencies) may not be as important as their orientation.
   - **Formula**: 
     $$
      \text{cosine\_similarity}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{||\mathbf{a}|| \, ||\mathbf{b}||}
      $$

#### 2. **Euclidean Distance**
   - **Description**: The "ordinary" straight-line distance between two points in Euclidean space.
   - **Range**: [0, ∞)
   - **Use Case**: Used when you care about the absolute distance between vectors. Common in physical space modeling and clustering algorithms like K-means.
   - **Formula**:
      $$
      \text{euclidean\_distance}(\mathbf{a}, \mathbf{b}) = \sqrt{\sum_{i=1}^{n} (a_i - b_i)^2}
      $$

#### 3. **Manhattan Distance (L1 Distance)**
   - **Description**: The sum of the absolute differences of their coordinates.
   - **Range**: [0, ∞)
   - **Use Case**: Useful when you need to compute distance in a grid-like path, such as in city blocks (hence the name).
   - **Formula**:
     $$
      \text{manhattan\_distance}(\mathbf{a}, \mathbf{b}) = \sum_{i=1}^{n} |a_i - b_i|
      $$

#### 4. **Minkowski Distance**
   - **Description**: Generalization of both Euclidean and Manhattan distances.
   - **Range**: [0, ∞)
   - **Use Case**: When you need a flexible distance metric that can be adjusted using a parameter \( p \). 
   - **Formula**:
     $$
      \text{minkowski\_distance}(\mathbf{a}, \mathbf{b}) = \left(\sum_{i=1}^{n} |a_i - b_i|^p\right)^{\frac{1}{p}}
      $$
      where \( p \) is the order of the norm.
   - **Special Cases**:
     - \( p = 1 \) is Manhattan distance.
     - \( p = 2 \) is Euclidean distance.

#### 5. **Chebyshev Distance**
   Not Tried

#### 6. **Jaccard Similarity**
   Not Tried

#### 7. **Hamming Distance**
   Not Tried

#### 8. **Mahalanobis Distance**
   Not Tried


### Attributes


### Using PDF documents


# References

https://blog.demir.io/advanced-rag-implementing-advanced-techniques-to-enhance-retrieval-augmented-generation-systems-0e07301e46f4