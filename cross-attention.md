# Cross-Attention

Now that we've covered self-attention and causal attention, let's explore **cross-attention** – another essential attention mechanism that enables different parts of a transformer to communicate with each other. While self-attention captures relationships within a single sequence, and causal attention ensures we don't look at future tokens, cross-attention allows us to bridge between two different sequences entirely.

## The Cross-Attention Concept

Cross-attention sounds complex, but it's really about letting one sequence "ask questions" about another sequence. Think of it like having a conversation where you're translating from English to Spanish – as you generate each Spanish word, you need to look back at the English sentence to see which parts are most relevant for what you're currently translating.[^1]

Unlike self-attention, where the query (Q), key (K), and value (V) matrices all come from the same input sequence, cross-attention operates on **two different sequences**:[^2][^1]

- The **queries (Q)** come from the target sequence (what we're generating)
- The **keys (K)** and **values (V)** come from the source sequence (what we're reading from)


## Step-by-Step Process

Let's walk through how cross-attention works, using machine translation as our example:

**1. Query Creation (The Question)**
The decoder creates a query for each word it's trying to generate. This query is essentially asking: "Which part of the source sentence should I focus on right now?"[^1]

**2. Keys and Values (The Answers)**
The encoder provides keys and values from the source sequence. The keys act like labels that help identify important parts, while the values contain the actual content of those parts.[^1]

**3. Matching (Finding Relevance)**
Cross-attention compares each query from the decoder with all the keys from the encoder. It calculates attention scores that measure how well each query matches each key – essentially finding which parts of the source are most relevant.[^1]

**4. Information Combination**
Once the best matches are found, cross-attention combines the relevant information from the encoder (using the values) to help the decoder generate the next word.[^1]

## The Mathematics

The mathematical formulation is identical to self-attention, but with a crucial difference in where the matrices come from:[^1]

```
Attention(Q, K, V) = softmax(QK^T/√d_k)V
```

Where:

- **Q** (queries) = W^Q × decoder_input
- **K** (keys) = W^K × encoder_output
- **V** (values) = W^V × encoder_output

This is the key distinction – while self-attention uses the same sequence for Q, K, and V, cross-attention splits them between decoder and encoder outputs.

## Implementation Considerations

From a computational perspective, cross-attention has some interesting properties. The attention matrix dimensions are determined by the lengths of both sequences – if we have a source sequence of length *n* and target sequence of length *m*, our attention matrix will be *m × n*. This is different from self-attention, where we always get an *n × n* matrix.

The time complexity remains **O(n·m·d)** where *d* is the dimensionality, and the space complexity is **O(n·m)** for storing the attention weights.[^1]

## Where Cross-Attention Lives

In the classic transformer architecture, you'll find cross-attention specifically in the **decoder layers**. Here's how it fits into the overall architecture:

1. **Encoder layers** use self-attention to understand relationships within the input sequence
2. **Decoder layers** use both:
    - Causal self-attention to understand relationships within the output sequence (without peeking at future tokens)
    - **Cross-attention** to focus on relevant parts of the encoder's output

This design allows the decoder to maintain the sequential nature of generation while still having access to the full context of the input.

## Real-World Applications

Cross-attention shines in tasks that involve connecting two different types of information:

- **Machine Translation**: Aligning source and target language words
- **Image Captioning**: Connecting visual features with text descriptions
- **Question Answering**: Linking questions with relevant passage content
- **Speech Recognition**: Matching audio features with text transcriptions[^1]


## Why Cross-Attention Matters

Cross-attention is what makes encoder-decoder architectures so powerful. Without it, the decoder would only have access to its own previous outputs and wouldn't be able to "see" the input that it's supposed to be processing. It's the bridge that allows information to flow from the encoder to the decoder, enabling the model to generate contextually appropriate outputs based on the input.[^2]

In essence, while self-attention asks "How do the words in this sequence relate to each other?" and causal attention adds "But only looking backwards," cross-attention asks the fundamentally different question: "How does what I'm generating relate to what I'm reading?"

This mechanism is what enables transformers to excel at sequence-to-sequence tasks, making it an indispensable component in modern language models and beyond.

<div style="text-align: center">⁂</div>

[^1]: https://www.geeksforgeeks.org/nlp/cross-attention-mechanism-in-transformers/

[^2]: https://aiml.com/explain-cross-attention-and-how-is-it-different-from-self-attention/

[^3]: Build_a_Large_Language_Model_-From_Scrat.pdf

[^4]: https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html

[^5]: https://www.youtube.com/watch?v=h94TQOK7NRA

[^6]: https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention

[^7]: https://www.gilesthomas.com/2025/03/llm-from-scratch-8-trainable-self-attention

[^8]: https://www.youtube.com/watch?v=mEsp94dOGgs

[^9]: https://www.gilesthomas.com/2025/03/llm-from-scratch-9-causal-attention

[^10]: https://www.reddit.com/r/LocalLLaMA/comments/1lue75q/day_1150_building_a_small_language_from_scratch/

[^11]: https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)

