# First Iteration Critique

The initial design embraces several advanced techniques, yet its simplicity introduces notable trade-offs:

1. **Offline RL**
   - *Pros:* avoids costly environment interaction; easy to implement as reward-weighted behaviour cloning.
   - *Cons:* performance heavily depends on the quality and coverage of the offline dataset. Without careful weighting, the model may overfit high-reward samples and degrade on unseen data. Lacking explicit state-action representations also limits the expressiveness of RL objectives.

2. **Reward-Augmented Decoding**
   - *Pros:* allows controllable generation by injecting domain preferences at inference time. Computational overhead is minor for small vocabularies.
   - *Cons:* requires a well-specified reward function. If the reward is misaligned or noisy, it can skew generation toward degenerate sequences. Balancing log-probability and reward via \(\beta\) is non-trivial and may require tuning per domain.

3. **Self-Alignment**
   - *Pros:* domain term embeddings guide the model to remain on-topic and respect specialised vocabulary.
   - *Cons:* the approach is lightweight and may not capture complex domain reasoning. Alignment relies on a manually curated dictionary; coverage gaps can reduce effectiveness. More sophisticated techniques (e.g., retrieval-augmented models) could achieve deeper alignment.

4. **Robust Estimation**
   - *Pros:* temperature scaling is simple and often effective for calibration. Entropy-based confidence gives a direct measure of uncertainty.
   - *Cons:* calibration is only as good as the validation set. Out-of-distribution inputs may still receive overconfident predictions. Additional methods such as Monte Carlo dropout or ensemble models could further improve reliability but would add complexity.

## Areas for Improvement

- Explore alternative offline RL objectives, e.g., conservative Q-learning or implicit reward models, to mitigate distributional shift.
- Investigate automatic tuning of the reward influence \(\beta\) using validation heuristics.
- Extend self-alignment by incorporating retrieval or knowledge graphs for richer domain context.
- Combine temperature scaling with other uncertainty estimators to better capture OOD behaviour.

Despite these limitations, the design establishes a foundation for controlled and reliable text generation suitable for experimentation on modest hardware.
