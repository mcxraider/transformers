# Transformer From Scratch 🤖

A comprehensive implementation of the Transformer architecture from the ground up, exploring both the foundational concepts and cutting-edge enhancements that have shaped modern NLP and AI.

## 🎯 Project Goals

This project is a deep dive into understanding how Transformers work by implementing them from scratch using PyTorch. Starting with the original "Attention Is All You Need" paper, I'll progressively add modern enhancements and optimizations to explore how the architecture has evolved.

### Learning Objectives
- **Understand the fundamentals**: Multi-head attention, positional encoding, feed-forward networks
- **Explore modern enhancements**: Flash Attention, RoPE, mixture of experts, and more
- **Compare architectures**: GPT, BERT, T5, LLaMA and their differences
- **Optimize for efficiency**: Memory usage, training speed, and inference performance
- **Build intuition**: Through visualization, ablation studies, and experimentation

## 🚀 Current Status

### ✅ Completed
- [ ] Repository setup and project structure
- [ ] Base Transformer components
- [ ] Basic training pipeline
- [ ] Initial documentation

### 🔄 In Progress
- [ ] Multi-head attention mechanism
- [ ] Positional encoding implementations
- [ ] Layer normalization

### 📋 Planned
- [ ] Complete base transformer
- [ ] Model variants (GPT, BERT)
- [ ] Enhancement implementations
- [ ] Performance benchmarking

## 🏗️ Architecture Overview

### Base Implementation
The foundation implements the original Transformer architecture with:

```
Transformer
├── Multi-Head Attention
├── Position-wise Feed-Forward Networks
├── Positional Encoding
├── Layer Normalization
└── Residual Connections
```

### Model Variants
- **GPT-style**: Decoder-only for autoregressive language modeling
- **BERT-style**: Encoder-only for bidirectional representations
- **T5-style**: Encoder-decoder for sequence-to-sequence tasks
- **LLaMA-style**: Modern decoder-only with enhancements

## 🔬 Planned Enhancements

### Attention Mechanisms
- **Flash Attention** - Memory-efficient attention computation
- **Multi-Query Attention (MQA)** - Shared key/value heads for faster inference
- **Grouped-Query Attention (GQA)** - Balance between MQA and standard attention
- **Linear Attention** - Linear complexity alternatives to quadratic attention

### Positional Encoding
- **Rotary Position Embedding (RoPE)** - Rotation-based position encoding
- **ALiBi** - Attention with Linear Biases
- **Relative Positional Encoding** - Learning relative positions

### Normalization Techniques
- **RMSNorm** - Root Mean Square normalization
- **LayerNorm variants** - Pre-norm vs post-norm architectures
- **Group Normalization** - Alternative normalization strategies

### Activation Functions
- **SwiGLU** - Gated Linear Units with Swish activation
- **GELU variants** - Different GELU approximations
- **Mish** - Self-regularized non-monotonic activation

### Architectural Innovations
- **Mixture of Experts (MoE)** - Sparse expert layers for scaling
- **Parallel Blocks** - Parallel attention and FFN computation
- **Mixture of Depths** - Dynamic depth routing
- **State Space Models** - Mamba and selective state spaces

### Optimization Techniques
- **Gradient Checkpointing** - Trading computation for memory
- **Mixed Precision Training** - fp16/bf16 for efficiency
- **Memory-Efficient Attention** - Various memory optimization strategies

## 📊 Benchmarking & Analysis

### Performance Metrics
- **Memory Usage**: Peak memory consumption during training/inference
- **Training Speed**: Tokens/second, wall-clock time per epoch
- **Model Quality**: Perplexity, downstream task performance
- **Scaling Behavior**: How improvements scale with model size

### Comparative Studies
- Base vs enhanced architectures
- Different attention mechanisms
- Normalization technique comparisons
- Activation function ablations

## 🛠️ Installation & Usage

### Prerequisites
```bash
Python 3.8+
PyTorch 2.0+
CUDA (optional, for GPU acceleration)
```

### Setup
```bash
git clone https://github.com/yourusername/transformer-from-scratch.git
cd transformer-from-scratch
pip install -r requirements.txt
pip install -e .
```

### Quick Start
```python
from src.transformer.base import Transformer
from src.transformer.models import GPT

# Basic transformer
model = Transformer(
    vocab_size=10000,
    d_model=512,
    n_heads=8,
    n_layers=6
)

# GPT-style model with enhancements
gpt_model = GPT(
    vocab_size=50000,
    d_model=768,
    n_heads=12,
    n_layers=12,
    use_flash_attention=True,
    use_rope=True,
    use_rms_norm=True
)
```

### Training
```bash
# Train base transformer
python experiments/scripts/train_base_model.py --config configs/base_transformer.yaml

# Train with enhancements
python experiments/scripts/train_enhanced_model.py --config configs/enhanced_transformer.yaml
```

## 📚 Learning Resources

### Key Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer
- [FlashAttention](https://arxiv.org/abs/2205.14135) - Memory-efficient attention
- [RoFormer](https://arxiv.org/abs/2104.09864) - Rotary Position Embedding
- [GLU Variants](https://arxiv.org/abs/2002.05202) - Gated Linear Units
- [Train Short, Test Long](https://arxiv.org/abs/2108.12409) - ALiBi

### Implementation References
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [Transformer Circuits Thread](https://transformer-circuits.pub/)
- [LLaMA Paper](https://arxiv.org/abs/2302.13971) - Modern architecture choices

## 🔍 Project Structure

```
src/transformer/
├── base/           # Core transformer components
├── enhancements/   # Modern improvements and optimizations
├── models/         # Complete model implementations
└── utils/          # Utilities and helpers

experiments/        # Training scripts and notebooks
tests/             # Comprehensive test suite
benchmarks/        # Performance analysis
docs/              # Documentation and learning notes
```

## 📈 Progress Tracking

### Implementation Phases

**Phase 1: Foundation** (Weeks 1-2)
- [ ] Multi-head attention
- [ ] Feed-forward networks
- [ ] Positional encoding
- [ ] Layer normalization
- [ ] Complete base transformer

**Phase 2: Model Variants** (Weeks 3-4)
- [ ] GPT implementation
- [ ] BERT implementation
- [ ] Training pipeline
- [ ] Basic evaluation

**Phase 3: Attention Enhancements** (Weeks 5-6)
- [ ] Flash Attention
- [ ] Multi-Query Attention
- [ ] Grouped-Query Attention
- [ ] Performance comparisons

**Phase 4: Positional & Normalization** (Weeks 7-8)
- [ ] RoPE implementation
- [ ] ALiBi implementation
- [ ] RMSNorm
- [ ] Architecture comparisons

**Phase 5: Advanced Features** (Weeks 9-12)
- [ ] SwiGLU activation
- [ ] Mixture of Experts
- [ ] Gradient checkpointing
- [ ] Mixed precision training

**Phase 6: Analysis & Documentation** (Ongoing)
- [ ] Comprehensive benchmarks
- [ ] Scaling studies
- [ ] Documentation completion
- [ ] Learning reflection

## 🤝 Contributing

This is primarily a learning project, but suggestions and discussions are welcome! Feel free to:
- Open issues for questions or clarifications
- Suggest additional enhancements to explore
- Share interesting papers or resources
- Point out bugs or improvements

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Vaswani et al. for the original Transformer architecture
- The PyTorch team for the excellent framework
- The research community for continuous innovations
- Various open-source implementations that provide inspiration

---

**Note**: This is an educational project focused on understanding and implementing transformer architectures. The implementations prioritize clarity and learning over production optimization.
