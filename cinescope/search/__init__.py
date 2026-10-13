"""Phase 8: scene search in plain English and "chat with the movie".

  vecindex.py      exact, IVF and IVF-PQ vector search from scratch (NumPy)
  dual_encoder.py  a small CLIP-style dual encoder trained with InfoNCE
  encoders.py      pluggable image/text encoders (open_clip, the from-scratch one)
  shotindex.py     clip -> shots -> key frames -> embeddings -> index on disk
  metrics.py       Recall@k, MRR, time-overlap relevance
  rag.py           retrieve shots for a question, assemble evidence, answer
"""
