# Legacy Associative Benchmark Report

This file preserves the older root-level associative benchmark report that was generated before warmups and repeated-run percentile reporting were added.

It should be treated as a historical artifact, not the current source of truth.

## Preserved Contents

# Micro-RAG Associative Multi-Hop Memory Benchmark

This benchmark evaluates the system's ability to recall multi-hop logical chains (contextually linked facts) where the second-hop target fact does not share semantic/keyword similarities with the user's query.

## 📝 Scenario Details

- **Fact A (Direct Match)**: `"Alice's favorite flower is Bob's favorite food."`
- **Fact B (Second-Hop Target)**: `"Bob's favorite food is Sushi."`
- **Graph Relation Link**: `Fact A` <--> `Fact B`
- **Query Prompt**: `"What is Alice's favorite flower?"`

## 📊 Performance & Recall Results

- **Pure Semantic Search**: `508.87 ms`, direct fact recalled, second-hop fact missed
- **Graph-Expanded Search**: `530.67 ms`, direct fact recalled, second-hop fact recalled
- **Traversal Overhead**: `21.80 ms`
