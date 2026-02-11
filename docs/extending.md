# Extending the Training Pipeline

This pipeline is designed to be extended without modification.

## Common extensions

- New modality
  → Write a new Tensor Adapter

- New task
  → Add a new model head and loss function

- New dataset
  → Reuse Trainer and Dataset abstractions

## Design rule

If you feel the need to change core training code,
consider whether a new adapter or wrapper is more appropriate.

## Goal

Enable future projects to inherit this pipeline with minimal friction.
