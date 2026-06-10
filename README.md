# Non-Autoregressive Translation with Pretrained mBART Encoder

This repository contains a PyTorch implementation of a Non-Autoregressive (NAT) Transformer designed for machine translation tasks. It initializes its encoder weights from a pretrained `mBART` model while concurrently running a custom target-length prediction network to manage parallel generation constraints.

## Features
* **Non-Autoregressive Decoding:** Decodes target tokens in parallel rather than token-by-token.
* **Target Length Prediction:** Incorporates a pooling-based predictor calculating categorical length differences between source and target sequences.
* **Pretrained Encoding Backbone:** Leverages `facebook/mbart-large-50-many-to-many-mmt`.

## Setup Instructions

1. Clone the repository and navigate into it:
   ```bash
   git clone [https://github.com/your-username/nat-translator.git](https://github.com/your-username/nat-translator.git)
   cd nat-translator