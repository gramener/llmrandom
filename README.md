# LLM Random Number Experiment

When picking a number between 1-100, do hashtag#LLMs pick randomly? Or pick like a human?

[Leniolabs](https://www.leniolabs.com/artificial-intelligence/2023/10/04/42-GPTs-answer-to-Life-the-Universe-and-Everything/) found that ChatGPT prefers 42.

When I re-ran the experiment, things changed a bit. Now, 47 is the new favorite.

But Claude 3 Haiku latched on to 42 as its favorite. Gemini's favorite is 72.

They all avoid multiples of 10 (10, 20, ...), repeated digits (11, 22, ...), single digits (1, 2, ...) and prefer 7-endings (27, 37, ...). These are clearly human biases -- avoiding regular / round numbers and seeking 7 as "random".

Strangely, they all avoid numbers ending with 1, and like 72, 73 and 56 a lot.

This repository contains the results of the experiment.

Here is the [code to generate the results](https://github.com/sanand0/ipython-notebooks/blob/master/llm-random-numbers.ipynb).

## 14 Sep 2026: GPT-4.1 Nano and GPT-5.6 Luna

I re-ran the experiment with newer OpenAI models through OpenRouter. `results.json` now contains the original 2024 data plus the new runs, with raw outputs, model/sampling metadata, token usage, provider routing, and cost.

Run it with:

```bash
# .env should contain OPENROUTER_API_KEY=...
# OPENROUTER_PERSONAL_API_KEY also works for my existing setup.
uv run generate-results.py
```

The script is resumable and idempotent: it saves after every successful response and skips samples/runs already present in `results.json`. By default it generates 200 responses per condition. GPT-4.1 Nano is tested at temperatures 0.0, 0.1, ..., 1.0. GPT-5.6 Luna does not expose `temperature` through OpenRouter, so it is tested once at the default sampling settings with reasoning effort set to `none`.

The result is emphatically not random:

- **GPT-4.1 Nano loves 57.** At temperature 0 it returned **57 all 200 times**. Even at temperature 1.0, 57 was still the mode at **78/200 (39%)**, and only **16 distinct numbers** appeared.
- **Higher temperature adds variety, not uniformity.** Nano goes from 1 distinct value at temperature 0 to 16 at temperature 1, but the same small cluster -- especially 57, 47, and 73 -- dominates throughout.
- **GPT-5.6 Luna prefers 47.** With reasoning disabled it returned **47 in 108/200 (54%)** runs and **73 in 58/200 (29%)**; only **6 distinct numbers** appeared.
- **The favorite changed with the model, not the phenomenon.** The 2024 experiment found GPT-3.5 Turbo favored 47, Claude 3 Haiku favored 42, and Gemini 1.0 Pro favored 72. Newer models still show a strong human-like preference for a few "random-looking" numbers rather than anything close to a uniform draw.

The 2,400 new OpenRouter calls all produced valid integers from 0 to 100 and cost **$0.0090684** in total. Costs for the original 2024 direct-API runs are unknown because token/billing data was not retained.
