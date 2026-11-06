# NOTICE

Sarvam-30B workspace lenses (J-lens, R-lens, J++, Indic-fit J-lens) and router.
Copyright 2026 Arsh Kumar Singh. Licensed under the Apache License, Version 2.0 (see LICENSE).

This release includes or is derived from the following works:

1. **Sarvam-30B** (`sarvamai/sarvam-30b`, revision 071ae95e933605ca1104a6b4524a36a98488efa4),
   Copyright Sarvam AI, Apache License 2.0. The lenses are Jacobians of this model's residual stream.
   No base-model weights are included.
2. **jpp_lens** (`safety-research/jpp_lens`, commit d7efc0de372e20589e899188c0cfe8434acc63ec),
   Copyright 2026 Kola Ayonrinde, Apache License 2.0. It includes the Jacobian lens reference
   implementation, Copyright 2026 Anthropic PBC, Apache License 2.0, and LRP rules adapted from
   Camila Blank and Agam Bhatia's R-lens. Used to fit every lens and the router.
3. **WikiText-103** (`Salesforce/wikitext`, `wikitext-103-raw-v1`), CC BY-SA, derived from
   Wikipedia. Activations on its text were used to fit the J-lens, R-lens, J++ lens and router. No
   text from it is included.
4. **Wikipedia** (`wikimedia/wikipedia`, dump 20231101; en, hi, bn, ta, te, mr, gu, kn, ml),
   CC BY-SA. Activations on 27 passages were used to fit the Indic J-lens. No text from it is
   included.

Code, logs and full provenance: https://github.com/arshkumarsingh/sarvam-lens.
