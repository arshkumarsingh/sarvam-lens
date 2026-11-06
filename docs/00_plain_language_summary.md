# Plain-language summary

## The question

Sarvam is an Indian AI model built to understand Tamil, Telugu, Hindi, Bengali and many other
Indian languages. When it translates a word from Tamil to Telugu, does it go straight from Tamil to
Telugu, or does it silently pass through English "in its head" first?

An analogy: a translator who grew up speaking English might think, while translating Tamil to
Telugu, *"நீர்… that's 'water'… so in Telugu that's నీరు."* The answer comes out in Telugu, but
English was the hidden middle step. We wanted to know whether Sarvam does the same.

## How we looked inside

An AI model processes text in stages, like an assembly line with stations (Sarvam-30B has 19). The
last station produces the answer. We used **"lenses"**, mathematical tools that let us peek at a
middle station and ask: *if the model had to speak right now, what word would it say?* We used four
different lenses. A result counts only if all four agree. They all read through the same last
station, though, so they aren't four independent witnesses; the deletion tests and a direct
comparison of the model's internal states (below) are the independent checks.

Then we went further than looking:
- **Removing:** we deleted the "English word" from the model's middle stations and checked whether
  its Telugu answer got worse.
- **Controls:** we repeated every check with a *different* word (e.g. "fire" when the word was
  "water") to make sure we weren't just seeing generic English.
- **A comparison model:** we ran the same tests on Qwen, a similar-sized model built mainly for
  English and Chinese.

## What we found

1. **Sarvam thinks in English in the middle.** At stations 14–16, the English word ("water") is
   among the model's top-10 candidate words for 93–100% of Tamil→Telugu, Bengali→Gujarati (etc.)
   translations, depending on the lens (about 98% under the main one), even though the prompt
   contains no English at all. The Telugu answer only appears at the end. With a *different*
   English word ("fire"), the score is 0–3%, so this is about the specific concept.
2. **The English step is used, but it isn't the only route.** Deleting the English word from the
   middle stations lowers the model's confidence in the right answer by ~6 points. Deleting a
   different word does nothing. But the model still gets ~95% of answers right afterwards, so it
   has other pathways too.
3. **Hindi is a quieter second middle language.** The right Hindi word also shows up often (41–83%
   of the time depending on the lens, with low weight), and deleting it hurts nearly as much as
   deleting English.
4. **Being built "for India" didn't reduce the English dependence.** Qwen (English/Chinese-centric)
   passes through English just as strongly, and also through Chinese. Each model's hidden middle
   languages track the languages it was mostly trained on.
5. **The same thing happens in ordinary sentences**, not just translation tasks: in 43 sentences the
   smaller model completed on its own (60 for the bigger model), and when it translates whole
   sentences in its chat format.
6. **The first ~60% of Sarvam's stations can't be read as words in any language.** Whatever happens
   there is not yet "verbal."

**A refinement, from a later check (doc 15).** "Thinks in English" is a handy summary, but the
more accurate picture is a shared "concept space" that all languages feed into. When we compared
the model's internal states directly, without asking which *word* they point to, Tamil "water",
Hindi "water" and English "water" all landed in the same place. English wasn't the centre;
English and Hindi were about equally central. English shows up so readily when we decode the
middle stations partly because most of Sarvam's vocabulary is English-like (Latin-script). The
Chinese-trained comparison model, by contrast, does have a clear English hub.

**Two more checks (doc 17).** English isn't the only language visible in the middle stations. If we
look only at words written in a given script, the right word in the other Indian languages shows up
for about 3 in 4 items. English is the easiest to read, not the only one there. And when we looked for
a near-synonym instead ("liquid" for "water"), it rarely showed up. What we read out is the word
itself, not just its meaning. Deleting the English, Hindi and target-language words all at once
still leaves 94% of translations right, so the concept is carried in other ways too.

## Things we found along the way

- **A real bug:** with the current version of a popular AI software library (transformers 5),
  Sarvam-30B silently loads *without its sense of word order*. It still answers simple questions,
  so nobody would notice, but it gets much worse at anything where order matters. We fixed it in
  our code.
- **A measurement quirk:** a popular visualization of how models organize their "thoughts" (the
  J-space CKA explorer) shows Sarvam as completely structureless. We traced that to a few unusually
  large numbers in Sarvam's final layer, which swamp that particular measurement. With them
  removed, Sarvam has structure, and its early/late boundary sits exactly where English appears.

## Later questions

- **All 22 official Indian languages, not just 8.** We added the other 14, including small
  languages like Kashmiri, Santali (with its own Ol Chiki script), Manipuri (Meitei Mayek script)
  and Bodo. The words came from Wiktionary. Every language we could test goes through English in
  the middle, on both Sarvam models: 21 of the 22 (Dogri had too few words to test). Deleting the
  English word hurts the six added languages we tested that way (Odia, Urdu, Kashmiri, Manipuri,
  Santali and Japanese) *more* than the original ones (about 13 points instead of 6, in a separate
  run). Where Sarvam struggles is producing the small
  languages: asked to translate from Telugu *into* Kashmiri, the smaller model got 1 word in 32
  right (13 in 32 from English). Even when it fails, the English word is still sitting in its
  middle stations. It understood the word and simply couldn't write it in Kashmiri, so it mostly
  wrote the Telugu word instead.
- **Math shows up in English too.** We gave Sarvam sums written only in Devanagari or Ol Chiki
  digits, like "३ + ४ =", with no English anywhere. In its middle stations the English word
  "seven" lit up for about three quarters of the problems under our main lens (about half under
  the simplest lens, and 4 in 10 on the bigger model); a wrong number's word never did. But
  deleting "seven" from those stations doesn't break the sum, and deleting all nine number words
  costs only 2–4 points (doc 17). The English word shows what the model is thinking about, but the
  sum doesn't depend on it. When
  Sarvam "thinks out loud" about a math problem, it does so mostly in English (90–97% of the
  letters are Latin script) in
  every one of 25 languages, small ones included (8 problems per language). It quotes the
  problem's own words inside English sentences. Is that "thinking out loud" honest? We slipped in
  a wrong hint ("a teacher says the answer is N", with N a wrong number near the right one). When
  Sarvam went along with it, its reasoning always admitted the hint, in every language. But
  usually the reasoning had already worked out the right answer, and when we cut the reasoning
  off and made it answer, Sarvam gave the teacher's number. Its reasoning was open about the
  hint; its final answer didn't follow its own reasoning. A rerun with three times the problems,
  letting the reasoning finish on its own, found the same: about a third of answers followed the
  hint, and about 4 in 10 among reasoning that finished by itself (doc 17).
- **Math questions aren't much "more English inside" than science or technical ones of the same
  length.** The apparent difference came mostly from math questions being longer.
- **Saying no happens in English first.** When someone asks Sarvam for something harmful in, say,
  Tamil, and Sarvam refuses in Tamil, the English "Sorry, I cannot…" shows up in its middle
  stations before any Tamil refusal words do. The Chinese-trained comparison model says "no" in
  Chinese inside. Sarvam also refuses harmful requests much less often in Indian languages than
  in English (about 42% vs 82%), even though a simple detector trained on English can often read
  from the model's internals that the request is harmful. The full refusal study comes with the
  second blog post.
- **A gap in Sarvam's vocabulary:** it can't represent 5 of the 10 Tamil digits or 3 of the 10
  Telugu digits as symbols, so it fails at sums written in them.

## Why it matters

- Even an India-focused AI mostly "thinks" in English, probably because most of its training text
  was English (the internet is mostly English). Concepts that don't translate cleanly into English
  may suffer.
- These tests are a ready-made yardstick. If someone builds a model that really doesn't route
  through English, this is how to check.

## Cost

About **$30** of rented GPU time on Modal, spread over 20 runs. The largest single item was fitting
one lens on 5 GPUs in parallel (~$19). The follow-up questions cost more on top; see
[`05_costs.md`](05_costs.md).
