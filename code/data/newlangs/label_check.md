# Language label check against each language's own Wikipedia

Checked 2026-10-19 with firecrawl. Article titles come from the Wikidata sitelink of each language item (wbgetentities with ascii=1, so code points are exact, decoded locally, not retyped). Leads were read with the TextExtracts API for ks, mni, kok, doi, brx. Labels are read from `code/languages.py`. Code points are hex.

| code | language | own Wikipedia | article title | bare name as written there | code points (bare) | current label | match | source URL | notes |
|---|---|---|---|---|---|---|---|---|---|
| as | Assamese | as.wikipedia | অসমীয়া ভাষা | অসমীয়া | 0985 09B8 09AE 09C0 09AF 09BC 09BE | অসমীয়া | YES (exact) | https://as.wikipedia.org/wiki/অসমীয়া_ভাষা | Dropped word: ভাষা (language). য় is written decomposed (09AF 09BC) in both title and label. MediaWiki autonym identical. |
| or | Odia | or.wikipedia | ଓଡ଼ିଆ ଭାଷା | ଓଡ଼ିଆ | 0B13 0B21 0B3C 0B3F 0B06 | ଓଡ଼ିଆ | YES (exact) | https://or.wikipedia.org/wiki/ଓଡ଼ିଆ_ଭାଷା | Dropped word: ଭାଷା. ଡ଼ is decomposed (0B21 0B3C) in both. MediaWiki autonym identical. |
| pa | Punjabi | pa.wikipedia | ਪੰਜਾਬੀ ਭਾਸ਼ਾ | ਪੰਜਾਬੀ | 0A2A 0A70 0A1C 0A3E 0A2C 0A40 | ਪੰਜਾਬੀ | YES (exact) | https://pa.wikipedia.org/wiki/ਪੰਜਾਬੀ_ਭਾਸ਼ਾ | Script: Gurmukhi. Uses tippi 0A70, same as label. Dropped word: ਭਾਸ਼ਾ. Shahmukhi Punjabi is a separate wiki (pnb.wikipedia), title پنجابی (067E 0646 062C 0627 0628 06CC). |
| ur | Urdu | ur.wikipedia | اردو | اردو | 0627 0631 062F 0648 | اردو | YES (exact) | https://ur.wikipedia.org/wiki/اردو | Title is the bare name. |
| ne | Nepali | ne.wikipedia | नेपाली भाषा | नेपाली | 0928 0947 092A 093E 0932 0940 | नेपाली | YES (exact) | https://ne.wikipedia.org/wiki/नेपाली_भाषा | Dropped word: भाषा. |
| sa | Sanskrit | sa.wikipedia | संस्कृतम् | संस्कृतम् | 0938 0902 0938 094D 0915 0943 0924 092E 094D | संस्कृतम् | YES (exact) | https://sa.wikipedia.org/wiki/संस्कृतम् | Title is the bare name, with final virama. |
| mai | Maithili | mai.wikipedia | मैथिली भाषा | मैथिली | 092E 0948 0925 093F 0932 0940 | मैथिली | YES (exact) | https://mai.wikipedia.org/wiki/मैथिली_भाषा | Dropped word: भाषा. |
| sd | Sindhi | sd.wikipedia | سنڌي ٻولي | سنڌي | 0633 0646 068C 064A | سنڌي | YES (exact) | https://sd.wikipedia.org/wiki/سنڌي_ٻولي | Script: Perso-Arabic (Sindhi alphabet). Final letter is Arabic yeh 064A, same as label. Dropped word: ٻولي (language). |
| ks | Kashmiri | ks.wikipedia | کٲشِر زَبان | کٲشِر | 06A9 0672 0634 0650 0631 | کٲشُر | NO vs title: label 06A9 0672 0634 064F 0631 vs 06A9 0672 0634 0650 0631; YES vs the lead's bold name کٲشُر (06A9 0672 0634 064F 0631) | https://ks.wikipedia.org/wiki/کٲشِر_زَبان | Script: Perso-Arabic. The wiki is inconsistent: the title has kasra 0650 (کٲشِر), but the bolded first word of the lead is کٲشُر (06A9 0672 0634 064F 0631, damma 064F), identical to the label; a later lead sentence uses the kasra form again. MediaWiki autonym for ks is also کٲشُر (damma). Dropped word: زَبان. |
| kok | Konkani | gom.wikipedia (Goan Konkani) | कोंकणी भास | कोंकणी | 0915 094B 0902 0915 0923 0940 | कोंकणी | YES (exact) | https://gom.wikipedia.org/wiki/कोंकणी_भास | Script: Devanagari for this article (gom.wikipedia also carries Roman-script Konkani; the lead gives Roman 'Konknni'). Dropped word: भास (language). Lead's first word is कोंकणी. MediaWiki name for the wiki language (gom-deva) is गोंयची कोंकणी ('Goan Konkani'). |
| doi | Dogri | NONE: no Dogri Wikipedia (dgowiki, doiwiki not recognised); Incubator test wiki Wp/dgo | डोगरी भाशा | डोगरी | 0921 094B 0917 0930 0940 | डोगरी | YES (exact) | https://incubator.wikimedia.org/wiki/Wp/dgo/डोगरी_भाशा | Incubator page, not a live Wikipedia. Dropped word: भाशा (Dogri spelling with श). Lead's first word is डोगरी. MediaWiki has no autonym for doi. |
| mni | Meitei (Manipuri) | mni.wikipedia | ꯃꯤꯇꯩꯂꯣꯟ | ꯃꯤꯇꯩꯂꯣꯟ | ABC3 ABE4 ABC7 ABE9 ABC2 ABE3 ABDF | ꯃꯤꯇꯩꯂꯣꯟ | YES (exact) | https://mni.wikipedia.org/wiki/ꯃꯤꯇꯩꯂꯣꯟ | ꯂꯣꯟ ('language') is attached with no space, so kept. Lead's bolded first word is ꯃꯤꯇꯩꯂꯣꯟ (same code points). MediaWiki autonym differs: ꯃꯤꯇꯩ ꯂꯣꯟ (ABC3 ABE4 ABC7 ABE9 0020 ABC2 ABE3 ABDF, with a space). |
| brx | Bodo | NONE: no Bodo Wikipedia (brxwiki not recognised); Incubator test wiki Wp/brx | बर' राव | बर' | 092C 0930 0027 | बर' | YES (exact) | https://incubator.wikimedia.org/wiki/Wp/brx/बर'_राव | Incubator page, not a live Wikipedia. Apostrophe is ASCII 0027 in the title and in running text (बर' मुंनि, बर' थुनलाइ). The bold opening of the lead shows 'बर राव' without the apostrophe (likely swallowed by bold wiki markup). Dropped word: राव (language). MediaWiki has no autonym for brx. |
| sat | Santali | sat.wikipedia | ᱥᱟᱱᱛᱟᱲᱤ ᱯᱟᱹᱨᱥᱤ | ᱥᱟᱱᱛᱟᱲᱤ | 1C65 1C5F 1C71 1C5B 1C5F 1C72 1C64 | ᱥᱟᱱᱛᱟᱲᱤ | YES (exact) | https://sat.wikipedia.org/wiki/ᱥᱟᱱᱛᱟᱲᱤ_ᱯᱟᱹᱨᱥᱤ | Script: Ol Chiki. Dropped word: ᱯᱟᱹᱨᱥᱤ (language). MediaWiki autonym identical. |
| ja | Japanese | ja.wikipedia | 日本語 | 日本語 | 65E5 672C 8A9E | 日本語 | YES (exact) | https://ja.wikipedia.org/wiki/日本語 | 語 ('language') is part of the word, so kept. |
| ru | Russian | ru.wikipedia | Русский язык | Русский | 0420 0443 0441 0441 043A 0438 0439 | Русский | YES (exact) | https://ru.wikipedia.org/wiki/Русский_язык | Dropped word: язык. Capitalised because it starts the title; MediaWiki autonym is lowercase русский. |
| ar | Arabic | ar.wikipedia | اللغة العربية | العربية | 0627 0644 0639 0631 0628 064A 0629 | العربية | YES (exact) | https://ar.wikipedia.org/wiki/اللغة_العربية | Dropped word: اللغة ('the language'). Bare form equals MediaWiki autonym. |

## Existence checks
- Wikidata `wbgetentities` warned 'Unrecognized values for parameter sitefilter: dgowiki, doiwiki' and 'Unrecognized value ... brxwiki', so there is no Dogri or Bodo Wikipedia. (`dgawiki` is recognised but is Southern Dagaare, not Dogri.)
- Incubator: `Wp/dgo` is an open test wiki (Dogri; macrolanguage doi; main page Wp/dgo/मुक्ख सफा). `Wp/doi` only points to Wp/dgo. `Wp/brx` is an open test wiki since 2009-01-14 (main page Wp/brx/Gahai bilai).

## API sources
- https://www.wikidata.org/w/api.php?action=wbgetentities&ids=Q29401|Q33810|Q58635|Q1617&props=sitelinks&sitefilter=aswiki|orwiki|pawiki|pnbwiki|urwiki&format=json&formatversion=2&ascii=1
- https://www.wikidata.org/w/api.php?action=wbgetentities&ids=Q33823|Q11059|Q36109|Q33997&props=sitelinks&sitefilter=newiki|sawiki|maiwiki|sdwiki&format=json&formatversion=2&ascii=1
- https://www.wikidata.org/w/api.php?action=wbgetentities&ids=Q33552|Q34239|Q5575236|Q32730&props=sitelinks&sitefilter=kswiki|gomwiki|dgowiki|doiwiki|dgawiki&format=json&formatversion=2&ascii=1
- https://www.wikidata.org/w/api.php?action=wbgetentities&ids=Q33868|Q33223|Q33965&props=sitelinks&sitefilter=mniwiki|brxwiki|satwiki&format=json&formatversion=2&ascii=1
- https://www.wikidata.org/w/api.php?action=wbgetentities&ids=Q5287|Q7737|Q13955&props=sitelinks&sitefilter=jawiki|ruwiki|arwiki&format=json&formatversion=2&ascii=1
- https://www.wikidata.org/w/api.php?action=query&meta=languageinfo&liprop=autonym&licode=as|or|pa|pnb|ur|ne|sa|mai|sd|ks|ks-arab|ks-deva|gom|gom-deva|gom-latn|kok|doi|dgo|mni|mni-beng|brx|sat|ja|ru|ar&format=json&formatversion=2&ascii=1 (MediaWiki interface autonyms)
- https://incubator.wikimedia.org/w/api.php?action=query&prop=revisions&rvprop=content&rvslots=main&titles=Wp/dgo|Wp/doi|Wp/brx&format=json&formatversion=2&ascii=1
- Lead extracts: ks.wikipedia, mni.wikipedia, gom.wikipedia and incubator (Wp/brx/बर' राव, Wp/dgo/डोगरी भाशा) via action=query&prop=extracts&exintro=1&explaintext=1&ascii=1

## Recommended label corrections
- None required. 16 of 17 labels equal, code point for code point, the bare name in the article title on the language's own Wikipedia (for doi and brx, the Incubator test-wiki article). The 17th, ks, equals the name the ks.wikipedia lead puts in bold.
- Judgement call, ks: the ks.wikipedia title spells the name with kasra (کٲشِر, 06A9 0672 0634 0650 0631). The lead's bold name and the MediaWiki autonym use the label's damma form (کٲشُر, 06A9 0672 0634 064F 0631). Keep کٲشُر. Switch to کٲشِر only if the rule is that the title spelling wins.
- Watch, mni: MediaWiki's interface autonym is spaced (ꯃꯤꯇꯩ ꯂꯣꯟ); the Wikipedia title and lead are unspaced like the label. Keep the label.
- Watch, doi and brx: no live Wikipedia; the labels match the Incubator test wikis only.
