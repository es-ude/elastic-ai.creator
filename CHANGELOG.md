# Changelog

<!--next-version-placeholder-->

## v0.6.0 (2022-05-25)
### Fix
* **vhdl:** Move missing files ([`e4ae3c2`](https://github.com/es-ude/elastic-ai.creator/commit/e4ae3c2815a33b8f4f33c9578ab5cae0842277aa))
* **vhdl:** Fix previously broken imports ([`bf694f8`](https://github.com/es-ude/elastic-ai.creator/commit/bf694f80fbd3a5478d99e8ae6b198a9e363569c9))

### Breaking
* move modules out of generator package  ([`bf694f8`](https://github.com/es-ude/elastic-ai.creator/commit/bf694f80fbd3a5478d99e8ae6b198a9e363569c9))

## v0.5.0 (2022-05-25)
### Feature
* **precomputation, typing:** Make IOTable iterable and fix types ([`faa1d77`](https://github.com/es-ude/elastic-ai.creator/commit/faa1d7799bd6e8223cc4953170286d425255bb7b))
* **vhdl:** Add multiline template expansion ([`309ea35`](https://github.com/es-ude/elastic-ai.creator/commit/309ea350fae2b4e54bf06101aadc28e227d30cbb))
* **vhdl:** Add multiline template expansion ([`0d7f91f`](https://github.com/es-ude/elastic-ai.creator/commit/0d7f91f7347a5501eba02ad40499a5c0fdcce3bc))
* **vhdl:** Add multiline template expansion ([`5779708`](https://github.com/es-ude/elastic-ai.creator/commit/5779708c2de34d9c32a18af689b4b094d513ef8d))
* **vhdl:** Add multiline template expansion ([`3177fcd`](https://github.com/es-ude/elastic-ai.creator/commit/3177fcd4e5f1830608e9f6590b5af312bb74b7a9))

### Documentation
* **readme:** Add git commit message scopes ([`fe8e328`](https://github.com/es-ude/elastic-ai.creator/commit/fe8e328eda5a5f9e4cac886fcbfc9388f13d3d0f))
* **readme:** Fix table of contents and section headers ([`ecdef5d`](https://github.com/es-ude/elastic-ai.creator/commit/ecdef5da63c2c10e61a159c144c5c3707a5699e8))
* **readme:** Shorten ([`e535ea0`](https://github.com/es-ude/elastic-ai.creator/commit/e535ea0fd9d783f29ebb32d756077289d8baa8c9))

## v0.4.2 (2022-05-24)
### Fix
* **number-repr:** Fix a bug and add some parameter checks ([`a78e9e8`](https://github.com/es-ude/elastic-ai.creator/commit/a78e9e8f669c477d0629695f5c7c8ad8628f0522))

## v0.4.1 (2022-05-24)
### Fix
* Minor errors ([`812809e`](https://github.com/es-ude/elastic-ai.creator/commit/812809e1d0e706df3a0514b3503dc283ea12d7a4))

## v0.4.0 (2022-05-23)
### Feature
* **vhdl:** Allow ToLogicEncoder to register symbols in batches ([`9209279`](https://github.com/es-ude/elastic-ai.creator/commit/9209279debe651b653d2fee44533ccbdae945b32))

### Fix
* **vhdl:** Improve names in scope of ToLogicEncoder ([`67f7312`](https://github.com/es-ude/elastic-ai.creator/commit/67f73129faefe343e9fb5e84563d125b1d36bab6))

### Breaking
* rename numerics attr to _symbols, mapping attribute to _mapping. rename add_numeric to register_symbol  ([`67f7312`](https://github.com/es-ude/elastic-ai.creator/commit/67f73129faefe343e9fb5e84563d125b1d36bab6))

## v0.3.10 (2022-05-23)
### Fix
* **types:** Add missing mlframework types ([`3b5cf5f`](https://github.com/es-ude/elastic-ai.creator/commit/3b5cf5f8be829e109db363c25ecff76634f9d94f))
* **typing:** Fix some mypy errors ([`35b8fdf`](https://github.com/es-ude/elastic-ai.creator/commit/35b8fdf4cb0736770d9592f86499192e1e84d673))

## v0.3.9 (2022-05-22)
### Fix
* **pre-commit:** Add missing commitlint.config.js ([`2251de8`](https://github.com/es-ude/elastic-ai.creator/commit/2251de83f60823d21346aedcc2b2e9aac4c27458))
* **pyproject:** Fix duplicate field ([`6616cab`](https://github.com/es-ude/elastic-ai.creator/commit/6616cab3b0342f0b5d0b8bbdbbdf719de56d5631))
* **gh-workflow:** Updat changelog correctly ([`e76a41c`](https://github.com/es-ude/elastic-ai.creator/commit/e76a41cf55463cbc2a4ffa5b2b233d49695302b9))
* Add missing tags_utils again ([`910c611`](https://github.com/es-ude/elastic-ai.creator/commit/910c6116600b82e2c52c7d46896d92b63954d7c7))
* **gh-workflow:** Install latest isort fixing broken imports ([`a61ef44`](https://github.com/es-ude/elastic-ai.creator/commit/a61ef445672f913ec4ebc4cc8b46c2ef9099bec7))
* **gh-workflow:** Set git user+mail to gh-actions ([`174ed47`](https://github.com/es-ude/elastic-ai.creator/commit/174ed478b04b846912d6b0315f1143f24bc94524))
* **input_domains:** Add missing import of itertools ([`a6b0344`](https://github.com/es-ude/elastic-ai.creator/commit/a6b0344ac4b933112b19b8603358a0adc7274533))
* **precomputation:** Correct numpy namespace for type alias ([`a6c5842`](https://github.com/es-ude/elastic-ai.creator/commit/a6c5842920c00ae6e53e226650e0fbfe48aac44a))
* **gh-workflow:** Fix job dependencies ([`6a7d3ee`](https://github.com/es-ude/elastic-ai.creator/commit/6a7d3eeb975ca303aa30fce21bd29d14cf9982d3))
* **gh-workflow:** Fix typo ([`7f86205`](https://github.com/es-ude/elastic-ai.creator/commit/7f8620502ee544917db42ea12c7cb2eadbaef8cc))
* **gh-workflow:** Typo unit-test -> unit-tests ([`1dbd71f`](https://github.com/es-ude/elastic-ai.creator/commit/1dbd71f5f3dae489b4752a5f6fdf9d10e4251a73))
* **gh-workflow:** Close brace ([`a6e4b99`](https://github.com/es-ude/elastic-ai.creator/commit/a6e4b999dadd163881fa96d03977c9c392a9267b))
* **gh-workflow:** Bump version ([`2cb3a72`](https://github.com/es-ude/elastic-ai.creator/commit/2cb3a72b2aa9a86c0b4da71e3d7bff962a5728f6))
* **gh-workflow:** Fix syntax error ([`895326d`](https://github.com/es-ude/elastic-ai.creator/commit/895326d67eb7ba1bb866a45c8b149778c93dc043))

### Documentation
* **readme:** Add brief explanation of pre-commit ([`3626bb0`](https://github.com/es-ude/elastic-ai.creator/commit/3626bb07cc1c8600b193bc380ae8275116ebaba8))
* Update changelog ([`e1aa8c9`](https://github.com/es-ude/elastic-ai.creator/commit/e1aa8c93554fc15c25a586b8e89eecda6dc03514))
