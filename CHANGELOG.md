# Changelog

<!--next-version-placeholder-->

## v0.10.1 (2022-06-29)
### Fix
* **gh-workflow:** Fix error in the main.yml ([`4a6ff5e`](https://github.com/es-ude/elastic-ai.creator/commit/4a6ff5e61f35661a3ef83ce4335c109333834d6d))
* **gh-workflow:** Try to fix the error with the semantic release tool ([`bc115f8`](https://github.com/es-ude/elastic-ai.creator/commit/bc115f899bd85e720448bfa67fe9964bb56c594b))

## v0.10.0 (2022-06-26)
### Feature
* **vhdl:** Format_vhdl function blocks the process until formatting is complete ([`a8a1bd0`](https://github.com/es-ude/elastic-ai.creator/commit/a8a1bd0e7a4db075d0cef4a9eb125a860a697719))

### Documentation
* **readme:** Remove compile instructions for onnx as they are no longer needed ([`3bee70a`](https://github.com/es-ude/elastic-ai.creator/commit/3bee70abe4185a0a6708ffc998bdc74004f90b8a))
* **vhdl:** Add a docstring with an example to the FixedPoint class ([`961d766`](https://github.com/es-ude/elastic-ai.creator/commit/961d76678d366730f57bbd69b43c38124c003bf7))

## v0.9.0 (2022-06-22)
### Feature
* **vhdl:** Integrate FixedPoint type ([`b67a609`](https://github.com/es-ude/elastic-ai.creator/commit/b67a6096023a51ff4882a8cdd03a7765884c8d93))
* **vhdl:** Add function to convert FixedPoint to signed int representation ([`03001ed`](https://github.com/es-ude/elastic-ai.creator/commit/03001ed608ac934e8bbdcdfa1acb2fc7c163a89a))
* **vhdl:** Verify total bits and frac bits for multiple lists ([`360d318`](https://github.com/es-ude/elastic-ai.creator/commit/360d318db0076d9077ceb94f3f7904d95e2b12f6))
* **vhdl:** Integrate FixedPoint datatype in the LSTM test bench classes ([`7cbb88a`](https://github.com/es-ude/elastic-ai.creator/commit/7cbb88a7f77728776e0e976dcc68505b4162f0cc))
* **vhdl:** Add function to convert a list of ints to a list of FixedPoint objects ([`abece1f`](https://github.com/es-ude/elastic-ai.creator/commit/abece1fd38af607c5f5734aeacd77a1743ff3411))
* **vhdl:** Add function to convert list of float values to a list of FixedPoint objects ([`02b26d8`](https://github.com/es-ude/elastic-ai.creator/commit/02b26d868cad2a5a5bed2350a2929cf362ccdca8))
* **vhdl:** Change Rom that it uses the FixedPoint datatype ([`876cdb8`](https://github.com/es-ude/elastic-ai.creator/commit/876cdb821ff0ac67ae2345c8a36e4a742cce0949))
* **vhdl:** Add a function to infer total and frac bits from a sequence of FixedPoint values ([`9cc2b72`](https://github.com/es-ude/elastic-ai.creator/commit/9cc2b721b147628b2abf524129eeaac8f68520d5))
* **vhdl:** Separate hex/bin representation from vhdl hex/bin representation ([`eb8fe60`](https://github.com/es-ude/elastic-ai.creator/commit/eb8fe60300ee7572500f9f9d11b62a9c5abff802))

### Fix
* **vhdl:** Correct usage of the lookup_table_generator_function according to the type hints ([`9812ee8`](https://github.com/es-ude/elastic-ai.creator/commit/9812ee85cd467e261af942b30493ac0e970ea5e4))
* **vhdl:** Remove old brevitas code ([`86a8104`](https://github.com/es-ude/elastic-ai.creator/commit/86a8104cb6049dc016a5c8da08a7d2abc011935b))
* **vhdl:** Change value so that it fits into the value range of a fixed point value ([`b4e973e`](https://github.com/es-ude/elastic-ai.creator/commit/b4e973ebb8a087351e07966821229f69dc345d79))

## v0.8.0 (2022-06-08)
### Feature
* **gh-workflow:** Increase python version ([`02403e6`](https://github.com/es-ude/elastic-ai.creator/commit/02403e6cb7d8c9acc4357d9649fd2ae0834030a0))
* **pyproject:** Drop brevitas support ([`103f188`](https://github.com/es-ude/elastic-ai.creator/commit/103f1882c8da81cdf114f10b1b76c2ce89a07cba))
* Bump python version to 3.10 ([`47f5f07`](https://github.com/es-ude/elastic-ai.creator/commit/47f5f0718460a966faaa937b2c6b016720434082))

### Fix
* **pyproject:** Update poetry lock file ([`0116934`](https://github.com/es-ude/elastic-ai.creator/commit/0116934b994c4e743b1be009172de0e07acd9182))
* **pyproject:** Update poetry lock file ([`9230672`](https://github.com/es-ude/elastic-ai.creator/commit/92306722dabe5c4196e79a7cbbebab1e75ac3e6d))
* **pyproject:** Correct deps ([`7935ba1`](https://github.com/es-ude/elastic-ai.creator/commit/7935ba19bcbda7e47ddbc358c12af3aa2a01df0a))
* **precomputation:** Change ModuleProto to Module ([`cfe418e`](https://github.com/es-ude/elastic-ai.creator/commit/cfe418e41889708a53c255a8a7abcd6f1648f8f2))
* **gh-workflow:** Set more explicit python version ([`9c44093`](https://github.com/es-ude/elastic-ai.creator/commit/9c44093c6cd41d05a2d178e6e113bd10f7b86016))
* **vhdl:** Fix import of Sequence type ([`2c463ac`](https://github.com/es-ude/elastic-ai.creator/commit/2c463acdbdae0ed7dc9fa99730f53db94deb7142))
* **pyproject:** Set correct version numbers and add protobuf dependency ([`260e5fb`](https://github.com/es-ude/elastic-ai.creator/commit/260e5fb31c425ad9ba2ec31f2fa292961fd28ffa))
* Specify exact python version in github workflow ([`f3ffb18`](https://github.com/es-ude/elastic-ai.creator/commit/f3ffb183e86b722cec5efb31e0937c4810542aef))
* Fix dependencies + onnx integration tests ([`f06d0f8`](https://github.com/es-ude/elastic-ai.creator/commit/f06d0f8436ca2a7ed3410aee4ad36df1cdad45c0))
* Resolve dependency version conflicts ([`32bd544`](https://github.com/es-ude/elastic-ai.creator/commit/32bd544b2e74b8b57497f3fd604deb5ed86ebb42))

## v0.7.0 (2022-06-05)
### Feature
* **vhdl:** Add rich comparison methods, multiple operators and a bit iterator to FixedPoint ([`116b006`](https://github.com/es-ude/elastic-ai.creator/commit/116b00647c05ef6854d3cbd1ab0f79c58f0c450d))
* **vhdl:** Start implementing FixedPoint datatype ([`8c4f420`](https://github.com/es-ude/elastic-ai.creator/commit/8c4f42097ff416f8e9056af430bda01a5bd42df5))

## v0.6.1 (2022-05-27)
### Fix
* **vhdl:** Saving generated examples to a directory instead of a giving an explicit file path ([`eb41d8d`](https://github.com/es-ude/elastic-ai.creator/commit/eb41d8db9af5171ac2826f41e98b5d85598b582d))

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
