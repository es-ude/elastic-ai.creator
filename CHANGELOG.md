# Changelog

<!--next-version-placeholder-->

## v0.17.0 (2022-10-22)
### Feature
* **examples:** Visualize model parameters ([`5e1b4fc`](https://github.com/es-ude/elastic-ai.creator/commit/5e1b4fc4c827c55d19cb9bc4206f706bcc737fba))

## v0.16.0 (2022-10-22)
### Feature
* **vhdl:** Implement qat for linear layer ([`d3ba49e`](https://github.com/es-ude/elastic-ai.creator/commit/d3ba49e266b2931c1b16677dd91f17a75f091501))
* **example:** Add the ability to plot the model parameters ([`b1b0b5e`](https://github.com/es-ude/elastic-ai.creator/commit/b1b0b5e7697992c4c53825c739e2fb2dcc903dac))
* **vhdl:** Move the input, weight and output quantization to the linear layer ([`0c8b259`](https://github.com/es-ude/elastic-ai.creator/commit/0c8b259ef688c606ebd4f8486ef7b6f48e0f8713))
* **examples:** Commit current state of the fixed point linear example ([`9b8ecae`](https://github.com/es-ude/elastic-ai.creator/commit/9b8ecae971bc1dedabf17e79272008a3cbfb5123))
* **vhdl:** Add feature to automatically derive fixed point parameters from a factory ([`70618d5`](https://github.com/es-ude/elastic-ai.creator/commit/70618d512718efd7e718491af52e1acbc6c86622))
* **examples:** Start implementing an example for learning a simple logic function ([`6cff6de`](https://github.com/es-ude/elastic-ai.creator/commit/6cff6deccd5c2080e930d93f5e145e4d7ea6a41e))
* **vhdl:** Move tracing example to example folder ([`b942155`](https://github.com/es-ude/elastic-ai.creator/commit/b942155a240a6f34f0f02361b6631b431a448443))
* **vhdl:** Make base linear package private ([`b3cfa55`](https://github.com/es-ude/elastic-ai.creator/commit/b3cfa55daff5c401bc036ffe2bba8b0c6b2f2554))
* **vhdl:** Add a type for fixed point factories ([`53b8499`](https://github.com/es-ude/elastic-ai.creator/commit/53b84991671832c2e7fa24e61d927b7c039832d9))
* **vhdl:** Implement custom linear layers that allows to do fixed point calculations ([`c2364f6`](https://github.com/es-ude/elastic-ai.creator/commit/c2364f6182bb8406e90a78d632bc868537705fd2))
* **vhdl:** Add function to get attribute names of an object matching a regex ([`acc8e29`](https://github.com/es-ude/elastic-ai.creator/commit/acc8e29e2771d5642e1371af6fb3c44f83b5ebc7))
* **vhdl:** Working further on the fixed point configuration finder ([`beb9da0`](https://github.com/es-ude/elastic-ai.creator/commit/beb9da0ec8c3fbc6bb4ff65a97e7424e4da6dd0d))
* **vhdl:** Start implementing fixed point evaluator ([`0f9c62a`](https://github.com/es-ude/elastic-ai.creator/commit/0f9c62a38f9df6ee4e84f1a3b5524df03511b438))
* **vhdl:** Implement signed fixed point integer to FixedPoint object ([`0a2fc79`](https://github.com/es-ude/elastic-ai.creator/commit/0a2fc7952dc13ea48c749856bf809a5540166598))
* **vhdl:** Implement automatic derivation of fixed point parameters in the lstm example ([`504008d`](https://github.com/es-ude/elastic-ai.creator/commit/504008d7ef3f402f8476bb77f02a4a37176d229e))

### Fix
* **vhdl:** Fix bug in the linear matix multiplication and rename _BaseLinear layer to _LinearBase ([`da11356`](https://github.com/es-ude/elastic-ai.creator/commit/da113561d69158ccc2a9266adb1eddcc79b1cb7d))

## v0.15.0 (2022-09-29)
### Feature
* **vhdl:** Implement clipped fixed point representation ([`8e53506`](https://github.com/es-ude/elastic-ai.creator/commit/8e53506fce0ba5adaa124ccd61de3b340bf1c95f))

## v0.14.0 (2022-09-28)
### Feature
* **vhdl:** Implement from_unsigned_int and from_signed_int function and remove unused function ([`aca77f5`](https://github.com/es-ude/elastic-ai.creator/commit/aca77f5eac396f21821b07706ff250b2589dd037))

### Fix
* **vhdl:** Reimplement unsigned_int_values_to_fixed_point function ([`cdd069e`](https://github.com/es-ude/elastic-ai.creator/commit/cdd069e6adffa882bb34fea2b7179891c282045b))

## v0.13.0 (2022-09-10)
### Feature
* **gh-workflow:** Explicitly set poetry version ([`82d202f`](https://github.com/es-ude/elastic-ai.creator/commit/82d202f0229e7931fc7371f69abe0d1fe3a58134))
* **vhdl:** Remove translatable protocol ([`ca52a92`](https://github.com/es-ude/elastic-ai.creator/commit/ca52a92d1bdc017773b872eaa5011b5117394472))
* **vhdl:** Remove translatable protocol ([`37412e8`](https://github.com/es-ude/elastic-ai.creator/commit/37412e87d89d16c9159cf12ef00032343119100c))

## v0.12.1 (2022-08-29)
### Fix
* **qat:** Reimplement binarize ([`9bbccdd`](https://github.com/es-ude/elastic-ai.creator/commit/9bbccddfc6ce6c2b928166cdfaf1112b294dba17))

## v0.12.0 (2022-08-26)
### Feature
* **vhdl:** Make work library name customizable ([`95fd8aa`](https://github.com/es-ude/elastic-ai.creator/commit/95fd8aa0d7e512aeb04893de2df2e58cc4b3e641))
* **vhdl:** Insert values in the updated lstm.vhd template ([`4d9dccb`](https://github.com/es-ude/elastic-ai.creator/commit/4d9dccbdb11afebb466f476c22539828bf5458b1))
* **examples:** Add linear layer to the translation example ([`5f1e1db`](https://github.com/es-ude/elastic-ai.creator/commit/5f1e1db8da7ce533cb592d56ca97e25ca563a60e))
* **vhdl:** Add an easy way to get a fixed point factory ([`d98ff03`](https://github.com/es-ude/elastic-ai.creator/commit/d98ff0351f739859ed668a2ec295421e29fd24ec))
* **vhdl:** Implement the translation of a linear1d layer ([`b627e78`](https://github.com/es-ude/elastic-ai.creator/commit/b627e780d054adcdd89009d87aa33fa31c913504))

### Fix
* **vhdl:** Remove some comments ([`13cc1a1`](https://github.com/es-ude/elastic-ai.creator/commit/13cc1a1ade14ccc7aa686523270dec20936ed14d))
* **vhdl:** Remove unused work library ([`c68fd9d`](https://github.com/es-ude/elastic-ai.creator/commit/c68fd9d00b152c5bdb70d2d2c90ca8d3e9f381d0))
* **vhdl:** Fix test ([`c125bf1`](https://github.com/es-ude/elastic-ai.creator/commit/c125bf16297ee9e39660ee904ab54268e8901d48))
* **vhdl:** Add changes from Chao after testing the translator ([`5a5d532`](https://github.com/es-ude/elastic-ai.creator/commit/5a5d5325a3f598e0163d4eac0601b5961c2f5780))
* **vhdl:** Pre-add input-hidden and hidden-hidden bias ([`750941c`](https://github.com/es-ude/elastic-ai.creator/commit/750941c3150cabefa2f393f6b12105a358a70f7f))
* **vhdl:** Fix calculation of the addr_width of the linear1d layer ([`6fa2b2a`](https://github.com/es-ude/elastic-ai.creator/commit/6fa2b2a3bc83d3a51eb955d1464501662f6676a8))

### Documentation
* **vhdl:** Adapt diagrams to the latest changes ([`c1750eb`](https://github.com/es-ude/elastic-ai.creator/commit/c1750eb19f92a705f8f36ccefc9729d3545f0743))
* **readme:** Move translator documentation to the vhdl package ([`9a90949`](https://github.com/es-ude/elastic-ai.creator/commit/9a90949528978ff4732f585986a71cedd44e82a5))
* **readme:** Small changes of the documentation ([`9e7699c`](https://github.com/es-ude/elastic-ai.creator/commit/9e7699ce617581f67f85cf4ef7d945d99df241be))
* **readme:** Update documentation according the newly added linear1d layer ([`41e2486`](https://github.com/es-ude/elastic-ai.creator/commit/41e24868aecbf310ee4c9ad815f6ccc0da3f9f9b))
* **readme:** Add documentation on how the translator works ([`91ebea3`](https://github.com/es-ude/elastic-ai.creator/commit/91ebea3fb7e7883f56b2cd9152769d151449a49a))

## v0.11.1 (2022-08-18)
### Fix
* **qat:** Remove deprecated threshold and codomain properties ([`5db9669`](https://github.com/es-ude/elastic-ai.creator/commit/5db9669fc3942851e65607a869bb822430df7836))

## v0.11.0 (2022-08-11)
### Feature
* **vhdl:** Adapt the example to the changes of the translation ([`6a5644e`](https://github.com/es-ude/elastic-ai.creator/commit/6a5644e30a7cd00ed1be1c2cb6fa2e0b4b114c1e))
* **vhdl:** Change translation from LSTMCell to LSTM ([`5e4f1cf`](https://github.com/es-ude/elastic-ai.creator/commit/5e4f1cff380fabd3685660a0c279b9098c4ef278))
* **vhdl:** Removed the possibility to get a build function from a type ([`dbc2e8f`](https://github.com/es-ude/elastic-ai.creator/commit/dbc2e8ffd95f5ddc2476fede9d170c9d4eb020c2))
* **vhdl:** Make build function mapping more general so that it can be reused for other frameworks ([`3369d7f`](https://github.com/es-ude/elastic-ai.creator/commit/3369d7fb6a7d08930514a7c0553c9efe65fc54b9))
* **vhdl:** Change build function mapping to a different approach ([`b1b79b2`](https://github.com/es-ude/elastic-ai.creator/commit/b1b79b2e5e9ea0cf627b16a41f1f75bf434b795e))
* **vhdl:** Add LSTMCellTranslationArguments to __init__.py file ([`061ead4`](https://github.com/es-ude/elastic-ai.creator/commit/061ead404dc82ddc79ac75c155328ad5733eb04a))
* **vhdl:** Pass an DTO to a translatable instead of raw arguments to fix typing errors ([`2c33869`](https://github.com/es-ude/elastic-ai.creator/commit/2c33869cce5bed725a90ea3a4980bc026aec1ac4))
* **vhdl:** Pass an DTO to a translatable instead of raw arguments to fix typing errors ([`4738725`](https://github.com/es-ude/elastic-ai.creator/commit/4738725d09ca9114064c4c42dd2818fc6d5c973b))
* **vhdl:** Implement a more functional build function mapping ([`1425e03`](https://github.com/es-ude/elastic-ai.creator/commit/1425e0304cf35617106199936d3b014c0d8ca483))
* **examples:** Add an example using the vhdl translator for pytorch ([`395adcd`](https://github.com/es-ude/elastic-ai.creator/commit/395adcd3e843b7f55f6156ba183dc8800055ef51))
* **vhdl:** Add the ability to infer the build function from a given layer object or type ([`306df14`](https://github.com/es-ude/elastic-ai.creator/commit/306df1427177d15c1b1e2c59b2e774a2a6e2c471))
* **vhdl:** First untested draft for the pytorch translator ([`7e59462`](https://github.com/es-ude/elastic-ai.creator/commit/7e5946259381af397e1ccd25006815af8256026f))
* **vhdl:** Implementation of the mapping of a torch module to the corresponding build function ([`b076fa3`](https://github.com/es-ude/elastic-ai.creator/commit/b076fa32cef3c64f8fcc45df24814f4333c90b5c))
* **vhdl:** Use __init__ files to simplify the usage ([`3cc07ee`](https://github.com/es-ude/elastic-ai.creator/commit/3cc07ee048a349ef5a6a5383dcd829d64b48de2d))
* **vhdl:** Add a build function to create an abstract LSTMCell object from a PyTorch LSTMCell ([`baca5bb`](https://github.com/es-ude/elastic-ai.creator/commit/baca5bb6c22692cf9bfc02a9147711b8869930fd))
* **vhdl:** Abstract LSTM cell takes float weights instead of FixedPoint weights ([`a5818cc`](https://github.com/es-ude/elastic-ai.creator/commit/a5818cc0edd918ef3ca49e843738823e988bfd79))
* **vhdl:** Introduce translation arguments ([`2c3a8c7`](https://github.com/es-ude/elastic-ai.creator/commit/2c3a8c72cfe8df70fd960e692d4fe037e2e86b6f))
* **vhdl:** Add a protocol specify a translatable layer ([`0fa966e`](https://github.com/es-ude/elastic-ai.creator/commit/0fa966e7f99ef2adb19321b3ca92202616b4c0a2))
* **vhdl:** Add ability to pass kwargs to the translate function of a translatable layer ([`196812e`](https://github.com/es-ude/elastic-ai.creator/commit/196812eecd0dc49a1b8c2d6675b9018ca07e003e))
* **vhdl:** Implementation of a LSTMCell class that can be translated to VHDL ([`ace37fe`](https://github.com/es-ude/elastic-ai.creator/commit/ace37fe4b215327bc5b43344ffcd0c44a4822dda))

### Fix
* **vhdl:** Remove print call ([`55164b7`](https://github.com/es-ude/elastic-ai.creator/commit/55164b78c61f37f4cdadde0385965ee540e4f555))
* **vhdl:** Rename LSTMCell translatable to LSTM ([`e05cd04`](https://github.com/es-ude/elastic-ai.creator/commit/e05cd042daf0420b2046607e00eeef3606a6defb))
* **examples:** Use LSTMTranslationArguments object instead of a dictionary ([`98a4d97`](https://github.com/es-ude/elastic-ai.creator/commit/98a4d97f8fbd217f67ed4009ab63ccc4705f720d))
* **vhdl:** Fix mypy typing errors ([`e1dba31`](https://github.com/es-ude/elastic-ai.creator/commit/e1dba317585c269ad58719184fb4764cc66485ae))
* **vhdl:** Fix wrong pytorch lstm cell class path ([`85a733c`](https://github.com/es-ude/elastic-ai.creator/commit/85a733cb5ff821bb602b5021f6438b7d5909382e))
* **vhdl:** Fix test ([`528910c`](https://github.com/es-ude/elastic-ai.creator/commit/528910cf3fe28958ebb7b246104e83df77bbf3f4))

### Documentation
* **vhdl:** Add some docstrings to the functions of the translator ([`6f9215e`](https://github.com/es-ude/elastic-ai.creator/commit/6f9215e5fc35287517d884a702bf887d7a09aa7f))
* **readme:** Fix commands of install dev dependencies ([`870e2de`](https://github.com/es-ude/elastic-ai.creator/commit/870e2de30f48223d8005bcf1240b624ebb314ad7))

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
