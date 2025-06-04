## 0.65.0 - 2025-06-04

### Chore

- Use git cliff instead of cog for changelogs ([83270cf]( https://github.com/es-ude/elastic-ai.creator/commit/83270cfacb9a5b27f059602f25fe05c7bb6d2a2f)) - [object]

### Docs

- Add explanation for skeleton generics/padding and example ([1c029b2]( https://github.com/es-ude/elastic-ai.creator/commit/1c029b26462fc13174f5dc012a936501c84a8743)) - [object]

### Feat

- Let users construct replacements based on match ([fd46b31]( https://github.com/es-ude/elastic-ai.creator/commit/fd46b315c6f5805d8e67a4901b90a1885c951c94)) - [object], BREAKING CHANGE:Users have to specify a constructor for
rewrite rules now

### Fix

- Import Template from string instead of ir2vhdl ([749e0d7]( https://github.com/es-ude/elastic-ai.creator/commit/749e0d7959b664a1a5c72677b1c10d577b7766b6)) - [object]
- Make lowering pass return correct type for static files ([203f10d]( https://github.com/es-ude/elastic-ai.creator/commit/203f10d5cbd7fb24ad4139723e71e999bdd70a42)) - [object]
- Improve error message when initializer is not found ([de6a809]( https://github.com/es-ude/elastic-ai.creator/commit/de6a8092b4119f1ad65202483c9ed564a93f820c)) - [object]

## 0.64.1 - 2025-05-26

[b675d80](b675d809fe9f2c51a64e0a71859d9011c7edde04)...[87f2fee](87f2fee7ee6f3f971cebca1c679ba35f46f44cc2)

### Chore

- Improve error message for missing fields in meta.toml ([fabf3bb]( https://github.com/es-ude/elastic-ai.creator/commit/fabf3bb0e5d70ce1eac76c493ffd43f3f1b5cd82)) - [object]
- V0.64.0 ([87f2fee]( https://github.com/es-ude/elastic-ai.creator/commit/87f2fee7ee6f3f971cebca1c679ba35f46f44cc2)) - [object]

### Docs

- Name plugin translation stages ([0d497f4]( https://github.com/es-ude/elastic-ai.creator/commit/0d497f402c5c0e75fa28bf343dec37922c883114)) - [object]
- Clarify purpose of the plugin system ([11db978]( https://github.com/es-ude/elastic-ai.creator/commit/11db978d217621257360cc3afc2d5d4f08095211)) - [object]
- Remove outdated docstring ([6ab932b]( https://github.com/es-ude/elastic-ai.creator/commit/6ab932b2f844d96672ebdff2444d6eaf7ecff4f8)) - [object]

### Feat

- Support nested module hierarchies ([23a4323]( https://github.com/es-ude/elastic-ai.creator/commit/23a43230dd4ab43e3e6f90fbc1a999bea23cbeb6)) - [object]
- Add function to sync implementation data with underlying graph ([56c198f]( https://github.com/es-ude/elastic-ai.creator/commit/56c198f4650d33de98f0eccf2c1661c49f12e190)) - [object]
- Introduce RewriteRules and rewriting for Implementations ([39baf11]( https://github.com/es-ude/elastic-ai.creator/commit/39baf11d6349466cb2265c04d456dfabec7325e9)) - [object]

### Fix

- Stop importing non-existent Template class in skeleton ([a0f7663]( https://github.com/es-ude/elastic-ai.creator/commit/a0f766355fd485bb50660431e368ea8ddc3d8d9e)) - [object]
- Fix bug that was preventing matches in some edge cases ([2c31179]( https://github.com/es-ude/elastic-ai.creator/commit/2c311799b7e8790b047228daaf204f53c5ef03c6)) - [object]

### Refactor

- Publicly expose NodeConstraintFn protocol ([9934fd5]( https://github.com/es-ude/elastic-ai.creator/commit/9934fd594ae2d243d495f0f4112b02898d8544ba)) - [object]
- Add a few missing type hints ([47344be]( https://github.com/es-ude/elastic-ai.creator/commit/47344be1c1353367d25f32ef1da7957085489d0c)) - [object]

### Test

- Add test changing node attribute ([079c8a1]( https://github.com/es-ude/elastic-ai.creator/commit/079c8a10395b7e901782d689f3947213f42d2bb7)) - [object]

## 0.63.0 - 2025-05-08

[3939e67](3939e676c60e2f070218e60601e156a038fafbbb)...[b675d80](b675d809fe9f2c51a64e0a71859d9011c7edde04)

### Chore

- Bump version v0.63.0 ([b675d80]( https://github.com/es-ude/elastic-ai.creator/commit/b675d809fe9f2c51a64e0a71859d9011c7edde04)) - [object]

### Docs

- List recent features ([ca818dd]( https://github.com/es-ude/elastic-ai.creator/commit/ca818dda9bec439f6a20bc51e9cdba1455996a2f)) - [object]
- Introduce upstream/downstream terminology ([d540be7]( https://github.com/es-ude/elastic-ai.creator/commit/d540be7590b159e300a36373a7e8c8bdf1ebbd6f)) - [object]

### Feat

- Add name of instantiated module as template parameter ([958eb26]( https://github.com/es-ude/elastic-ai.creator/commit/958eb26abc6fbc990ed03b528926e5970225df1d)) - [object]
- Find all non-overlapping subgraphs ([bdd01a6]( https://github.com/es-ude/elastic-ai.creator/commit/bdd01a64e18ce9018e01b1c41a7bba3949ea9985)) - [object]
- Decouple core data types from identifiers in graphs ([dde4558]( https://github.com/es-ude/elastic-ai.creator/commit/dde45585d2e1ecef6b3d99983b08bf65dd0938ac)) - [object], BREAKING CHANGE:will break almost every client
that has been using the serialized ir and
Node/Edge constructors. Most node/edge
constructors now need explicit
name/src, dst arguments respectively.
The serialized IR now uses dicts
instead of lists for edges and nodes.
Edge/Node data does not include name/(src,dst)
anymore.
- Find all subgraph matches ([916d54c]( https://github.com/es-ude/elastic-ai.creator/commit/916d54ce203d49e118b47586a9fcabd23ed5a925)) - [object]
- Filter subgraph matches for rewrite ([573f514]( https://github.com/es-ude/elastic-ai.creator/commit/573f5148469ab97da4cb0bfd390943e0e9bb400e)) - [object]
- Extract patterns into sub implementations ([3eb29c9]( https://github.com/es-ude/elastic-ai.creator/commit/3eb29c958ab15798442ecd8e52cbc15166bcb105)) - [object]

### Fix

- Update verilog templates to correctly replace arrays ([c639e8d]( https://github.com/es-ude/elastic-ai.creator/commit/c639e8daa4eb50112cef37a94d04d6ca3651697d)) - [object]
- Make subgraph matching independent of node iteration order ([ec45d11]( https://github.com/es-ude/elastic-ai.creator/commit/ec45d1102686f6236235bfc13d78971afcd34caf)) - [object]

### Refactor

- Rename sink to destination/dst ([5b40309]( https://github.com/es-ude/elastic-ai.creator/commit/5b40309b37c623bad5815543950bd7a333301629)) - [object], BREAKING CHANGE:every client that was using
`sink` as a keyword param or attribute
- Rename dfs_preorder to dfs_iter ([96bcffd]( https://github.com/es-ude/elastic-ai.creator/commit/96bcffd2a0336b6e47b5f860c03334efa58e64ff)) - [object]
- Only require a single match in rewriter ([e57b400]( https://github.com/es-ude/elastic-ai.creator/commit/e57b4005d942554c30d43cbf0fdc19c2bb6afb2a)) - [object]
- Add rewrite function ([5dc4816]( https://github.com/es-ude/elastic-ai.creator/commit/5dc48169d31ac98d1d1123b3300d95570250074e)) - [object]
- Remove interface arg from rewrite function ([db1e312]( https://github.com/es-ude/elastic-ai.creator/commit/db1e3128108cb3082d3705e95b2660a7b5119e99)) - [object]
- Use simpler rewrite function instead of GraphRewriter ([f0540cd]( https://github.com/es-ude/elastic-ai.creator/commit/f0540cdc8dbf15a05095f0c5a7a31eeab896db8d)) - [object]

## 0.62.0 - 2025-02-26

[0b5a266](0b5a2667c4fa89bb3e56e4ba16a760a0d782eabc)...[3939e67](3939e676c60e2f070218e60601e156a038fafbbb)

### Chore

- V0.62.0 ([3939e67]( https://github.com/es-ude/elastic-ai.creator/commit/3939e676c60e2f070218e60601e156a038fafbbb)) - [object]

### Ci

- Update deprecated upload gh-action ([f81c05a]( https://github.com/es-ude/elastic-ai.creator/commit/f81c05aca6ae9c6e034c38cfb0010392b191562c)) - [object]

### Docs

- Fix links in docstrings ([fc7fc79]( https://github.com/es-ude/elastic-ai.creator/commit/fc7fc7902e7f8ffd7d294da72bdadf082b9e102a)) - [object]

### Feat

- Add lutron implementation ([f04aa63]( https://github.com/es-ude/elastic-ai.creator/commit/f04aa63bc6cc9d6888e0c988263a7fe706f37e81)) - [object]
- Add combinatorial components ([f3f6745]( https://github.com/es-ude/elastic-ai.creator/commit/f3f67458b4c7b073b2540d73b9966a31bd1bbb72)) - [object]
- Add grouped filter plugin ([5794938]( https://github.com/es-ude/elastic-ai.creator/commit/5794938ad4dd0f5e97cbda16c88d3b03b47620fd)) - [object]
- Add time multiplexed sequential plugin ([080af51]( https://github.com/es-ude/elastic-ai.creator/commit/080af5109b1be09130015952fc46a417aab12db3)) - [object]
- Provide replaced and resulting subgraph from rewriting ([1b9aab8]( https://github.com/es-ude/elastic-ai.creator/commit/1b9aab8349657eca5e313fe14c014109e669753e)) - [object]
- Support reordering sequential subgraphs ([b356307]( https://github.com/es-ude/elastic-ai.creator/commit/b356307121f4a964de60cea99c05f2ef66077be6)) - [object]
- Add support verilog translation ([f40439b]( https://github.com/es-ude/elastic-ai.creator/commit/f40439bf56d5b7cb9e1e67792b11126defbdc6e7)) - [object]

### Fix

- Fix __eq__ and __repr__ for IrData ([a18a207]( https://github.com/es-ude/elastic-ai.creator/commit/a18a207852c71a8ca205b6f64ab6184b0947e131)) - [object]
- Return str instead of str.Template from builder ([dfd9d8e]( https://github.com/es-ude/elastic-ai.creator/commit/dfd9d8e7c8b73691a84f747fb81210248c10f4cf)) - [object]

### Refactor

- Allow strings to specify src_sink_indices ([5868dcb]( https://github.com/es-ude/elastic-ai.creator/commit/5868dcb55c77136bb02128e408e009aaab0f2e6d)) - [object]
- Turn Graph into Implementation ([065aece]( https://github.com/es-ude/elastic-ai.creator/commit/065aece5e6f0730085b45395f3da4b015f332834)) - [object], BREAKING CHANGE:breaks every client that was previously
importing ir.Graph directly
- Separate graph from ir ([3c09f1f]( https://github.com/es-ude/elastic-ai.creator/commit/3c09f1fa29bf7075124eedb858e4e414391fc93e)) - [object]
- Use mappings instead of functions for lhs/rhs ([99fc7ec]( https://github.com/es-ude/elastic-ai.creator/commit/99fc7ec7382f2be085b658b191cca717d6465dc1)) - [object]
- Make pattern/graph args kw only ([7a7ea1e]( https://github.com/es-ude/elastic-ai.creator/commit/7a7ea1e4c4b1ff1ba5a9d5292e8afa58e55ced3d)) - [object]
- Automatically pass node/edge_fn to impl ([2817325]( https://github.com/es-ude/elastic-ai.creator/commit/28173258a13185a921496d81356f0889bc06ed56)) - [object]
- Simplify template API ([be6de71]( https://github.com/es-ude/elastic-ai.creator/commit/be6de71516b94b3684275cce29d706c74152148e)) - [object], BREAKING CHANGE:impacts only clients that were
   using the template parameter classes directly

## 0.61.0 - 2025-02-21

[ffc8529](ffc85292dceeafe56e752fc522d6e732fd7d57c2)...[0b5a266](0b5a2667c4fa89bb3e56e4ba16a760a0d782eabc)

### Build

- Replace poetry by uv ([47d9538]( https://github.com/es-ude/elastic-ai.creator/commit/47d9538782766f33ae553e2425a6fe21956dd4cb)) - [object]
- Add uv.lock and set .python-version to 3.10 ([533add1]( https://github.com/es-ude/elastic-ai.creator/commit/533add17585846ea7f6c49ce0c954df24c0c88a5)) - [object]
- Update ruff version and uv.lock file ([9229e30]( https://github.com/es-ude/elastic-ai.creator/commit/9229e301bb48e50e2445dd0790e30857ea1b68c8)) - [object]
- Add plugins namespace to pytest discovery ([fbf5c07]( https://github.com/es-ude/elastic-ai.creator/commit/fbf5c07505df674f8d5077b0cac021595376509e)) - [object]
- Autoupdate uv.lock ([2a59a5c]( https://github.com/es-ude/elastic-ai.creator/commit/2a59a5c69ab026311f137ea2f30c7a59399830d8)) - [object]
- Use git to dynamically extract version ([223a13e]( https://github.com/es-ude/elastic-ai.creator/commit/223a13e8014cf320f90386453be9daa29d00e22b)) - [object]
- Add kroki plugin to antora to build diagrams ([380ab02]( https://github.com/es-ude/elastic-ai.creator/commit/380ab021458c5827edddec5d8e4e2a18014279e1)) - [object]
- Use cog instead of cliff for changelogs ([eabf4e1]( https://github.com/es-ude/elastic-ai.creator/commit/eabf4e1b57657e4e0131d676be606d67d8618487)) - [object]
- Fix includes for hatch ([58c8f8e]( https://github.com/es-ude/elastic-ai.creator/commit/58c8f8e11e5772d89895ddf67314c5dd21acb4bd)) - [object]
- Drop support for py 3.10 and move to 3.11 ([284d476]( https://github.com/es-ude/elastic-ai.creator/commit/284d476aa66102aedd0ee839a70cc414f2f53048)) - [object], BREAKING CHANGE:every dependant that uses python
3.10
- Add VUnit to testing dependencies ([8bf1572]( https://github.com/es-ude/elastic-ai.creator/commit/8bf15720342826dfe4e5013a6792a562a2b191f8)) - [object]
- Add hypothesis for testing ([a0f5eba]( https://github.com/es-ude/elastic-ai.creator/commit/a0f5eba394177e8a622436446d91fd533f37fa2f)) - [object]

### Chore

- Add basic support for devenv ([7df56a1]( https://github.com/es-ude/elastic-ai.creator/commit/7df56a17428833ecdf63cc3f62509c04e76c557a)) - [object]
- Add bash scripts to help creating new plugins ([a1b7f4a]( https://github.com/es-ude/elastic-ai.creator/commit/a1b7f4ad618654d1b9d82bf5ae2e2fd90d1d6d5d)) - [object]
- Use importlib mode for pytest to avoid errors about missing __init__ files in test folders ([687f0c3]( https://github.com/es-ude/elastic-ai.creator/commit/687f0c3e896c3388b9babb4a86162140aef904b8)) - [object]
- Lint unorganized imports ([f31196a]( https://github.com/es-ude/elastic-ai.creator/commit/f31196aa42bda753f49d35b42744ca1cc39b4085)) - [object]
- Rename FunctionDecoratorFactory to FunctionDecorator ([85464cf]( https://github.com/es-ude/elastic-ai.creator/commit/85464cfcbef68d8fb588e5b23c6577685ccf4ffe)) - [object]
- Support vivado and add options to ghdl module ([0ddb41f]( https://github.com/es-ude/elastic-ai.creator/commit/0ddb41fa88c4a3932d0f4e613eb39c72bc0023fe)) - [object]
- Add jj/git/pikchr ([9c8ebdc]( https://github.com/es-ude/elastic-ai.creator/commit/9c8ebdc5d1711feefc1fd991ab59649bf97628d1)) - [object]
- Remove runner and utils ([fa268cf]( https://github.com/es-ude/elastic-ai.creator/commit/fa268cf1b1dce2b4e822985c4fdcf08ec6b7b2bc)) - [object]
- Add devenv tasks to run all checks in parallel ([cc17635]( https://github.com/es-ude/elastic-ai.creator/commit/cc1763557d02747b2a87979759a44bc1ade9396c)) - [object]
- Update devenv.lock ([640a01e]( https://github.com/es-ude/elastic-ai.creator/commit/640a01e2308ccc271a6c05f7aecb69c14c23c08b)) - [object]
- Include tests in ruff linting ([9e3e38a]( https://github.com/es-ude/elastic-ai.creator/commit/9e3e38a8b7a04775a0d56f87797ff65c72a2841e)) - [object]
- Remove statements to run py311 to generate docs ([0841dc3]( https://github.com/es-ude/elastic-ai.creator/commit/0841dc3b32f044892da781c72bd73b5ac4ca38d6)) - [object]
- Use tach to adhere to architecture ([9c845ae]( https://github.com/es-ude/elastic-ai.creator/commit/9c845aebb8eda4cc84943157a8e32f8010f313c4)) - [object]
- Replace antora/asciidoc by sphinx/markdown ([39b38d4]( https://github.com/es-ude/elastic-ai.creator/commit/39b38d491a7b4e44dfab9cdefdb16afeafe0b59c)) - [object]
- Move docs source do `docs/` and adjust some docstrings ([1325b12]( https://github.com/es-ude/elastic-ai.creator/commit/1325b12e64b9e1f80027002912f289ea3a40554c)) - [object]
- Add new 'slow' marker ([4da2334]( https://github.com/es-ude/elastic-ai.creator/commit/4da23341706824bc557e3eb0b31e4bec5b901678)) - [object]
- Add issue and pr templates ([f9fe969]( https://github.com/es-ude/elastic-ai.creator/commit/f9fe969f09dfe0a25aaaa7c00ee74c5c7f025143)) - [object]
- Print class name in instead of class in IrData repr ([168f45c]( https://github.com/es-ude/elastic-ai.creator/commit/168f45c46e911963371bdd8afe928bd7469c35d2)) - [object]
- Configure mypy ([32cd72e]( https://github.com/es-ude/elastic-ai.creator/commit/32cd72e8e85464c0786cb9b287b4208d7793d939)) - [object]
- Add wavedrom extension for sphinx ([4e3514d]( https://github.com/es-ude/elastic-ai.creator/commit/4e3514d0e0866e4608a2a7c4eabc80af34ef61b1)) - [object]
- Change docs theme to pydata ([9f23518]( https://github.com/es-ude/elastic-ai.creator/commit/9f235182a1c94e3d5ea3f7fc56e96261dd6bb7dd)) - [object]
- Remove and ignore autogenerated files ([4a29ee9]( https://github.com/es-ude/elastic-ai.creator/commit/4a29ee969bbfe8d8098b390777721363283deecf)) - [object]
- Fine tune change log template ([04cf4fe]( https://github.com/es-ude/elastic-ai.creator/commit/04cf4fe294e3ba1862ed85006fad1eb659a41686)) - [object]
- V0.61.0 ([0b5a266]( https://github.com/es-ude/elastic-ai.creator/commit/0b5a2667c4fa89bb3e56e4ba16a760a0d782eabc)) - [object]

### Ci

- Correct name in workflow uploading to github ([22a7867]( https://github.com/es-ude/elastic-ai.creator/commit/22a7867513f16e32ce508c20a19ef4f4da062777)) - [object]
- Remove beautify pipeline step ([e2c9faf]( https://github.com/es-ude/elastic-ai.creator/commit/e2c9fafdd9df712bab2c2d76d05dc5e0afa5b5a2)) - [object]
- Fix pypi publish gh-action version ([0e18cb8]( https://github.com/es-ude/elastic-ai.creator/commit/0e18cb86b91435a7454d04754beb61b3cafe9ca5)) - [object]
- Build docs ([fe0b992]( https://github.com/es-ude/elastic-ai.creator/commit/fe0b99291f692350f4283465ee4902dcc15454ac)) - [object]
- Run ghdl testbenches for plugins ([6b6a046]( https://github.com/es-ude/elastic-ai.creator/commit/6b6a0465ca82a02f23d34a872d73b09fefa55310)) - [object]
- Use devenv to run all tests in ci ([8220954]( https://github.com/es-ude/elastic-ai.creator/commit/82209547897c5572f553cb8ecb4fbefa42673f6b)) - [object]
- Remove broken task dependencies ([111ab50]( https://github.com/es-ude/elastic-ai.creator/commit/111ab5019a45855bef7e222c57d8de017b27fa6b)) - [object]
- Check types with mypy ([df71257]( https://github.com/es-ude/elastic-ai.creator/commit/df712574487f4d06f25566545756a78a279413ac)) - [object]
- Use devenv from unstable channel v1.4 ([d7ace47]( https://github.com/es-ude/elastic-ai.creator/commit/d7ace47e70f0f85209da63c7e202fca67b05a562)) - [object]
- Deploy self-contained html to PR for preview ([03eb5ba]( https://github.com/es-ude/elastic-ai.creator/commit/03eb5ba00220aabadef46c8c165ee3c69bf223f4)) - [object]

### Docs

- Explain commit/pr policy in contribution guide ([e04630c]( https://github.com/es-ude/elastic-ai.creator/commit/e04630cfab084cb4d8e2ec40a9aa863cbe6a5bdb)) - [object]
- Move dev docs from readme into contribution.md ([3a56226]( https://github.com/es-ude/elastic-ai.creator/commit/3a5622604b07b21a9665d271013529e6366b655b)) - [object]
- Fix wrong package name in install command ([22426ab]( https://github.com/es-ude/elastic-ai.creator/commit/22426aba973377b89388da9e0d0c171b03eb1fb0)) - [object]
- Explain how to annotate fields ([4176c58]( https://github.com/es-ude/elastic-ai.creator/commit/4176c58985a954e5bbc9c63d6203d2cd039ba428)) - [object]
- Improve formatting ([1c81917]( https://github.com/es-ude/elastic-ai.creator/commit/1c81917bea6eb7870edcc80f229769cb252bd6e8)) - [object]
- Improve wording in ghdl test-runner docs ([868a51f]( https://github.com/es-ude/elastic-ai.creator/commit/868a51f84ea7ee6edd60c055fe93c799366af7bf)) - [object]
- Fix asciidoc refs and formatting in docstrings ([b6ff358]( https://github.com/es-ude/elastic-ai.creator/commit/b6ff358d282acec8474128483e011d305a1f0a20)) - [object]
- Clarified installation via PyPi ([037d0e2]( https://github.com/es-ude/elastic-ai.creator/commit/037d0e2be285695e3aa31719a1656314ca74a375)) - [object]
- Do not build api docs for plugins ([7de975d]( https://github.com/es-ude/elastic-ai.creator/commit/7de975d216d81f1e872ec20151649172a50f52be)) - [object]
- Fix missing table end ([6ec2343]( https://github.com/es-ude/elastic-ai.creator/commit/6ec2343f13c8e49e736030737cb5055f6f6b8e30)) - [object]
- Extend ir2vhdl documentaion ([c5eafde]( https://github.com/es-ude/elastic-ai.creator/commit/c5eafde83fb72ceac0e0a07517723d0a717906ff)) - [object]
- Explain the core concepts of ir ([1d54159]( https://github.com/es-ude/elastic-ai.creator/commit/1d54159db87af8b6815ac99ab45485c9eb9ce01f)) - [object]
- Add GitHub repository link to documentation header ([78d110d]( https://github.com/es-ude/elastic-ai.creator/commit/78d110d631048da5f79dcc3af22dd50eb064c855)) - [object]
- Add previously missing ir2vhdl.adoc file ([05606c7]( https://github.com/es-ude/elastic-ai.creator/commit/05606c7d8b449792ca832e146c21d2577c4021c2)) - [object]
- Fix incorrect class names in examples ([99968b8]( https://github.com/es-ude/elastic-ai.creator/commit/99968b885b1b3715c31ee96fc55f778f0725afe7)) - [object]
- Improve readability (hopefully) ([7be9894]( https://github.com/es-ude/elastic-ai.creator/commit/7be9894ecf563a6c5bc54725b8e0be81f2679d50)) - [object]
- Update autogenerated docs ([3ce97f4]( https://github.com/es-ude/elastic-ai.creator/commit/3ce97f4c1545b9bdfc26d8a0f5dd3b716b18cd8e)) - [object]

### Feat

- Discover/compile/run/log vhdl testbenches from plugins ([c485099]( https://github.com/es-ude/elastic-ai.creator/commit/c4850990de72533374f854c5617606470359751f)) - [object]
- Added sgd ([#429](https://github.com/es-ude/elastic-ai.creator/issues/429)) ([6d74b48]( https://github.com/es-ude/elastic-ai.creator/commit/6d74b486b5a00ffd09e308c4ed1710c2ac4c8828)) - [object]
- Added momentum and weight decay ([44a0732]( https://github.com/es-ude/elastic-ai.creator/commit/44a073273cc10644c4465b7dd0896b19187da32b)) - [object]
- Improve error handling when loading plugins ([924d63c]( https://github.com/es-ude/elastic-ai.creator/commit/924d63c15f7cb5e1f186ad1ed2448c83f0315dac)) - [object]
- Improve typing and allow to pass pathlib.Path as working_dir ([79c6115]( https://github.com/es-ude/elastic-ai.creator/commit/79c611512de072b2768b40b1088692fc4ee27b89)) - [object]
- Make the plugin system more intuitive ([ac13c15]( https://github.com/es-ude/elastic-ai.creator/commit/ac13c15e8d42d92d23a74f0484030f79a64128ed)) - [object]
- Add FilterParameters and Shape ([09443f4]( https://github.com/es-ude/elastic-ai.creator/commit/09443f495460b86764299daf8a20fd567d6df2f3)) - [object]
- Adds support for ir2vhdl plugins ([3239641]( https://github.com/es-ude/elastic-ai.creator/commit/3239641f87ab109908be0be24c0b5f0922b492d6)) - [object]
- Add support for execution on specific devices ([2964e62]( https://github.com/es-ude/elastic-ai.creator/commit/2964e62d7a145c62c19fa9c6ad2f2f20c819aadc)) - [object]
- Extend signature of FunctionDecorator ([b972cf4]( https://github.com/es-ude/elastic-ai.creator/commit/b972cf4923f05f2ee8f091df3a3de85e9acf7313)) - [object]
- Make plugin loader more flexible ([aae96e6]( https://github.com/es-ude/elastic-ai.creator/commit/aae96e6392aae03cc6efc832057cf72bb9b6cbe5)) - [object]
- Create static files from plugins ([ee23798]( https://github.com/es-ude/elastic-ai.creator/commit/ee23798ca8376983c877927203921297a874ab1e)) - [object]
- Add middleware ([1c7b82b]( https://github.com/es-ude/elastic-ai.creator/commit/1c7b82b8f66f6e3ad214db165687543c20e3e535)) - [object]
- Added parametrization and module quantization ([4e551e9]( https://github.com/es-ude/elastic-ai.creator/commit/4e551e958abef5a17f74807cb4f8204b39daa6cd)) - [object]
- Turn ir.Graph into a IrData object ([8d44f1b]( https://github.com/es-ude/elastic-ai.creator/commit/8d44f1b9ab1fb18212e03b6980fc44af1205cd49)) - [object], BREAKING CHANGE:Might break clients that were using
`asdict` `fromdict` methods of `ir2vhdl.Implementation`
previously. Clients using the old version of the
Graph constructor will receive a deprecation warning.
- Add HwFunctionIdUpdater ([09aef74]( https://github.com/es-ude/elastic-ai.creator/commit/09aef74eb6bdef899e1ad323500f42119800de5c)) - [object]
- Add basic test runner for vunit testbenches ([b6b48ee]( https://github.com/es-ude/elastic-ai.creator/commit/b6b48ee103436062c36edbf2731aca55286ae691)) - [object]
- Add counter ([7e04f9a]( https://github.com/es-ude/elastic-ai.creator/commit/7e04f9a95447e101993bf3f80d02f26f30d05706)) - [object]
- Add padding ([b71a83a]( https://github.com/es-ude/elastic-ai.creator/commit/b71a83ad09de36095a025eb075993b01e05445b4)) - [object]
- Support simple torch 2 ir conversion w/o model state ([e06e028]( https://github.com/es-ude/elastic-ai.creator/commit/e06e02893ee6541204f090319dd325106ab55e1b)) - [object]
- Add Ir2Torch and support linear+relu ([df47f74]( https://github.com/es-ude/elastic-ai.creator/commit/df47f7471ac3f708b04d2928237e0e1d4c3a3aa9)) - [object]
- Support overriding registered type handlers ([fec5313]( https://github.com/es-ude/elastic-ai.creator/commit/fec5313bb155627a26a2237b13b2ab62fd33a119)) - [object]
- Make ir2torch a lowering pass ([74942a4]( https://github.com/es-ude/elastic-ai.creator/commit/74942a40d862ef6cb5a11614cc846822409675f3)) - [object], BREAKING CHANGE:impacts all code that has been
using the ir2torch module
- Implement __eq__ for nodes/edges attributes of Graphs ([47e37dd]( https://github.com/es-ude/elastic-ai.creator/commit/47e37ddf46f2e477a84feb2c16a751870f679492)) - [object]
- Support decorating methods as required fields ([569d0b2]( https://github.com/es-ude/elastic-ai.creator/commit/569d0b2a7a90f41427bb59a49615cd48d57a7603)) - [object]
- Support loading model state dicts ([a1eacaf]( https://github.com/es-ude/elastic-ai.creator/commit/a1eacaf490e07334a14fbbb252b2e0005319b0d2)) - [object]
- Add `find_subgraph` ([97e84ec]( https://github.com/es-ude/elastic-ai.creator/commit/97e84ece6f5452023dac79628a47a7f34e2e56f2)) - [object]
- Add `get_empty_copy` fn for `ir.Graph` ([1e28c14]( https://github.com/es-ude/elastic-ai.creator/commit/1e28c149777a2eee7f51a26b503159b3a071617e)) - [object]
- Support pattern based graph rewriting ([79dc9f4]( https://github.com/es-ude/elastic-ai.creator/commit/79dc9f4e25eed23e15c3b0322e793f1422b23cbe)) - [object]
- Add shift register ([6b5a5b2]( https://github.com/es-ude/elastic-ai.creator/commit/6b5a5b2f5448cab825d0fb97311ded09b15fed5f)) - [object]
- Add sliding window ([d22336e]( https://github.com/es-ude/elastic-ai.creator/commit/d22336e1a249422b1dda9a4ef5562b9df07e5a9e)) - [object]
- Add striding shift register ([7f25038]( https://github.com/es-ude/elastic-ai.creator/commit/7f25038659825f5942f17a924f4a8eb68b1b6ecb)) - [object]
- Add a new skeleton ([e4bc616]( https://github.com/es-ude/elastic-ai.creator/commit/e4bc6161feb5d9ecc9ff3ab6a7db74a90e90b9c4)) - [object]

### Fix

- Fix several minor errors that were discoverd by mypy ([5645da0]( https://github.com/es-ude/elastic-ai.creator/commit/5645da0f9a60cbb8071b22a7346a95880e4a9521)) - [object]
- Remove incorrect use of Self type variable ([0f8a738]( https://github.com/es-ude/elastic-ai.creator/commit/0f8a738fe67074e04db3ae0768b3bead27e7d718)) - [object]
- Use the pytest tmp_dir fixture to avoid creating files when running simulations ([06dbd62]( https://github.com/es-ude/elastic-ai.creator/commit/06dbd629d84c11a800526197696a808f7dba0814)) - [object]
- Provide useful error message for <py311 ([940ccfe]( https://github.com/es-ude/elastic-ai.creator/commit/940ccfe08ee3c07d40bfc039466381667ec08c0e)) - [object]
- Exclude required fields for node.attributes | dict() ([5ef4802]( https://github.com/es-ude/elastic-ai.creator/commit/5ef480250e5fe5495c3c40cf705d1fcc8d3e38e6)) - [object]
- Error from IrDataMeta if inheriting class has no annotations ([42fbea2]( https://github.com/es-ude/elastic-ai.creator/commit/42fbea2efd75c011b2f6590ac83eb103d168113e)) - [object]
- Do not auto create __init__ from IrDataMeta by default ([6d4b87d]( https://github.com/es-ude/elastic-ai.creator/commit/6d4b87d8d8ca49374a95b77cf7b6b7c8cdc80fef)) - [object]
- Do not raise error on unexpected plugin fields ([e3d4445]( https://github.com/es-ude/elastic-ai.creator/commit/e3d4445a0a2091d67007596461b3c32e6656474c)) - [object]
- Properly deal with nested folders ([79e5f9e]( https://github.com/es-ude/elastic-ai.creator/commit/79e5f9e948e9dd60d0a5a3c662743c92decc21d8)) - [object]
- Correct command for building docs ([0344afd]( https://github.com/es-ude/elastic-ai.creator/commit/0344afda66ec1f02755e34bb2a3d65c37adfb526)) - [object]
- Correct link to repo in html theme ([dabecc1]( https://github.com/es-ude/elastic-ai.creator/commit/dabecc111cf19f914e7fed977c7dcd178d19ebc7)) - [object]
- Fix typo ([13ebdfb]( https://github.com/es-ude/elastic-ai.creator/commit/13ebdfb0f86e92273b8190ec5757eb9abb15fd7a)) - [object]

### Refactor

- Automatically fix linted problems ([3862f21]( https://github.com/es-ude/elastic-ai.creator/commit/3862f218fe0f4ec5dc16b50239d61eac1c3edd42)) - [object]
- Removed test_utils ([ff3cd5a]( https://github.com/es-ude/elastic-ai.creator/commit/ff3cd5a5510fda21a972ab26f2e6a7944ef960ba)) - [object]
- Use more a precise return type for the create_design function ([079328a]( https://github.com/es-ude/elastic-ai.creator/commit/079328a8260f4cedf1196e28333f7cac2e0c0b7f)) - [object]
- Move ghdl_msg parsing to its own module ([124cad1]( https://github.com/es-ude/elastic-ai.creator/commit/124cad1aafe825fb9f3726b648e7dedd08311acd)) - [object]
- Refactored interface for autograd and mathoperations ([#456](https://github.com/es-ude/elastic-ai.creator/issues/456)) ([e745f3b]( https://github.com/es-ude/elastic-ai.creator/commit/e745f3b3448e8f4001e20498f2e7635590d7a69f)) - [object]
- Move tests from the Elasticai Creator source to the tests folder to maintain a consistent location for tests ([b0b892f]( https://github.com/es-ude/elastic-ai.creator/commit/b0b892f26defed59e12884c8fea6c641f8dc2605)) - [object]
- Move tests from the elasticai.creator source tree to the tests folder to maintain a consistent tests location ([7ea58c4]( https://github.com/es-ude/elastic-ai.creator/commit/7ea58c4894c9ec4b97d978effb975e61418900ce)) - [object]
- Add iter_edges method to GraphDelegate for consistency ([87313b7]( https://github.com/es-ude/elastic-ai.creator/commit/87313b70bc9eeaeec51442060acf117902facc5a)) - [object]
- Add iterator methods to ir.Graph ([dd6f077]( https://github.com/es-ude/elastic-ai.creator/commit/dd6f077343a2f22d25023c1e25ab641b7939f054)) - [object]
- Move ir2vhdl to dedicated subpackage ([65034d5]( https://github.com/es-ude/elastic-ai.creator/commit/65034d51c218ea3d28e1c7d477dc28681d33e88e)) - [object], BREAKING CHANGE:impacts every client that imported
the elasticai.creator.vhdl_template package
- Move fixed point mac to nn.fixed_point ([21f5fd6]( https://github.com/es-ude/elastic-ai.creator/commit/21f5fd62023e4a2c756c89036a3be6dd73077239)) - [object], BREAKING CHANGE:will impact everything that was
importing from elasticai.creator.vhdl.shared_designs.mac
- Put ir graph tests into same module ([d1441e6]( https://github.com/es-ude/elastic-ai.creator/commit/d1441e6db5c24a8c86999ea3b4ba77699726a303)) - [object]
- Clean up type hints ([8047010]( https://github.com/es-ude/elastic-ai.creator/commit/80470101f912ff1799002b1843e5a96382a8688a)) - [object]
- Expose nodes/edges via ir data fields ([d861c53]( https://github.com/es-ude/elastic-ai.creator/commit/d861c536ad0a1d46be5c2c98f89ec82f8c797720)) - [object]

### Style

- Fix formatting using ruff ([faf97cb]( https://github.com/es-ude/elastic-ai.creator/commit/faf97cb6f2a163f27ce5602c4ff383183d88ea2a)) - [object]
- Improve formatting and add missing text in workflows ([11b0160]( https://github.com/es-ude/elastic-ai.creator/commit/11b0160224165e200f1a5e42198de4bdb9802655)) - [object]
- Apply safe ruff format for imports ([f42651e]( https://github.com/es-ude/elastic-ai.creator/commit/f42651e973bb4de492a41fe60ab242820a894e2c)) - [object]
- Make imports in __init__.py files explicit ([60149d1]( https://github.com/es-ude/elastic-ai.creator/commit/60149d162639427e7341fed82744fd4ebd4f62c4)) - [object]
- Improve type hints ([c0701d0]( https://github.com/es-ude/elastic-ai.creator/commit/c0701d0f363eedfb270f2018492127552f28dfb7)) - [object]
- Apply safe ruff fixes to tests ([ed2ec86]( https://github.com/es-ude/elastic-ai.creator/commit/ed2ec86e006d1d64f378a7b4734d3a2851082db5)) - [object]
- Apply unsafe ruff fixes to tests ([c588232]( https://github.com/es-ude/elastic-ai.creator/commit/c5882324d7b3332fe17e5bd6a2d84aed67cba243)) - [object]

### Test

- Fix torch warnings during test runs ([4022702]( https://github.com/es-ude/elastic-ai.creator/commit/402270294973726858ad75429cfa8e1f5230e688)) - [object]
- Fix torch warnings resulting from using the deprecated torch.range function ([2225375]( https://github.com/es-ude/elastic-ai.creator/commit/22253759190d3c3561cd5cc2e65ef257eeacff18)) - [object]
- Add test cases for linear and batchnormed linear layer ([31a19f1]( https://github.com/es-ude/elastic-ai.creator/commit/31a19f1f0a27bd172b006d44b843ec155c31d0a5)) - [object]
- Remove broken echo server test ([b0f8e72]( https://github.com/es-ude/elastic-ai.creator/commit/b0f8e720741642e2a2a5a7572d962ccacab6e315)) - [object]
- Pass command line args to vunit test facade ([d36a94e]( https://github.com/es-ude/elastic-ai.creator/commit/d36a94ef9304524c9c4f66f9c74e51a2396f79ca)) - [object]

## 0.60.0 - 2024-12-18

[a8aaa00](a8aaa00ce80b9ffc2e3684648a63c47759974023)...[ffc8529](ffc85292dceeafe56e752fc522d6e732fd7d57c2)

### Build

- Add missing pytest-cov ([1707f07]( https://github.com/es-ude/elastic-ai.creator/commit/1707f07eb91693a03e3e33e1f3a4cc1ab12037c2)) - [object]
- Clean up pyproject ([fa27806]( https://github.com/es-ude/elastic-ai.creator/commit/fa27806b53b5d27f84c4ccee57e7abefbe242cf6)) - [object]
- Omit *_test in main folder for coverage ([da8104d]( https://github.com/es-ude/elastic-ai.creator/commit/da8104da6da54438104cd6bdecd68cc06d08cadd)) - [object]

### Bump

- 0.59.2 -> 0.60.0 ([a38a0ce]( https://github.com/es-ude/elastic-ai.creator/commit/a38a0ce001a397ce73d8e6f86fb839258ceaddc4)) - [object]

### Chore

- Only throw a warning if commit message exceeds char limit ([3e1d509]( https://github.com/es-ude/elastic-ai.creator/commit/3e1d509e23d8aa5302e708bb309514294b9d7984)) - [object]
- Use python3.10 to run the tests ([3151ba2]( https://github.com/es-ude/elastic-ai.creator/commit/3151ba269b6265c1f90bf12b125f0c62e5e969f0)) - [object]
- Synchronize pyproject.toml and poetry.lock ([76a8baa]( https://github.com/es-ude/elastic-ai.creator/commit/76a8baa01b52daf59248c33994e747f7b3dc4cb8)) - [object]
- Added package.json to fix the verison of commitlint ([f8a7a0f]( https://github.com/es-ude/elastic-ai.creator/commit/f8a7a0f3f888e4937baa8d0c7423637facaf443d)) - [object]
- Fixed dependency of elasticai-runtime-env5 from develop branch to specific commit ([a5ac0df]( https://github.com/es-ude/elastic-ai.creator/commit/a5ac0dfb7e0db20f21daaa75d4dd1e162f298cea)) - [object]
- Update pre-commit (necessary to fix broken black deps) ([9102843]( https://github.com/es-ude/elastic-ai.creator/commit/9102843a898829a0dea0001f009a41265b4cf919)) - [object]
- Add devenv/direnv/uvlock ([476cefa]( https://github.com/es-ude/elastic-ai.creator/commit/476cefa326eb3084a74717c872981dbbf97feff0)) - [object]
- Create coverage report for develop as well ([8c11c01]( https://github.com/es-ude/elastic-ai.creator/commit/8c11c01ae66485d70538f07effbb637533de085f)) - [object]
- Clean up external deps ([2f83d46]( https://github.com/es-ude/elastic-ai.creator/commit/2f83d46e0a39a079f9f3884ee71eb37a87d972c0)) - [object]
- Run black with python3.12 to support new syntax ([ff51308]( https://github.com/es-ude/elastic-ai.creator/commit/ff5130872f82206e68075f9b6c99b53dfa746a39)) - [object]
- Remove redundant tests ([3f0c243]( https://github.com/es-ude/elastic-ai.creator/commit/3f0c243ad1f35172e199e94d3fe060b86b943661)) - [object]
- Add test/build/style as types ([7d00767]( https://github.com/es-ude/elastic-ai.creator/commit/7d0076794361fc99aeaaf7b80474679e8cd6d257)) - [object]
- Allow 'bump' for commit msg type ([8b341ca]( https://github.com/es-ude/elastic-ai.creator/commit/8b341caf19271235caaab3f161b1b561b6a8fbf5)) - [object]

### Ci

- Publish on pushing release tags ([4b16c98]( https://github.com/es-ude/elastic-ai.creator/commit/4b16c98ed20599385e49a49e44a995cf4ea73dc6)) - [object]
- Install only testing group for unit-tests job ([73b3bf8]( https://github.com/es-ude/elastic-ai.creator/commit/73b3bf822e298fde23a045d4297ff3ca48773383)) - [object]
- Perform coverage when running checks ([75b520f]( https://github.com/es-ude/elastic-ai.creator/commit/75b520f300e4ac0bd36ed08c7c92ace70d44c64c)) - [object]
- Don't install lsp deps in test pipeline ([07853aa]( https://github.com/es-ude/elastic-ai.creator/commit/07853aa973d5e290d869b96c3168ed5bda1cde7d)) - [object]
- Remove publishing to test.pypi.org ([44d50d4]( https://github.com/es-ude/elastic-ai.creator/commit/44d50d446b8bc6a1f41a563fc3cb52ad61bf04fe)) - [object]
- Remove old release workflow ([8d0b5af]( https://github.com/es-ude/elastic-ai.creator/commit/8d0b5afc48bbee40e00924860075a6fa6641fea6)) - [object]
- Fix triggering release on tag push ([cc5c5ef]( https://github.com/es-ude/elastic-ai.creator/commit/cc5c5efaa3c185abb2e6f75e8339121afbc4f5fc)) - [object]
- Remove errornous check for tag from release pipeline ([1847b55]( https://github.com/es-ude/elastic-ai.creator/commit/1847b550cfa700891ce1fc77f15b1358588b3238)) - [object]
- Fix repo name in release workflow ([ffc8529]( https://github.com/es-ude/elastic-ai.creator/commit/ffc85292dceeafe56e752fc522d6e732fd7d57c2)) - [object]

### Docs

- Add register documentation ([59f7ed4]( https://github.com/es-ude/elastic-ai.creator/commit/59f7ed4c044b6062d37d66c7dc97cb31a056939b)) - [object]
- Add more middleware/skeleton specification ([b62d982]( https://github.com/es-ude/elastic-ai.creator/commit/b62d982f13360adffdfbd6a041ae30aaf83f7571)) - [object]
- Add timing diagram to skeleton/middleware spec ([574116b]( https://github.com/es-ude/elastic-ai.creator/commit/574116b529b20334c6646de1fd20f3e95dc47218)) - [object]
- Explain we need to read each result byte two times ([96572fb]( https://github.com/es-ude/elastic-ai.creator/commit/96572fb21958c7b79505e1ea004cdf9681e8097d)) - [object]
- Fix hw function id length ([3539c9f]( https://github.com/es-ude/elastic-ai.creator/commit/3539c9f65a6bccf72c4cfb0312a4b2408e0b4fb9)) - [object]
- Specified required versions for dev dependencies ([3b39f0e]( https://github.com/es-ude/elastic-ai.creator/commit/3b39f0e8c6a8e7a68a748df9217ad81f638705fc)) - [object]
- Started documenting supported features ([91c5167]( https://github.com/es-ude/elastic-ai.creator/commit/91c5167056b88402190b5f51b59e49d8db7090d3)) - [object]
- Added preliminary documentation for creating new modules ([1d47f05]( https://github.com/es-ude/elastic-ai.creator/commit/1d47f05290811af95c62e22dab7605b608209a8d)) - [object]
- Listed deprecated modules and those in development ([9c7c12c]( https://github.com/es-ude/elastic-ai.creator/commit/9c7c12c59ed6d84eee5caf70fb5ca8722964a139)) - [object]
- Removed unnecessary comments ([25cdf90]( https://github.com/es-ude/elastic-ai.creator/commit/25cdf904571da8d4f60418ce52447c8959b4c87b)) - [object]
- Added comments to parsing functions in testbenches ([55c9f4d]( https://github.com/es-ude/elastic-ai.creator/commit/55c9f4de6ce269f60f25edd91592aea1debe8701)) - [object]
- Added more context for the parse reported content functions ([70c8b4b]( https://github.com/es-ude/elastic-ai.creator/commit/70c8b4bbd01aaf272ccf6a91af4d91a333dce41f)) - [object]

### Feat

- Added a bash script to automatically build the vivado file with the help of vivado 2021.1 on a server ([eb8c835]( https://github.com/es-ude/elastic-ai.creator/commit/eb8c835529d736037b085f5ede0490ca342bac3e)) - [object]
- Set more specific return type for create_testbench function ([7e3f54b]( https://github.com/es-ude/elastic-ai.creator/commit/7e3f54b5cbc7aa6d026d60b28f8c32c05768aa0c)) - [object]
- Add general skeleton class ([b4ffacb]( https://github.com/es-ude/elastic-ai.creator/commit/b4ffacb1847685851def6beb9f53044fe5dbd75f)) - [object]
- New firmware that does not save testbenches ([8ad3272]( https://github.com/es-ude/elastic-ai.creator/commit/8ad3272c80a350348df1ce7a562df6f51928a4ee)) - [object]
- Test that firmware generates skeleton correctly ([3a18656]( https://github.com/es-ude/elastic-ai.creator/commit/3a1865642e60fcd4f4fbf49917d0930663bc38aa)) - [object]
- Create separate LSTMFirmwareENv5 ([17a274c]( https://github.com/es-ude/elastic-ai.creator/commit/17a274c6bb23fb5721b3e07cd16916bcbd3889c8)) - [object]
- Add skeleton for sequential layer ([34e8202]( https://github.com/es-ude/elastic-ai.creator/commit/34e8202281e0be4d79d78df66cbcffc9b4db3878)) - [object]
- Add support for less than 8 bit in skeleton ([231f0ca]( https://github.com/es-ude/elastic-ai.creator/commit/231f0ca808248b740421e5bb516b71e5f0c434ce)) - [object]
- Convert negative numbers to bit patterns using two's complement ([c94dc3b]( https://github.com/es-ude/elastic-ai.creator/commit/c94dc3ba59698e87eac2efe702a43dfc925401bd)) - [object]
- Added skeleton version 2 to project ([6ed2c94]( https://github.com/es-ude/elastic-ai.creator/commit/6ed2c94abbcb0d090eac3844fb59e27983f7ed11)) - [object]
- Remove restriction to pytorch versions < 2.0.1 ([bb47705]( https://github.com/es-ude/elastic-ai.creator/commit/bb477058440e07e2bdd6c467e328219519510771)) - [object]
- Allow python3.10 ([0628024]( https://github.com/es-ude/elastic-ai.creator/commit/0628024ba826ebbdbe5b5deda4aac67d81876248)) - [object]
- Added an basic example for a network using skeleton v2 ([6d94158]( https://github.com/es-ude/elastic-ai.creator/commit/6d941584c40d41049bd27e9da8c2dc204f79080b)) - [object]
- Allow '${key}' placeholders for multiline templates ([d25eef1]( https://github.com/es-ude/elastic-ai.creator/commit/d25eef1369c754911e56ed5aa4a92f62b2716325)) - [object]
- Allow python3.12 ([46d6cfb]( https://github.com/es-ude/elastic-ai.creator/commit/46d6cfb52eb2c6de471d9dd0310a9152200ec0db)) - [object]
- Abstract class DesignCreator inherits from torch.nn.Module ([d6e70ed]( https://github.com/es-ude/elastic-ai.creator/commit/d6e70ed1c84d025bb2eebbfda45c67ad9ba1f987)) - [object]
- Added conv1d. Simulation works. End-to-end system test is still pending ([93e5ecd]( https://github.com/es-ude/elastic-ai.creator/commit/93e5ecdaae2987d34772a504bc843019419bd845)) - [object]
- Added simulation for linear layer ([aac395b]( https://github.com/es-ude/elastic-ai.creator/commit/aac395b9413943d6252bf5cb4866d96173216ae8)) - [object]
- Added enV5 usb library to development pyproject.toml. This will be used in the future to do system tests ([3089341]( https://github.com/es-ude/elastic-ai.creator/commit/3089341849008fbfb1ff66029ea522702cc4303f)) - [object]
- Added a generator for echo server with skeleton #378 ([2c3faf5]( https://github.com/es-ude/elastic-ai.creator/commit/2c3faf575df21f1aca236138257097ddd2320bff)) - [object]
- Added an example for the echoserver with skeleton v2 ([3f46780]( https://github.com/es-ude/elastic-ai.creator/commit/3f46780d44dc6f0d220b3a3d82f71e33ae38fdac)) - [object]
- Echo server works now ([a4359a0]( https://github.com/es-ude/elastic-ai.creator/commit/a4359a0f08fa5a620ce414c8de6e133613427a65)) - [object]
- Linear layer system test with elastic node works now ([238964a]( https://github.com/es-ude/elastic-ai.creator/commit/238964a119b31db57c44085c336c8605e10c8e9a)) - [object]
- Add graph delegate and iterators ([868a188]( https://github.com/es-ude/elastic-ai.creator/commit/868a188aa4ce0be92608997bbfdb2916e7f8603e)) - [object]
- Add abstract ir data class and nodes ([bb81f0d]( https://github.com/es-ude/elastic-ai.creator/commit/bb81f0dea0ee8ea8646e57d85cc070baddf91e8a)) - [object]
- Added fixed point config, autograd and quantize ([97bb203]( https://github.com/es-ude/elastic-ai.creator/commit/97bb203898e6d689fff54c73da722584aca6882f)) - [object]
- Added basic layers ([82db217]( https://github.com/es-ude/elastic-ai.creator/commit/82db217242fed30a955dcf7a69eb98a56e4b931a)) - [object]
- Introduce read only field for IrData ([be1e8fb]( https://github.com/es-ude/elastic-ai.creator/commit/be1e8fb3a659d30882b77399016fa8c21d8f0e6b)) - [object]
- Make suc/predecessors sorted/deterministic ([8825607]( https://github.com/es-ude/elastic-ai.creator/commit/88256079288bae95668ffe60aced222e920419c3)) - [object]
- Add basic graph data structure ([e1dfef2]( https://github.com/es-ude/elastic-ai.creator/commit/e1dfef26688ffc819fe07ae7831df6b899c4b7f3)) - [object]
- Add basic function registry ([3785db0]( https://github.com/es-ude/elastic-ai.creator/commit/3785db0d42f2f7a9118b9b5c3e60d1f83f9bbd86)) - [object]
- Add LoweringPass class ([d598021]( https://github.com/es-ude/elastic-ai.creator/commit/d5980214c2441b52eef063bed865de2eecd52f10)) - [object]
- Add automatic deterministic skeleton id generation ([eb7e59f]( https://github.com/es-ude/elastic-ai.creator/commit/eb7e59f7aa7506206e6807ef6a649e8b458930b4)) - [object]
- Tweak api for skel id computation ([f7d9a77]( https://github.com/es-ude/elastic-ai.creator/commit/f7d9a7786e06a500dafcc6cbf3f08f81083c6166)) - [object]
- Move hw accel meta to dedicated module ([9f65b8d]( https://github.com/es-ude/elastic-ai.creator/commit/9f65b8dce91fcdfd87f7f1229a06c2d3776f8ad5)) - [object]
- Load plugin description from package ([05a99c3]( https://github.com/es-ude/elastic-ai.creator/commit/05a99c3e71a3c408a4ab273b7fe453b215d39ef9)) - [object]
- Load plugin and call generated fn ([0492e0b]( https://github.com/es-ude/elastic-ai.creator/commit/0492e0b94eae88eb536bd7b85859527026ec273d)) - [object]
- Move plugin_loader and type_handler decorators ([6bba61d]( https://github.com/es-ude/elastic-ai.creator/commit/6bba61d5f7758f1b9db14a0938e29f4c163c52b9)) - [object]
- Add plugin loader and improve function registry ([0a8ac61]( https://github.com/es-ude/elastic-ai.creator/commit/0a8ac61fef8792ab177f7635d86d4f9ae23029b1)) - [object]
- Add basic but flexible templating component ([2ae0506]( https://github.com/es-ude/elastic-ai.creator/commit/2ae050611b9bc2cc93624e99bad7c1244dd2b6c4)) - [object]
- Remove ir2vhdl (shouldnt have been committed) ([20fb891]( https://github.com/es-ude/elastic-ai.creator/commit/20fb8916f78ea3a78ea7eeef9af1d3f071168ca2)) - [object]

### Fix

- Save testbench to separate folder ([9937431]( https://github.com/es-ude/elastic-ai.creator/commit/99374317966a0db05c147bf99d322da5b14b0f5a)) - [object]
- Xil to work lib ([005ed36]( https://github.com/es-ude/elastic-ai.creator/commit/005ed36a4ff8bac6bb1ba1ed29e5e9cfe0be6c73)) - [object]
- Fix tables and language ([63a1b9d]( https://github.com/es-ude/elastic-ai.creator/commit/63a1b9d42aa7a8f3866b978516a7269cec10e61b)) - [object]
- Fix wrong signal name in integration test ([74ebc32]( https://github.com/es-ude/elastic-ai.creator/commit/74ebc32e938936d2d60c3da09773439c5675106d)) - [object]
- Added skeleton_1.vhd needs to be changed ([632bf89]( https://github.com/es-ude/elastic-ai.creator/commit/632bf8974ace775c8289351d03c026e587c237ed)) - [object]
- Add expected newline to end of skeleton ([dcb20b7]( https://github.com/es-ude/elastic-ai.creator/commit/dcb20b712945439b1c3db799404beb33d8587e4f)) - [object]
- Fix skeleton for mlp use case ([e4b67cc]( https://github.com/es-ude/elastic-ai.creator/commit/e4b67ccbc4178629f35f3f3a89259d2bfae3aba0)) - [object]
- Transmit high byte first instead of low ([f2bd5af]( https://github.com/es-ude/elastic-ai.creator/commit/f2bd5af1cf6d9e4f40da8c89a89c61663cf12086)) - [object]
- Correct counter in example code ([1860657]( https://github.com/es-ude/elastic-ai.creator/commit/1860657968f0828502eecfb892c6e58fab93bf10)) - [object]
- Update deps to resolve security issues ([6568d28]( https://github.com/es-ude/elastic-ai.creator/commit/6568d2830120922f77d2e183aa5764369143135f)) - [object]
- Fixed the test for the old skeleton and added another one for skeleton v2 ([87b11a4]( https://github.com/es-ude/elastic-ai.creator/commit/87b11a4455ec059ed8e3fdb3a405a976435facd6)) - [object]
- Added an exception raise for the skeleton for not supported configurations ([11d006c]( https://github.com/es-ude/elastic-ai.creator/commit/11d006c146b9eaa809460729132222a0595e6793)) - [object]
- Warn when using skeleton v1 ([5a46331]( https://github.com/es-ude/elastic-ai.creator/commit/5a4633193cf2c6db699fa19c47ddfbc53599c1fe)) - [object]
- Implement function only supported in python3.11 and higher ([5223adf]( https://github.com/es-ude/elastic-ai.creator/commit/5223adfdf6551f9758ee4dbdef9df1f2aed36377)) - [object]
- Fix incorrectly sorted inputs, add input/output widths and assert them ([74f5e26]( https://github.com/es-ude/elastic-ai.creator/commit/74f5e265fb388249a3d46c1743bc3b3e38366a78)) - [object]
- Fix wrong path that leads to temporary files created in the project tree ([0ff9c0b]( https://github.com/es-ude/elastic-ai.creator/commit/0ff9c0b04442e6da0d76f3991254cae63bf260e8)) - [object]
- #374 remove unnecessary data_buf from top module. It is not used anywhere so should have no effect ([e8cd5a7]( https://github.com/es-ude/elastic-ai.creator/commit/e8cd5a797c92b4f29b94dad1a9b7de4d090a98ae)) - [object]
- Fixed fxp_mac reset. This modules reset is now working at any point in time independent of the other inputs ([054b8ab]( https://github.com/es-ude/elastic-ai.creator/commit/054b8ab4d3569b3ae105b791ea0e8f116a8ddfd6)) - [object]
- Linear layer uses signals now. So simulation works ([7f5d30c]( https://github.com/es-ude/elastic-ai.creator/commit/7f5d30c6c066506b66112c9ba15fe367ce33f9a8)) - [object]
- Fixed the dependency for the runtime utils ([cfaf318]( https://github.com/es-ude/elastic-ai.creator/commit/cfaf318915a046f0b5707a56c2fcbdb9e312f1dc)) - [object]
- Fixed the poetry lock file ([31868ca]( https://github.com/es-ude/elastic-ai.creator/commit/31868caefb3959d0966c93d01a45c234a9041b55)) - [object]
- Revert changes in linear.tpl.vhd ([f11b606]( https://github.com/es-ude/elastic-ai.creator/commit/f11b6061ff88c4b16e70131508b9f10758c9b90d)) - [object]
- Fixed error in template ([948af39]( https://github.com/es-ude/elastic-ai.creator/commit/948af39669a9070518a96bd3611cb6d43b405986)) - [object]
- Fixed an error in the skeleton v2 template ([dc2ee96]( https://github.com/es-ude/elastic-ai.creator/commit/dc2ee964c8898a359fafe75b2bcb216ab39ebf2a)) - [object]
- Fixed the test for the firmware with skelton v2 ([75ef96a]( https://github.com/es-ude/elastic-ai.creator/commit/75ef96a21236c9fc3a6e830aedc5978a9e033c9e)) - [object]
- Fixed error in test ([2d7f140]( https://github.com/es-ude/elastic-ai.creator/commit/2d7f140bc9382be47074c1cbda3015f10ecdfaab)) - [object]
- Fixed error in convolution ([390656c]( https://github.com/es-ude/elastic-ai.creator/commit/390656cc00cd4f827c27badae232ac8073f480a2)) - [object]
- Fixed test for changes in sensitivity list and for rising/falling edge clock ([088bc1f]( https://github.com/es-ude/elastic-ai.creator/commit/088bc1f7a206739701d6bf9735b3974add0262c0)) - [object]
- Fixed code generation test for linear layer ([7f8f445]( https://github.com/es-ude/elastic-ai.creator/commit/7f8f4455ea1352d0c30fea05e22fb7ce561d654c)) - [object]
- Fix bug where iterator was not remembering visited nodes ([3da0dbc]( https://github.com/es-ude/elastic-ai.creator/commit/3da0dbc5e84748fdd7db5ee78b9cd40636f19e7e)) - [object]
- Fix conceptual problems with abstract ir data type ([1e6210d]( https://github.com/es-ude/elastic-ai.creator/commit/1e6210db742f3e1b9b2613126cc48262e6eddee4)) - [object]
- Remove dead code and fix typing ([609eb51]( https://github.com/es-ude/elastic-ai.creator/commit/609eb51c45c298e190a1e6f2133623b456e9ee2c)) - [object]
- Make graph iterators deterministic ([2c3b27a]( https://github.com/es-ude/elastic-ai.creator/commit/2c3b27a0e8afbf7bdbea3ce8e45abbbc65408184)) - [object]
- Removed init in elasticAi ([be65e7c]( https://github.com/es-ude/elastic-ai.creator/commit/be65e7c223e339b5ec03fc3b54ec3e4782a58d98)) - [object]
- Remove toplevel __init__.py ([c7f0a78]( https://github.com/es-ude/elastic-ai.creator/commit/c7f0a7820c094789d8ae7e4bc9076c5cda167f8d)) - [object]
- Make type hints 3.10 compatible ([db8a0f8]( https://github.com/es-ude/elastic-ai.creator/commit/db8a0f8dc836e09bc4cc978e574b4d22be798954)) - [object]
- Remove outdated srcs in skeleton plugin ([57ae044]( https://github.com/es-ude/elastic-ai.creator/commit/57ae0442099fe36a2e8c31fe100d2eba59779093)) - [object]

### Refactor

- Rename _integration_test to example ([6828739]( https://github.com/es-ude/elastic-ai.creator/commit/6828739ea5ee27cbe077b5c31ffbf14d66d5f480)) - [object]
- Remove unnecessary files ([26b8afa]( https://github.com/es-ude/elastic-ai.creator/commit/26b8afaa457969777b03f04e537470f1f6917055)) - [object]
- Added __init__.py ([ad897c2]( https://github.com/es-ude/elastic-ai.creator/commit/ad897c20df13c550eb8110299c2c85f3ba960eeb)) - [object]
- Renamed tests folder to test_utils and moved to elasticai/creator ([e2b86a8]( https://github.com/es-ude/elastic-ai.creator/commit/e2b86a8f8b556cd3d285f59052f11a9c173592a8)) - [object]
- Move design_creator module to nn package ([f17a6da]( https://github.com/es-ude/elastic-ai.creator/commit/f17a6dac5e612ba99bfc906862de3e3048aa7a17)) - [object]
- Rename the module design_creator to design_creator_module ([49d3ab4]( https://github.com/es-ude/elastic-ai.creator/commit/49d3ab4e1d244c13d4994df65dda2f47a846aaad)) - [object]
- Better ghdl simulation class ([873fd42]( https://github.com/es-ude/elastic-ai.creator/commit/873fd421db4f0bfb179479c43ee459b71dbeee01)) - [object]
- Made the name a property so the name is already set correctly and still accessible ([593682b]( https://github.com/es-ude/elastic-ai.creator/commit/593682b933642a809496dd2a9b00fdce0e9ba19d)) - [object]
- Changing the wake_up signal to best practice method ([04221ec]( https://github.com/es-ude/elastic-ai.creator/commit/04221ec93724939dcc3bc4d3e28428ca1afffe28)) - [object]
- Making the test a bit more convinient ([6c778bc]( https://github.com/es-ude/elastic-ai.creator/commit/6c778bca80ed20c85f2ed2d00701e3b0f2152486)) - [object]
- Add new replacement variable in log2 calculation of linear layer ([082b8fd]( https://github.com/es-ude/elastic-ai.creator/commit/082b8fd23ba37f7428b49ab7dcbfabb056c37544)) - [object]
- Changed sensitivity list to clock only ([01dd3c5]( https://github.com/es-ude/elastic-ai.creator/commit/01dd3c5095d2d515cd1516b3a7ca56fe370bee6d)) - [object]
- Moved the opening of the serial port to context manager ([8431bc7]( https://github.com/es-ude/elastic-ai.creator/commit/8431bc74cd1f8444df5a0b6f5b184e52979ffc95)) - [object]
- Moved mac operators to vhdl shared design ([4925d67]( https://github.com/es-ude/elastic-ai.creator/commit/4925d673e8086fceb5af31d1e577c56a003f1dd2)) - [object]
- Moved simulated layer. MAC operator design simulations do not work ([3b927b6]( https://github.com/es-ude/elastic-ai.creator/commit/3b927b693e989a4e82f97b925df28641b7b33fab)) - [object]
- Removed unnecessary print statements and added type hint ([c3615bd]( https://github.com/es-ude/elastic-ai.creator/commit/c3615bddd1a0fab13001dc2457b5f9094c7a91e7)) - [object]
- Avoid keeping two registries ([21a951e]( https://github.com/es-ude/elastic-ai.creator/commit/21a951ed97229f2451d4ddf28c52db794e6f86be)) - [object]
- Decouple fn registering and calling ([ab737b9]( https://github.com/es-ude/elastic-ai.creator/commit/ab737b9bc4f86781e45d5d0b2d804ab5892a495d)) - [object]
- Use new descriptor for registering fns in lowerable ([2bd382d]( https://github.com/es-ude/elastic-ai.creator/commit/2bd382ddd78212aaf925592b9c7f7838c85e89cb)) - [object]

### Style

- Beautify commit # [9035010](https://github.com/es-ude/elastic-ai.creator/commit/903501083d6acaee8b472f22e7bf24cddb3647b8) ([cc6fda5]( https://github.com/es-ude/elastic-ai.creator/commit/cc6fda534289fdf4359c46e9e08ba984c7638a07)) - [object]
- Beautify commit # [b61c618](https://github.com/es-ude/elastic-ai.creator/commit/b61c6180a56de314e569338eebbbdbe45a889f42) ([ce92f2b]( https://github.com/es-ude/elastic-ai.creator/commit/ce92f2b593d3b705e34867a8b87d90e3f4a7d9a9)) - [object]

## 0.59.2 - 2023-10-06

[9e690eb](9e690eb35378b3915fe8c12226fc12e2b64974c1)...[a8aaa00](a8aaa00ce80b9ffc2e3684648a63c47759974023)

### Fix

- Copy model to cpu for quantized inference ([0c5d88e]( https://github.com/es-ude/elastic-ai.creator/commit/0c5d88e26e55eb11d2a729c5a7bf6b865927b61f)) - [object]

## 0.59.1 - 2023-10-06

[3314acf](3314acfafe001683a31ad211bcb9599c54acb886)...[9e690eb](9e690eb35378b3915fe8c12226fc12e2b64974c1)

### Docs

- Explain relationship between LSTM, LSTMNetwork and their sw/hw impl ([7db8974]( https://github.com/es-ude/elastic-ai.creator/commit/7db8974e11a586e60c7f154e7cbbbf27b75a9c41)) - [object]

### Feat

- Reintegrate lstm implementation ([9440fbb]( https://github.com/es-ude/elastic-ai.creator/commit/9440fbb1ed9c81218e62f4e60917127e128d3856)) - [object]
- Added skeleton, middleware, top module and Vivado constraints fopr env5 LSTM example ([67117a4]( https://github.com/es-ude/elastic-ai.creator/commit/67117a443032cbafd3bcf8abcab7177a801fd659)) - [object]
- Add lstm_network testbench ([37d7921]( https://github.com/es-ude/elastic-ai.creator/commit/37d79212f52b949634f7af321e7b7bc56306ffeb)) - [object]
- Inject network to FirmwareENv5 ([bf2c53f]( https://github.com/es-ude/elastic-ai.creator/commit/bf2c53f554bfc312875f08cb99d95e028364667b)) - [object]

### Fix

- Turn `vhdl.top` into a package ([2a272ea]( https://github.com/es-ude/elastic-ai.creator/commit/2a272ea4e0b470bf8684098cf48e8121ee92d27f)) - [object]
- Add saving constraints and sources to plug&play ENV5 ([41aa4f1]( https://github.com/es-ude/elastic-ai.creator/commit/41aa4f105fb59bcdc76b031800af916eb0c76f35)) - [object]
- Added missing file for network skeleton tpl ([38fe8a7]( https://github.com/es-ude/elastic-ai.creator/commit/38fe8a7337479e5b7c02d66d682431e95b03e190)) - [object]
- Parametrize names ([c13576d]( https://github.com/es-ude/elastic-ai.creator/commit/c13576d380c58f658c4054a645f8407041a95faf)) - [object]
- Fix lstm names ([da279f7]( https://github.com/es-ude/elastic-ai.creator/commit/da279f742409b9c31d690d385c4d04896e3afbb4)) - [object]
- Fix lstm test bench file name ([f879b4d]( https://github.com/es-ude/elastic-ai.creator/commit/f879b4de526bc7c7cd93371b5874c1eac2f465f5)) - [object]
- Correct `create_testbench` for lstm ([e82af52]( https://github.com/es-ude/elastic-ai.creator/commit/e82af52217f0ef1874abfcd0b43f1d905ed3e4bb)) - [object]
- Move `create_testbench` to correct class ([53bc568]( https://github.com/es-ude/elastic-ai.creator/commit/53bc5684682abbc721969f357b0810175a89a25f)) - [object]
- Names and templates for lstm ([3ad358c]( https://github.com/es-ude/elastic-ai.creator/commit/3ad358c698bca447376b910bdd275bc806eb6db6)) - [object]
- Remove unnecessary instance name templ variables ([7da4b5a]( https://github.com/es-ude/elastic-ai.creator/commit/7da4b5a473e9976392eda1d4e686cc4ff9b12d0d)) - [object]
- Don't save uut in testbench ([7f09a2a]( https://github.com/es-ude/elastic-ai.creator/commit/7f09a2ab3549d309759e531dd5b6ec4051a9d3e7)) - [object]
- Add skeleton, etc. to generated files ([2bbd588]( https://github.com/es-ude/elastic-ai.creator/commit/2bbd588ceaec6408f61f48c2f61289c90afffef9)) - [object]
- Fix fxp mac test ([baad73b]( https://github.com/es-ude/elastic-ai.creator/commit/baad73b8e97f5c7b65e52a1c9755eb2086de02aa)) - [object]
- Fix skeleton test ([12b7c27]( https://github.com/es-ude/elastic-ai.creator/commit/12b7c274d1d9116d06be711066f2d5ee1cf5725e)) - [object]
- Use linear layer name ([9d48f09]( https://github.com/es-ude/elastic-ai.creator/commit/9d48f098e60ead2854af61a6c1394e824e762538)) - [object]
- Skeleton naming ([21d057d]( https://github.com/es-ude/elastic-ai.creator/commit/21d057d2497d817c09c12e90f43047aeed71e6d8)) - [object]
- Set step_lut as non trainable parameter ([95954c2]( https://github.com/es-ude/elastic-ai.creator/commit/95954c2cbd52a85f762128ce3a88259085431536)) - [object]
- Do not copy input to cpu ([586b774]( https://github.com/es-ude/elastic-ai.creator/commit/586b77458e9529dc8f12023fbefb6a3747fd222e)) - [object]
- Add step_lut to state_dict ([e18f46f]( https://github.com/es-ude/elastic-ai.creator/commit/e18f46f0a70b1000e8e6d0ea3ecdddce2ad325d5)) - [object]

## 0.59.0 - 2023-10-02

[5e4f666](5e4f6666f6be1c68f7b99bb9f10c055ab96a80e5)...[3314acf](3314acfafe001683a31ad211bcb9599c54acb886)

### Chore

- Add simulation test tag ([f7fda58]( https://github.com/es-ude/elastic-ai.creator/commit/f7fda587861e83c729eaa90791b34efc4f9b433d)) - [object]

### Docs

- Add documentation for number conversion ([617e2c6]( https://github.com/es-ude/elastic-ai.creator/commit/617e2c61fab2b21459a1a62c65c184cdbcc35e09)) - [object]

### Feat

- Lstm reintegration ([0ffa9b0]( https://github.com/es-ude/elastic-ai.creator/commit/0ffa9b029636ab36570019e9f99fd5c788281e26)) - [object]
- Add number_conversion ([1fde323]( https://github.com/es-ude/elastic-ai.creator/commit/1fde32308a83a3de71af736eda47a81bb5bc468b)) - [object]
- Parse ghdl output ([3f7e05f]( https://github.com/es-ude/elastic-ai.creator/commit/3f7e05f44f86776ea01b28b176116730d94a9354)) - [object]
- Handle colons in ghdl sim parsing ([f485531]( https://github.com/es-ude/elastic-ai.creator/commit/f485531881b0874eb14df58ccb3873889cb1cac6)) - [object]
- Basic fxp mac + hw/sw simulation tests ([f34a1ed]( https://github.com/es-ude/elastic-ai.creator/commit/f34a1edc70c34d456daa222bed537d793ee0c29e)) - [object]
- Add xnor-popcount based mac bin impl ([6a63eb3]( https://github.com/es-ude/elastic-ai.creator/commit/6a63eb358ce3279bcdbca468aed25445ab0be13e)) - [object]

### Fix

- Remove need for csv in testbench ([f905bbf]( https://github.com/es-ude/elastic-ai.creator/commit/f905bbf6e1f9119677b7f73caf58b35343e0d7bb)) - [object]
- Ignore one more line for ghdl out parsing ([c48ec8f]( https://github.com/es-ude/elastic-ai.creator/commit/c48ec8f42f906222223b7282c677ffdfe5cd06ec)) - [object]
- Exclude simulations from coverage ([3b31a45]( https://github.com/es-ude/elastic-ai.creator/commit/3b31a45bccb262d60a1d920618457514a2bd8a95)) - [object]
- Make mac impl use round to zero logic ([f8c674f]( https://github.com/es-ude/elastic-ai.creator/commit/f8c674f5a2382a1d8b9b1630a8af35977dc0c9c9)) - [object]

### Refactor

- Simplify number conversion implementations ([715d6c7]( https://github.com/es-ude/elastic-ai.creator/commit/715d6c7540748fb53e93f78782f653ee45e6bdb4)) - [object]
- Simplify simulating test benches ([b607211]( https://github.com/es-ude/elastic-ai.creator/commit/b6072118e1d84e59b1faac73ec3c75c6aca88ee9)) - [object]
- Rename hw_integ_test.py ([8324d1a]( https://github.com/es-ude/elastic-ai.creator/commit/8324d1abaad92a12d5fcdee9925c58b9c9743aff)) - [object]
- Move number conversion modules ([ed3086c]( https://github.com/es-ude/elastic-ai.creator/commit/ed3086c149388876e8cd243cd535199bded8f9f5)) - [object]
- Add create_simulation method to layer ([6c93c81]( https://github.com/es-ude/elastic-ai.creator/commit/6c93c81b5910dfcc2eb71d64c137be5f9b8d0fad)) - [object]

### Style

- Beautify commit # [f94e16f](https://github.com/es-ude/elastic-ai.creator/commit/f94e16fd03d289124dd20dd844776d517fb91e4a) ([db35406]( https://github.com/es-ude/elastic-ai.creator/commit/db354066168885e5ae91d18efd705d239319b81a)) - [object]

## 0.58.0 - 2023-09-29

[a5aecb3](a5aecb3cbc65e1231d1e2de957f28a1fdaba427c)...[5e4f666](5e4f6666f6be1c68f7b99bb9f10c055ab96a80e5)

### Docs

- Use create_design function instead of translate in minimal example ([b9351ca]( https://github.com/es-ude/elastic-ai.creator/commit/b9351ca28b37a0ee7f34187b01986c3ba11c6827)) - [object]

### Feat

- Add tests for conv1d layer ([c645297]( https://github.com/es-ude/elastic-ai.creator/commit/c6452975ef02d6c2a46ca4550311238a46917636)) - [object]
- Add parameter getters ([3c184c0]( https://github.com/es-ude/elastic-ai.creator/commit/3c184c075cdb94ffae05ea7424e33dd98a4c09f9)) - [object]
- Add tests for the conv1d design ([c2c94bd]( https://github.com/es-ude/elastic-ai.creator/commit/c2c94bd91e09e507120cfcf71e28b0797dca419c)) - [object]
- Use bias as default ([14b01be]( https://github.com/es-ude/elastic-ai.creator/commit/14b01be0e3aad2315daa1864588701ce2fd8dff7)) - [object]
- Add tests for fixed point linear layer ([2e959b6]( https://github.com/es-ude/elastic-ai.creator/commit/2e959b619ea9ef66dfede042131f7116b18f3532)) - [object]
- Add tests for linear design ([55a366a]( https://github.com/es-ude/elastic-ai.creator/commit/55a366a34049708ee62d0c440f67640435716900)) - [object]
- Add small integration test to verify that linear layer creates correct design ([f71e43f]( https://github.com/es-ude/elastic-ai.creator/commit/f71e43f92cbfbf2c7a735a610c3dbca790ea8299)) - [object]
- Add small integration test to verify that conv1d layer generates correct design ([da45cc3]( https://github.com/es-ude/elastic-ai.creator/commit/da45cc30b481ee27e9bd9d5d172e29a0bd0b519f)) - [object]

### Fix

- Remove unnecessary output quantization of the SiLU base module ([350faa5]( https://github.com/es-ude/elastic-ai.creator/commit/350faa52d93994738a68cc0222dfb907b5174f12)) - [object]
- Remove stride and padding as supported parameters for conv1d ([05b57d1]( https://github.com/es-ude/elastic-ai.creator/commit/05b57d1d2e112a21259fa2221f2204b3f6d87bfe)) - [object]
- Remove already dropped padding, stride and dilation ([79494fe]( https://github.com/es-ude/elastic-ai.creator/commit/79494fe6116921ff7b4d1287bd67451ba7498ecd)) - [object]

### Refactor

- Rename SiLUWithTrainableScaleBeta to AdaptableSiLU ([fdf34b0]( https://github.com/es-ude/elastic-ai.creator/commit/fdf34b0c04eae35af3867c8f9109cf24400b4d33)) - [object]
- Add test for the fixed point silu ([2b0da94]( https://github.com/es-ude/elastic-ai.creator/commit/2b0da947ff8cddf282294f4139a74b4b723cc4cb)) - [object]
- Split batch normed conv1d and conv1d layers in seperate files and add parameter getters ([924e5f3]( https://github.com/es-ude/elastic-ai.creator/commit/924e5f3646c151667c881360ed68aad890dc5a67)) - [object]
- Split layer.py into multiple files to improve readability ([37128c9]( https://github.com/es-ude/elastic-ai.creator/commit/37128c95d0a28268b1889783b31e536074d01ab9)) - [object]
- Remove unused import ([24164b0]( https://github.com/es-ude/elastic-ai.creator/commit/24164b0ff52f9e26a21a2342c9be2d5079a270e6)) - [object]
- Remove not necessary fixed weights ([d9482ce]( https://github.com/es-ude/elastic-ai.creator/commit/d9482cec23e20d4b370be8037fb528552b92c2cf)) - [object]
- Rename Translatable protocol to DesignCreator ([e60bf7f]( https://github.com/es-ude/elastic-ai.creator/commit/e60bf7f0688ba6e875d821c8cab4c34f4054cdec)) - [object]
- Rename translatable module to design_creator ([0aa4d72]( https://github.com/es-ude/elastic-ai.creator/commit/0aa4d72458aef4a91c87334048c357a276627d3d)) - [object]
- Remove failing test ([e56a07f]( https://github.com/es-ude/elastic-ai.creator/commit/e56a07feb90b8853f3b2901503bbe034fa7b4f16)) - [object]

### Style

- Beautify commit # [4b05c4a](https://github.com/es-ude/elastic-ai.creator/commit/4b05c4a847ee33048b9552a93f816d6fec3c404f) ([c10028b]( https://github.com/es-ude/elastic-ai.creator/commit/c10028bd5a2019c8de67f95bf1c25c930c5ec8d0)) - [object]

## 0.57.1 - 2023-08-29

[6858178](68581787183a4ce4058d7db87e366c1376bdf08c)...[a5aecb3](a5aecb3cbc65e1231d1e2de957f28a1fdaba427c)

### Fix

- Try to exclude test files from build ([f282ac0]( https://github.com/es-ude/elastic-ai.creator/commit/f282ac06aae45451d3787d74cda54b51e7f28200)) - [object]

## 0.57.0 - 2023-08-29

[535c4c1](535c4c16f9c82922f4c6dc07c9737995f66f72fb)...[6858178](68581787183a4ce4058d7db87e366c1376bdf08c)

### Feat

- Global math operations depend on math operations of supported layers ([127ffdb]( https://github.com/es-ude/elastic-ai.creator/commit/127ffdb29853587e4f819d75077e524e7a168bc5)) - [object]
- Exclude tests from build ([72b8e0a]( https://github.com/es-ude/elastic-ai.creator/commit/72b8e0af0e3fddc154be5763b26f5174cc49d7f4)) - [object]

### Refactor

- Move unit tests to the elasticai package ([bb0ab8b]( https://github.com/es-ude/elastic-ai.creator/commit/bb0ab8b8b8636c07318bae5e662836a07b5f33ec)) - [object]
- Rename test files from test_*.py to *_test.py to improve readabilitly ([b7d3557]( https://github.com/es-ude/elastic-ai.creator/commit/b7d3557338617c90b9b70459c4eeff12cc1c4623)) - [object]

## 0.56.0 - 2023-08-28

[882c9c3](882c9c3bf5fc1173bd7722bfc2d9581e86575828)...[535c4c1](535c4c16f9c82922f4c6dc07c9737995f66f72fb)

### Chore

- Remove fixed commit scopes ([9f6b2e7]( https://github.com/es-ude/elastic-ai.creator/commit/9f6b2e793e7a952f5ddff7efc89677a2f00c935e)) - [object]

### Docs

- Start glossary ([b2d82cd]( https://github.com/es-ude/elastic-ai.creator/commit/b2d82cdcc663d13492b611a700321c4bbcf452be)) - [object]
- Update minimal example to reflect the most recent changes ([8f122a0]( https://github.com/es-ude/elastic-ai.creator/commit/8f122a04699c42ff7abb7179d3cb7412cf94c0ef)) - [object]
- Add glossary entry ([8855e02]( https://github.com/es-ude/elastic-ai.creator/commit/8855e0248e97a2628c1ff73e69538c969f52d685)) - [object]

### Feat

- Add layers to __init__ file in fixed_point package to improve usability ([93917d8]( https://github.com/es-ude/elastic-ai.creator/commit/93917d8654794b7f5baa005e989f2984d6c846e3)) - [object]
- Add a public quantize function to allow initial quantization of model inputs ([c8a170c]( https://github.com/es-ude/elastic-ai.creator/commit/c8a170cb1ba529855790dcaf2dad97c38171174e)) - [object]

### Fix

- Outdated imports ([650a71e]( https://github.com/es-ude/elastic-ai.creator/commit/650a71e36f9c12db1160b86f82ad4d37715b19d7)) - [object]
- Remove deprecated layers ([d55c041]( https://github.com/es-ude/elastic-ai.creator/commit/d55c041e0141fe30bf3038074c061704ed057682)) - [object]
- Fix broken imports and other errors that leads to failing tests ([6eac561]( https://github.com/es-ude/elastic-ai.creator/commit/6eac56189e1e449d378867a4b4d8003967f48689)) - [object]

### Refactor

- Remove mlframework/typing.py ([6858537]( https://github.com/es-ude/elastic-ai.creator/commit/6858537f7906b75495376828c4713690b14cb461)) - [object]
- Remove empty folders, start docs improvements ([28a9a2d]( https://github.com/es-ude/elastic-ai.creator/commit/28a9a2d96521fbcb53ca889b746470bb99aef20f)) - [object]
- Improve separation of core packages ([b5f469f]( https://github.com/es-ude/elastic-ai.creator/commit/b5f469f1fbb354547560c9fefcd88e271382ae91)) - [object]
- Restructure packages ([2fa7a4f]( https://github.com/es-ude/elastic-ai.creator/commit/2fa7a4f58a481868edca1bdd3a568686130873dd)) - [object]
- Move batchnormed layers to their base versions ([b1e0feb]( https://github.com/es-ude/elastic-ai.creator/commit/b1e0feb6e45fd4f300065397ab2698382135c4b5)) - [object]
- Separate interface for conv1d ([76ba0ac]( https://github.com/es-ude/elastic-ai.creator/commit/76ba0ac054d7acd07ace2eb9875e9bd3473eeca3)) - [object]
- Rename and move modules to fit our new scheme ([8effe1a]( https://github.com/es-ude/elastic-ai.creator/commit/8effe1ac03fceebd324e4ad07f9d305c8e7d0c08)) - [object]
- Remove arithmetics and autograd_functions from base_modules ([1e23c74]( https://github.com/es-ude/elastic-ai.creator/commit/1e23c7406ffb0c0ab0a62aa31f8d61a502a4886f)) - [object]
- Rename inputs parameter to x in forward parameter lists ([314b747]( https://github.com/es-ude/elastic-ai.creator/commit/314b747e72509a2d34e72cc1d7738e2e26c18bd3)) - [object]
- Adapt the structure of the tests directory to the latest changes ([b3cd5cc]( https://github.com/es-ude/elastic-ai.creator/commit/b3cd5cc5c3cb68e7b5adb8136322426830d6db40)) - [object]
- Reformat code ([1fb5ed6]( https://github.com/es-ude/elastic-ai.creator/commit/1fb5ed6663424a98a36db3ad6ef899d62c74b75c)) - [object]

### Style

- Beautify commit # [fcb153e](https://github.com/es-ude/elastic-ai.creator/commit/fcb153ea3aa32a73e07dd1f71d148634698a6cda) ([6515ab0]( https://github.com/es-ude/elastic-ai.creator/commit/6515ab0225bd4b55b4e8a7ad1a5e4acb2d397ea3)) - [object]

## 0.55.2 - 2023-08-12

[7642f80](7642f80fec4c7eb59ea9a3f6b403731bc1f304a4)...[882c9c3](882c9c3bf5fc1173bd7722bfc2d9581e86575828)

### Fix

- Add dummy batch dimension to meet the requirements of the batch norm ([0f499f0]( https://github.com/es-ude/elastic-ai.creator/commit/0f499f048c0606de3e14163f16e8bf049708e6f1)) - [object]

## 0.55.1 - 2023-08-11

[e1b7d90](e1b7d909bf7e496690a34a53a23276ccf4cc1d90)...[7642f80](7642f80fec4c7eb59ea9a3f6b403731bc1f304a4)

### Fix

- Fix non existing in_channels variable and remove unused import ([0e73c2d]( https://github.com/es-ude/elastic-ai.creator/commit/0e73c2dc6876772d0caba46639af77bd5ac53b62)) - [object]

## 0.55.0 - 2023-08-11

[b276d40](b276d40440bcf6f44b270d51bd44cccfe47fb1ab)...[e1b7d90](e1b7d909bf7e496690a34a53a23276ccf4cc1d90)

### Feat

- Implemented batch normed conv1d layer ([cd6836c]( https://github.com/es-ude/elastic-ai.creator/commit/cd6836cc72b3fecee1b522f9b8934fabefd46d63)) - [object]

### Fix

- Typing and errors ([af6f859]( https://github.com/es-ude/elastic-ai.creator/commit/af6f85913fd6111bcc7164a106a9cbb8d4b7b9a0)) - [object]

### Refactor

- Set alias for Design/FPLinear ([559395c]( https://github.com/es-ude/elastic-ai.creator/commit/559395cde0dca73344cd162df04fea510a621b49)) - [object]

## 0.54.0 - 2023-08-09

[85fe235](85fe235a2329c110bd3842757ef4ae14f7aa37c2)...[b276d40](b276d40440bcf6f44b270d51bd44cccfe47fb1ab)

### Fix

- Use same bit width for all rom values ([cd609e6]( https://github.com/es-ude/elastic-ai.creator/commit/cd609e65f306e62110fbdc4113f4bb330f960f19)) - [object]

### Refactor

- Rename precomputed monotonic increasing module ([ab8dfdf]( https://github.com/es-ude/elastic-ai.creator/commit/ab8dfdf4c19646ae14dd787203a380eda47c281d)) - [object]

## 0.53.0 - 2023-08-02

[2d8ecd7](2d8ecd7f02f2bcd1eb4b2e11572d5f54cf40c99e)...[85fe235](85fe235a2329c110bd3842757ef4ae14f7aa37c2)

### Feat

- Implement fixed point one dimensional convolution ([2ea9389]( https://github.com/es-ude/elastic-ai.creator/commit/2ea9389a37eac7be62e26a9727b8824b47fc2085)) - [object]

### Fix

- Fix missing parameter in tests for conv1d ([d8f8d4c]( https://github.com/es-ude/elastic-ai.creator/commit/d8f8d4c40ec1576c5dc58a38b2b80d9d4130b4fd)) - [object]

### Refactor

- Simplify string ([de8d3ec]( https://github.com/es-ude/elastic-ai.creator/commit/de8d3ec98a6105d48630a0b2e6d82f15c3e75a9e)) - [object]

## 0.52.0 - 2023-08-02

[4f690ee](4f690ee8b2209914bac8e0a7a175756b7832342f)...[2d8ecd7](2d8ecd7f02f2bcd1eb4b2e11572d5f54cf40c99e)

### Feat

- Implemeted base module for SiLU aka Swish activation function ([93b5954]( https://github.com/es-ude/elastic-ai.creator/commit/93b59544c1d164de2a9f9362f0aefe1aaae8d7d8)) - [object]
- Added nn module for Swish activation function ([b7579c9]( https://github.com/es-ude/elastic-ai.creator/commit/b7579c9f1111521064a7fd0366647da7a45e2d7a)) - [object]
- Added Swish activation function precomputed ([fd487b5]( https://github.com/es-ude/elastic-ai.creator/commit/fd487b57bb7d3e7525f935f1533f815e58f1dc0d)) - [object]

### Refactor

- Changed names of learnable parameters in the swish function ([bb2b7a8]( https://github.com/es-ude/elastic-ai.creator/commit/bb2b7a81bf6365a54590575f39a447b6cd769cd9)) - [object]
- Delete files that aren´t necessary ([53aebf3]( https://github.com/es-ude/elastic-ai.creator/commit/53aebf3702c9c3511ef81e8fd9a1fcca018bf26d)) - [object]
- Removed unnecessary file ([fe58c0f]( https://github.com/es-ude/elastic-ai.creator/commit/fe58c0f22d38284e289542b6f3e58fbff60963f9)) - [object]
- Deleted unnecessary File test_silu.py from branch ([80e8919]( https://github.com/es-ude/elastic-ai.creator/commit/80e8919078ba32dd9af0146a94bd38b63bc761b1)) - [object]

## 0.51.0 - 2023-07-28

[04bf4ba](04bf4ba8ae38c96008475c1c5ec22ffa63f94de1)...[4f690ee](4f690ee8b2209914bac8e0a7a175756b7832342f)

### Feat

- Rename custom float to float to improve readability ([794bfe7]( https://github.com/es-ude/elastic-ai.creator/commit/794bfe79a6a821050b33cc246e9e1cad09e7e682)) - [object]
- Add debug messages ([ee09864]( https://github.com/es-ude/elastic-ai.creator/commit/ee09864686d87471617aae4ae65118096d31a6ff)) - [object]
- Enable debug messages ([36ca597]( https://github.com/es-ude/elastic-ai.creator/commit/36ca597ded38bb3c5343e872ed7cf9cb09065a6f)) - [object]
- Seperate semantic release run into multiple steps ([475b425]( https://github.com/es-ude/elastic-ai.creator/commit/475b425910c4a124c34ca9a68fd5c49b4789541b)) - [object]
- Apply proposed semantic release migration procedure ([d5ea981]( https://github.com/es-ude/elastic-ai.creator/commit/d5ea981cd8852e5790c77d9667187168c34c81e3)) - [object]
- Increase semantic release version to v8.0.4 ([bb29612]( https://github.com/es-ude/elastic-ai.creator/commit/bb2961243f20ade3e7c4a142601f58fca6e9b5ad)) - [object]
- Revert changes and explicitly set semantic release version to v7 instead of v8 ([2ecf0db]( https://github.com/es-ude/elastic-ai.creator/commit/2ecf0db3c22ce034c4f36a26c96027f0229a4bf0)) - [object]

### Fix

- Remove not implemented jvp function ([0ea4834]( https://github.com/es-ude/elastic-ai.creator/commit/0ea48341c02d116dd3ef2a94e0997ce8e0641b60)) - [object]
- Try to fix semantic release ([0eab187]( https://github.com/es-ude/elastic-ai.creator/commit/0eab187389b3d435be473671d4a593ead8586e78)) - [object]

### Refactor

- Remove noise comment ([f6be240]( https://github.com/es-ude/elastic-ai.creator/commit/f6be240b03484876627f5f7de5198fd1332d6ba7)) - [object]
- Remove newline ([33fa0a9]( https://github.com/es-ude/elastic-ai.creator/commit/33fa0a932b5c2126a004b429702bcda72e696069)) - [object]

## 0.50.0 - 2023-07-11

[aad3340](aad33400ae4879477375ecb4bf507033eebca4b6)...[04bf4ba](04bf4ba8ae38c96008475c1c5ec22ffa63f94de1)

### Feat

- Implement RoundToCustomFloat autograd function ([0794a8e]( https://github.com/es-ude/elastic-ai.creator/commit/0794a8e900d6f87edc03dbd71162e7300e13b5ae)) - [object]
- Implement custom float arithmetics ([b72713e]( https://github.com/es-ude/elastic-ai.creator/commit/b72713e3db1e15957e865ed95216a2f180523114)) - [object]

### Fix

- Return wrong number of values in the backward pass ([6bcfa4e]( https://github.com/es-ude/elastic-ai.creator/commit/6bcfa4eff8d9b7c0c0461a61800ef68ef6b0cb62)) - [object]

### Refactor

- Rename FloatArithmetics to TorchArithmetics ([5cd7a3b]( https://github.com/es-ude/elastic-ai.creator/commit/5cd7a3b6913b14456f524a9486bac2c42dc72412)) - [object]
- Rename CustomFloatArithmetics to FloatArithmetics ([824b029]( https://github.com/es-ude/elastic-ai.creator/commit/824b029971789c951a243937d942d7597225e829)) - [object]

## 0.49.0 - 2023-07-01

[1211442](1211442c4b3de0dd47a2491530fd70f14a02fe3d)...[aad3340](aad33400ae4879477375ecb4bf507033eebca4b6)

### Docs

- Complete table of contents ([cf0ef63]( https://github.com/es-ude/elastic-ai.creator/commit/cf0ef63eb628521f14406fb7d59cee53c71c8d60)) - [object]
- Add minimal example that demonstrates the usage of the creator ([64030f2]( https://github.com/es-ude/elastic-ai.creator/commit/64030f2eb129ff8275022ab0b8bf4945d42626a8)) - [object]

### Feat

- Update readme and add small improvements ([8f2bbd0]( https://github.com/es-ude/elastic-ai.creator/commit/8f2bbd093e18c15421abab20ecb0f9afbc6d12a1)) - [object]

## 0.48.1 - 2023-06-24

[f9a851e](f9a851eea1a3c6b458b2432ba83391aa3fc3ed48)...[1211442](1211442c4b3de0dd47a2491530fd70f14a02fe3d)

### Fix

- Only create coverage reports in PR ([1bd728f]( https://github.com/es-ude/elastic-ai.creator/commit/1bd728f4e8edb6595a35dafd71c5d68263a7358f)) - [object]

## 0.48.0 - 2023-06-24

[da12e09](da12e09be6b0198577720191340266c94244cf39)...[f9a851e](f9a851eea1a3c6b458b2432ba83391aa3fc3ed48)

### Feat

- Use binary values instead of hex values to fill the rom template ([af56c02]( https://github.com/es-ude/elastic-ai.creator/commit/af56c02da42433c2db1a9a2a6ddb3705d213d765)) - [object]
- Add pytest-cov dependency ([a737729]( https://github.com/es-ude/elastic-ai.creator/commit/a7377290ffee7359f6f8c0392960d7038fe2a73b)) - [object]
- Add coverage workflow to create reports ([3f6caca]( https://github.com/es-ude/elastic-ai.creator/commit/3f6caca6a626923ec3d8078320fa9b70092495ee)) - [object]
- Only trigger coverage report when pushing to main ([b4b23c9]( https://github.com/es-ude/elastic-ai.creator/commit/b4b23c988803165895c14a8357427a3069f09233)) - [object]

### Fix

- Use poetry run to run pytest ([7058e42]( https://github.com/es-ude/elastic-ai.creator/commit/7058e42cc7fa0849841578f2bafd6a3fc6155f2a)) - [object]

### Refactor

- Improve readability ([e4de568]( https://github.com/es-ude/elastic-ai.creator/commit/e4de5682419829675a92ff95f8e853dc28cf181e)) - [object]
- Remove unused to_vhdl_hex_string function ([24ccbf1]( https://github.com/es-ude/elastic-ai.creator/commit/24ccbf1a1d9ff3d270faba19581a6f72eadb751e)) - [object]

## 0.47.2 - 2023-06-23

[cc4279c](cc4279cfdff992551fb274818964c74e216c64ec)...[da12e09](da12e09be6b0198577720191340266c94244cf39)

### Fix

- Fix error when passing a cuda tensor to the IdentityStepFunction ([7f49617]( https://github.com/es-ude/elastic-ai.creator/commit/7f496171a547bae17c69976c35d437428022447f)) - [object]

## 0.47.1 - 2023-06-16

[f0e1eee](f0e1eeefa6f972574a9324885f00af052924f295)...[cc4279c](cc4279cfdff992551fb274818964c74e216c64ec)

### Chore

- Add do_not_commit path to prevent files from being committed by mistake ([af13e16]( https://github.com/es-ude/elastic-ai.creator/commit/af13e1687f57fc3545d0c114263ed439b78973cd)) - [object]

### Fix

- Remove wrongly committed files ([4fdea0c]( https://github.com/es-ude/elastic-ai.creator/commit/4fdea0c9ff2db5e8af3f208bbd83d995332d5b85)) - [object]

### Refactor

- Merge fp quant and fp dequant into a roundtofixedpoint autograd function ([b986a62]( https://github.com/es-ude/elastic-ai.creator/commit/b986a62ea7a0a58e6479aa5082ddd2de11ed27d7)) - [object]

## 0.47.0 - 2023-06-16

[f029667](f029667c2c9f61acacb0638f1a9bc85177c9e553)...[f0e1eee](f0e1eeefa6f972574a9324885f00af052924f295)

### Feat

- Simplify project structure ([81cbcb3]( https://github.com/es-ude/elastic-ai.creator/commit/81cbcb343b26473290609c7715051059127a924b)) - [object]

### Refactor

- Remove unused manifest module ([55f8e6d]( https://github.com/es-ude/elastic-ai.creator/commit/55f8e6deac74d953b97b031a22e0dd9a73ecf20c)) - [object]

### Style

- Beautify commit # [7beebdb](https://github.com/es-ude/elastic-ai.creator/commit/7beebdbc67074dc6f8e8a0320563385ee49a7915) ([1c7fead]( https://github.com/es-ude/elastic-ai.creator/commit/1c7feadd0825be6648702c7ecffcdb1c2ce974f5)) - [object]

## 0.46.1 - 2023-06-13

[dd9a1b9](dd9a1b95a9a7a1a3b5f0074c18ed9c3d8aec3e73)...[f029667](f029667c2c9f61acacb0638f1a9bc85177c9e553)

### Fix

- Fix wrong port definitions ([9a4c8af]( https://github.com/es-ude/elastic-ai.creator/commit/9a4c8af6f8f8be2bf6fff49c25fc0ca12cbea45a)) - [object]

## 0.46.0 - 2023-06-13

[8e6c6fb](8e6c6fbbae0e5b3d32eecc95a0877a5c288218fe)...[dd9a1b9](dd9a1b95a9a7a1a3b5f0074c18ed9c3d8aec3e73)

### Feat

- Add the ability to sum over dimension ([c45c0e6]( https://github.com/es-ude/elastic-ai.creator/commit/c45c0e676e1df70bf99c4c943874168781ef2a93)) - [object]
- Test that conv1d uses different arithmetics ([7eb01db]( https://github.com/es-ude/elastic-ai.creator/commit/7eb01dbaa2afbbb02410e6fc6272ba02fec7878a)) - [object]
- Add conv1d function to arithmetics ([1cab190]( https://github.com/es-ude/elastic-ai.creator/commit/1cab1901e324eb100f1cbccf6d54fae429210b33)) - [object]
- Use conv1d arithmetics function to implement conv1d module ([69778be]( https://github.com/es-ude/elastic-ai.creator/commit/69778be7fd1becab2ad5099ebb8d64d4a0db0de5)) - [object]

### Fix

- Fix some syntax errors ([3997bbd]( https://github.com/es-ude/elastic-ai.creator/commit/3997bbdb134a94defd4e32ad1a2eb3aa236d6b96)) - [object]
- Quantize weights before inference ([61153e6]( https://github.com/es-ude/elastic-ai.creator/commit/61153e60d6854bacf0bd2501d96efc3f6e62714e)) - [object]

### Refactor

- Remove debug print call ([f85172e]( https://github.com/es-ude/elastic-ai.creator/commit/f85172ebddfceb98a7c661cd3f57db60b19b61c0)) - [object]
- Improve readablility ([004c736]( https://github.com/es-ude/elastic-ai.creator/commit/004c736cab22b4e8eed5eb867c203b4b62e7e235)) - [object]
- Remove redundant tests ([c828d53]( https://github.com/es-ude/elastic-ai.creator/commit/c828d536110205b3e00f61a33e31d0cae1eaee6f)) - [object]

## 0.45.0 - 2023-06-10

[6839439](6839439f318cd10e304a625ba3120c81dc292be7)...[8e6c6fb](8e6c6fbbae0e5b3d32eecc95a0877a5c288218fe)

### Chore

- Update dependencies ([a2558b5]( https://github.com/es-ude/elastic-ai.creator/commit/a2558b5649d5416c730cfdfebdd4d38ce48a6a88)) - [object]

### Feat

- Simplify usage for the elasticai.creator.nn.vhdl package by adding layers to __init__ ([2c7c968]( https://github.com/es-ude/elastic-ai.creator/commit/2c7c96858ec9d935389a960baee46e8c506f9b5c)) - [object]

### Fix

- Fix broken import in base template generator and move it with its template to own folder ([9eb1f70]( https://github.com/es-ude/elastic-ai.creator/commit/9eb1f70cff10e075712d5bf7e3fc9fcfed2aae19)) - [object]

### Refactor

- Remove unused template resources ([d58f267]( https://github.com/es-ude/elastic-ai.creator/commit/d58f267772839df6c254b9d749b8e5653b9a20e1)) - [object]
- Rename sequential layer module according to our convention ([ae1da5e]( https://github.com/es-ude/elastic-ai.creator/commit/ae1da5e5aced255e38f0c13691a1d42f90dd5cb3)) - [object]
- Remove unused and redundant port definition ([b376b75]( https://github.com/es-ude/elastic-ai.creator/commit/b376b757f6dd0e6400813688a2dfdf6ca392a6f9)) - [object]
- Rename template and remove some newlines ([707310b]( https://github.com/es-ude/elastic-ai.creator/commit/707310b3202ec1b48f847a228455f8cd77436219)) - [object]
- Remove some newlines, use create_port function and fix wrong template ([1bc4a70]( https://github.com/es-ude/elastic-ai.creator/commit/1bc4a70f173c9f380a76438842ba6708d1659aad)) - [object]
- Transform bdd test to pytest test ([475ec7b]( https://github.com/es-ude/elastic-ai.creator/commit/475ec7bd12ed0f43b65438a2ef62aa97d3ca8b14)) - [object]
- Remove unused pytest-bdd dependency ([e9203a0]( https://github.com/es-ude/elastic-ai.creator/commit/e9203a0223ef3adfcbd40af841e569438684e1c8)) - [object]
- Rename monotonously increasing scalar function ([baff8b2]( https://github.com/es-ude/elastic-ai.creator/commit/baff8b2fd8569c60b51906458c2d541e1371f111)) - [object]
- Better separation of designs and modules ([44f22ae]( https://github.com/es-ude/elastic-ai.creator/commit/44f22ae25a02c0c4810e64c970cdc5dd28135c89)) - [object]
- Create rom design folder ([9e40f5f]( https://github.com/es-ude/elastic-ai.creator/commit/9e40f5fa9b40c2542e4ef99cf02d1b6004ad2a60)) - [object]
- Remove deprecated documentation ([349a9f8]( https://github.com/es-ude/elastic-ai.creator/commit/349a9f866e001ce0494f9876d894ef0c5833817d)) - [object]
- Remove unused base signal definition ([14dc275]( https://github.com/es-ude/elastic-ai.creator/commit/14dc275beea7f3c757433eb9b3872c895fc6fca3)) - [object]
- Rename ports module to port_definitions ([b5a64b8]( https://github.com/es-ude/elastic-ai.creator/commit/b5a64b812145a34dd1dd0d20cb2ca31f18804a1f)) - [object]
- Use container types from collections.abc instead of typing because they are deprecated ([7a45e67]( https://github.com/es-ude/elastic-ai.creator/commit/7a45e672cdcc47b426a57a8297febc8aa9664744)) - [object]
- Remove unused imports ([e8881d3]( https://github.com/es-ude/elastic-ai.creator/commit/e8881d31e11d3e27489322deabb3c29d420e568b)) - [object]
- Use Identity class from base_modules instead of torch ([8f179f0]( https://github.com/es-ude/elastic-ai.creator/commit/8f179f0bbbb2510b294665e0502715b6b69346c8)) - [object]

## 0.44.0 - 2023-06-09

[d317c39](d317c390c7394785626886970de8f6dd3b37f470)...[6839439](6839439f318cd10e304a625ba3120c81dc292be7)

### Fix

- Port def and impl of monotonous function design ([2d423d4]( https://github.com/es-ude/elastic-ai.creator/commit/2d423d46faa86fbf43cb8ba1d01aafe92c5bfa23)) - [object]
- Use new Sequential constructor ([6bb111b]( https://github.com/es-ude/elastic-ai.creator/commit/6bb111b748567502c23a48a52d7e477645969996)) - [object]

### Refactor

- Cleanup imports ([c402a03]( https://github.com/es-ude/elastic-ai.creator/commit/c402a031f5996c6f7a1b3a5199e1cf9697e7dc5a)) - [object]

### Style

- Beautify commit # [95ca255](https://github.com/es-ude/elastic-ai.creator/commit/95ca25571e9757d932a45749e9cf92531c13ab36) ([cdf44ce]( https://github.com/es-ude/elastic-ai.creator/commit/cdf44cec1a9a656ce6b3a9d19a717a9e7163d1b6)) - [object]

## 0.43.0 - 2023-06-09

[d37ed1c](d37ed1ca6c6e0a49c7c25ca11e06f9a9566fb70a)...[d317c39](d317c390c7394785626886970de8f6dd3b37f470)

### Feat

- Introduce FPMonotonouslyIncreasingModule to easily add new activations ([b78c922]( https://github.com/es-ude/elastic-ai.creator/commit/b78c9225f7f70ec329bee5705c11d9e7b1392c41)) - [object]
- Add tests for the FPMonotonouslyIncreasingModule ([9ba64ae]( https://github.com/es-ude/elastic-ai.creator/commit/9ba64ae3d253db76a6368c5e561ce28bcec2aab5)) - [object]

### Fix

- Increase default sampling intervall ([07620d3]( https://github.com/es-ude/elastic-ai.creator/commit/07620d3e2ee9db1bc6aa081a15274cb79b5ee4b0)) - [object]
- Use elsif in lookup table ([f375ba3]( https://github.com/es-ude/elastic-ai.creator/commit/f375ba3784bf92887e689f77f592dfc2fa2c7e2c)) - [object]
- Set correct signal names for x and y address ([5354a2a]( https://github.com/es-ude/elastic-ai.creator/commit/5354a2a0e85bc0788f5d74377c1a685e9d0e0de7)) - [object]

### Refactor

- Move all arithmetics to arithmetics folder in base_modules ([de0fd46]( https://github.com/es-ude/elastic-ai.creator/commit/de0fd460eae7d7d155188d2e73dd4cc82b913718)) - [object]
- Remove unnecessary tests ([c0756b3]( https://github.com/es-ude/elastic-ai.creator/commit/c0756b3d7a7468aa0e3d7c55e126170790bae076)) - [object]

## 0.42.0 - 2023-06-08

[59e48f1](59e48f1e0b609088fc7aa43479cfa5f7a785c9a8)...[d37ed1c](d37ed1ca6c6e0a49c7c25ca11e06f9a9566fb70a)

### Feat

- Make sure that inplace parameter is fixed defined ([79b7a1e]( https://github.com/es-ude/elastic-ai.creator/commit/79b7a1eea0cb71f5a838cfebf02970927410f594)) - [object]
- Add working hardsigmoid implementation ([db03ff0]( https://github.com/es-ude/elastic-ai.creator/commit/db03ff080f878c9b9fe54303ead97c673022f3a1)) - [object]
- Reimplement hard tanh activation function ([9b86f9d]( https://github.com/es-ude/elastic-ai.creator/commit/9b86f9d440cc991d624a6f3492a3caf7419bdbf3)) - [object]

## 0.41.0 - 2023-06-08

[1800a76](1800a76e78e9289eeb9741da75a367744c0fa4bb)...[59e48f1](59e48f1e0b609088fc7aa43479cfa5f7a785c9a8)

### Feat

- Add fixed point ReLU module ([62c1555]( https://github.com/es-ude/elastic-ai.creator/commit/62c15557fc515c89644c674aef9fc39d22ab672f)) - [object]

## 0.40.0 - 2023-06-04

[7b9b42d](7b9b42dc6eb1414666dcdc32b954a8a9acd6bc02)...[1800a76](1800a76e78e9289eeb9741da75a367744c0fa4bb)

### Feat

- Add a function to easily compare tensors with pytest ([24e737e]( https://github.com/es-ude/elastic-ai.creator/commit/24e737eaea48044df3e8addaca0d1cc804a3b6f4)) - [object]
- Implement autograd fn to map inputs to a subset of inputs ([26c6ec7]( https://github.com/es-ude/elastic-ai.creator/commit/26c6ec7a203eea4fed4c3eb3d5c3e4893acb545f)) - [object]
- Rename autograd function and pass step lut to autograd function ([d607e98]( https://github.com/es-ude/elastic-ai.creator/commit/d607e98bd14dfa1ae23e9726b2046baaede21361)) - [object]
- Pass step lut to identity step function and improve readablility ([c1b6747]( https://github.com/es-ude/elastic-ai.creator/commit/c1b67473c33ddc27590068472dcff6969f9e7135)) - [object]
- Implement bufferless component interface for precomputed scalar function ([f701a57]( https://github.com/es-ude/elastic-ai.creator/commit/f701a57db54e0d5f3e5e43047725b28646cb5f15)) - [object]
- Add quantized tanh implementation with lookup tables ([3a1fb10]( https://github.com/es-ude/elastic-ai.creator/commit/3a1fb10944e566ca33e3e745b939b6700421fdb9)) - [object]
- Improve performance of the identity step autograd function ([46f036c]( https://github.com/es-ude/elastic-ai.creator/commit/46f036c8fb2d007d21e32214ac92d4d9aa2fe9d1)) - [object]
- Simplify the use of the sequential layer (same as in torch) ([9fad15d]( https://github.com/es-ude/elastic-ai.creator/commit/9fad15d774f3573fb26f168295f9bd2ae5cdd046)) - [object]

### Fix

- Fix missing creation of a subpath in the save_to function ([2a4dbdf]( https://github.com/es-ude/elastic-ai.creator/commit/2a4dbdf2f6fce4de567281002dd4640ff3ae54ed)) - [object]
- Fix that last io pair was dropped when calling save_to function ([2bc46ac]( https://github.com/es-ude/elastic-ai.creator/commit/2bc46ac9c535b65ef7a3dc5cbe12b27d253c3b37)) - [object]

### Refactor

- Move torch dependency to base_moduels ([06d1aca]( https://github.com/es-ude/elastic-ai.creator/commit/06d1aca6e3ca95a1e371253aa97dee831119250c)) - [object]
- Remove unused base modules ([97d1e7d]( https://github.com/es-ude/elastic-ai.creator/commit/97d1e7dbc181fc03562ccbcde976eb9e661c381e)) - [object]
- Small change of the folder structure ([58783a8]( https://github.com/es-ude/elastic-ai.creator/commit/58783a83a891d85c50c43a6af2ac3efa3e634657)) - [object]
- Remove unnecessary tests ([23f78db]( https://github.com/es-ude/elastic-ai.creator/commit/23f78db7aec7efeef669a32ebe76ea3ebcb6b133)) - [object]
- Remove default sampling intervall ([9d7caea]( https://github.com/es-ude/elastic-ai.creator/commit/9d7caeae98408d2eaf0c97032dae0b5b4b312429)) - [object]
- Remove unused import ([4de2055]( https://github.com/es-ude/elastic-ai.creator/commit/4de205551938c7a284af78b5c2c418fdf95358f6)) - [object]
- Change indentations ([d5f5bf0]( https://github.com/es-ude/elastic-ai.creator/commit/d5f5bf07b85d7b1902d474975da58d29bc615f6d)) - [object]
- Remove the leading underscore of the class name ([6643bf1]( https://github.com/es-ude/elastic-ai.creator/commit/6643bf13dfbe50f7b98c0a49a238041c49fa8b89)) - [object]

## 0.39.0 - 2023-05-19

[f06c144](f06c1440f8e45535b9d1a56142356a9acea31bfd)...[7b9b42d](7b9b42dc6eb1414666dcdc32b954a8a9acd6bc02)

### Feat

- Make precomputed scalar functions bufferless ([89986fa]( https://github.com/es-ude/elastic-ai.creator/commit/89986fad041c89d0543fe9a22946e5f5f49e2b61)) - [object]
- Port expansion/template based on autowiring protocol ([0d14618]( https://github.com/es-ude/elastic-ai.creator/commit/0d146181c8b789b09871af43654ca2d83ea55ddb)) - [object]
- Add basic vhdl parsing ([5df2a3f]( https://github.com/es-ude/elastic-ai.creator/commit/5df2a3ff4e9ba7ec33398a267cd983ad886d1fe7)) - [object]
- Add standalone parser module ([5a9b141]( https://github.com/es-ude/elastic-ai.creator/commit/5a9b141285fefecf61f581417061428cda382ad5)) - [object]
- Support parsing partial files ([8170012]( https://github.com/es-ude/elastic-ai.creator/commit/817001208b774e57cfb27fb4d4ee9d704541c9f8)) - [object]
- Add intermediate symbols to rule definitions ([624b310]( https://github.com/es-ude/elastic-ai.creator/commit/624b310fc9beb130902fdf3269e3f30714fe0c3f)) - [object]
- Add AutoWirer ([f4159c8]( https://github.com/es-ude/elastic-ai.creator/commit/f4159c800fe54cc0fe73fbebdf2ac0410ddac635)) - [object]
- Check for autowiring protocol violation ([3f17e00]( https://github.com/es-ude/elastic-ai.creator/commit/3f17e002e050dc92516e4ff5468041f06ebd6760)) - [object]
- Add experimental precomputed tanh in fixed point ([0e76d03]( https://github.com/es-ude/elastic-ai.creator/commit/0e76d03b6d0f23d8932b94bb7728cbeea2de0289)) - [object]
- Implement batch normed linear layer ([9322f6f]( https://github.com/es-ude/elastic-ai.creator/commit/9322f6f699f9884273c3f9815b9a026c9f7840ae)) - [object]

### Fix

- Correct tuple type annotation ([f0e7da0]( https://github.com/es-ude/elastic-ai.creator/commit/f0e7da0cf186015004970102f2b9b57a9f839585)) - [object]
- Adjust tests to follow previous change ([c328bd5]( https://github.com/es-ude/elastic-ai.creator/commit/c328bd565d6ba84a9d1fab788051c3e884ea2094)) - [object]
- Remove obsolete parsing functionality ([7f85d05]( https://github.com/es-ude/elastic-ai.creator/commit/7f85d05aa3da2e0fd7c266bfc9c1aad573adecc4)) - [object]
- Children of sequential layer determine signal widths ([3dd5c0c]( https://github.com/es-ude/elastic-ai.creator/commit/3dd5c0cc4f7a52c7b3a86cec437005b86aa0a509)) - [object]
- Remove dequantize ([c111022]( https://github.com/es-ude/elastic-ai.creator/commit/c111022854ce6965b705b3a3de296e032d7ff107)) - [object]
- Allow to set affine and bias equals false in translate function ([b351284]( https://github.com/es-ude/elastic-ai.creator/commit/b351284335a77caec838a8f4ea57684e429cc35b)) - [object]

### Refactor

- Remove obsolete module ([5adc999]( https://github.com/es-ude/elastic-ai.creator/commit/5adc999c3f4fb5a45e569680fa466694127688da)) - [object]
- Make identity layer/design names more specific ([0aed47e]( https://github.com/es-ude/elastic-ai.creator/commit/0aed47ebd3dbd784156a949822b8fc7c117e07c0)) - [object]
- Remove obsolete test helper code ([17e4e12]( https://github.com/es-ude/elastic-ai.creator/commit/17e4e1250c1b94b3f72ac9dba57f7ee66825f381)) - [object]
- Pull up tokenize functions ([ace6f1e]( https://github.com/es-ude/elastic-ai.creator/commit/ace6f1eb5d0162d7454d56a5baf6f3fb59f3dc06)) - [object]
- Pull up parse function ([1b8f187]( https://github.com/es-ude/elastic-ai.creator/commit/1b8f1874eff63130e71c1754257d5bb3d05bb827)) - [object]
- Move sequential layer to nn.vhdl ([caea325]( https://github.com/es-ude/elastic-ai.creator/commit/caea325588f8c87cc28d5df248129b0e73111e3d)) - [object]
- Move binarize autograd function to autograd_functions folder ([03d5bc8]( https://github.com/es-ude/elastic-ai.creator/commit/03d5bc86462b36be30c2887593360ec48a908ab1)) - [object]
- Rename FPLinear1d design to FPLinear ([238f167]( https://github.com/es-ude/elastic-ai.creator/commit/238f1671a28b9b5735ca7e01360d4dda7122a2a7)) - [object]
- Remove redundant quantize function ([02094cf]( https://github.com/es-ude/elastic-ai.creator/commit/02094cf412f2846821c9c2925bedcdc585fe8a8d)) - [object]

## 0.38.0 - 2023-05-09

[b752226](b752226a16f65806bad09e88a2f65f7fffe43168)...[f06c144](f06c1440f8e45535b9d1a56142356a9acea31bfd)

### Chore

- Remove unused workflow ([dd08e08]( https://github.com/es-ude/elastic-ai.creator/commit/dd08e08b0af74c4d7ba927c892de6081717657db)) - [object]

### Feat

- Write function of InMemoryFile and OnDiskFile now takes Template object ([a867ea1]( https://github.com/es-ude/elastic-ai.creator/commit/a867ea15980b8ca1390327f2999c4d7b91ef3041)) - [object]
- Add function to get all unfilled variables of a template ([d635cb6]( https://github.com/es-ude/elastic-ai.creator/commit/d635cb6098735b451aea259a8a6f15619bfcd64f)) - [object]
- Add check that all variables are filled when saving a template ([c988d2b]( https://github.com/es-ude/elastic-ai.creator/commit/c988d2bc203790ba8ab900e8a2de6996b22d6fcb)) - [object]

### Fix

- Fix not inserted process name ([dbabea0]( https://github.com/es-ude/elastic-ai.creator/commit/dbabea07c888a5309d9ca55cd2c01ae0debea57d)) - [object]
- Add variable ([229d452]( https://github.com/es-ude/elastic-ai.creator/commit/229d452d0c2f798ee1dd0124f50be8f01d69ede4)) - [object]
- Remove broken lstm implementation ([c524ca2]( https://github.com/es-ude/elastic-ai.creator/commit/c524ca20cc49333007c4e0bbfa167912580e5c01)) - [object]

### Refactor

- Temporarily rename template class ([6fb83a2]( https://github.com/es-ude/elastic-ai.creator/commit/6fb83a2d773bb474bf96f4c248de8537f91673aa)) - [object]
- Rename TemplateConfig protocol to Template ([33d01ee]( https://github.com/es-ude/elastic-ai.creator/commit/33d01eef31e7c9cb919a9684150dfba8ce1c60a5)) - [object]
- Remove InProjectVHDLTemplate and InMemoryVHDLTemplate ([e625399]( https://github.com/es-ude/elastic-ai.creator/commit/e6253997447b0976de4ed60ec671de80ec6740a6)) - [object]
- Remove RawTemplate class ([eb91cd8]( https://github.com/es-ude/elastic-ai.creator/commit/eb91cd81475a6a9aa94fc8ab4ccf3457cef55d01)) - [object]
- Remove deprecated and broken relu and tanh implementations ([286686c]( https://github.com/es-ude/elastic-ai.creator/commit/286686cd6a2a185a94c03585f41d15dea794b1a2)) - [object]

## 0.37.2 - 2023-05-07

[4797aaa](4797aaa00b066104eb17dc8977bd6f47ee112396)...[b752226](b752226a16f65806bad09e88a2f65f7fffe43168)

### Fix

- Try manual publishing ([c8b6c35]( https://github.com/es-ude/elastic-ai.creator/commit/c8b6c355896c1f3b0630c227af8414f281b5d3ff)) - [object]

## 0.37.1 - 2023-05-07

[7fe6bba](7fe6bbaf37ee509a703ed4eeb446206eb1e3024c)...[4797aaa](4797aaa00b066104eb17dc8977bd6f47ee112396)

### Fix

- Try to fix semantic release ([2625e89]( https://github.com/es-ude/elastic-ai.creator/commit/2625e8982c021cbf5b778e95194cc53170ab0afb)) - [object]

## 0.37.0 - 2023-05-05

[f2707c4](f2707c47f99a1f9d11addbb3fa966054ed1a0b8f)...[7fe6bba](7fe6bbaf37ee509a703ed4eeb446206eb1e3024c)

### Chore

- Add  force-publish workflow ([b59268d]( https://github.com/es-ude/elastic-ai.creator/commit/b59268d15b8ef605c6dbb48e606f5b1ad746548f)) - [object]
- Update force publish workflow ([9a0a7ac]( https://github.com/es-ude/elastic-ai.creator/commit/9a0a7aca438f92e728c0310ec16adb0ded902f29)) - [object]
- Update force-publish workflow ([c7b011c]( https://github.com/es-ude/elastic-ai.creator/commit/c7b011cd289baa1615cde11224f2a0ec25221e15)) - [object]

### Feat

- Assert that all inserted variables exists in template and remove AbstractBaseTemplate ([51f1a08]( https://github.com/es-ude/elastic-ai.creator/commit/51f1a0883a8d0a54caee66080ef85f84049ad806)) - [object]

### Refactor

- Remove unused parameter ([89ca654]( https://github.com/es-ude/elastic-ai.creator/commit/89ca65467a983230a1dc54d8b1502e82185f2acc)) - [object]
- Remove duplicated test ([cfd304e]( https://github.com/es-ude/elastic-ai.creator/commit/cfd304e630ba4f13ee87fc074c7d05fd99b1c98a)) - [object]

## 0.36.0 - 2023-04-26

[2c91d42](2c91d4293eee0aab1af6b2c796936db8e0d93807)...[f2707c4](f2707c47f99a1f9d11addbb3fa966054ed1a0b8f)

### Chore

- Adjust main.yml ([93550cc]( https://github.com/es-ude/elastic-ai.creator/commit/93550cccd7eda401dc7f759da8efe048661c2573)) - [object]

### Feat

- Introduce abstract Translatable class ([5d9fa2d]( https://github.com/es-ude/elastic-ai.creator/commit/5d9fa2d167a8c46c301bb4a0da25718b1fcf0dee)) - [object]
- Sequential layer can have a name ([9e46938]( https://github.com/es-ude/elastic-ai.creator/commit/9e46938e9e5fc6960e70bef26aa72ec51566a007)) - [object]
- Test all subdesigns generated by sequential layer gets a unique name ([009405b]( https://github.com/es-ude/elastic-ai.creator/commit/009405bc64cd5e8a86909330bb450ee58ee98289)) - [object]
- Test signal definitions, layer connections and instantiations separately ([65201c8]( https://github.com/es-ude/elastic-ai.creator/commit/65201c83bae07c62efcd705f67f34d9ff88da557)) - [object]
- Add tests for sequential model with two layer ([df73a4f]( https://github.com/es-ude/elastic-ai.creator/commit/df73a4fb27a8867a4b633c4ffdd737ead34d2f16)) - [object]
- Autogenerate sequential signal connections ([6dfca07]( https://github.com/es-ude/elastic-ai.creator/commit/6dfca078b735a3387b65c20de601426ea27371c6)) - [object]

### Fix

- Add missing save_to function ([ef24ee2]( https://github.com/es-ude/elastic-ai.creator/commit/ef24ee21672099359867bc4a74f5804af0c10158)) - [object]
- Fix syntax errors ([f9b57e4]( https://github.com/es-ude/elastic-ai.creator/commit/f9b57e4f8173dc0bd52c21b1da351304ceb5a122)) - [object]
- Fix syntax error ([396f5c4]( https://github.com/es-ude/elastic-ai.creator/commit/396f5c45b382454d6cc97e4be573fcfe45a4592a)) - [object]
- Fix that test ignores parameter ([f448919]( https://github.com/es-ude/elastic-ai.creator/commit/f448919ff4882696c0991d6aec3608616e258596)) - [object]
- Correct expected connections ([2fb0f8e]( https://github.com/es-ude/elastic-ai.creator/commit/2fb0f8edc45a7a38e2a9b7433dee90f139b10006)) - [object]

### Refactor

- Remove unused translatable protocol and rename module ([9d59f8c]( https://github.com/es-ude/elastic-ai.creator/commit/9d59f8cd533b32baf6f90365e0db5a8b18d1c5a7)) - [object]
- Remove unused import ([602c137]( https://github.com/es-ude/elastic-ai.creator/commit/602c1376cefe7dc4a95ef7cf04b9f67b0e2cf1e3)) - [object]
- Fix/add missing type annotations ([d47a8c1]( https://github.com/es-ude/elastic-ai.creator/commit/d47a8c1c8919066e557a702f3bccc3928f35fa69)) - [object]
- Use identity instead of linear layer to simplify test ([28a75c3]( https://github.com/es-ude/elastic-ai.creator/commit/28a75c337734b6bed887b1a3f9fc0369d92d330b)) - [object]
- Rename FPLinear1d to FPLinear ([5550dd9]( https://github.com/es-ude/elastic-ai.creator/commit/5550dd97956171f53edc59e534dd02161c463133)) - [object]
- Reduce code duplication ([ae65808]( https://github.com/es-ude/elastic-ai.creator/commit/ae65808bc66ebd2982a80ec3b6c5d70f749723d8)) - [object]

## 0.35.0 - 2023-04-17

[1dd7d74](1dd7d74f0d4795ca6141da7b9b0957298b2a604d)...[2c91d42](2c91d4293eee0aab1af6b2c796936db8e0d93807)

### Chore

- Move to python3.11 ([389e4ec]( https://github.com/es-ude/elastic-ai.creator/commit/389e4ec6d60dbf594026993bf8f7d94d4bea1da8)) - [object]
- Upgrade to python3.11 ([f39c779]( https://github.com/es-ude/elastic-ai.creator/commit/f39c7798f4ccc3799c707c8dcefbd176f9b6813b)) - [object]

### Feat

- Add translate_to_vhdl function ([ba0edc2]( https://github.com/es-ude/elastic-ai.creator/commit/ba0edc25b93075cbb2d104c2216dcc15df36c13c)) - [object]
- Implement translatable identity module ([54327fa]( https://github.com/es-ude/elastic-ai.creator/commit/54327fa3e45ca3617d642134ca8d842e7d2afc4c)) - [object]
- Add indentations to template ([aa254d1]( https://github.com/es-ude/elastic-ai.creator/commit/aa254d12f38712e798db9b31a5a58e197a44121a)) - [object]
- Generate first base template ([a65d72e]( https://github.com/es-ude/elastic-ai.creator/commit/a65d72ea1ad2dd87a0443b56711d11ce321d14b6)) - [object]
- Generate template from manifest.toml ([51276a0]( https://github.com/es-ude/elastic-ai.creator/commit/51276a01de5ff37bedc598f5c758e3dc681aa49c)) - [object]
- Use fixed base template ([432dfd9]( https://github.com/es-ude/elastic-ai.creator/commit/432dfd9518a0a33a7ba08cf95436f9472b274b52)) - [object]

### Fix

- Set correct resource options in rom and fix signal definitions ([2c2964c]( https://github.com/es-ude/elastic-ai.creator/commit/2c2964ceaa746163ebbeaef09181e09c06ecb4f2)) - [object]
- Fix tests and remove hard sigmoid test in sequential test case ([a1ada6f]( https://github.com/es-ude/elastic-ai.creator/commit/a1ada6f0ceec750bb80abf866d28f96719f2f1f9)) - [object]

### Refactor

- Remove unused imports ([d9592ec]( https://github.com/es-ude/elastic-ai.creator/commit/d9592ecb3677ba8050cb737bbc112987e72f25b5)) - [object]
- Remove superfluous module protocols ([4e25dc6]( https://github.com/es-ude/elastic-ai.creator/commit/4e25dc65dfa0c226c298f5e589a6c887d72a3c19)) - [object]

## 0.34.0 - 2023-04-06

[165f434](165f4343832b770386b1c98649aa69878bbaaf33)...[1dd7d74](1dd7d74f0d4795ca6141da7b9b0957298b2a604d)

### Chore

- Remove unneeded import ([e3df52a]( https://github.com/es-ude/elastic-ai.creator/commit/e3df52a091e4673460f7b1ad733d766bad4afd02)) - [object]
- Add mypy and pylint to pyproject.toml ([aad5549]( https://github.com/es-ude/elastic-ai.creator/commit/aad5549c7bbfbaf648fc3bbab0f77cd6c0ad49ca)) - [object]

### Feat

- Binary_arithmetics ([54e38d5]( https://github.com/es-ude/elastic-ai.creator/commit/54e38d57f27db2d8d0baff5fee3c35a91e26ecd9)) - [object]
- Make precomputed scalar functions use unified interface ([6b59da5]( https://github.com/es-ude/elastic-ai.creator/commit/6b59da53a896db7676119de2f74129bcc47287ed)) - [object]

### Fix

- Correct import paths ([169f868]( https://github.com/es-ude/elastic-ai.creator/commit/169f8686108845702f01482170df53e3fabbfe8b)) - [object]

## 0.33.3 - 2023-04-06

[463ff20](463ff20af4c7f5aa97c1cb9453713804e6ddef2b)...[165f434](165f4343832b770386b1c98649aa69878bbaaf33)

### Docs

- Remove deprecated documentation ([11b9945]( https://github.com/es-ude/elastic-ai.creator/commit/11b9945bf3b6bf96899a09751963a93eb98d846d)) - [object]

### Fix

- Set correct rom names ([9570826]( https://github.com/es-ude/elastic-ai.creator/commit/95708269900ca99b79da9ba37078f593724e5d17)) - [object]
- Remove DualPort2ClockRam design ([f9224c6]( https://github.com/es-ude/elastic-ai.creator/commit/f9224c6809b3a6f72bfe0405419de494b099b17c)) - [object]

### Refactor

- Rename nn to base_modules ([44207a8]( https://github.com/es-ude/elastic-ai.creator/commit/44207a8f72e426fcd1cb4acc5b3c53c4ac8fa2f2)) - [object]
- Rename translatable_modules to nn ([333ac57]( https://github.com/es-ude/elastic-ai.creator/commit/333ac5776788367ed3a8c17632fa20e11556f43e)) - [object]
- Move hardware specific lstm parts to nn package ([bfe575c]( https://github.com/es-ude/elastic-ai.creator/commit/bfe575c50291388eb2f8b243d3411ff9e847490c)) - [object]
- Reorder class definitions to avoid the usage of quotes ([780c1fe]( https://github.com/es-ude/elastic-ai.creator/commit/780c1fe67d18893400226e8acc6e77504da6a6ad)) - [object]
- Move lstm designs in designs directory ([36a807b]( https://github.com/es-ude/elastic-ai.creator/commit/36a807b00794bac42a5018759e2ec09238bf043e)) - [object]

## 0.33.2 - 2023-03-23

[aea7083](aea7083d41dd77f30849efd56aa5f493614a4d13)...[463ff20](463ff20af4c7f5aa97c1cb9453713804e6ddef2b)

### Chore

- Allow all torch versions >= 1.11 and < 2.0 ([7321d7c]( https://github.com/es-ude/elastic-ai.creator/commit/7321d7cf5694588a607975d13958edbfa5a3b331)) - [object]

### Fix

- Fix failing unittests that are using the linear1d layer and design ([ff582e1]( https://github.com/es-ude/elastic-ai.creator/commit/ff582e185ea01cc6282cb4553e14701e88a9d8f8)) - [object]
- Fix type annotation ([8da1107]( https://github.com/es-ude/elastic-ai.creator/commit/8da1107b2640d695816c71dd3980c0783b522122)) - [object]
- Add missing ROMs and set correct names in fp_linear1d template ([ad4c6f0]( https://github.com/es-ude/elastic-ai.creator/commit/ad4c6f095102965ff1dffa83dab4f2cb9749ce49)) - [object]
- Add missing rom files and calculate correct twos complement ([f700409]( https://github.com/es-ude/elastic-ai.creator/commit/f70040956b7637844a471a5eff171d9cc6ba4c72)) - [object]
- Small import fix ([07d2e29]( https://github.com/es-ude/elastic-ai.creator/commit/07d2e29c36e60d35066d2145782223aa42d64519)) - [object]

### Refactor

- Small file and folder renames ([9602a86]( https://github.com/es-ude/elastic-ai.creator/commit/9602a868e6067889e2386c764e173c36f33e304c)) - [object]

## 0.33.1 - 2023-03-15

[1fb0bcb](1fb0bcb25d068e5819e7be00f94062328aa444d8)...[aea7083](aea7083d41dd77f30849efd56aa5f493614a4d13)

### Fix

- Wrong fixed point config object used for linear layers ([3626113]( https://github.com/es-ude/elastic-ai.creator/commit/36261136add4b4d378598dc8c9e858240f6557c5)) - [object]
- Usage of lstm output in lstm_network impl ([2e16141]( https://github.com/es-ude/elastic-ai.creator/commit/2e1614184cdaa073fdcc686b891748861fe5c7cc)) - [object]

## 0.33.0 - 2023-03-15

[a640437](a6404377d7f47c699be41fcd5ce3ea2f1f1db43c)...[1fb0bcb](1fb0bcb25d068e5819e7be00f94062328aa444d8)

### Feat

- Add rom design for saving weights ([75862b7]( https://github.com/es-ude/elastic-ai.creator/commit/75862b7db4e64173daf7e6cdcb8413b0f510d396)) - [object]

### Fix

- Correctly pad rom memory ([fe768d5]( https://github.com/es-ude/elastic-ai.creator/commit/fe768d5f93c34ade65c24479c70f3528c66b0408)) - [object]

### Refactor

- Rom design ([975ad7e]( https://github.com/es-ude/elastic-ai.creator/commit/975ad7e139a15466338cff72cfedeedf0c532f75)) - [object]
- Use rom design in implementation ([a8bfe4a]( https://github.com/es-ude/elastic-ai.creator/commit/a8bfe4a2395a9bd81aa33f1989154f84a21bf001)) - [object]
- Move conversions to twos complement from designs to translatable modules ([50ada18]( https://github.com/es-ude/elastic-ai.creator/commit/50ada185de5a081295515e16773b7fefdaa107eb)) - [object]

## 0.32.1 - 2023-03-14

[01be016](01be016011cef672e8489252f59e73dd36b533d1)...[a640437](a6404377d7f47c699be41fcd5ce3ea2f1f1db43c)

### Fix

- Set library for lstm_cell ([2b3a565]( https://github.com/es-ude/elastic-ai.creator/commit/2b3a565039672ca89a1c5f593db5a5f32742f771)) - [object]
- Typo in test for lstm cell designs ([2ffeaec]( https://github.com/es-ude/elastic-ai.creator/commit/2ffeaecf3ba7c3c0946c57ab3bee92af55746887)) - [object]

## 0.32.0 - 2023-03-14

[b605967](b6059676e662e63937af9eaf96feb7cb6c111533)...[01be016](01be016011cef672e8489252f59e73dd36b533d1)

### Chore

- Update gh workflow to match new tests location ([58b7151]( https://github.com/es-ude/elastic-ai.creator/commit/58b71513d05aa0bbf34533dc72b070ceaee34e83)) - [object]
- Update gh-workflow ([b1d714d]( https://github.com/es-ude/elastic-ai.creator/commit/b1d714d4d408917ddd389db7fa29eed6c0230684)) - [object]

### Feat

- Sequential layer with bufferless layers ([d7cea69]( https://github.com/es-ude/elastic-ai.creator/commit/d7cea69ad0696f63e00762991e7407ad09d8a94c)) - [object]
- Add support for single buffered module to sequential ([5402782]( https://github.com/es-ude/elastic-ai.creator/commit/5402782c0c37a6838b77b19d8040d256217d72ba)) - [object]
- Add linear layer to lstm network ([48982f0]( https://github.com/es-ude/elastic-ai.creator/commit/48982f0aca675098b77edb2c8419b09ebc388835)) - [object]

### Fix

- Tests and remove type annotations leading to deps ([75ed6cc]( https://github.com/es-ude/elastic-ai.creator/commit/75ed6cc4f3a92b80656433b8209c0c932595900e)) - [object]
- Correct values for x/y_address_width ([c7af1af]( https://github.com/es-ude/elastic-ai.creator/commit/c7af1af71ef9319ed2ee7fffd7afcbaa5ffda580)) - [object]

### Refactor

- Move modules ([24e522f]( https://github.com/es-ude/elastic-ai.creator/commit/24e522fb10224bbd4065d841b2df97fa0f561021)) - [object]
- Replace fixed point factory by fixed point config ([b5a08ac]( https://github.com/es-ude/elastic-ai.creator/commit/b5a08acc11453ad550e2457836f1f4a2f5cbbae1)) - [object]
- Start moving relevant tests to top-level tests dir ([577f43d]( https://github.com/es-ude/elastic-ai.creator/commit/577f43d16a30fb1e6cc73c7dca7a4d6391559f79)) - [object]
- Tweak module hierarchy ([40bc371]( https://github.com/es-ude/elastic-ai.creator/commit/40bc371d6602c504ed6e69542ef3a51d525fda70)) - [object]
- Remove code generation dependency on fixed point data types ([4d83d1b]( https://github.com/es-ude/elastic-ai.creator/commit/4d83d1bc8f1a91de6dfd8995373155151d74fc25)) - [object]
- Refactor autowiring for sequential network module ([431862f]( https://github.com/es-ude/elastic-ai.creator/commit/431862f21b6f074021973a88789a654461ae269e)) - [object]
- Lstm roms ([a2e08ec]( https://github.com/es-ude/elastic-ai.creator/commit/a2e08ec2f1492cd0efc9f4e60b76b4a42c0d093f)) - [object]

## 0.31.0 - 2023-02-22

[2e64ede](2e64eded06119897f311dff39761dbd7acd14d43)...[b605967](b6059676e662e63937af9eaf96feb7cb6c111533)

### Chore

- Introduce private package import lint rule ([b497e1c]( https://github.com/es-ude/elastic-ai.creator/commit/b497e1ca3c512d2414cc0736305e19a867251741)) - [object]
- Tweak import contract ([306de20]( https://github.com/es-ude/elastic-ai.creator/commit/306de20163ad6e751b5e8d5e66601e90d1856b50)) - [object]
- Update deps ([00700fe]( https://github.com/es-ude/elastic-ai.creator/commit/00700fe92b86442cc7e0db29794fa78d20ba48f9)) - [object]
- Add class diagram for vhdldesign ([01c63e0]( https://github.com/es-ude/elastic-ai.creator/commit/01c63e02759ca71c93dc3f985d416d3ffa2c31af)) - [object]
- Clean up external deps ([d1be65a]( https://github.com/es-ude/elastic-ai.creator/commit/d1be65aee7144be24c79a280c93537115acd2e31)) - [object]

### Feat

- Add connectable base in/out signals ([7ad67f9]( https://github.com/es-ude/elastic-ai.creator/commit/7ad67f916815b692daddae98d4c93b9a5eb21641)) - [object]
- Add logic and logic vector signals ([1947baa]( https://github.com/es-ude/elastic-ai.creator/commit/1947baac032e1b3958344779a00b84615b5581a1)) - [object]
- Introduce vhdl_design class ([20566f6]( https://github.com/es-ude/elastic-ai.creator/commit/20566f600383ccb68fed60483bede9db5436913f)) - [object]
- Add data flow node, sink node and source node ([9a511de]( https://github.com/es-ude/elastic-ai.creator/commit/9a511de4d2618c3131abcd3c481b918ffa96545e)) - [object]
- Add missing suffixes ([cb05d0f]( https://github.com/es-ude/elastic-ai.creator/commit/cb05d0f3f8665ac98c0cff70cbb2dbd8d2a5b2f2)) - [object]

### Fix

- Fix incorrect vector signal initialization ([3c68255]( https://github.com/es-ude/elastic-ai.creator/commit/3c68255057dad325ab4ba89601f6f1e2384f0d95)) - [object]
- Type annotations for tracing module ([da598a9]( https://github.com/es-ude/elastic-ai.creator/commit/da598a92fc8f76b3c19d0b960d77122b82d171ac)) - [object]
- Fix unit tests after major rebase ([3b596e9]( https://github.com/es-ude/elastic-ai.creator/commit/3b596e9c20e302bbf42efda7577e01498c05bc6c)) - [object]
- Typing ([b0bfa39]( https://github.com/es-ude/elastic-ai.creator/commit/b0bfa39b98555b37f0d2626a235ac74987e2c9ad)) - [object]

### Refactor

- Merge utilities for testing code ([333c09a]( https://github.com/es-ude/elastic-ai.creator/commit/333c09a9b396f450e24d7d2390daa8b502b5cdac)) - [object]
- Move file reading to CodeTestCase ([3cc9c5e]( https://github.com/es-ude/elastic-ai.creator/commit/3cc9c5e4c67fea3e8bea566eeb1a30feea7c1b56)) - [object]
- Remove unintended print statement ([b43befd]( https://github.com/es-ude/elastic-ai.creator/commit/b43befdb529389a8cc8c08d087631ca45163f51c)) - [object]
- Move code test utility files ([d390af1]( https://github.com/es-ude/elastic-ai.creator/commit/d390af12f9658952fd08b4493b467ee820c45f5f)) - [object]
- Rename test_logic_signals ([f817425]( https://github.com/es-ude/elastic-ai.creator/commit/f817425f96895cdf52ff184f7cc32473e3c85fe9)) - [object]
- Simplify architecture ([1f5f1f1]( https://github.com/es-ude/elastic-ai.creator/commit/1f5f1f19510f6dd9282e5bdda5beab904b2328b3)) - [object]
- Remove obsolete graph package ([ac53d76]( https://github.com/es-ude/elastic-ai.creator/commit/ac53d7684135e3bab4d940d1c80951b297d19d77)) - [object]
- Remove obsolete vhdl_design module ([d4e61bd]( https://github.com/es-ude/elastic-ai.creator/commit/d4e61bd7440d42a878f7539af7c256d637c2b7ba)) - [object]
- Simplify signals and move classes ([aacb702]( https://github.com/es-ude/elastic-ai.creator/commit/aacb7021bcb83cb96053092640a7b7cdc6e2077d)) - [object]
- Use relative imports inside packages ([ef8d588]( https://github.com/es-ude/elastic-ai.creator/commit/ef8d58878058b2eb6ef5f177171350c6759132f7)) - [object]
- Simplify data flow node ([82c8ba8]( https://github.com/es-ude/elastic-ai.creator/commit/82c8ba825bfa3b5d367bc3d6f473d2055ef217d6)) - [object]
- Remove/move/merge protocols ([8391a1c]( https://github.com/es-ude/elastic-ai.creator/commit/8391a1c7e459bbf176840976a741317a28f3abd6)) - [object]
- Only return file object from package without opening it ([2c57287]( https://github.com/es-ude/elastic-ai.creator/commit/2c572879a98a4af72978bbd471704395606b96fc)) - [object]
- Separate template from file ([73f00e0]( https://github.com/es-ude/elastic-ai.creator/commit/73f00e0e2e1e6302f2d8325fe9075d9bd51c25a3)) - [object]
- Remove deprecated vhdl.language module ([e29f6da]( https://github.com/es-ude/elastic-ai.creator/commit/e29f6da7e76018dce7d32f9698a7973de6e5e832)) - [object]
- Move modules/classes to fix dependency issues ([22564d7]( https://github.com/es-ude/elastic-ai.creator/commit/22564d7ce4b05770d49078c0d5ce13fe3ace231d)) - [object]
- Move more modules/classes to fix dependency issues ([ae82c14]( https://github.com/es-ude/elastic-ai.creator/commit/ae82c143100ddb9a49a7cfae36d8ea5289789fa4)) - [object]
- Adjust architecture in design.md and move modules accordingly ([236e6c3]( https://github.com/es-ude/elastic-ai.creator/commit/236e6c3457cbbb413b8fd79015bfe1e97c49563d)) - [object]
- Simplify signals ([884ad64]( https://github.com/es-ude/elastic-ai.creator/commit/884ad648fde4381a4dd892542bf576a7cd2d090b)) - [object]
- Simplify ports ([4bdf84a]( https://github.com/es-ude/elastic-ai.creator/commit/4bdf84a4f72f1b99d89afa84de234c74a637fcd0)) - [object]
- Remove superfluous protocol ([741c53b]( https://github.com/es-ude/elastic-ai.creator/commit/741c53baf3ca0ee9ccb27d5cf5a64d172eac7781)) - [object]

## 0.30.4 - 2023-02-16

[fdb1cea](fdb1cea8779ce59a81669b5cbe8e65e746214353)...[2e64ede](2e64eded06119897f311dff39761dbd7acd14d43)

### Fix

- Get rid of the duplicated suffix on rom component ([9cd0e0b]( https://github.com/es-ude/elastic-ai.creator/commit/9cd0e0be9481a286820eea5c8d5bdc9d28fcc0d8)) - [object]

## 0.30.3 - 2023-02-16

[f6e8c11](f6e8c119ff366f3269bddc733f9c9f4d82167693)...[fdb1cea](fdb1cea8779ce59a81669b5cbe8e65e746214353)

### Fix

- Linear layer template ([96bdf03]( https://github.com/es-ude/elastic-ai.creator/commit/96bdf030ca4c27d67a4978e3b8609ef57c40a01e)) - [object]
- Add rounding to prevent tests from failing due to floating point loss ([b7314b7]( https://github.com/es-ude/elastic-ai.creator/commit/b7314b797ef39c2f693554821ec7bb3d96689661)) - [object]

## 0.30.2 - 2023-02-15

[a0d6266](a0d6266c487f1ccafa44516c001eef027766ad98)...[f6e8c11](f6e8c119ff366f3269bddc733f9c9f4d82167693)

### Chore

- Remove unused dependencies and update poetry lock ([7b4b658]( https://github.com/es-ude/elastic-ai.creator/commit/7b4b658c2649500809ade7efd716e8dca4153576)) - [object]

### Fix

- Ignore single import mypy error ([dd85159]( https://github.com/es-ude/elastic-ai.creator/commit/dd851590719ec76ab66dc9d908493991fc235e7e)) - [object]
- Use non-static path to example folder ([613a152]( https://github.com/es-ude/elastic-ai.creator/commit/613a152e65fbe0f7116a1f772fea8a3836d888af)) - [object]

### Refactor

- Remove deprecated examples ([eec3f0e]( https://github.com/es-ude/elastic-ai.creator/commit/eec3f0e75a7875a8a2d1da9c2ffe586a4a18ebf9)) - [object]
- Remove unused module ([d2e643b]( https://github.com/es-ude/elastic-ai.creator/commit/d2e643b1368a5776829a0353730afa5039c19590)) - [object]
- Move test in the unit folder ([89df933]( https://github.com/es-ude/elastic-ai.creator/commit/89df933b50eb35e0528042f81a37a59ba8630ff5)) - [object]
- Create integration test from POS tagger example ([cb73343]( https://github.com/es-ude/elastic-ai.creator/commit/cb73343957c6b75df2a741b08c66c11545b86f2d)) - [object]
- Remove non-deterministic test ([ebed2a7]( https://github.com/es-ude/elastic-ai.creator/commit/ebed2a73beaba1f9e6abdc843eb5771cc1d34061)) - [object]
- Remove deprecated example ([008241c]( https://github.com/es-ude/elastic-ai.creator/commit/008241c8d5414cbe9478e1cdb226c22c48b2c663)) - [object]
- Move tensor_test_case in tests directory ([3cf635b]( https://github.com/es-ude/elastic-ai.creator/commit/3cf635b2d5ecbad524cfed75d4d4b7543c2dbcc2)) - [object]
- Delete not relevant example ([3c0fce9]( https://github.com/es-ude/elastic-ai.creator/commit/3c0fce95db8c078b8e37e34d0018872164402c4f)) - [object]
- Rename example ([84d4792]( https://github.com/es-ude/elastic-ai.creator/commit/84d479296c1930f4e7f334ae1d2fd89ba84b595a)) - [object]

## 0.30.1 - 2023-02-04

[a6f9577](a6f957796c627bcf7af43d78cfdc767331d6092a)...[a0d6266](a0d6266c487f1ccafa44516c001eef027766ad98)

### Chore

- Remove vhdl scope ([5c9571b]( https://github.com/es-ude/elastic-ai.creator/commit/5c9571b384588551c7439f3e45ad63d8f718b79f)) - [object]

### Fix

- Make test more deterministic ([97fd410]( https://github.com/es-ude/elastic-ai.creator/commit/97fd4101af93cf17d446cb0cb38a419080d5bee6)) - [object]

## 0.30.0 - 2023-02-04

[46f013b](46f013ba3fb9e90856882296095a011c00457ad8)...[a6f9577](a6f957796c627bcf7af43d78cfdc767331d6092a)

### Chore

- Relax commitlint rules ([108e361]( https://github.com/es-ude/elastic-ai.creator/commit/108e361f763f23843b72c5620cbebd0c171a9433)) - [object]

### Docs

- Add commit types and scopes ([e759fd3]( https://github.com/es-ude/elastic-ai.creator/commit/e759fd38fb41d413ccf03617f84f87f6df9aeb12)) - [object]

### Feat

- Add unit tests for the LSTMBase layer ([589f803]( https://github.com/es-ude/elastic-ai.creator/commit/589f803fd858b22985485d795f4441a9abf97742)) - [object]
- Add unit tests for the fixed point quant/dequant autograd functions ([f82431c]( https://github.com/es-ude/elastic-ai.creator/commit/f82431c164b9536899d0cca9b391a057add8187a)) - [object]
- Improve TensorTestCase class ([d4273a6]( https://github.com/es-ude/elastic-ai.creator/commit/d4273a60c169669ddba5f80636d1430b69c77d90)) - [object]
- Rename quant_typings module to quantization and implement FakeQuant ([0e5f24a]( https://github.com/es-ude/elastic-ai.creator/commit/0e5f24aeb9f43258f9e971ffa777c585faff05f0)) - [object]
- Integrate arithmetics for the linear layer ([a961558]( https://github.com/es-ude/elastic-ai.creator/commit/a9615581159ba4b962fac8458d9b76de0a61d98f)) - [object]
- Convert example parametrize_convolution to automated integration test ([3dde1c2]( https://github.com/es-ude/elastic-ai.creator/commit/3dde1c250fa4ebb617bbd543c9b26cb320d430f7)) - [object]
- Convert example translate_linear_model to automated integration test ([5d92d0b]( https://github.com/es-ude/elastic-ai.creator/commit/5d92d0b15d8c0a1d76f842fd7a8bbc591bd1cf18)) - [object]
- Remove input_quant and param_quant and add quantize function to arithmetics ([ee91e42]( https://github.com/es-ude/elastic-ai.creator/commit/ee91e42801b0d1163a0d52130fc578477da60c74)) - [object]
- Implement concept of arithmetics ([e7ad504]( https://github.com/es-ude/elastic-ai.creator/commit/e7ad50471e2ac7300e0db781bd37cbba1364a5e6)) - [object]
- Remove quantized_forward function and adopt tests ([c865c73]( https://github.com/es-ude/elastic-ai.creator/commit/c865c73a53e89c40ecebc9c4b49ba6d5c14256c1)) - [object]
- Add example to demonstrate that the new kinds of layers are trainable ([231e325]( https://github.com/es-ude/elastic-ai.creator/commit/231e325815c469596c63259c5f345dc9afb0f3b7)) - [object]
- Lstm uses fp hard sigmoid ([fd265ac]( https://github.com/es-ude/elastic-ai.creator/commit/fd265ac3e1ef7f11e28236705e4a38760462bddc)) - [object]
- Integrate hard tanh layer ([eb74d3a]( https://github.com/es-ude/elastic-ai.creator/commit/eb74d3a3671616db37ba8f554332ca1ddc33dffe)) - [object]
- Small example for translating combination of lstm and linear layer ([12e7101]( https://github.com/es-ude/elastic-ai.creator/commit/12e7101e8c62e8424bc2ed580cfbe645e8d33510)) - [object]

### Fix

- Fix imports and use new FixedPointFactory features ([e8c74c3]( https://github.com/es-ude/elastic-ai.creator/commit/e8c74c34ec1c5a4b5189d74f2a19a993a5ae9779)) - [object]
- Adapt basic qtorch example to recent changes of the creator ([a17d900]( https://github.com/es-ude/elastic-ai.creator/commit/a17d9006240a67da97b8a539620aa1974e07e942)) - [object]
- Add similar concept of translation arguments to fix the translation process ([e387ae2]( https://github.com/es-ude/elastic-ai.creator/commit/e387ae26918fbe8e4a0ee01ccc4361849746bd66)) - [object]
- Fix unit and integration tests to use the new layers correctly ([0553017]( https://github.com/es-ude/elastic-ai.creator/commit/05530178cf7fb64dc88cab82b89c24b2a1406e8d)) - [object]
- Remove unused OperationType type and FakeQuant class ([596dbd8]( https://github.com/es-ude/elastic-ai.creator/commit/596dbd8cdf3cde67eedea2779a35ff682c9ac9f7)) - [object]
- Change torch LSTM layer to our FixedPointLSTM layer ([5e7a39a]( https://github.com/es-ude/elastic-ai.creator/commit/5e7a39a78684c09a1d374476f8fb611019ae994f)) - [object]
- Infer fixed_point_factory of linear and lstm in build functions ([81df686]( https://github.com/es-ude/elastic-ai.creator/commit/81df686fe13db5f85c91b65c73713b7da8e6c64f)) - [object]
- Fix LSTMCell raises Error for unbatched input data and add a test for this case ([5ce3e21]( https://github.com/es-ude/elastic-ai.creator/commit/5ce3e2125b4bcd1115d77ebe5c833e52d58bad77)) - [object]
- Rename to .tpl.vhd ([fe3c85c]( https://github.com/es-ude/elastic-ai.creator/commit/fe3c85cd77d0f2fefb90f2d3ff6eadde8570d000)) - [object]
- Remove sigmoid_resolution ([dd4f033]( https://github.com/es-ude/elastic-ai.creator/commit/dd4f03366920f1a3774772a16a49efaa8756d249)) - [object]
- Use model.children() instead of model.modules() to avoid recursion ([a3c349b]( https://github.com/es-ude/elastic-ai.creator/commit/a3c349b13af0fef383b494850973d8ff9ac2dd68)) - [object]
- Fix some mypy errors and remove unused imports ([08e2362]( https://github.com/es-ude/elastic-ai.creator/commit/08e2362fa32efd13e388140ad58c93b0e79229b3)) - [object]
- Change not existing layer_id field to layer_name ([f7425c5]( https://github.com/es-ude/elastic-ai.creator/commit/f7425c515395243962db1517116b9961b1668cd7)) - [object]
- Add layer_name to all vhdl templates and components ([2d9c47d]( https://github.com/es-ude/elastic-ai.creator/commit/2d9c47dc60642d94efeb58cc3014f6a7790a6f26)) - [object]
- Fix errors in the lstm template and remove lstm_common component ([c4a28ce]( https://github.com/es-ude/elastic-ai.creator/commit/c4a28ce2f40dc84e7a5e4470c62a40911b73901f)) - [object]

### Refactor

- Move unit test to correct location ([c03c362]( https://github.com/es-ude/elastic-ai.creator/commit/c03c3621c6cdef58a44e1c3e279d025ebdf34aa6)) - [object]
- Remove unnecessary print statement ([2f8a0a7]( https://github.com/es-ude/elastic-ai.creator/commit/2f8a0a75b602d6d7621f310e33ccf0bf0d5c1e28)) - [object]
- Remove examples belonging to the removed precomputation package ([4dc681b]( https://github.com/es-ude/elastic-ai.creator/commit/4dc681b18207dd92d767c97df2c70e2fd3e6cd2e)) - [object]
- Move integration test to more specific location ([0115399]( https://github.com/es-ude/elastic-ai.creator/commit/01153996ac556eb9a96f404e8efed2af5bbdf1dd)) - [object]
- Remove default bias value from linear layer ([8d55471]( https://github.com/es-ude/elastic-ai.creator/commit/8d5547180a50f07ee259f37cd8cd89ffe496e421)) - [object]
- Add more precise type annotation ([0c47fe0]( https://github.com/es-ude/elastic-ai.creator/commit/0c47fe0b485cb71662ef017b7c454b848baa0b4f)) - [object]
- Remove outdated evaluators ([8c0009a]( https://github.com/es-ude/elastic-ai.creator/commit/8c0009ae54dfed9f24223ca01a6b146ee0c06f04)) - [object]
- Add fixed_point_factory property to fp layers and remove FixedPointLSTMCell ([9f0a5d3]( https://github.com/es-ude/elastic-ai.creator/commit/9f0a5d3505dc05d53aaf9fa9fb1c607049c661fd)) - [object]

### Style

- Beautify commit # [6209df2](https://github.com/es-ude/elastic-ai.creator/commit/6209df2bbc3c693f1829ce8b93822fc84152f69b) ([423b081]( https://github.com/es-ude/elastic-ai.creator/commit/423b081476868df0a7f90fbcaeec16203670551f)) - [object]

## 0.29.0 - 2022-12-16

[58f37d2](58f37d22b9b09eb48c2b3a604d3365f7520653e8)...[46f013b](46f013ba3fb9e90856882296095a011c00457ad8)

### Chore

- Tighten commitlint rules ([47a35da]( https://github.com/es-ude/elastic-ai.creator/commit/47a35da220ba1c6081af11b0a6e7945978f2fe77)) - [object]

### Feat

- Set pypi project api token ([37ba8c9]( https://github.com/es-ude/elastic-ai.creator/commit/37ba8c9794acc6b4bdf64087c98c61172446fcb6)) - [object]

## 0.28.0 - 2022-12-16

[6346bfe](6346bfe40ac06023434683c8ec2e8d73e9e246ed)...[58f37d2](58f37d22b9b09eb48c2b3a604d3365f7520653e8)

### Chore

- Use gh-action provided by python-semantic-release ([0d0321e]( https://github.com/es-ude/elastic-ai.creator/commit/0d0321e44455d40c3b04929df13cccfe7056c35c)) - [object]
- Add noop to semantic-release and trigger on workflow call ([ecdb463]( https://github.com/es-ude/elastic-ai.creator/commit/ecdb463514c0e5b8b0d0d22818071c728e6997e2)) - [object]
- Add github action for test environment setup ([8a38722]( https://github.com/es-ude/elastic-ai.creator/commit/8a3872210155601a900b8ac59757808974961999)) - [object]
- Rename actions yml ([7882524]( https://github.com/es-ude/elastic-ai.creator/commit/78825240f3cd78863110f516574d915781f3a4c5)) - [object]
- Add commit hash to action reference ([459a4cc]( https://github.com/es-ude/elastic-ai.creator/commit/459a4ccd2487762c67a1be86f2ae071dc89396e8)) - [object]
- Fetch repo in job instead of action ([05d8bd1]( https://github.com/es-ude/elastic-ai.creator/commit/05d8bd14a7c287c90755ffb68f2c899d3d182ad2)) - [object]
- Specify shell in gh-action ([a5fb59e]( https://github.com/es-ude/elastic-ai.creator/commit/a5fb59e35e8b557011559ba5d55b68a452574710)) - [object]
- Create cache-dir in action ([f0ecc17]( https://github.com/es-ude/elastic-ai.creator/commit/f0ecc17eedd1e9acdc6c0d4baa713eee6a5e2495)) - [object]
- Reorder poetry calls and cache setup for action ([2a0fb0d]( https://github.com/es-ude/elastic-ai.creator/commit/2a0fb0d65d5bf80746b65d1d5f29f63cc59f36f1)) - [object]
- Add missing argument to poetry configuration ([1567b0c]( https://github.com/es-ude/elastic-ai.creator/commit/1567b0c7269f14a1454b206b959c2c33862fe239)) - [object]
- Enable semantic release for main again ([6c93920]( https://github.com/es-ude/elastic-ai.creator/commit/6c939203995883b390a20bc98b098a252563c669)) - [object]
- Temporary relax commitlint rules ([437c3d7]( https://github.com/es-ude/elastic-ai.creator/commit/437c3d7cec0487f5754ec357fb4d313343fd2cbc)) - [object]

### Refactor

- Remove unused import ([14d1d60]( https://github.com/es-ude/elastic-ai.creator/commit/14d1d60bb7b56c2c6bdd00feb767a8248a09699c)) - [object]
- Rename package from qat to nn ([e211ae6]( https://github.com/es-ude/elastic-ai.creator/commit/e211ae63d9ee7fdc2c0fad15a40730399fac7654)) - [object]
- Rename _init_quantizable_convolution function ([2b57dbc]( https://github.com/es-ude/elastic-ai.creator/commit/2b57dbcaa02f202c7654d8d15b53c84a0210ee1f)) - [object]

### Revert

- "chore: add commit hash to action reference" ([e42d010]( https://github.com/es-ude/elastic-ai.creator/commit/e42d01029b403029334dc2ed1a3311631361f9fb)) - [object]

## 0.27.0 - 2022-12-15

[913658d](913658d4df2f93a54270d7591b974a65fd34b34a)...[6346bfe](6346bfe40ac06023434683c8ec2e8d73e9e246ed)

### Chore

- Set correct path to unit and integration tests ([538eb2f]( https://github.com/es-ude/elastic-ai.creator/commit/538eb2f036f24ea99135f0e66ad59c3738e60231)) - [object]
- Remove superfluous line ([71edbc4]( https://github.com/es-ude/elastic-ai.creator/commit/71edbc4369a59c90c561ff3e8b335bd85ecbba7e)) - [object]
- More specific commitlint rules ([bbb88e9]( https://github.com/es-ude/elastic-ai.creator/commit/bbb88e9080ecd873209f99aa01473b9d57bd2012)) - [object]
- Update poetry.lock ([0f78c4b]( https://github.com/es-ude/elastic-ai.creator/commit/0f78c4bfdddad038bd69b1a92f3b1fba4c5ab9f8)) - [object]
- Don't install extras prior publishing ([effa8c0]( https://github.com/es-ude/elastic-ai.creator/commit/effa8c004a2d8356a96e3869763e85e58ee92924)) - [object]
- Tweak pyproject and commitlint ([addc521]( https://github.com/es-ude/elastic-ai.creator/commit/addc521744804fb8a6deeadde8510bd9fe37d87b)) - [object]
- Add style again to pyproject and commitlint ([d7aaf28]( https://github.com/es-ude/elastic-ai.creator/commit/d7aaf28042881c272f851e5402135d15a149ec42)) - [object]

### Ci

- Add commitlint constraints ([d345351]( https://github.com/es-ude/elastic-ai.creator/commit/d345351c96c0ce6d0a5bcf52ae9bca8eacdafd6b)) - [object]
- Adjust commitlint config ([e518975]( https://github.com/es-ude/elastic-ai.creator/commit/e518975728140ab29692bea341ea015cfcfb59df)) - [object]
- Temporarily disable commitlint constraints ([87eaa63]( https://github.com/es-ude/elastic-ai.creator/commit/87eaa632c644d70c5ac693b2f5f6aac6a3625acc)) - [object]
- Temporarily further relax commitlint ([08eab5b]( https://github.com/es-ude/elastic-ai.creator/commit/08eab5b817d772457dfa983846adb36a8f1b64d3)) - [object]
- Don't install onnx ([7b164cd]( https://github.com/es-ude/elastic-ai.creator/commit/7b164cdd95a5af672f78c7c22e267c3364fe4d0a)) - [object]
- Clean up pyproject.toml ([547d724]( https://github.com/es-ude/elastic-ai.creator/commit/547d724db5c14392b148c8cf9e0a5714b1052a4d)) - [object]

### Doc

- Add doc to VHDLFile ([5fcf78b]( https://github.com/es-ude/elastic-ai.creator/commit/5fcf78b87edf75ff3e9e818b1511aef00ffbf46a)) - [object]

### Docs

- Move tests and remove deprecated lines ([4a074a8]( https://github.com/es-ude/elastic-ai.creator/commit/4a074a87fb31df535d415c2ab6aede7e4d7d8949)) - [object]

### Feat

- Add constraint type ([dc4c4e5]( https://github.com/es-ude/elastic-ai.creator/commit/dc4c4e57a9615a9be6941ecc750d3838458ff919)) - [object]
- Update qlstm sine wave example to the correctly implemented QLSTM layer ([dc62cd2]( https://github.com/es-ude/elastic-ai.creator/commit/dc62cd2aa05067b164009301ab7c5e110797c503)) - [object]
- Remove constraints ([6b7b483]( https://github.com/es-ude/elastic-ai.creator/commit/6b7b4835dc9f9f6b6fc83bc619727aa948c19161)) - [object]
- Support generation of layer connections ([1d43c42]( https://github.com/es-ude/elastic-ai.creator/commit/1d43c4212ef54c5488df7e7dc3829df31a7e8484)) - [object]
- Generate portmap output_address ([c6a26a6]( https://github.com/es-ude/elastic-ai.creator/commit/c6a26a61d98c90fa29b02e6619116e67a4a67ac5)) - [object]
- Add hw equivalent module tracer ([3f2c2c7]( https://github.com/es-ude/elastic-ai.creator/commit/3f2c2c7acc5046131d420d513a4bb3d3981ac0c5)) - [object]
- Tracer records reference to module for call_module nodes ([20ed7da]( https://github.com/es-ude/elastic-ai.creator/commit/20ed7dab9677e476925a8b1250cbbc2004d43246)) - [object]
- Generate vhdl signal definitions ([53408f6]( https://github.com/es-ude/elastic-ai.creator/commit/53408f6cb9daa5c44931e880fda0712c2924b822)) - [object]
- Generate layer instantiations ([7a75fc3]( https://github.com/es-ude/elastic-ai.creator/commit/7a75fc31780a6173424ffdcf3129bc60d5a83e59)) - [object]
- Introduce HWBlocks ([ab03eaf]( https://github.com/es-ude/elastic-ai.creator/commit/ab03eaf28c74483fcd9dbd78d247d39e248bdea1)) - [object]
- Extend code file with parameters ([4833f8b]( https://github.com/es-ude/elastic-ai.creator/commit/4833f8b2d5553cf02d322b8485587612cd67a9e8)) - [object]
- Implement HWBlocks interface for sigmoid,linear ([0177373]( https://github.com/es-ude/elastic-ai.creator/commit/0177373eeddfa9c32100777bbcd7a94765dc1122)) - [object]
- Add module_nodes to graph decorator ([6d0a612]( https://github.com/es-ude/elastic-ai.creator/commit/6d0a61217b36b9db8e9df19210e5f0d3aeed4ef2)) - [object]
- Introduce HWEquivalentGraph ([844bb84]( https://github.com/es-ude/elastic-ai.creator/commit/844bb84a2d36e50f3de7ae4b713d370011d3240e)) - [object]
- Introduce HWBlockCollection ([a80bda2]( https://github.com/es-ude/elastic-ai.creator/commit/a80bda2d705992030b18649ff99f3a6ce75d7ef3)) - [object]
- Distinguish x/y width ([2f52100]( https://github.com/es-ude/elastic-ai.creator/commit/2f52100d32502520ce66a240bae90dd48e070ebd)) - [object]

### Fix

- Fix circular dependency ([1d5615b]( https://github.com/es-ude/elastic-ai.creator/commit/1d5615bf81757bf16904eb75c33fead69a68dd43)) - [object]
- Fix the problem of wrong shapes for the QLSTM layer ([b75f478]( https://github.com/es-ude/elastic-ai.creator/commit/b75f47804016a3dfdad3f8d2dd575f4252cac5ff)) - [object]
- Fix error when passing flat input data to _QLSTMBase and batch_first set to True ([29918d1]( https://github.com/es-ude/elastic-ai.creator/commit/29918d11c508e3e91fe00a0e07988be0ed198b35)) - [object]
- Remove unmaintained onnx support ([dc773d3]( https://github.com/es-ude/elastic-ai.creator/commit/dc773d39fe2c0ea5785e3fb0bf7a43f3bf83495f)) - [object]
- Remove obsolete vhdl formatter ([83d81e3]( https://github.com/es-ude/elastic-ai.creator/commit/83d81e348152e047482ccc45a2ccaf6173f772d9)) - [object]

### Refactor

- Using code function to generate code instead of call ([843ad64]( https://github.com/es-ude/elastic-ai.creator/commit/843ad64d33e2018da9c88fd487ccd46fd598c58f)) - [object]
- Add missing type annotations ([c83a746]( https://github.com/es-ude/elastic-ai.creator/commit/c83a7466cecfc043dd95800c69b6ee5df8b5bd4f)) - [object]
- Add missing type annotations and remove unused parts ([6fb622b]( https://github.com/es-ude/elastic-ai.creator/commit/6fb622b3cff4caf8d849c8df8696275fa38fa9bb)) - [object]
- Rename call function to code in the language module ([4cf795e]( https://github.com/es-ude/elastic-ai.creator/commit/4cf795ee1cc679ea6b4b7cf51198cb536a5d9af5)) - [object]
- Rename call function to code in the precomputed scalar functions and test benches ([a40553e]( https://github.com/es-ude/elastic-ai.creator/commit/a40553e05f64d9bd57473fed4d40b269858ef65f)) - [object]
- Fix some mypy errors ([cd9899f]( https://github.com/es-ude/elastic-ai.creator/commit/cd9899f55476cd91993f7276cdda02fc7e3d7b26)) - [object]
- Move all unit and integration tests in a tests folder ([8afb751]( https://github.com/es-ude/elastic-ai.creator/commit/8afb751a5dc9f7fd4e2fa4a1dd1167682efe590f)) - [object]
- Move BatchNormedActivatedConv1d from layers module to blocks module ([6269522]( https://github.com/es-ude/elastic-ai.creator/commit/6269522bcdb978a905c84693e6c9fa4bdc32bfa7)) - [object]
- Split LayersTest class into classes for each layer ([55c12b3]( https://github.com/es-ude/elastic-ai.creator/commit/55c12b36ce0b9807ffa4f5dd8344e3b8143f1212)) - [object]
- Remove unused import ([b6cf349]( https://github.com/es-ude/elastic-ai.creator/commit/b6cf3494b36cca9d2fd732a24952423b68ad6c46)) - [object]
- Remove unused code ([43e5992]( https://github.com/es-ude/elastic-ai.creator/commit/43e5992f1e48d078113bba7863c0ac5e3e967ada)) - [object]
- Remove unused code and make Identity quantizer public ([dcd726e]( https://github.com/es-ude/elastic-ai.creator/commit/dcd726e183c5b74b05c27155ec64cc08f395802e)) - [object]
- Remove noise comments and remove default quantizer from QLSTM and QLSTMCell layer ([4a57ca9]( https://github.com/es-ude/elastic-ai.creator/commit/4a57ca900a6c5dad1710f6d558c1ade17527d2b4)) - [object]
- Remove default quantizer from QLSTM and QLSTMCell layer ([cce2f8f]( https://github.com/es-ude/elastic-ai.creator/commit/cce2f8f1d22c384f136583f74d3a2b396500b0e0)) - [object]
- Create a _QLSTMCellBase and _QLSTMBase class to avoid default parameters ([98ba0b7]( https://github.com/es-ude/elastic-ai.creator/commit/98ba0b78090c120070e38bf9b0502b3027e0fa33)) - [object]
- Remove noise comments ([ccc3979]( https://github.com/es-ude/elastic-ai.creator/commit/ccc397911d899bdc49b917d0438336e17e37d100)) - [object]
- Reading text from resources returns list[str] ([361c5a5]( https://github.com/es-ude/elastic-ai.creator/commit/361c5a571432f24b8b2be0327fdc1b4edea1c6fe)) - [object]
- Use streams for reading templates ([2af5d2a]( https://github.com/es-ude/elastic-ai.creator/commit/2af5d2a41d72406a4abcbb2c55f3c0c01150cad4)) - [object]
- Remove CodeModule/CodeComponent ([89e27dd]( https://github.com/es-ude/elastic-ai.creator/commit/89e27dd4e1bd77502d69054c15a3277f0a4b0826)) - [object]
- Use new vhdl file class for hard_sigmoid ([a9e1f6c]( https://github.com/es-ude/elastic-ai.creator/commit/a9e1f6ccba2f978b8290be94c754872432d3c311)) - [object]
- Refactor FPHardSigmoidFile ([3c54418]( https://github.com/es-ude/elastic-ai.creator/commit/3c544184877d3416445f064d33bee3e95d78ac31)) - [object]
- Use VHDLFile for root module generation ([e2b8423]( https://github.com/es-ude/elastic-ai.creator/commit/e2b842310f237ff8802bd92ac4f7f537d9ede707)) - [object]
- Move files ([01999c4]( https://github.com/es-ude/elastic-ai.creator/commit/01999c4049f21e357390c2fc88a09bfc987d0cb6)) - [object]
- Rename BaseHWBlockInterface to BaseHWBlock ([4c69682]( https://github.com/es-ude/elastic-ai.creator/commit/4c696826e559603ac54681eae7ff50e34d22a1ac)) - [object]
- Move classes out of hw_equivalent_layers.__init__ ([0713211]( https://github.com/es-ude/elastic-ai.creator/commit/071321164dee3e9a9db3d113848c6ce3dd960b1c)) - [object]
- Sort imports ([2c114f1]( https://github.com/es-ude/elastic-ai.creator/commit/2c114f1b8c839ef939051f5a1c5b6d40585908cb)) - [object]
- Move files, simplify, correct types, merge ([57d3754]( https://github.com/es-ude/elastic-ai.creator/commit/57d37541fed53c29ad9e6a665aa72b99fe5a2df0)) - [object]
- Move last tests out of old test folder structure ([a3e12c1]( https://github.com/es-ude/elastic-ai.creator/commit/a3e12c11df45b4de8babb2f1862ea92a1778c92a)) - [object]

### Style

- Remove deprecated code and move/rename ([f6b8020]( https://github.com/es-ude/elastic-ai.creator/commit/f6b8020f8a5dfc5d9226578efa5f2512b84223e5)) - [object]
- Introduce template code file interface ([69fb2b6]( https://github.com/es-ude/elastic-ai.creator/commit/69fb2b681497289d230f8d203f5f430c91a3ff54)) - [object]
- Beautify commit # [5fcfc23](https://github.com/es-ude/elastic-ai.creator/commit/5fcfc23c342983a98efc1d527648ef17644c472c) ([a228cc0]( https://github.com/es-ude/elastic-ai.creator/commit/a228cc00bf0a44327a858835e9a73531af56e59e)) - [object]

### Test

- Start splitting large LayersTest TestCase class into smaller classes for each layer ([905f165]( https://github.com/es-ude/elastic-ai.creator/commit/905f165fabf01e0a5897ffc41bff01acac3175b2)) - [object]
- Start making network.vhd tests more specific ([ea260f0]( https://github.com/es-ude/elastic-ai.creator/commit/ea260f0bc989a0bc477b425f14364bdb7756a37b)) - [object]
- Add multiline begin-end extraction for code ([e958557]( https://github.com/es-ude/elastic-ai.creator/commit/e958557fccaabf46d4020cf23fae14e20c4452ee)) - [object]
- Introduce more fine grained testing for generated vhd files ([5a27cdc]( https://github.com/es-ude/elastic-ai.creator/commit/5a27cdc16ed7b3cbb880c022ac65e1adcfdca563)) - [object]

## 0.26.1 - 2022-11-30

[dd5511f](dd5511f9df92c33e31c8449f7e9a957a206e93c9)...[913658d](913658d4df2f93a54270d7591b974a65fd34b34a)

### Feat

- Implement quantized forward function of the fixed point lstm cell ([7818e15]( https://github.com/es-ude/elastic-ai.creator/commit/7818e15bc6c41454090b77fe5df7a8e7930ab570)) - [object]
- Start implementing lstm base module ([b154ca5]( https://github.com/es-ude/elastic-ai.creator/commit/b154ca5525c00f735150c21f64324da87328ba5e)) - [object]

### Fix

- Remove layer_name parameter ([7a83b1e]( https://github.com/es-ude/elastic-ai.creator/commit/7a83b1eed3095a8b7f90438c78ba24bba6e44958)) - [object]

### Refactor

- Rename lstm module to lstm_cell ([97bf791]( https://github.com/es-ude/elastic-ai.creator/commit/97bf791a8a3fbab68179f5a9a20e9410c3bcccf7)) - [object]
- Remove examples that are not relevant anymore ([3b241e2]( https://github.com/es-ude/elastic-ai.creator/commit/3b241e2ddfe14a248e411f0b8da9ec6cf85cc8bc)) - [object]
- Remove vhdl formatter that is not used anymore ([007a8c4]( https://github.com/es-ude/elastic-ai.creator/commit/007a8c4ec4c42382390b0af034a2f5f3226fea86)) - [object]

## 0.26.0 - 2022-11-23

[d1a9b56](d1a9b56c4661bed43895e83b54051f9888a1d634)...[dd5511f](dd5511f9df92c33e31c8449f7e9a957a206e93c9)

### Chore

- Remove minor python versions from gh workflow ([fc517a6]( https://github.com/es-ude/elastic-ai.creator/commit/fc517a6bb81fb037f2b9d3466d32506aa0573020)) - [object]

### Feat

- Merge from main ([fefd3ba]( https://github.com/es-ude/elastic-ai.creator/commit/fefd3ba4ab1fa8ae9d09bfc6185f906175f7a6ff)) - [object]
- Make linear layers better timing ([1c6a3ae]( https://github.com/es-ude/elastic-ai.creator/commit/1c6a3aeeeaee929affbb092eb485c1cf7a323355)) - [object]
- Clean the code ([d737d02]( https://github.com/es-ude/elastic-ai.creator/commit/d737d02122207bcd24f4b7c960b71db095d34a26)) - [object]

### Fix

- Fix error during integrating to a MLP model ([0e2b89c]( https://github.com/es-ude/elastic-ai.creator/commit/0e2b89c898497f35a2ad840bd3065429799bdf61)) - [object]

## 0.25.0 - 2022-11-22

[0c8fdc2](0c8fdc25eb6977b2bfeb0d9523efa11ae8167f08)...[d1a9b56](d1a9b56c4661bed43895e83b54051f9888a1d634)

### Feat

- Add expand_template function that fills string templates instead of format strings ([eb9ee98]( https://github.com/es-ude/elastic-ai.creator/commit/eb9ee987f73ffb26e8280ec3c32b32e38896d3c1)) - [object]
- Apply the expand_template function to the already existing templates ([c958f54]( https://github.com/es-ude/elastic-ai.creator/commit/c958f545f4c2cf2414a007753b416ec73c410458)) - [object]

### Fix

- Remove the layer name in the example file ([767b5f9]( https://github.com/es-ude/elastic-ai.creator/commit/767b5f9c62d493d35e5a294b1363c861d5438fa5)) - [object]
- Fix small error in the template file ([fe94518]( https://github.com/es-ude/elastic-ai.creator/commit/fe94518ff2e5e44f7c1ff8f9bf8b4ff8f0b5cf41)) - [object]
- Fix the error from merging braches ([c386766]( https://github.com/es-ude/elastic-ai.creator/commit/c386766ea654852c5ad5254cefc1fab28f544c66)) - [object]

## 0.24.0 - 2022-11-22

[a7eebbb](a7eebbb669e5329ba9731b0282e3e2a2eca612a6)...[0c8fdc2](0c8fdc25eb6977b2bfeb0d9523efa11ae8167f08)

### Chore

- Update action versions and remove usage of set-output ([d106116]( https://github.com/es-ude/elastic-ai.creator/commit/d1061167f9da09f7aa2a191340652ae56e3335e0)) - [object]
- Set correct commit action ([a7a2439]( https://github.com/es-ude/elastic-ai.creator/commit/a7a2439d8b8a1358983d18c22de6e57820757d82)) - [object]
- Set correct parameters for the commit action ([373ffd2]( https://github.com/es-ude/elastic-ai.creator/commit/373ffd20d331152cedde6083d72fdb40d83c741d)) - [object]

### Feat

- Implement FixedPointHardTanh layer ([ed72810]( https://github.com/es-ude/elastic-ai.creator/commit/ed728101fb596a08e1a76d936d04306a066c50b5)) - [object]
- Start implementing lstm base layer ([39ce891]( https://github.com/es-ude/elastic-ai.creator/commit/39ce891d56be59d5a20a36889b0e9c2f13e00bd1)) - [object]
- Implement and test lstm cell base class and start implementing fp lstm cell ([f458fb6]( https://github.com/es-ude/elastic-ai.creator/commit/f458fb6c216385a119774a3f98788941e13ed5c9)) - [object]
- Add layer_id parameter to build function and set it to a unique value during translation ([cfdf949]( https://github.com/es-ude/elastic-ai.creator/commit/cfdf9492190e24230293e3b0b1b312bfc9710952)) - [object]

### Fix

- Fix wrong return type ([eb53ed9]( https://github.com/es-ude/elastic-ai.creator/commit/eb53ed972ec9078f6c405ecd7c92043eaf8ed419)) - [object]
- Remove duplicated key ([5a4bcd6]( https://github.com/es-ude/elastic-ai.creator/commit/5a4bcd6fb6de9cff6c639866db1dd50918f3039b)) - [object]

### Refactor

- Move common helper functions to a separate utils.py module ([b459f0a]( https://github.com/es-ude/elastic-ai.creator/commit/b459f0af4b8b9884c03277928d7a0437b68f9716)) - [object]
- Move OperationType type in the typing module ([bf0d3fe]( https://github.com/es-ude/elastic-ai.creator/commit/bf0d3feacdfdf461527056b8a24aec63907a2578)) - [object]

## 0.23.0 - 2022-11-15

[c66096a](c66096a44b6a99d090c2f0754d893ec6737fd100)...[a7eebbb](a7eebbb669e5329ba9731b0282e3e2a2eca612a6)

### Docs

- Change documentation ([d3fb540]( https://github.com/es-ude/elastic-ai.creator/commit/d3fb5402c7acb09cee3df535671f22d5011f2f47)) - [object]

### Feat

- Merge main to current working branch ([35db3c5]( https://github.com/es-ude/elastic-ai.creator/commit/35db3c56608493c6b33d05e0c2250cedb0374c8e)) - [object]
- Enable multiple linear layers in the same model, by adding layer_name ([3a99a30]( https://github.com/es-ude/elastic-ai.creator/commit/3a99a3059dd53b913e7d619cbce28014007bf854)) - [object]
- Remove the previous linear_1d implementation ([0f1b9aa]( https://github.com/es-ude/elastic-ai.creator/commit/0f1b9aa2f1c12f5c0fc1fe6a3db482f40041c057)) - [object]

## 0.22.0 - 2022-11-13

[518bbb6](518bbb6fcc78c870f9cf82133cae298e12d918c1)...[c66096a](c66096a44b6a99d090c2f0754d893ec6737fd100)

### Docs

- Add missing parameter in docstring of the translate_model function ([458a02c]( https://github.com/es-ude/elastic-ai.creator/commit/458a02c38402a0860500d5821b68890fcc78c01a)) - [object]

### Feat

- Raise an exception if the build folder already exists ([d09bfa1]( https://github.com/es-ude/elastic-ai.creator/commit/d09bfa105d909b58432cf8883ee55a6b11639add)) - [object]

## 0.21.0 - 2022-11-13

[f71ab12](f71ab12692dbd71a8227087c1c4eb885f8d674aa)...[518bbb6](518bbb6fcc78c870f9cf82133cae298e12d918c1)

### Feat

- Add fp_linear_component and its template unittest is passed ([6e97316]( https://github.com/es-ude/elastic-ai.creator/commit/6e973168ca244e4cf407c48b31406d2eed73b4b0)) - [object]
- Add fp_linear_module and test passed ([241fd65]( https://github.com/es-ude/elastic-ai.creator/commit/241fd652495d6ce582873f1bcc297302f3d61764)) - [object]
- Add fp_linear build function, and test passed ([ffcbb1d]( https://github.com/es-ude/elastic-ai.creator/commit/ffcbb1d57408ad03e91bd1228bc6d3289f1d0c66)) - [object]
- Add default build function mapping and small changes ([b1d6f2a]( https://github.com/es-ude/elastic-ai.creator/commit/b1d6f2ac1040e63781d5f4af7ee29e486d9b6d69)) - [object]
- Check the component interface ([53791c5]( https://github.com/es-ude/elastic-ai.creator/commit/53791c5eb9a72793b16a0a41eb79ed8932b8e32d)) - [object]
- Add fixed point relu to translator ([80935ce]( https://github.com/es-ude/elastic-ai.creator/commit/80935ce550a2e99267a55b41ad272906faf211a5)) - [object]
- Add default build mapping for fp_hard_sigmoid and fp_relu ([c9c4d9f]( https://github.com/es-ude/elastic-ai.creator/commit/c9c4d9f329ed2c56d47f2b698dbe1d3b34c1c8a5)) - [object]

### Refactor

- Change the interface name of the template ([a693041]( https://github.com/es-ude/elastic-ai.creator/commit/a693041a050ef77828a4f4dba791b0a38a845184)) - [object]
- Get rid of some comments ([ced1b12]( https://github.com/es-ude/elastic-ai.creator/commit/ced1b127031c02d8576ccc35fdd9f143017c3368)) - [object]

### Test

- Add test coverage of relu component ([88e0d10]( https://github.com/es-ude/elastic-ai.creator/commit/88e0d10d4cb0a64ac397cee1a9e42db9184c6139)) - [object]

## 0.20.1 - 2022-11-10

[51dd88a](51dd88a341570fcd4eeaab201c112ba44c43de5b)...[f71ab12](f71ab12692dbd71a8227087c1c4eb885f8d674aa)

### Fix

- Fix incompatible signature of the forward function ([ff6c165]( https://github.com/es-ude/elastic-ai.creator/commit/ff6c165cd0bf17477051548018b791809fff33c9)) - [object]

### Refactor

- Small change of the FixedPointFactory type ([c7629bd]( https://github.com/es-ude/elastic-ai.creator/commit/c7629bd05764de09f03fd8445437dee671518d38)) - [object]
- Remove usage of deprecated assertEquals function ([6a6f4f3]( https://github.com/es-ude/elastic-ai.creator/commit/6a6f4f3af28735e27fc70b51f857324cd1ead7ef)) - [object]

## 0.20.0 - 2022-11-08

[7568be6](7568be66864623be1b31c27db21573fb18efe0e6)...[51dd88a](51dd88a341570fcd4eeaab201c112ba44c43de5b)

### Docs

- Add documentation for the quantized_modules package ([9da4a0d]( https://github.com/es-ude/elastic-ai.creator/commit/9da4a0d380304a7ab8834049ad93bed547816ddb)) - [object]

### Feat

- Add example using quantized modules to verify the current state of the translator ([0c55e00]( https://github.com/es-ude/elastic-ai.creator/commit/0c55e00657c0d260766155995b75f25bff642e24)) - [object]
- Integrate fixed point hard sigmoid to the translator ([0a07cee]( https://github.com/es-ude/elastic-ai.creator/commit/0a07ceeb3d238456dad08448b543f4a075873322)) - [object]

### Refactor

- Change name of the output_quant parameter to output_dequant to be more precise ([3186b5b]( https://github.com/es-ude/elastic-ai.creator/commit/3186b5b848e4b7be8e0bc0d94a1897722d2e2397)) - [object]
- Rename example to fit its actual content ([9ac5a83]( https://github.com/es-ude/elastic-ai.creator/commit/9ac5a83a558887d0cf4830a6f7ba94ede92de594)) - [object]
- Remove unused import ([fbb684d]( https://github.com/es-ude/elastic-ai.creator/commit/fbb684daaeb8376ec7a56b413959cb9e9f2dc600)) - [object]

### Test

- Add tests to test the quant and dequant parameters of the LinearBase class ([ad49bc6]( https://github.com/es-ude/elastic-ai.creator/commit/ad49bc68ef5f38e6047d569d8e90513f50698a27)) - [object]

## 0.19.0 - 2022-11-05

[51f771e](51f771e64860e869cf9f35ffc21a027e4d1a0f72)...[7568be6](7568be66864623be1b31c27db21573fb18efe0e6)

### Feat

- Merge translate_model and generate_code functions ([c12562e]( https://github.com/es-ude/elastic-ai.creator/commit/c12562ee4a55c61b5ef82b5ef37568fe32e8f525)) - [object]

## 0.18.0 - 2022-11-04

[bba0d93](bba0d93501bfad3cc38a640c33c0afbc71f7c7f6)...[51f771e](51f771e64860e869cf9f35ffc21a027e4d1a0f72)

### Feat

- Refactoring and start implementing hard sigmoid activation function ([ff94c9d]( https://github.com/es-ude/elastic-ai.creator/commit/ff94c9dd1d1297f02e82a0d1f7f203f80c8d2732)) - [object]
- Fix wrong calculation of fixed point values and add quantized forward functions ([93046d3]( https://github.com/es-ude/elastic-ai.creator/commit/93046d3b93d1a977c4106cf56e7f98847a47aa00)) - [object]
- Implement a version of relu for qat and quantized inference ([ddd9607]( https://github.com/es-ude/elastic-ai.creator/commit/ddd9607e8dbf333817112dfe24f795ac717f609e)) - [object]
- Use fixed point hard sigmoid and relu in the example ([90350b9]( https://github.com/es-ude/elastic-ai.creator/commit/90350b91b9ac917c8c1f0ab50c2744fb09671947)) - [object]
- Implement evaluator for simulation of a quantized inference ([353e82e]( https://github.com/es-ude/elastic-ai.creator/commit/353e82e798359c3b15a42a02dcdc63e071b2d34e)) - [object]
- Implement evaluator that evaluates a model according to a given metric ([a0b089a]( https://github.com/es-ude/elastic-ai.creator/commit/a0b089ad1f7c32acc0c4522bf830080442e8414d)) - [object]
- Add simulated fixed point inference to the example ([4f81d8d]( https://github.com/es-ude/elastic-ai.creator/commit/4f81d8d3d44f1c677fc1a12edf94b7b614d72efb)) - [object]
- Add clamp to min or max fixed point integer for overflowing values ([ca3fc19]( https://github.com/es-ude/elastic-ai.creator/commit/ca3fc19aec062d4de34a4698c9e0a9351b41c761)) - [object]

### Refactor

- Create a better module structure ([b2dfeee]( https://github.com/es-ude/elastic-ai.creator/commit/b2dfeee795d980aabc2822e3b1470f2e41d63416)) - [object]
- Removed unfinished fixed point configuration finder ([fa6dc44]( https://github.com/es-ude/elastic-ai.creator/commit/fa6dc44e0a02f57993f08b381b62297c2682b167)) - [object]
- Small changes to make the code easier to understand ([7168ba1]( https://github.com/es-ude/elastic-ai.creator/commit/7168ba145243616d247006f56d99de3c21e91401)) - [object]
- Make floating point values more explicit ([231b903]( https://github.com/es-ude/elastic-ai.creator/commit/231b903127dcdc0b90bc6a4e29ccd29543033935)) - [object]
- Remove unused line of code ([f020554]( https://github.com/es-ude/elastic-ai.creator/commit/f020554dc3f1ae6ed4ac025711e6ec1025ba8964)) - [object]
- Rename fixed point linear layer example ([59216da]( https://github.com/es-ude/elastic-ai.creator/commit/59216da6973daca87e80c105513586df1c682ba6)) - [object]

### Test

- Write tests for the evaluators and do some refactoring ([2641578]( https://github.com/es-ude/elastic-ai.creator/commit/2641578eb1820793e4e2117563dc11607707e11d)) - [object]

## 0.17.0 - 2022-10-22

[0ba60d8](0ba60d8bdc85599868bbb96281d94acf2d47b39e)...[bba0d93](bba0d93501bfad3cc38a640c33c0afbc71f7c7f6)

### Feat

- Visualize model parameters ([5e1b4fc]( https://github.com/es-ude/elastic-ai.creator/commit/5e1b4fc4c827c55d19cb9bc4206f706bcc737fba)) - [object]

## 0.16.0 - 2022-10-22

[01d8c35](01d8c3518096459326be48435e0df35b4960e105)...[0ba60d8](0ba60d8bdc85599868bbb96281d94acf2d47b39e)

### Chore

- Skip some tests that are not relevant at the moment ([2dbfd55]( https://github.com/es-ude/elastic-ai.creator/commit/2dbfd55a805166e97b640fafd5ebc1214288a863)) - [object]

### Feat

- Add function to get attribute names of an object matching a regex ([acc8e29]( https://github.com/es-ude/elastic-ai.creator/commit/acc8e29e2771d5642e1371af6fb3c44f83b5ebc7)) - [object]
- Implement custom linear layers that allows to do fixed point calculations ([c2364f6]( https://github.com/es-ude/elastic-ai.creator/commit/c2364f6182bb8406e90a78d632bc868537705fd2)) - [object]
- Add a type for fixed point factories ([53b8499]( https://github.com/es-ude/elastic-ai.creator/commit/53b84991671832c2e7fa24e61d927b7c039832d9)) - [object]
- Make base linear package private ([b3cfa55]( https://github.com/es-ude/elastic-ai.creator/commit/b3cfa55daff5c401bc036ffe2bba8b0c6b2f2554)) - [object]
- Move tracing example to example folder ([b942155]( https://github.com/es-ude/elastic-ai.creator/commit/b942155a240a6f34f0f02361b6631b431a448443)) - [object]
- Start implementing an example for learning a simple logic function ([6cff6de]( https://github.com/es-ude/elastic-ai.creator/commit/6cff6deccd5c2080e930d93f5e145e4d7ea6a41e)) - [object]
- Add feature to automatically derive fixed point parameters from a factory ([70618d5]( https://github.com/es-ude/elastic-ai.creator/commit/70618d512718efd7e718491af52e1acbc6c86622)) - [object]
- Commit current state of the fixed point linear example ([9b8ecae]( https://github.com/es-ude/elastic-ai.creator/commit/9b8ecae971bc1dedabf17e79272008a3cbfb5123)) - [object]
- Move the input, weight and output quantization to the linear layer ([0c8b259]( https://github.com/es-ude/elastic-ai.creator/commit/0c8b259ef688c606ebd4f8486ef7b6f48e0f8713)) - [object]
- Add the ability to plot the model parameters ([b1b0b5e]( https://github.com/es-ude/elastic-ai.creator/commit/b1b0b5e7697992c4c53825c739e2fb2dcc903dac)) - [object]
- Implement qat for linear layer ([d3ba49e]( https://github.com/es-ude/elastic-ai.creator/commit/d3ba49e266b2931c1b16677dd91f17a75f091501)) - [object]

### Fix

- Fix bug in the linear matix multiplication and rename _BaseLinear layer to _LinearBase ([da11356]( https://github.com/es-ude/elastic-ai.creator/commit/da113561d69158ccc2a9266adb1eddcc79b1cb7d)) - [object]

### Refactor

- Remove unused typevars and change typing ([2770991]( https://github.com/es-ude/elastic-ai.creator/commit/2770991c00a9395697180fcec98e733164efde24)) - [object]

## 0.15.0 - 2022-09-29

[393d96b](393d96b92bba7c9c47f8ee222e2dda619ed40259)...[01d8c35](01d8c3518096459326be48435e0df35b4960e105)

### Feat

- Implement clipped fixed point representation ([8e53506]( https://github.com/es-ude/elastic-ai.creator/commit/8e53506fce0ba5adaa124ccd61de3b340bf1c95f)) - [object]

## 0.14.0 - 2022-09-28

[5581b87](5581b878ae1302451ab51af81cecee3d6e9c60ed)...[393d96b](393d96b92bba7c9c47f8ee222e2dda619ed40259)

### Feat

- Implement automatic derivation of fixed point parameters in the lstm example ([504008d]( https://github.com/es-ude/elastic-ai.creator/commit/504008d7ef3f402f8476bb77f02a4a37176d229e)) - [object]
- Implement signed fixed point integer to FixedPoint object ([0a2fc79]( https://github.com/es-ude/elastic-ai.creator/commit/0a2fc7952dc13ea48c749856bf809a5540166598)) - [object]
- Start implementing fixed point evaluator ([0f9c62a]( https://github.com/es-ude/elastic-ai.creator/commit/0f9c62a38f9df6ee4e84f1a3b5524df03511b438)) - [object]
- Working further on the fixed point configuration finder ([beb9da0]( https://github.com/es-ude/elastic-ai.creator/commit/beb9da0ec8c3fbc6bb4ff65a97e7424e4da6dd0d)) - [object]
- Implement from_unsigned_int and from_signed_int function and remove unused function ([aca77f5]( https://github.com/es-ude/elastic-ai.creator/commit/aca77f5eac396f21821b07706ff250b2589dd037)) - [object]

### Fix

- Reimplement unsigned_int_values_to_fixed_point function ([cdd069e]( https://github.com/es-ude/elastic-ai.creator/commit/cdd069e6adffa882bb34fea2b7179891c282045b)) - [object]

## 0.13.0 - 2022-09-10

[653835a](653835a3ddb383bfed60613928eab732b5f93855)...[5581b87](5581b878ae1302451ab51af81cecee3d6e9c60ed)

### Feat

- Remove translatable protocol ([37412e8]( https://github.com/es-ude/elastic-ai.creator/commit/37412e87d89d16c9159cf12ef00032343119100c)) - [object]
- Explicitly set poetry version ([82d202f]( https://github.com/es-ude/elastic-ai.creator/commit/82d202f0229e7931fc7371f69abe0d1fe3a58134)) - [object]

### Style

- Beautify commit # [2024c7e](https://github.com/es-ude/elastic-ai.creator/commit/2024c7e9a8aa9ed2487f58586aa41beabd6f63d2) ([7a58041]( https://github.com/es-ude/elastic-ai.creator/commit/7a58041b71ccd258d9fcb16b1ac1a15be32e212d)) - [object]

## 0.12.1 - 2022-08-29

[02d0ba3](02d0ba3850d892626ee39e0adb31939605b5836e)...[653835a](653835a3ddb383bfed60613928eab732b5f93855)

### Fix

- Reimplement binarize ([9bbccdd]( https://github.com/es-ude/elastic-ai.creator/commit/9bbccddfc6ce6c2b928166cdfaf1112b294dba17)) - [object]

### Style

- Beautify commit # [7a60a04](https://github.com/es-ude/elastic-ai.creator/commit/7a60a043e83aedcdf281ec9357ee9f274aca59dd) ([0723cb1]( https://github.com/es-ude/elastic-ai.creator/commit/0723cb12e8fc6290403efd68b6d552a81ad69a99)) - [object]

## 0.12.0 - 2022-08-26

[ff37f26](ff37f26d0f570d154e912c51cfffe84a755484fd)...[02d0ba3](02d0ba3850d892626ee39e0adb31939605b5836e)

### Chore

- Add a dockerfile ([e2f54b3]( https://github.com/es-ude/elastic-ai.creator/commit/e2f54b373bf26ee6c94b0c0c448b6e34affb2e64)) - [object]

### Docs

- Update documentation according the newly added linear1d layer ([41e2486]( https://github.com/es-ude/elastic-ai.creator/commit/41e24868aecbf310ee4c9ad815f6ccc0da3f9f9b)) - [object]
- Small changes of the documentation ([9e7699c]( https://github.com/es-ude/elastic-ai.creator/commit/9e7699ce617581f67f85cf4ef7d945d99df241be)) - [object]
- Move translator documentation to the vhdl package ([9a90949]( https://github.com/es-ude/elastic-ai.creator/commit/9a90949528978ff4732f585986a71cedd44e82a5)) - [object]
- Adapt diagrams to the latest changes ([c1750eb]( https://github.com/es-ude/elastic-ai.creator/commit/c1750eb19f92a705f8f36ccefc9729d3545f0743)) - [object]

### Feat

- Insert values in the updated lstm.vhd template ([4d9dccb]( https://github.com/es-ude/elastic-ai.creator/commit/4d9dccbdb11afebb466f476c22539828bf5458b1)) - [object]
- Make work library name customizable ([95fd8aa]( https://github.com/es-ude/elastic-ai.creator/commit/95fd8aa0d7e512aeb04893de2df2e58cc4b3e641)) - [object]

### Fix

- Fix calculation of the addr_width of the linear1d layer ([6fa2b2a]( https://github.com/es-ude/elastic-ai.creator/commit/6fa2b2a3bc83d3a51eb955d1464501662f6676a8)) - [object]
- Pre-add input-hidden and hidden-hidden bias ([750941c]( https://github.com/es-ude/elastic-ai.creator/commit/750941c3150cabefa2f393f6b12105a358a70f7f)) - [object]
- Add changes from Chao after testing the translator ([5a5d532]( https://github.com/es-ude/elastic-ai.creator/commit/5a5d5325a3f598e0163d4eac0601b5961c2f5780)) - [object]
- Fix test ([c125bf1]( https://github.com/es-ude/elastic-ai.creator/commit/c125bf16297ee9e39660ee904ab54268e8901d48)) - [object]
- Remove unused work library ([c68fd9d]( https://github.com/es-ude/elastic-ai.creator/commit/c68fd9d00b152c5bdb70d2d2c90ca8d3e9f381d0)) - [object]
- Remove some comments ([13cc1a1]( https://github.com/es-ude/elastic-ai.creator/commit/13cc1a1ade14ccc7aa686523270dec20936ed14d)) - [object]

### Refactor

- Simplify code and reuse components ([37cffd7]( https://github.com/es-ude/elastic-ai.creator/commit/37cffd76953e1f7756de8ec7ebc5b356fb89f1ad)) - [object]

## 0.11.1 - 2022-08-18

[9f2babe](9f2babede009ac27b1587175403d8fe10c735f16)...[ff37f26](ff37f26d0f570d154e912c51cfffe84a755484fd)

### Build

- Perform tests with more verbose output ([8d7b50b]( https://github.com/es-ude/elastic-ai.creator/commit/8d7b50b7ae2c6d513f67027e5f209cf3115e0964)) - [object]

### Docs

- Add documentation on how the translator works ([91ebea3]( https://github.com/es-ude/elastic-ai.creator/commit/91ebea3fb7e7883f56b2cd9152769d151449a49a)) - [object]

### Feat

- Implement the translation of a linear1d layer ([b627e78]( https://github.com/es-ude/elastic-ai.creator/commit/b627e780d054adcdd89009d87aa33fa31c913504)) - [object]
- Add an easy way to get a fixed point factory ([d98ff03]( https://github.com/es-ude/elastic-ai.creator/commit/d98ff0351f739859ed668a2ec295421e29fd24ec)) - [object]
- Add linear layer to the translation example ([5f1e1db]( https://github.com/es-ude/elastic-ai.creator/commit/5f1e1db8da7ce533cb592d56ca97e25ca563a60e)) - [object]

### Fix

- Remove deprecated threshold and codomain properties ([5db9669]( https://github.com/es-ude/elastic-ai.creator/commit/5db9669fc3942851e65607a869bb822430df7836)) - [object]

### Refactor

- Change naming of the translator components ([fdf5586]( https://github.com/es-ude/elastic-ai.creator/commit/fdf5586da727542be6bfab57fba4a98d8ec482d7)) - [object]
- Change naming for better understanding ([17d8a3d]( https://github.com/es-ude/elastic-ai.creator/commit/17d8a3d89dcbbb4882c62953ddb928d268945852)) - [object]
- Small naming changes ([fd5c9b4]( https://github.com/es-ude/elastic-ai.creator/commit/fd5c9b4f9fccb95b9fd4a0223e87a791fd02224c)) - [object]

### Style

- Beautify commit # [52e7e3e](https://github.com/es-ude/elastic-ai.creator/commit/52e7e3e55053a9e95e786bf899056148753cddfc) ([a5c17b4]( https://github.com/es-ude/elastic-ai.creator/commit/a5c17b428c20f8c55b7c9350e5d9a33ef8b76822)) - [object]

### Test

- Add tests for rom and linear1d component ([fc1f20e]( https://github.com/es-ude/elastic-ai.creator/commit/fc1f20e30bf91f6aa96ee800e2e66aa5b3c217ad)) - [object]

## 0.11.0 - 2022-08-11

[0288e04](0288e04e546dbc91d60884588fe5a31e0f81fa7f)...[9f2babe](9f2babede009ac27b1587175403d8fe10c735f16)

### Docs

- Fix commands of install dev dependencies ([870e2de]( https://github.com/es-ude/elastic-ai.creator/commit/870e2de30f48223d8005bcf1240b624ebb314ad7)) - [object]
- Add some docstrings to the functions of the translator ([6f9215e]( https://github.com/es-ude/elastic-ai.creator/commit/6f9215e5fc35287517d884a702bf887d7a09aa7f)) - [object]

### Feat

- Implementation of a LSTMCell class that can be translated to VHDL ([ace37fe]( https://github.com/es-ude/elastic-ai.creator/commit/ace37fe4b215327bc5b43344ffcd0c44a4822dda)) - [object]
- Add ability to pass kwargs to the translate function of a translatable layer ([196812e]( https://github.com/es-ude/elastic-ai.creator/commit/196812eecd0dc49a1b8c2d6675b9018ca07e003e)) - [object]
- Add a protocol specify a translatable layer ([0fa966e]( https://github.com/es-ude/elastic-ai.creator/commit/0fa966e7f99ef2adb19321b3ca92202616b4c0a2)) - [object]
- Introduce translation arguments ([2c3a8c7]( https://github.com/es-ude/elastic-ai.creator/commit/2c3a8c72cfe8df70fd960e692d4fe037e2e86b6f)) - [object]
- Abstract LSTM cell takes float weights instead of FixedPoint weights ([a5818cc]( https://github.com/es-ude/elastic-ai.creator/commit/a5818cc0edd918ef3ca49e843738823e988bfd79)) - [object]
- Add a build function to create an abstract LSTMCell object from a PyTorch LSTMCell ([baca5bb]( https://github.com/es-ude/elastic-ai.creator/commit/baca5bb6c22692cf9bfc02a9147711b8869930fd)) - [object]
- Use __init__ files to simplify the usage ([3cc07ee]( https://github.com/es-ude/elastic-ai.creator/commit/3cc07ee048a349ef5a6a5383dcd829d64b48de2d)) - [object]
- Implementation of the mapping of a torch module to the corresponding build function ([b076fa3]( https://github.com/es-ude/elastic-ai.creator/commit/b076fa32cef3c64f8fcc45df24814f4333c90b5c)) - [object]
- First untested draft for the pytorch translator ([7e59462]( https://github.com/es-ude/elastic-ai.creator/commit/7e5946259381af397e1ccd25006815af8256026f)) - [object]
- Add the ability to infer the build function from a given layer object or type ([306df14]( https://github.com/es-ude/elastic-ai.creator/commit/306df1427177d15c1b1e2c59b2e774a2a6e2c471)) - [object]
- Add an example using the vhdl translator for pytorch ([395adcd]( https://github.com/es-ude/elastic-ai.creator/commit/395adcd3e843b7f55f6156ba183dc8800055ef51)) - [object]
- Implement a more functional build function mapping ([1425e03]( https://github.com/es-ude/elastic-ai.creator/commit/1425e0304cf35617106199936d3b014c0d8ca483)) - [object]
- Pass an DTO to a translatable instead of raw arguments to fix typing errors ([4738725]( https://github.com/es-ude/elastic-ai.creator/commit/4738725d09ca9114064c4c42dd2818fc6d5c973b)) - [object]
- Add LSTMCellTranslationArguments to __init__.py file ([061ead4]( https://github.com/es-ude/elastic-ai.creator/commit/061ead404dc82ddc79ac75c155328ad5733eb04a)) - [object]
- Change build function mapping to a different approach ([b1b79b2]( https://github.com/es-ude/elastic-ai.creator/commit/b1b79b2e5e9ea0cf627b16a41f1f75bf434b795e)) - [object]
- Make build function mapping more general so that it can be reused for other frameworks ([3369d7f]( https://github.com/es-ude/elastic-ai.creator/commit/3369d7fb6a7d08930514a7c0553c9efe65fc54b9)) - [object]
- Removed the possibility to get a build function from a type ([dbc2e8f]( https://github.com/es-ude/elastic-ai.creator/commit/dbc2e8ffd95f5ddc2476fede9d170c9d4eb020c2)) - [object]
- Change translation from LSTMCell to LSTM ([5e4f1cf]( https://github.com/es-ude/elastic-ai.creator/commit/5e4f1cff380fabd3685660a0c279b9098c4ef278)) - [object]
- Adapt the example to the changes of the translation ([6a5644e]( https://github.com/es-ude/elastic-ai.creator/commit/6a5644e30a7cd00ed1be1c2cb6fa2e0b4b114c1e)) - [object]

### Fix

- Fix test ([528910c]( https://github.com/es-ude/elastic-ai.creator/commit/528910cf3fe28958ebb7b246104e83df77bbf3f4)) - [object]
- Fix wrong pytorch lstm cell class path ([85a733c]( https://github.com/es-ude/elastic-ai.creator/commit/85a733cb5ff821bb602b5021f6438b7d5909382e)) - [object]
- Fix mypy typing errors ([e1dba31]( https://github.com/es-ude/elastic-ai.creator/commit/e1dba317585c269ad58719184fb4764cc66485ae)) - [object]
- Use LSTMTranslationArguments object instead of a dictionary ([98a4d97]( https://github.com/es-ude/elastic-ai.creator/commit/98a4d97f8fbd217f67ed4009ab63ccc4705f720d)) - [object]
- Rename LSTMCell translatable to LSTM ([e05cd04]( https://github.com/es-ude/elastic-ai.creator/commit/e05cd042daf0420b2046607e00eeef3606a6defb)) - [object]
- Remove print call ([55164b7]( https://github.com/es-ude/elastic-ai.creator/commit/55164b78c61f37f4cdadde0385965ee540e4f555)) - [object]

### Refactor

- Remove custom template mapping, fixed point args and use protocols instead of abc ([fed2658]( https://github.com/es-ude/elastic-ai.creator/commit/fed26585c8ccd123e590476b8e0a8ec4df8891f6)) - [object]
- Change typings and Translatable yield VHDLComponent ([eedacb1]( https://github.com/es-ude/elastic-ai.creator/commit/eedacb16afaf805eb6a990aa1ad40273722e02a3)) - [object]
- Use better typing ([86a019d]( https://github.com/es-ude/elastic-ai.creator/commit/86a019d6d3db8696850b65047481e9566da66cd8)) - [object]
- Correct name of a test ([73d360f]( https://github.com/es-ude/elastic-ai.creator/commit/73d360f9f3c9fc6fdf5380ff45c947b49f475199)) - [object]
- Vhdl module type is now an iterable instead of an iterator to be more flexible ([1b471ca]( https://github.com/es-ude/elastic-ai.creator/commit/1b471ca3a8f5b3ff3c7c28e105ae3f7f2419367d)) - [object]
- Change some names to make the code more understandable ([a8d8b0c]( https://github.com/es-ude/elastic-ai.creator/commit/a8d8b0c2fd3a27911a530db20dc3596113fc80e8)) - [object]
- Change the name of an example ([606c0a3]( https://github.com/es-ude/elastic-ai.creator/commit/606c0a30e37e5bd7d7ddc5529c770594debd7605)) - [object]
- Remove empty module ([03edaca]( https://github.com/es-ude/elastic-ai.creator/commit/03edaca097759ff381b012f757631662c4b5fe3a)) - [object]

### Test

- Add tests for the abstract LSTMCell layer ([643a91f]( https://github.com/es-ude/elastic-ai.creator/commit/643a91fa8f569fb002200039a253c9a9a79e5373)) - [object]
- Add tests for the translator and the lstm_cell build function ([a92987d]( https://github.com/es-ude/elastic-ai.creator/commit/a92987dfff0e3e2ab7646e984d5309383a0f9681)) - [object]
- Add test that should pass in the future to check the correct layer ordering ([3e5b452]( https://github.com/es-ude/elastic-ai.creator/commit/3e5b45266454ce8df5858990314e8702a0db0345)) - [object]
- Add tests for the build function mapping ([4e4fee2]( https://github.com/es-ude/elastic-ai.creator/commit/4e4fee2971a4131174dc6286ff85d9f4e0795611)) - [object]
- Fixed tests of the build function mapping ([7885cdb]( https://github.com/es-ude/elastic-ai.creator/commit/7885cdb30a413070816442c0e6daf2c0400b2743)) - [object]

## 0.10.1 - 2022-06-29

[aaf7ff9](aaf7ff97e6661209388845c25686f2f2e88f702a)...[0288e04](0288e04e546dbc91d60884588fe5a31e0f81fa7f)

### Chore

- Try to fix/investigate the error with the semantic release tool ([7697877]( https://github.com/es-ude/elastic-ai.creator/commit/7697877c44fa382bf0fd3838077078d61b5117dc)) - [object]

### Fix

- Try to fix the error with the semantic release tool ([bc115f8]( https://github.com/es-ude/elastic-ai.creator/commit/bc115f899bd85e720448bfa67fe9964bb56c594b)) - [object]
- Fix error in the main.yml ([4a6ff5e]( https://github.com/es-ude/elastic-ai.creator/commit/4a6ff5e61f35661a3ef83ce4335c109333834d6d)) - [object]

## 0.10.0 - 2022-06-26

[de00f90](de00f90e6d4c9a17ec5831ad73b371aa7a9ac822)...[aaf7ff9](aaf7ff97e6661209388845c25686f2f2e88f702a)

### Chore

- Update numpy, onnx and add pre-commit to dev dependencies ([a23c00a]( https://github.com/es-ude/elastic-ai.creator/commit/a23c00ad3faef0ed5e2318f83553ad243749c920)) - [object]
- Add matplotlib dependency for the qlstm example ([dadbc20]( https://github.com/es-ude/elastic-ai.creator/commit/dadbc20f5e4328d6475c418277b08059b9ba1391)) - [object]
- Removing the compilation steps for onnx as they are no longer needed ([1118ee4]( https://github.com/es-ude/elastic-ai.creator/commit/1118ee4ad89713a56d8a36fb93be46f0a2a33a32)) - [object]

### Ci

- Use python3.10 for semantic release ([d7c5b6b]( https://github.com/es-ude/elastic-ai.creator/commit/d7c5b6b6fc59ca88532defeb48894a0c792601d6)) - [object]

### Docs

- Add a docstring with an example to the FixedPoint class ([961d766]( https://github.com/es-ude/elastic-ai.creator/commit/961d76678d366730f57bbd69b43c38124c003bf7)) - [object]
- Remove compile instructions for onnx as they are no longer needed ([3bee70a]( https://github.com/es-ude/elastic-ai.creator/commit/3bee70abe4185a0a6708ffc998bdc74004f90b8a)) - [object]

### Feat

- Format_vhdl function blocks the process until formatting is complete ([a8a1bd0]( https://github.com/es-ude/elastic-ai.creator/commit/a8a1bd0e7a4db075d0cef4a9eb125a860a697719)) - [object]

### Refactor

- Remove unused file ([577c91e]( https://github.com/es-ude/elastic-ai.creator/commit/577c91ed4279ed7dbcdae71b5f4e8f868f6092ab)) - [object]
- Apply python3.10 typing, renaming functions/classes, remove unused imports ([15f4b8a]( https://github.com/es-ude/elastic-ai.creator/commit/15f4b8a52c78a680e3ad95fc70dbd85864282606)) - [object]
- Apply python3.10 typing, remove unused imports ([f2a31c6]( https://github.com/es-ude/elastic-ai.creator/commit/f2a31c6d7d75e1f545ea63cd1ed6f19dc7be7249)) - [object]
- Move _int_to_bin_str to ToLogicEncoder class and refactor the class ([c6495a0]( https://github.com/es-ude/elastic-ai.creator/commit/c6495a05c77962ce4cfb4a4110bf0add74d11869)) - [object]
- Set correct typings ([600f6fb]( https://github.com/es-ude/elastic-ai.creator/commit/600f6fb9db4e908e7c6eda4652af858258c903aa)) - [object]
- Add missing typing ([1e58596]( https://github.com/es-ude/elastic-ai.creator/commit/1e58596b12eef51de75fe01f60529271f4caaa6b)) - [object]

## 0.9.0 - 2022-06-22

[0e62c95](0e62c950b6f56592db4a61fe2af7aae9b649d4e3)...[de00f90](de00f90e6d4c9a17ec5831ad73b371aa7a9ac822)

### Feat

- Separate hex/bin representation from vhdl hex/bin representation ([eb8fe60]( https://github.com/es-ude/elastic-ai.creator/commit/eb8fe60300ee7572500f9f9d11b62a9c5abff802)) - [object]
- Add a function to infer total and frac bits from a sequence of FixedPoint values ([9cc2b72]( https://github.com/es-ude/elastic-ai.creator/commit/9cc2b721b147628b2abf524129eeaac8f68520d5)) - [object]
- Change Rom that it uses the FixedPoint datatype ([876cdb8]( https://github.com/es-ude/elastic-ai.creator/commit/876cdb821ff0ac67ae2345c8a36e4a742cce0949)) - [object]
- Add function to convert list of float values to a list of FixedPoint objects ([02b26d8]( https://github.com/es-ude/elastic-ai.creator/commit/02b26d868cad2a5a5bed2350a2929cf362ccdca8)) - [object]
- Add function to convert a list of ints to a list of FixedPoint objects ([abece1f]( https://github.com/es-ude/elastic-ai.creator/commit/abece1fd38af607c5f5734aeacd77a1743ff3411)) - [object]
- Integrate FixedPoint datatype in the LSTM test bench classes ([7cbb88a]( https://github.com/es-ude/elastic-ai.creator/commit/7cbb88a7f77728776e0e976dcc68505b4162f0cc)) - [object]
- Verify total bits and frac bits for multiple lists ([360d318]( https://github.com/es-ude/elastic-ai.creator/commit/360d318db0076d9077ceb94f3f7904d95e2b12f6)) - [object]
- Add function to convert FixedPoint to signed int representation ([03001ed]( https://github.com/es-ude/elastic-ai.creator/commit/03001ed608ac934e8bbdcdfa1acb2fc7c163a89a)) - [object]
- Integrate FixedPoint type ([b67a609]( https://github.com/es-ude/elastic-ai.creator/commit/b67a6096023a51ff4882a8cdd03a7765884c8d93)) - [object]

### Fix

- Change value so that it fits into the value range of a fixed point value ([b4e973e]( https://github.com/es-ude/elastic-ai.creator/commit/b4e973ebb8a087351e07966821229f69dc345d79)) - [object]
- Remove old brevitas code ([86a8104]( https://github.com/es-ude/elastic-ai.creator/commit/86a8104cb6049dc016a5c8da08a7d2abc011935b)) - [object]
- Correct usage of the lookup_table_generator_function according to the type hints ([9812ee8]( https://github.com/es-ude/elastic-ai.creator/commit/9812ee85cd467e261af942b30493ac0e970ea5e4)) - [object]

### Refactor

- Apply python 3.10 typing ([1c73b26]( https://github.com/es-ude/elastic-ai.creator/commit/1c73b265bb8c935d8618f0355c50ca42d1b47168)) - [object]
- Small code quality improvement by using the chain function ([6517cdd]( https://github.com/es-ude/elastic-ai.creator/commit/6517cdd4090574d2be5bbbdf6ae68571d6679f05)) - [object]
- Use resource_utils instead of importlib directly ([a7598b4]( https://github.com/es-ude/elastic-ai.creator/commit/a7598b4779d98dc0a843f6a90d459f70c6d632f3)) - [object]
- Merge gen_func_for_one_lstm_cell and gen_func_for_lstm_layer in one module ([06158a9]( https://github.com/es-ude/elastic-ai.creator/commit/06158a92feca1023dd1b781da691a6965529c842)) - [object]
- Remove no longer needed fixed point converter classes ([4fcf0d1]( https://github.com/es-ude/elastic-ai.creator/commit/4fcf0d16cdcac082c19dd654210b5d37991f9139)) - [object]

## 0.8.0 - 2022-06-08

[b9afe97](b9afe9718b1ca01aeac3f49a41c8ae6967b8047e)...[0e62c95](0e62c950b6f56592db4a61fe2af7aae9b649d4e3)

### Feat

- Bump python version to 3.10 ([47f5f07]( https://github.com/es-ude/elastic-ai.creator/commit/47f5f0718460a966faaa937b2c6b016720434082)) - [object]
- Drop brevitas support ([103f188]( https://github.com/es-ude/elastic-ai.creator/commit/103f1882c8da81cdf114f10b1b76c2ce89a07cba)) - [object]
- Increase python version ([02403e6]( https://github.com/es-ude/elastic-ai.creator/commit/02403e6cb7d8c9acc4357d9649fd2ae0834030a0)) - [object]

### Fix

- Resolve dependency version conflicts ([32bd544]( https://github.com/es-ude/elastic-ai.creator/commit/32bd544b2e74b8b57497f3fd604deb5ed86ebb42)) - [object]
- Fix dependencies + onnx integration tests ([f06d0f8]( https://github.com/es-ude/elastic-ai.creator/commit/f06d0f8436ca2a7ed3410aee4ad36df1cdad45c0)) - [object]
- Specify exact python version in github workflow ([f3ffb18]( https://github.com/es-ude/elastic-ai.creator/commit/f3ffb183e86b722cec5efb31e0937c4810542aef)) - [object]
- Set correct version numbers and add protobuf dependency ([260e5fb]( https://github.com/es-ude/elastic-ai.creator/commit/260e5fb31c425ad9ba2ec31f2fa292961fd28ffa)) - [object]
- Fix import of Sequence type ([2c463ac]( https://github.com/es-ude/elastic-ai.creator/commit/2c463acdbdae0ed7dc9fa99730f53db94deb7142)) - [object]
- Set more explicit python version ([9c44093]( https://github.com/es-ude/elastic-ai.creator/commit/9c44093c6cd41d05a2d178e6e113bd10f7b86016)) - [object]
- Change ModuleProto to Module ([cfe418e]( https://github.com/es-ude/elastic-ai.creator/commit/cfe418e41889708a53c255a8a7abcd6f1648f8f2)) - [object]
- Correct deps ([7935ba1]( https://github.com/es-ude/elastic-ai.creator/commit/7935ba19bcbda7e47ddbc358c12af3aa2a01df0a)) - [object]
- Update poetry lock file ([9230672]( https://github.com/es-ude/elastic-ai.creator/commit/92306722dabe5c4196e79a7cbbebab1e75ac3e6d)) - [object]

### Style

- Beautify commit # [6772407](https://github.com/es-ude/elastic-ai.creator/commit/6772407f9929e398f7e03858e91b02c52bc8e3ec) ([ecb21e2]( https://github.com/es-ude/elastic-ai.creator/commit/ecb21e271271e52d63c268b311d598fb8c86af15)) - [object]

## 0.7.0 - 2022-06-05

[f07eebc](f07eebcadd5db5e2748ea6a1539bc0498cc4ed09)...[b9afe97](b9afe9718b1ca01aeac3f49a41c8ae6967b8047e)

### Feat

- Start implementing FixedPoint datatype ([8c4f420]( https://github.com/es-ude/elastic-ai.creator/commit/8c4f42097ff416f8e9056af430bda01a5bd42df5)) - [object]
- Add rich comparison methods, multiple operators and a bit iterator to FixedPoint ([116b006]( https://github.com/es-ude/elastic-ai.creator/commit/116b00647c05ef6854d3cbd1ab0f79c58f0c450d)) - [object]

## 0.6.1 - 2022-05-27

[329f779](329f779e97bcb9e175409564b733f7a756996143)...[f07eebc](f07eebcadd5db5e2748ea6a1539bc0498cc4ed09)

### Fix

- Saving generated examples to a directory instead of a giving an explicit file path ([eb41d8d]( https://github.com/es-ude/elastic-ai.creator/commit/eb41d8db9af5171ac2826f41e98b5d85598b582d)) - [object]

## 0.6.0 - 2022-05-25

[77adbfd](77adbfd77513f261edfa6743f2be45dd59208046)...[329f779](329f779e97bcb9e175409564b733f7a756996143)

### Fix

- Fix previously broken imports ([bf694f8]( https://github.com/es-ude/elastic-ai.creator/commit/bf694f80fbd3a5478d99e8ae6b198a9e363569c9)) - [object], BREAKING CHANGE:move modules out of generator package
- Move missing files ([e4ae3c2]( https://github.com/es-ude/elastic-ai.creator/commit/e4ae3c2815a33b8f4f33c9578ab5cae0842277aa)) - [object]

### Refactor

- Remove usage of protected functions in tests ([47ca401]( https://github.com/es-ude/elastic-ai.creator/commit/47ca401e9c19f3f80140bc9c06c1a3e162c6849c)) - [object]

## 0.5.0 - 2022-05-25

[9f78ef1](9f78ef1d5de772e05a25affd7ee37788613110a4)...[77adbfd](77adbfd77513f261edfa6743f2be45dd59208046)

### Chore

- Remove deprecation warning about Q* layers ([c696596]( https://github.com/es-ude/elastic-ai.creator/commit/c6965961f37a5154356a9b299fc1de36888cd184)) - [object]

### Ci

- Remove outdated gitlab ci configs ([07832c8]( https://github.com/es-ude/elastic-ai.creator/commit/07832c85f62e3a71bed507d100685923d70bf424)) - [object]

### Docs

- Shorten ([e535ea0]( https://github.com/es-ude/elastic-ai.creator/commit/e535ea0fd9d783f29ebb32d756077289d8baa8c9)) - [object]
- Fix table of contents and section headers ([ecdef5d]( https://github.com/es-ude/elastic-ai.creator/commit/ecdef5da63c2c10e61a159c144c5c3707a5699e8)) - [object]
- Add git commit message scopes ([fe8e328]( https://github.com/es-ude/elastic-ai.creator/commit/fe8e328eda5a5f9e4cac886fcbfc9388f13d3d0f)) - [object]

### Feat

- Add multiline template expansion ([309ea35]( https://github.com/es-ude/elastic-ai.creator/commit/309ea350fae2b4e54bf06101aadc28e227d30cbb)) - [object]
- Make IOTable iterable and fix types ([faa1d77]( https://github.com/es-ude/elastic-ai.creator/commit/faa1d7799bd6e8223cc4953170286d425255bb7b)) - [object]

### Refactor

- Make IOTable grouping an IOTable method ([c97ec8c]( https://github.com/es-ude/elastic-ai.creator/commit/c97ec8c40e1f525a19cdc6838f73be312c209b10)) - [object]
- Move implementations to packages corresponding to scopes ([fa4487b]( https://github.com/es-ude/elastic-ai.creator/commit/fa4487b6d491f2f3b089000aca7fe04366b441d0)) - [object]
- Use correct numpy typing ([3d3ce3f]( https://github.com/es-ude/elastic-ai.creator/commit/3d3ce3fe11e96c882e5392cc98ca059addb2b145)) - [object]
- Move type defs ([1796473]( https://github.com/es-ude/elastic-ai.creator/commit/1796473e2cfb6c0e97c9562844f811878b2b518d)) - [object]

### Test

- Start implementation/integration of truth table design ([40f5396]( https://github.com/es-ude/elastic-ai.creator/commit/40f5396f6a207cb72b961a2900dbbefd59dbc5f1)) - [object]
- Create list from grouped tables to allow subscript ([0027a3d]( https://github.com/es-ude/elastic-ai.creator/commit/0027a3d7faf88d7c2c91f325685953f5fde4e347)) - [object]
- Move some test files ([dc7056c]( https://github.com/es-ude/elastic-ai.creator/commit/dc7056c177dff7517d16f3adf5fbfe568eeb85f1)) - [object]

## 0.4.2 - 2022-05-24

[12d377b](12d377b32d2fb4ecc8cece066a09dbba3df96cd3)...[9f78ef1](9f78ef1d5de772e05a25affd7ee37788613110a4)

### Fix

- Fix a bug and add some parameter checks ([a78e9e8]( https://github.com/es-ude/elastic-ai.creator/commit/a78e9e8f669c477d0629695f5c7c8ad8628f0522)) - [object]

### Test

- Add tests for the _int_to_bin_str and _int_to_hex_str functions ([002d7e2]( https://github.com/es-ude/elastic-ai.creator/commit/002d7e2cc20d6646973eb343d787392b28d65b26)) - [object]

## 0.4.1 - 2022-05-24

[0357cb8](0357cb84e621c99a6b01ecad9124b38456841f2f)...[12d377b](12d377b32d2fb4ecc8cece066a09dbba3df96cd3)

### Fix

- Minor errors ([812809e]( https://github.com/es-ude/elastic-ai.creator/commit/812809e1d0e706df3a0514b3503dc283ea12d7a4)) - [object]

### Test

- Fix errors in the intergation test of generating testbench ([d8378bf]( https://github.com/es-ude/elastic-ai.creator/commit/d8378bfa6afaaf84949b284c6b53884d5b5d4ff6)) - [object]

## 0.4.0 - 2022-05-23

[a7b76dd](a7b76dd4701ce32e19537d1abd2f577699203b48)...[0357cb8](0357cb84e621c99a6b01ecad9124b38456841f2f)

### Feat

- Allow ToLogicEncoder to register symbols in batches ([9209279]( https://github.com/es-ude/elastic-ai.creator/commit/9209279debe651b653d2fee44533ccbdae945b32)) - [object]

### Fix

- Improve names in scope of ToLogicEncoder ([67f7312]( https://github.com/es-ude/elastic-ai.creator/commit/67f73129faefe343e9fb5e84563d125b1d36bab6)) - [object], BREAKING CHANGE:rename numerics attr to _symbols,
mapping attribute to _mapping.
rename add_numeric to register_symbol

## 0.3.10 - 2022-05-23

[800fc4e](800fc4ec7ac3cbd717aaadbd3652207ee51760ef)...[a7b76dd](a7b76dd4701ce32e19537d1abd2f577699203b48)

### Chore

- Add coverage and onnx outputs ([a034785]( https://github.com/es-ude/elastic-ai.creator/commit/a03478528034243c1cbe8358890bb65a2845423c)) - [object]
- Add htmlcov produced from coverage ([5179c0d]( https://github.com/es-ude/elastic-ai.creator/commit/5179c0d526dd549fe06101342a33f89117acc022)) - [object]

### Ci

- Ignore main branch for checks ([56a6640]( https://github.com/es-ude/elastic-ai.creator/commit/56a6640d9839447880ca6b8e6ca495615bc86454)) - [object]
- Use pypi auth token instead of testpypi ([fa2faae]( https://github.com/es-ude/elastic-ai.creator/commit/fa2faae0e9fc4dc09b9de9f8b9c5032dfc104ecb)) - [object]
- Remove main branch from pull_request branch-ignore filter for checks.yml ([c081fd9]( https://github.com/es-ude/elastic-ai.creator/commit/c081fd984f3f8ea8bfa3bc4552980608d73badc3)) - [object]
- Publish release on github ([7c0fb1c]( https://github.com/es-ude/elastic-ai.creator/commit/7c0fb1cc74956257ce3fa93288c7f4dffacfef54)) - [object]

### Fix

- Fix some mypy errors ([35b8fdf]( https://github.com/es-ude/elastic-ai.creator/commit/35b8fdf4cb0736770d9592f86499192e1e84d673)) - [object]
- Add missing mlframework types ([3b5cf5f]( https://github.com/es-ude/elastic-ai.creator/commit/3b5cf5f8be829e109db363c25ecff76634f9d94f)) - [object]

### Test

- Move brevitas tests to integration tests ([7ba7757]( https://github.com/es-ude/elastic-ai.creator/commit/7ba7757e21245f6c418b5c12f5e3d1cc0bee9a7e)) - [object]

## 0.3.9 - 2022-05-22

[3238589](323858945e70baff146ee45a76206029f3d5537f)...[800fc4e](800fc4ec7ac3cbd717aaadbd3652207ee51760ef)

### Chore

- Update version number ([94bdcab]( https://github.com/es-ude/elastic-ai.creator/commit/94bdcabfa74fe4e634932eac6e4b5e36a02df236)) - [object]

### Ci

- Enable test-and-publish for main ([78870c1]( https://github.com/es-ude/elastic-ai.creator/commit/78870c1400b72c7d847b3da2d346f8eaf14fd619)) - [object]

## 0.3.8 - 2022-05-20

[3c4c2e1](3c4c2e181384367c6af92c6d9edd30f5110ee718)...[3238589](323858945e70baff146ee45a76206029f3d5537f)

### Chore

- Try semantic release ([8a23dbf]( https://github.com/es-ude/elastic-ai.creator/commit/8a23dbfeafeae82f1332c3cd28c5cbf72215a9c8)) - [object]
- Setup semantic versioning ([ac06cf2]( https://github.com/es-ude/elastic-ai.creator/commit/ac06cf26b4cfd66a3b49b82106ace1f236c01eb4)) - [object]
- Use emojis for automatic semver ([93a60cc]( https://github.com/es-ude/elastic-ai.creator/commit/93a60cc2755c08098b9c1a1f8ff5dfecee289c76)) - [object]
- Remove tag_utils.py ([a8baca4]( https://github.com/es-ude/elastic-ai.creator/commit/a8baca48073d6efa1330f82f87f23ec205ac02e9)) - [object]
- Revert to angular style commit messages ([55f99dd]( https://github.com/es-ude/elastic-ai.creator/commit/55f99ddd6f809169f91d707a51f29477523f26b0)) - [object]
- Automatically update changelog ([45bfef3]( https://github.com/es-ude/elastic-ai.creator/commit/45bfef38bd0dc3e86a9a291553b1f3ea5570dc9e)) - [object]
- Deploy to pypi instead of testpypi ([18aee87]( https://github.com/es-ude/elastic-ai.creator/commit/18aee872212ba9f066d579e4c2a5edd11e5b4a59)) - [object]
- Add node_modules ([23e1234]( https://github.com/es-ude/elastic-ai.creator/commit/23e12348b598edea69cf0a79e4bee26c45f62f43)) - [object]
- Build via semantic-release ([e78e882]( https://github.com/es-ude/elastic-ai.creator/commit/e78e882bb02b0a7adc9ff10c437a37bf6cc08dbc)) - [object]
- Upload to github ([cd1a8d5]( https://github.com/es-ude/elastic-ai.creator/commit/cd1a8d5a14a462db4bde32b769f88fa15aebaebc)) - [object]
- Manually trigger precommit workflow ([f6611d9]( https://github.com/es-ude/elastic-ai.creator/commit/f6611d9360f2b8a9ece7ace714050c85884fd6ce)) - [object]
- Trigger precommit on push ([391cc8e]( https://github.com/es-ude/elastic-ai.creator/commit/391cc8ef81c2d92bf432adb7814cbe95e9961c38)) - [object]
- Configure hook stages ([1d9ffc5]( https://github.com/es-ude/elastic-ai.creator/commit/1d9ffc57bdad9832c286d70412a2dcccce866f29)) - [object]
- Correct token for  test pypi ([112eb37]( https://github.com/es-ude/elastic-ai.creator/commit/112eb374c0f4b43b61b4988e8425a82881bd6802)) - [object]
- Default install commit-msg stage ([28c4716]( https://github.com/es-ude/elastic-ai.creator/commit/28c4716672a446623ad957b5f7f090f1eff211af)) - [object]
- Remove pre-commit usage ([9cd3f34]( https://github.com/es-ude/elastic-ai.creator/commit/9cd3f34c8e8b6ef1dc0904f071b1d2e3a2c0e684)) - [object]
- Put mypy+dead into manual stage ([98d9620]( https://github.com/es-ude/elastic-ai.creator/commit/98d9620f3a33a42a14d3dae04841660e82187609)) - [object]
- Add mypy cache ([0a8c31e]( https://github.com/es-ude/elastic-ai.creator/commit/0a8c31e0045ad244192bdb9fc91803a5d6470de1)) - [object]
- Add npm files ([352c38f]( https://github.com/es-ude/elastic-ai.creator/commit/352c38f3c83982b3abd52eb0d2bb1a654ff9bb57)) - [object]

### Ci

- Add test publish workflow ([473e533]( https://github.com/es-ude/elastic-ai.creator/commit/473e533dd8046e922416e121e60971a9ce2e9641)) - [object]
- Update repository url to test pypi ([fab7dbb]( https://github.com/es-ude/elastic-ai.creator/commit/fab7dbba3bd800189b04e1f13434440b6b1be603)) - [object]
- Use pre-commit in checks ([c3b741a]( https://github.com/es-ude/elastic-ai.creator/commit/c3b741a9554dd63ec02ee1044a96bfd2fa658bfe)) - [object]
- Move unit/integration tests to checks.yml ([a3ffe51]( https://github.com/es-ude/elastic-ai.creator/commit/a3ffe5137af975d516d00240f1517f58fbed9196)) - [object]
- Disable mypy typechecking ([f63ded1]( https://github.com/es-ude/elastic-ai.creator/commit/f63ded1fa71a9137d9e2502e7e9b693682116302)) - [object]
- Add next version placeholder to CHANGELOG.md ([2c94335]( https://github.com/es-ude/elastic-ai.creator/commit/2c94335e57dcac4cdfa1d612a4858e5fa0c34a8b)) - [object]

### Docs

- Update changelog ([e1aa8c9]( https://github.com/es-ude/elastic-ai.creator/commit/e1aa8c93554fc15c25a586b8e89eecda6dc03514)) - [object]
- Add brief explanation of pre-commit ([3626bb0]( https://github.com/es-ude/elastic-ai.creator/commit/3626bb07cc1c8600b193bc380ae8275116ebaba8)) - [object]

### Fix

- Fix syntax error ([895326d]( https://github.com/es-ude/elastic-ai.creator/commit/895326d67eb7ba1bb866a45c8b149778c93dc043)) - [object]
- Bump version ([2cb3a72]( https://github.com/es-ude/elastic-ai.creator/commit/2cb3a72b2aa9a86c0b4da71e3d7bff962a5728f6)) - [object]
- Close brace ([a6e4b99]( https://github.com/es-ude/elastic-ai.creator/commit/a6e4b999dadd163881fa96d03977c9c392a9267b)) - [object]
- Typo unit-test -> unit-tests ([1dbd71f]( https://github.com/es-ude/elastic-ai.creator/commit/1dbd71f5f3dae489b4752a5f6fdf9d10e4251a73)) - [object]
- Fix typo ([7f86205]( https://github.com/es-ude/elastic-ai.creator/commit/7f8620502ee544917db42ea12c7cb2eadbaef8cc)) - [object]
- Fix job dependencies ([6a7d3ee]( https://github.com/es-ude/elastic-ai.creator/commit/6a7d3eeb975ca303aa30fce21bd29d14cf9982d3)) - [object]
- Correct numpy namespace for type alias ([a6c5842]( https://github.com/es-ude/elastic-ai.creator/commit/a6c5842920c00ae6e53e226650e0fbfe48aac44a)) - [object]
- Add missing import of itertools ([a6b0344]( https://github.com/es-ude/elastic-ai.creator/commit/a6b0344ac4b933112b19b8603358a0adc7274533)) - [object]
- Set git user+mail to gh-actions ([174ed47]( https://github.com/es-ude/elastic-ai.creator/commit/174ed478b04b846912d6b0315f1143f24bc94524)) - [object]
- Install latest isort fixing broken imports ([a61ef44]( https://github.com/es-ude/elastic-ai.creator/commit/a61ef445672f913ec4ebc4cc8b46c2ef9099bec7)) - [object]
- Add missing tags_utils again ([910c611]( https://github.com/es-ude/elastic-ai.creator/commit/910c6116600b82e2c52c7d46896d92b63954d7c7)) - [object]
- Updat changelog correctly ([e76a41c]( https://github.com/es-ude/elastic-ai.creator/commit/e76a41cf55463cbc2a4ffa5b2b233d49695302b9)) - [object]
- Fix duplicate field ([6616cab]( https://github.com/es-ude/elastic-ai.creator/commit/6616cab3b0342f0b5d0b8bbdbbdf719de56d5631)) - [object]
- Add missing commitlint.config.js ([2251de8]( https://github.com/es-ude/elastic-ai.creator/commit/2251de83f60823d21346aedcc2b2e9aac4c27458)) - [object]

### Style

- Beautify commit # [6fe04ec](https://github.com/es-ude/elastic-ai.creator/commit/6fe04eccb8dc55714b78e1a7222113c93a0b258c) ([919ac6e]( https://github.com/es-ude/elastic-ai.creator/commit/919ac6ecfc5702c9a705f3da181916c2b9265366)) - [object]
- Beautify commit # [1d617cd](https://github.com/es-ude/elastic-ai.creator/commit/1d617cd289068f3c6552da1bd6e9468759cb5747) ([0bb5d39]( https://github.com/es-ude/elastic-ai.creator/commit/0bb5d39e73c6b4e746f1fb0308b863273d86b7f3)) - [object]
- Sort imports ([de31b33]( https://github.com/es-ude/elastic-ai.creator/commit/de31b335ed9ee8cf04d3823d0b9058e54df07eb9)) - [object]
- Run pre-commit tools on all files ([c22eecf]( https://github.com/es-ude/elastic-ai.creator/commit/c22eecf97792e104596e6575d692c6f4564e66c2)) - [object]

