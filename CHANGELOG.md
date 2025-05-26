## [v0.63.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.62.0..v0.63.0) - 2025-05-08
#### Features
- **(graph)** filter subgraph matches for rewrite - ([573f514](https://github.com/es-ude/elastic-ai.creator/commit/573f5148469ab97da4cb0bfd390943e0e9bb400e))
- **(graph)** find all subgraph matches - ([916d54c](https://github.com/es-ude/elastic-ai.creator/commit/916d54ce203d49e118b47586a9fcabd23ed5a925))
- **(graph)** find all non-overlapping subgraphs - ([bdd01a6](https://github.com/es-ude/elastic-ai.creator/commit/bdd01a64e18ce9018e01b1c41a7bba3949ea9985))
- **(ir)** decouple core data types from identifiers in graphs - ([dde4558](https://github.com/es-ude/elastic-ai.creator/commit/dde45585d2e1ecef6b3d99983b08bf65dd0938ac))
  - **BREAKING CHANGE**: will break almost every client
    that has been using the serialized ir and
    Node/Edge constructors. Most node/edge
    constructors now need explicit
    name/src, dst arguments respectively.
    The serialized IR now uses dicts
    instead of lists for edges and nodes.
    Edge/Node data does not include name/(src,dst)
    anymore.
- **(ir2verilog)** add name of instantiated module as template parameter - ([958eb26](https://github.com/es-ude/elastic-ai.creator/commit/958eb26abc6fbc990ed03b528926e5970225df1d))
- **(ir_transforms)** extract patterns into sub implementations - ([3eb29c9](https://github.com/es-ude/elastic-ai.creator/commit/3eb29c958ab15798442ecd8e52cbc15166bcb105))
#### Bug Fixes
- **(graph)** make subgraph matching independent of node iteration order - ([ec45d11](https://github.com/es-ude/elastic-ai.creator/commit/ec45d1102686f6236235bfc13d78971afcd34caf))
- **(verilog)** update verilog templates to correctly replace arrays - ([c639e8d](https://github.com/es-ude/elastic-ai.creator/commit/c639e8daa4eb50112cef37a94d04d6ca3651697d))
#### Miscellaneous Chores

- bump version v0.63.0 - ([f2b1bda](https://github.com/es-ude/elastic-ai.creator/commit/f2b1bda12241b37a0bf69087ea9071424af3c645))
#### Documentation
- **(ir)** introduce upstream/downstream terminology - ([d540be7](https://github.com/es-ude/elastic-ai.creator/commit/d540be7590b159e300a36373a7e8c8bdf1ebbd6f))
- **(readme)** list recent features - ([ca818dd](https://github.com/es-ude/elastic-ai.creator/commit/ca818dda9bec439f6a20bc51e9cdba1455996a2f))
#### Refactoring
- **(graph)** use simpler rewrite function instead of GraphRewriter - ([f0540cd](https://github.com/es-ude/elastic-ai.creator/commit/f0540cdc8dbf15a05095f0c5a7a31eeab896db8d))
- **(graph)** remove interface arg from rewrite function - ([db1e312](https://github.com/es-ude/elastic-ai.creator/commit/db1e3128108cb3082d3705e95b2660a7b5119e99))
- **(graph)** add rewrite function - ([5dc4816](https://github.com/es-ude/elastic-ai.creator/commit/5dc48169d31ac98d1d1123b3300d95570250074e))
- **(graph)** only require a single match in rewriter - ([e57b400](https://github.com/es-ude/elastic-ai.creator/commit/e57b4005d942554c30d43cbf0fdc19c2bb6afb2a))
- **(graph)** rename dfs_preorder to dfs_iter - ([96bcffd](https://github.com/es-ude/elastic-ai.creator/commit/96bcffd2a0336b6e47b5f860c03334efa58e64ff))
- **(ir)** rename sink to destination/dst - ([5b40309](https://github.com/es-ude/elastic-ai.creator/commit/5b40309b37c623bad5815543950bd7a333301629))
  - **BREAKING CHANGE**: every client that was using
    `sink` as a keyword param or attribute

- - -
## [v0.64.0](https://github.com/es-ude/elastic-ai.creator/compare/23a43230dd4ab43e3e6f90fbc1a999bea23cbeb6..v0.64.0) - 2025-05-26
#### Features
- **(ir)** introduce RewriteRules and rewriting for Implementations - ([39baf11](https://github.com/es-ude/elastic-ai.creator/commit/39baf11d6349466cb2265c04d456dfabec7325e9))
- **(ir)** add function to sync implementation data with underlying graph - ([56c198f](https://github.com/es-ude/elastic-ai.creator/commit/56c198f4650d33de98f0eccf2c1661c49f12e190))
- **(ir2torch)** support nested module hierarchies - ([23a4323](https://github.com/es-ude/elastic-ai.creator/commit/23a43230dd4ab43e3e6f90fbc1a999bea23cbeb6))
#### Bug Fixes
- **(graph)** fix bug that was preventing matches in some edge cases - ([2c31179](https://github.com/es-ude/elastic-ai.creator/commit/2c311799b7e8790b047228daaf204f53c5ef03c6))
- **(ir2vhdl)** stop importing non-existent Template class in skeleton - ([a0f7663](https://github.com/es-ude/elastic-ai.creator/commit/a0f766355fd485bb50660431e368ea8ddc3d8d9e))
#### Miscellaneous Chores
- **(plugin)** improve error message for missing fields in meta.toml - ([fabf3bb](https://github.com/es-ude/elastic-ai.creator/commit/fabf3bb0e5d70ce1eac76c493ffd43f3f1b5cd82))
#### Documentation
- **(graph)** remove outdated docstring - ([6ab932b](https://github.com/es-ude/elastic-ai.creator/commit/6ab932b2f844d96672ebdff2444d6eaf7ecff4f8))
- **(plugins)** clarify purpose of the plugin system - ([11db978](https://github.com/es-ude/elastic-ai.creator/commit/11db978d217621257360cc3afc2d5d4f08095211))
- **(plugins)** name plugin translation stages - ([0d497f4](https://github.com/es-ude/elastic-ai.creator/commit/0d497f402c5c0e75fa28bf343dec37922c883114))
#### Refactoring
- **(graph)** add a few missing type hints - ([47344be](https://github.com/es-ude/elastic-ai.creator/commit/47344be1c1353367d25f32ef1da7957085489d0c))
- **(graph)** publicly expose NodeConstraintFn protocol - ([9934fd5](https://github.com/es-ude/elastic-ai.creator/commit/9934fd594ae2d243d495f0f4112b02898d8544ba))

- - -


## [v0.62.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.61.0..v0.62.0) - 2025-05-08
#### Features
- **(graph)** provide replaced and resulting subgraph from rewriting - ([1b9aab8](https://github.com/es-ude/elastic-ai.creator/commit/1b9aab8349657eca5e313fe14c014109e669753e))
- **(ir2ir)** add time multiplexed sequential plugin - ([080af51](https://github.com/es-ude/elastic-ai.creator/commit/080af5109b1be09130015952fc46a417aab12db3))
- **(ir2ir)** add grouped filter plugin - ([5794938](https://github.com/es-ude/elastic-ai.creator/commit/5794938ad4dd0f5e97cbda16c88d3b03b47620fd))
- **(ir2verilog)** add support verilog translation - ([f40439b](https://github.com/es-ude/elastic-ai.creator/commit/f40439bf56d5b7cb9e1e67792b11126defbdc6e7))
- **(ir_transform)** support reordering sequential subgraphs - ([b356307](https://github.com/es-ude/elastic-ai.creator/commit/b356307121f4a964de60cea99c05f2ef66077be6))
- **(vhdl-plugins)** add combinatorial components - ([f3f6745](https://github.com/es-ude/elastic-ai.creator/commit/f3f67458b4c7b073b2540d73b9966a31bd1bbb72))
- **(vhdl-plugins)** add lutron implementation - ([f04aa63](https://github.com/es-ude/elastic-ai.creator/commit/f04aa63bc6cc9d6888e0c988263a7fe706f37e81))
#### Bug Fixes
- **(ir)** fix __eq__ and __repr__ for IrData - ([a18a207](https://github.com/es-ude/elastic-ai.creator/commit/a18a207852c71a8ca205b6f64ab6184b0947e131))
- **(template)** return str instead of str.Template from builder - ([dfd9d8e](https://github.com/es-ude/elastic-ai.creator/commit/dfd9d8e7c8b73691a84f747fb81210248c10f4cf))
#### Miscellaneous Chores
- **(version)** v0.62.0 - ([3939e67](https://github.com/es-ude/elastic-ai.creator/commit/3939e676c60e2f070218e60601e156a038fafbbb))
#### Documentation
- **(ir)** fix links in docstrings - ([fc7fc79](https://github.com/es-ude/elastic-ai.creator/commit/fc7fc7902e7f8ffd7d294da72bdadf082b9e102a))
#### Refactoring
- **(graph)** make pattern/graph args kw only - ([7a7ea1e](https://github.com/es-ude/elastic-ai.creator/commit/7a7ea1e4c4b1ff1ba5a9d5292e8afa58e55ced3d))
- **(graph)** use mappings instead of functions for lhs/rhs - ([99fc7ec](https://github.com/es-ude/elastic-ai.creator/commit/99fc7ec7382f2be085b658b191cca717d6465dc1))
- **(ir)** separate graph from ir - ([3c09f1f](https://github.com/es-ude/elastic-ai.creator/commit/3c09f1fa29bf7075124eedb858e4e414391fc93e))
- **(ir)** turn Graph into Implementation - ([065aece](https://github.com/es-ude/elastic-ai.creator/commit/065aece5e6f0730085b45395f3da4b015f332834))
  - **BREAKING CHANGE**: breaks every client that was previously
    importing ir.Graph directly
- **(ir2vhdl)** allow strings to specify src_sink_indices - ([5868dcb](https://github.com/es-ude/elastic-ai.creator/commit/5868dcb55c77136bb02128e408e009aaab0f2e6d))
- **(template)** simplify template API - ([be6de71](https://github.com/es-ude/elastic-ai.creator/commit/be6de71516b94b3684275cce29d706c74152148e))
  - **BREAKING CHANGE**: impacts only clients that were
       using the template parameter classes directly
- **(torch2ir)** automatically pass node/edge_fn to impl - ([2817325](https://github.com/es-ude/elastic-ai.creator/commit/28173258a13185a921496d81356f0889bc06ed56))

- - -

## [v0.61.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.60.0..v0.61.0) - 2025-05-08
#### Features
- **(function-utils)** extend signature of FunctionDecorator - ([b972cf4](https://github.com/es-ude/elastic-ai.creator/commit/b972cf4923f05f2ee8f091df3a3de85e9acf7313))
- **(ir)** support pattern based graph rewriting - ([79dc9f4](https://github.com/es-ude/elastic-ai.creator/commit/79dc9f4e25eed23e15c3b0322e793f1422b23cbe))
- **(ir)** add `get_empty_copy` fn for `ir.Graph` - ([1e28c14](https://github.com/es-ude/elastic-ai.creator/commit/1e28c149777a2eee7f51a26b503159b3a071617e))
- **(ir)** add `find_subgraph` - ([97e84ec](https://github.com/es-ude/elastic-ai.creator/commit/97e84ece6f5452023dac79628a47a7f34e2e56f2))
- **(ir)** support decorating methods as required fields - ([569d0b2](https://github.com/es-ude/elastic-ai.creator/commit/569d0b2a7a90f41427bb59a49615cd48d57a7603))
- **(ir)** implement __eq__ for nodes/edges attributes of Graphs  - ([47e37dd](https://github.com/es-ude/elastic-ai.creator/commit/47e37ddf46f2e477a84feb2c16a751870f679492))
- **(ir)** support overriding registered type handlers - ([fec5313](https://github.com/es-ude/elastic-ai.creator/commit/fec5313bb155627a26a2237b13b2ab62fd33a119))
- **(ir)** turn ir.Graph into a IrData object - ([8d44f1b](https://github.com/es-ude/elastic-ai.creator/commit/8d44f1b9ab1fb18212e03b6980fc44af1205cd49))
  - **BREAKING CHANGE**: Might break clients that were using
    `asdict` `fromdict` methods of `ir2vhdl.Implementation`
    previously. Clients using the old version of the
    Graph constructor will receive a deprecation warning.
- **(ir)** add FilterParameters and Shape - ([09443f4](https://github.com/es-ude/elastic-ai.creator/commit/09443f495460b86764299daf8a20fd567d6df2f3))
- **(ir2torch)** support loading model state dicts - ([a1eacaf](https://github.com/es-ude/elastic-ai.creator/commit/a1eacaf490e07334a14fbbb252b2e0005319b0d2))
- **(ir2torch)** make ir2torch a lowering pass - ([74942a4](https://github.com/es-ude/elastic-ai.creator/commit/74942a40d862ef6cb5a11614cc846822409675f3))
  - **BREAKING CHANGE**: impacts all code that has been
    using the ir2torch module
- **(ir2torch)** add Ir2Torch and support linear+relu - ([df47f74](https://github.com/es-ude/elastic-ai.creator/commit/df47f7471ac3f708b04d2928237e0e1d4c3a3aa9))
- **(ir2vhdl)** add basic test runner for vunit testbenches - ([b6b48ee](https://github.com/es-ude/elastic-ai.creator/commit/b6b48ee103436062c36edbf2731aca55286ae691))
- **(ir2vhdl)** create static files from plugins - ([ee23798](https://github.com/es-ude/elastic-ai.creator/commit/ee23798ca8376983c877927203921297a874ab1e))
- **(ir2vhdl)** adds support for ir2vhdl plugins - ([3239641](https://github.com/es-ude/elastic-ai.creator/commit/3239641f87ab109908be0be24c0b5f0922b492d6))
- **(nn-qgrad)** added momentum and weight decay - ([44a0732](https://github.com/es-ude/elastic-ai.creator/commit/44a073273cc10644c4465b7dd0896b19187da32b))
- **(nn-qgrad)** added sgd (#429) - ([6d74b48](https://github.com/es-ude/elastic-ai.creator/commit/6d74b486b5a00ffd09e308c4ed1710c2ac4c8828))
- **(plugin)** make plugin loader more flexible - ([aae96e6](https://github.com/es-ude/elastic-ai.creator/commit/aae96e6392aae03cc6efc832057cf72bb9b6cbe5))
- **(plugins)** make the plugin system more intuitive - ([ac13c15](https://github.com/es-ude/elastic-ai.creator/commit/ac13c15e8d42d92d23a74f0484030f79a64128ed))
- **(qgrads)** added parametrization and module quantization - ([4e551e9](https://github.com/es-ude/elastic-ai.creator/commit/4e551e958abef5a17f74807cb4f8204b39daa6cd))
- **(qgrads)** add support for execution on specific devices - ([2964e62](https://github.com/es-ude/elastic-ai.creator/commit/2964e62d7a145c62c19fa9c6ad2f2f20c819aadc))
- **(torch2ir)** support simple torch 2 ir conversion w/o model state - ([e06e028](https://github.com/es-ude/elastic-ai.creator/commit/e06e02893ee6541204f090319dd325106ab55e1b))
- **(vhdl)** add HwFunctionIdUpdater - ([09aef74](https://github.com/es-ude/elastic-ai.creator/commit/09aef74eb6bdef899e1ad323500f42119800de5c))
- **(vhdl-plugins)** add a new skeleton - ([e4bc616](https://github.com/es-ude/elastic-ai.creator/commit/e4bc6161feb5d9ecc9ff3ab6a7db74a90e90b9c4))
- **(vhdl-plugins)** add striding shift register - ([7f25038](https://github.com/es-ude/elastic-ai.creator/commit/7f25038659825f5942f17a924f4a8eb68b1b6ecb))
- **(vhdl-plugins)** add sliding window - ([d22336e](https://github.com/es-ude/elastic-ai.creator/commit/d22336e1a249422b1dda9a4ef5562b9df07e5a9e))
- **(vhdl-plugins)** add shift register - ([6b5a5b2](https://github.com/es-ude/elastic-ai.creator/commit/6b5a5b2f5448cab825d0fb97311ded09b15fed5f))
- **(vhdl-plugins)** add padding - ([b71a83a](https://github.com/es-ude/elastic-ai.creator/commit/b71a83ad09de36095a025eb075993b01e05445b4))
- **(vhdl-plugins)** add counter - ([7e04f9a](https://github.com/es-ude/elastic-ai.creator/commit/7e04f9a95447e101993bf3f80d02f26f30d05706))
- **(vhdl-plugins)** add middleware - ([1c7b82b](https://github.com/es-ude/elastic-ai.creator/commit/1c7b82b8f66f6e3ad214db165687543c20e3e535))

- improve typing and allow to pass pathlib.Path as working_dir - ([79c6115](https://github.com/es-ude/elastic-ai.creator/commit/79c611512de072b2768b40b1088692fc4ee27b89))

- improve error handling when loading plugins - ([924d63c](https://github.com/es-ude/elastic-ai.creator/commit/924d63c15f7cb5e1f186ad1ed2448c83f0315dac))

- discover/compile/run/log vhdl testbenches from plugins - ([c485099](https://github.com/es-ude/elastic-ai.creator/commit/c4850990de72533374f854c5617606470359751f))
#### Bug Fixes
- **(HwFunctionId)** fix typo - ([13ebdfb](https://github.com/es-ude/elastic-ai.creator/commit/13ebdfb0f86e92273b8190ec5757eb9abb15fd7a))
- **(ci)** correct command for building docs - ([0344afd](https://github.com/es-ude/elastic-ai.creator/commit/0344afda66ec1f02755e34bb2a3d65c37adfb526))
- **(docs)** correct link to repo in html theme - ([dabecc1](https://github.com/es-ude/elastic-ai.creator/commit/dabecc111cf19f914e7fed977c7dcd178d19ebc7))
- **(ghdl-test-runner)** provide useful error message for <py311 - ([940ccfe](https://github.com/es-ude/elastic-ai.creator/commit/940ccfe08ee3c07d40bfc039466381667ec08c0e))
- **(ir)** do not auto create __init__ from IrDataMeta by default - ([6d4b87d](https://github.com/es-ude/elastic-ai.creator/commit/6d4b87d8d8ca49374a95b77cf7b6b7c8cdc80fef))
- **(ir)** error from IrDataMeta if inheriting class has no annotations - ([42fbea2](https://github.com/es-ude/elastic-ai.creator/commit/42fbea2efd75c011b2f6590ac83eb103d168113e))
- **(ir)** exclude required fields for node.attributes | dict() - ([5ef4802](https://github.com/es-ude/elastic-ai.creator/commit/5ef480250e5fe5495c3c40cf705d1fcc8d3e38e6))
- **(plugin)** do not raise error on unexpected plugin fields - ([e3d4445](https://github.com/es-ude/elastic-ai.creator/commit/e3d4445a0a2091d67007596461b3c32e6656474c))
- **(skeleton_id)** properly deal with nested folders - ([79e5f9e](https://github.com/es-ude/elastic-ai.creator/commit/79e5f9e948e9dd60d0a5a3c662743c92decc21d8))

- use the pytest tmp_dir fixture to avoid creating files when running simulations - ([06dbd62](https://github.com/es-ude/elastic-ai.creator/commit/06dbd629d84c11a800526197696a808f7dba0814))

- remove incorrect use of Self type variable - ([0f8a738](https://github.com/es-ude/elastic-ai.creator/commit/0f8a738fe67074e04db3ae0768b3bead27e7d718))

- fix several minor errors that were discoverd by mypy - ([5645da0](https://github.com/es-ude/elastic-ai.creator/commit/5645da0f9a60cbb8071b22a7346a95880e4a9521))
#### Miscellaneous Chores
- **(build)** configure mypy - ([32cd72e](https://github.com/es-ude/elastic-ai.creator/commit/32cd72e8e85464c0786cb9b287b4208d7793d939))
- **(changelog)** fine tune change log template - ([04cf4fe](https://github.com/es-ude/elastic-ai.creator/commit/04cf4fe294e3ba1862ed85006fad1eb659a41686))
- **(devenv)** update devenv.lock - ([09dd5f1](https://github.com/es-ude/elastic-ai.creator/commit/09dd5f18bd9182b67e8dfd52ed111dd78b240b5d))
- **(devenv)** remove statements to run py311 to generate docs - ([0841dc3](https://github.com/es-ude/elastic-ai.creator/commit/0841dc3b32f044892da781c72bd73b5ac4ca38d6))
- **(devenv)** update devenv.lock - ([640a01e](https://github.com/es-ude/elastic-ai.creator/commit/640a01e2308ccc271a6c05f7aecb69c14c23c08b))
- **(devenv)** add devenv tasks to run all checks in parallel - ([cc17635](https://github.com/es-ude/elastic-ai.creator/commit/cc1763557d02747b2a87979759a44bc1ade9396c))
- **(devenv)** add jj/git/pikchr - ([9c8ebdc](https://github.com/es-ude/elastic-ai.creator/commit/9c8ebdc5d1711feefc1fd991ab59649bf97628d1))
- **(devenv)** support vivado and add options to ghdl module - ([0ddb41f](https://github.com/es-ude/elastic-ai.creator/commit/0ddb41fa88c4a3932d0f4e613eb39c72bc0023fe))
- **(docs)** remove and ignore autogenerated files - ([4a29ee9](https://github.com/es-ude/elastic-ai.creator/commit/4a29ee969bbfe8d8098b390777721363283deecf))
- **(docs)** change docs theme to pydata - ([9f23518](https://github.com/es-ude/elastic-ai.creator/commit/9f235182a1c94e3d5ea3f7fc56e96261dd6bb7dd))
- **(docs)** move docs source do `docs/` and adjust some docstrings - ([1325b12](https://github.com/es-ude/elastic-ai.creator/commit/1325b12e64b9e1f80027002912f289ea3a40554c))
- **(docs)** replace antora/asciidoc by sphinx/markdown - ([39b38d4](https://github.com/es-ude/elastic-ai.creator/commit/39b38d491a7b4e44dfab9cdefdb16afeafe0b59c))
- **(ghdl-tb-runner)** remove runner and utils - ([fa268cf](https://github.com/es-ude/elastic-ai.creator/commit/fa268cf1b1dce2b4e822985c4fdcf08ec6b7b2bc))
- **(github)** add issue and pr templates - ([f9fe969](https://github.com/es-ude/elastic-ai.creator/commit/f9fe969f09dfe0a25aaaa7c00ee74c5c7f025143))
- **(ir)** print class name in instead of class in IrData repr - ([168f45c](https://github.com/es-ude/elastic-ai.creator/commit/168f45c46e911963371bdd8afe928bd7469c35d2))
- **(plugins)** add bash scripts to help creating new plugins - ([a1b7f4a](https://github.com/es-ude/elastic-ai.creator/commit/a1b7f4ad618654d1b9d82bf5ae2e2fd90d1d6d5d))
- **(pyproject)** add wavedrom extension for sphinx - ([4e3514d](https://github.com/es-ude/elastic-ai.creator/commit/4e3514d0e0866e4608a2a7c4eabc80af34ef61b1))
- **(pytest)** add new 'slow' marker - ([4da2334](https://github.com/es-ude/elastic-ai.creator/commit/4da23341706824bc557e3eb0b31e4bec5b901678))
- **(version)** v0.61.0 - ([0b5a266](https://github.com/es-ude/elastic-ai.creator/commit/0b5a2667c4fa89bb3e56e4ba16a760a0d782eabc))

- use tach to adhere to architecture - ([9c845ae](https://github.com/es-ude/elastic-ai.creator/commit/9c845aebb8eda4cc84943157a8e32f8010f313c4))

- include tests in ruff linting - ([9e3e38a](https://github.com/es-ude/elastic-ai.creator/commit/9e3e38a8b7a04775a0d56f87797ff65c72a2841e))

- rename FunctionDecoratorFactory to FunctionDecorator - ([85464cf](https://github.com/es-ude/elastic-ai.creator/commit/85464cfcbef68d8fb588e5b23c6577685ccf4ffe))

- lint unorganized imports - ([f31196a](https://github.com/es-ude/elastic-ai.creator/commit/f31196aa42bda753f49d35b42744ca1cc39b4085))

- use importlib mode for pytest to avoid errors about missing __init__ files in test folders - ([687f0c3](https://github.com/es-ude/elastic-ai.creator/commit/687f0c3e896c3388b9babb4a86162140aef904b8))

- add basic support for devenv - ([bbb3073](https://github.com/es-ude/elastic-ai.creator/commit/bbb30738014cb44f90cfee2e30a25019503358e6))

- add basic support for devenv - ([7df56a1](https://github.com/es-ude/elastic-ai.creator/commit/7df56a17428833ecdf63cc3f62509c04e76c557a))
#### Documentation
- **(contribution)** improve readability (hopefully) - ([7be9894](https://github.com/es-ude/elastic-ai.creator/commit/7be9894ecf563a6c5bc54725b8e0be81f2679d50))
- **(function_utils)** fix incorrect class names in examples - ([99968b8](https://github.com/es-ude/elastic-ai.creator/commit/99968b885b1b3715c31ee96fc55f778f0725afe7))
- **(ir)** explain the core concepts of ir - ([1d54159](https://github.com/es-ude/elastic-ai.creator/commit/1d54159db87af8b6815ac99ab45485c9eb9ce01f))
- **(ir)** explain how to annotate fields - ([4176c58](https://github.com/es-ude/elastic-ai.creator/commit/4176c58985a954e5bbc9c63d6203d2cd039ba428))
- **(ir2vhdl)** add previously missing ir2vhdl.adoc file - ([05606c7](https://github.com/es-ude/elastic-ai.creator/commit/05606c7d8b449792ca832e146c21d2577c4021c2))

- update autogenerated docs - ([3ce97f4](https://github.com/es-ude/elastic-ai.creator/commit/3ce97f4c1545b9bdfc26d8a0f5dd3b716b18cd8e))

- add GitHub repository link to documentation header - ([78d110d](https://github.com/es-ude/elastic-ai.creator/commit/78d110d631048da5f79dcc3af22dd50eb064c855))

- extend ir2vhdl documentaion - ([c5eafde](https://github.com/es-ude/elastic-ai.creator/commit/c5eafde83fb72ceac0e0a07517723d0a717906ff))

- Clarified installation via PyPi - ([037d0e2](https://github.com/es-ude/elastic-ai.creator/commit/037d0e2be285695e3aa31719a1656314ca74a375))

- fix missing table end - ([6ec2343](https://github.com/es-ude/elastic-ai.creator/commit/6ec2343f13c8e49e736030737cb5055f6f6b8e30))

- do not build api docs for plugins - ([7de975d](https://github.com/es-ude/elastic-ai.creator/commit/7de975d216d81f1e872ec20151649172a50f52be))

- fix asciidoc refs and formatting in docstrings - ([b6ff358](https://github.com/es-ude/elastic-ai.creator/commit/b6ff358d282acec8474128483e011d305a1f0a20))

- improve wording in ghdl test-runner docs - ([868a51f](https://github.com/es-ude/elastic-ai.creator/commit/868a51f84ea7ee6edd60c055fe93c799366af7bf))

- improve formatting - ([1c81917](https://github.com/es-ude/elastic-ai.creator/commit/1c81917bea6eb7870edcc80f229769cb252bd6e8))

- fix wrong package name in install command - ([22426ab](https://github.com/es-ude/elastic-ai.creator/commit/22426aba973377b89388da9e0d0c171b03eb1fb0))

- move dev docs from readme into contribution.md - ([3a56226](https://github.com/es-ude/elastic-ai.creator/commit/3a5622604b07b21a9665d271013529e6366b655b))

- explain commit/pr policy in contribution guide - ([e04630c](https://github.com/es-ude/elastic-ai.creator/commit/e04630cfab084cb4d8e2ec40a9aa863cbe6a5bdb))
#### Refactoring
- **(ir)** expose nodes/edges via ir data fields - ([d861c53](https://github.com/es-ude/elastic-ai.creator/commit/d861c536ad0a1d46be5c2c98f89ec82f8c797720))
- **(ir)** clean up type hints - ([8047010](https://github.com/es-ude/elastic-ai.creator/commit/80470101f912ff1799002b1843e5a96382a8688a))
- **(ir)** add iterator methods to ir.Graph - ([dd6f077](https://github.com/es-ude/elastic-ai.creator/commit/dd6f077343a2f22d25023c1e25ab641b7939f054))
- **(ir)** add iter_edges method to GraphDelegate for consistency - ([87313b7](https://github.com/es-ude/elastic-ai.creator/commit/87313b70bc9eeaeec51442060acf117902facc5a))
- **(ir2vhdl)** move ir2vhdl to dedicated subpackage - ([65034d5](https://github.com/es-ude/elastic-ai.creator/commit/65034d51c218ea3d28e1c7d477dc28681d33e88e))
  - **BREAKING CHANGE**: impacts every client that imported
    the elasticai.creator.vhdl_template package
- **(qgrads)** refactored interface for autograd and mathoperations (#456) - ([e745f3b](https://github.com/es-ude/elastic-ai.creator/commit/e745f3b3448e8f4001e20498f2e7635590d7a69f))
- **(test)** put ir graph tests into same module - ([d1441e6](https://github.com/es-ude/elastic-ai.creator/commit/d1441e6db5c24a8c86999ea3b4ba77699726a303))
- **(tests)** move tests from the elasticai.creator source tree to the tests folder to maintain a consistent tests location - ([7ea58c4](https://github.com/es-ude/elastic-ai.creator/commit/7ea58c4894c9ec4b97d978effb975e61418900ce))
- **(tests)** move tests from the Elasticai Creator source to the tests folder to maintain a consistent location for tests - ([b0b892f](https://github.com/es-ude/elastic-ai.creator/commit/b0b892f26defed59e12884c8fea6c641f8dc2605))
- **(utils)** move ghdl_msg parsing to its own module - ([124cad1](https://github.com/es-ude/elastic-ai.creator/commit/124cad1aafe825fb9f3726b648e7dedd08311acd))
- **(vhdl-mac)** move fixed point mac to nn.fixed_point - ([21f5fd6](https://github.com/es-ude/elastic-ai.creator/commit/21f5fd62023e4a2c756c89036a3be6dd73077239))
  - **BREAKING CHANGE**: will impact everything that was
    importing from elasticai.creator.vhdl.shared_designs.mac

- use more a precise return type for the create_design function - ([079328a](https://github.com/es-ude/elastic-ai.creator/commit/079328a8260f4cedf1196e28333f7cac2e0c0b7f))

- Removed test_utils - ([ff3cd5a](https://github.com/es-ude/elastic-ai.creator/commit/ff3cd5a5510fda21a972ab26f2e6a7944ef960ba))

- automatically fix linted problems - ([3862f21](https://github.com/es-ude/elastic-ai.creator/commit/3862f218fe0f4ec5dc16b50239d61eac1c3edd42))
#### Build system
- **(devenv)** add kroki plugin to antora to build diagrams - ([380ab02](https://github.com/es-ude/elastic-ai.creator/commit/380ab021458c5827edddec5d8e4e2a18014279e1))
- **(pyproject)** add hypothesis for testing - ([a0f5eba](https://github.com/es-ude/elastic-ai.creator/commit/a0f5eba394177e8a622436446d91fd533f37fa2f))
- **(versioning)** use cog instead of cliff for changelogs - ([eabf4e1](https://github.com/es-ude/elastic-ai.creator/commit/eabf4e1b57657e4e0131d676be606d67d8618487))

- add VUnit to testing dependencies - ([8bf1572](https://github.com/es-ude/elastic-ai.creator/commit/8bf15720342826dfe4e5013a6792a562a2b191f8))

- drop support for py 3.10 and move to 3.11 - ([284d476](https://github.com/es-ude/elastic-ai.creator/commit/284d476aa66102aedd0ee839a70cc414f2f53048))
  - **BREAKING CHANGE**: every dependant that uses python
    3.10

- fix includes for hatch - ([58c8f8e](https://github.com/es-ude/elastic-ai.creator/commit/58c8f8e11e5772d89895ddf67314c5dd21acb4bd))

- use git to dynamically extract version - ([223a13e](https://github.com/es-ude/elastic-ai.creator/commit/223a13e8014cf320f90386453be9daa29d00e22b))

- autoupdate uv.lock - ([2a59a5c](https://github.com/es-ude/elastic-ai.creator/commit/2a59a5c69ab026311f137ea2f30c7a59399830d8))

- add plugins namespace to pytest discovery - ([fbf5c07](https://github.com/es-ude/elastic-ai.creator/commit/fbf5c07505df674f8d5077b0cac021595376509e))

- update ruff version and uv.lock file - ([9229e30](https://github.com/es-ude/elastic-ai.creator/commit/9229e301bb48e50e2445dd0790e30857ea1b68c8))

- add uv.lock and set .python-version to 3.10 - ([533add1](https://github.com/es-ude/elastic-ai.creator/commit/533add17585846ea7f6c49ce0c954df24c0c88a5))

- replace poetry by uv - ([47d9538](https://github.com/es-ude/elastic-ai.creator/commit/47d9538782766f33ae553e2425a6fe21956dd4cb))
#### Style
- **(ir)** improve type hints - ([c0701d0](https://github.com/es-ude/elastic-ai.creator/commit/c0701d0f363eedfb270f2018492127552f28dfb7))

- apply unsafe ruff fixes to tests - ([c588232](https://github.com/es-ude/elastic-ai.creator/commit/c5882324d7b3332fe17e5bd6a2d84aed67cba243))

- apply safe ruff fixes to tests - ([ed2ec86](https://github.com/es-ude/elastic-ai.creator/commit/ed2ec86e006d1d64f378a7b4734d3a2851082db5))

- make imports in __init__.py files explicit - ([60149d1](https://github.com/es-ude/elastic-ai.creator/commit/60149d162639427e7341fed82744fd4ebd4f62c4))

- apply safe ruff format for imports - ([f42651e](https://github.com/es-ude/elastic-ai.creator/commit/f42651e973bb4de492a41fe60ab242820a894e2c))

- improve formatting and add missing text in workflows - ([11b0160](https://github.com/es-ude/elastic-ai.creator/commit/11b0160224165e200f1a5e42198de4bdb9802655))

- fix formatting using ruff - ([faf97cb](https://github.com/es-ude/elastic-ai.creator/commit/faf97cb6f2a163f27ce5602c4ff383183d88ea2a))

- - -

## [v0.60.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.59.2..v0.60.0) - 2025-05-08
#### Features
- **(examples)** added an basic example for a network using skeleton v2 - ([6d94158](https://github.com/es-ude/elastic-ai.creator/commit/6d941584c40d41049bd27e9da8c2dc204f79080b))
- **(firmware)** create separate LSTMFirmwareENv5 - ([17a274c](https://github.com/es-ude/elastic-ai.creator/commit/17a274c6bb23fb5721b3e07cd16916bcbd3889c8))
- **(firmware)** test that firmware generates skeleton correctly - ([3a18656](https://github.com/es-ude/elastic-ai.creator/commit/3a1865642e60fcd4f4fbf49917d0930663bc38aa))
- **(firmware)** new firmware that does not save testbenches - ([8ad3272](https://github.com/es-ude/elastic-ai.creator/commit/8ad3272c80a350348df1ce7a562df6f51928a4ee))
- **(ir)** add LoweringPass class - ([d598021](https://github.com/es-ude/elastic-ai.creator/commit/d5980214c2441b52eef063bed865de2eecd52f10))
- **(ir)** add basic function registry - ([3785db0](https://github.com/es-ude/elastic-ai.creator/commit/3785db0d42f2f7a9118b9b5c3e60d1f83f9bbd86))
- **(ir)** add basic graph data structure - ([e1dfef2](https://github.com/es-ude/elastic-ai.creator/commit/e1dfef26688ffc819fe07ae7831df6b899c4b7f3))
- **(ir)** make suc/predecessors sorted/deterministic - ([2fd8e5e](https://github.com/es-ude/elastic-ai.creator/commit/2fd8e5e5a60fa7d7d3c69f1d8fcb0e783eade399))
- **(ir)** make suc/predecessors sorted/deterministic - ([8825607](https://github.com/es-ude/elastic-ai.creator/commit/88256079288bae95668ffe60aced222e920419c3))
- **(ir)** introduce read only field for IrData - ([be1e8fb](https://github.com/es-ude/elastic-ai.creator/commit/be1e8fb3a659d30882b77399016fa8c21d8f0e6b))
- **(ir)** add abstract ir data class and nodes - ([bb81f0d](https://github.com/es-ude/elastic-ai.creator/commit/bb81f0dea0ee8ea8646e57d85cc070baddf91e8a))
- **(ir)** add graph delegate and iterators - ([1cd2a35](https://github.com/es-ude/elastic-ai.creator/commit/1cd2a353288bf08f2982bfafbfa13923da0729bb))
- **(ir)** add graph delegate and iterators - ([868a188](https://github.com/es-ude/elastic-ai.creator/commit/868a188aa4ce0be92608997bbfdb2916e7f8603e))
- **(lstm)** set more specific return type for create_testbench function - ([7e3f54b](https://github.com/es-ude/elastic-ai.creator/commit/7e3f54b5cbc7aa6d026d60b28f8c32c05768aa0c))
- **(nn)** added enV5 usb library to development pyproject.toml. This will be used in the future to do system tests - ([3089341](https://github.com/es-ude/elastic-ai.creator/commit/3089341849008fbfb1ff66029ea522702cc4303f))
- **(nn)** added simulation for linear layer - ([aac395b](https://github.com/es-ude/elastic-ai.creator/commit/aac395b9413943d6252bf5cb4866d96173216ae8))
- **(nn)** added conv1d. Simulation works. End-to-end system test is still pending - ([93e5ecd](https://github.com/es-ude/elastic-ai.creator/commit/93e5ecdaae2987d34772a504bc843019419bd845))
- **(nn)** added simulation for linear layer - ([a2cd0a0](https://github.com/es-ude/elastic-ai.creator/commit/a2cd0a07854f746353009af3c63b22e03f9fcabb))
- **(nn)** added conv1d. Simulation works. End-to-end system test is still pending - ([d8ca219](https://github.com/es-ude/elastic-ai.creator/commit/d8ca2193fec46161c3f77d60773ad19396a7090c))
- **(nn-qgrad)** added basic layers - ([82db217](https://github.com/es-ude/elastic-ai.creator/commit/82db217242fed30a955dcf7a69eb98a56e4b931a))
- **(nn-qgrad)** added fixed point config, autograd and quantize - ([97bb203](https://github.com/es-ude/elastic-ai.creator/commit/97bb203898e6d689fff54c73da722584aca6882f))
- **(plugin)** move plugin_loader and type_handler decorators - ([6bba61d](https://github.com/es-ude/elastic-ai.creator/commit/6bba61d5f7758f1b9db14a0938e29f4c163c52b9))
- **(plugin)** load plugin and call generated fn - ([0492e0b](https://github.com/es-ude/elastic-ai.creator/commit/0492e0b94eae88eb536bd7b85859527026ec273d))
- **(plugins)** load plugin description from package - ([7dfae73](https://github.com/es-ude/elastic-ai.creator/commit/7dfae73b226d6cec7d5e0660b4da0fd78bef4439))
- **(plugins)** load plugin description from package - ([05a99c3](https://github.com/es-ude/elastic-ai.creator/commit/05a99c3e71a3c408a4ab273b7fe453b215d39ef9))
- **(pyproject)** allow python3.10 - ([0628024](https://github.com/es-ude/elastic-ai.creator/commit/0628024ba826ebbdbe5b5deda4aac67d81876248))
- **(pyproject)** remove restriction to pytorch versions < 2.0.1 - ([bb47705](https://github.com/es-ude/elastic-ai.creator/commit/bb477058440e07e2bdd6c467e328219519510771))
- **(skeleton)** add general skeleton class - ([b4ffacb](https://github.com/es-ude/elastic-ai.creator/commit/b4ffacb1847685851def6beb9f53044fe5dbd75f))
- **(skeleton_id)** move hw accel meta to dedicated module - ([9f65b8d](https://github.com/es-ude/elastic-ai.creator/commit/9f65b8dce91fcdfd87f7f1229a06c2d3776f8ad5))
- **(skeleton_id)** tweak api for skel id computation - ([f7d9a77](https://github.com/es-ude/elastic-ai.creator/commit/f7d9a7786e06a500dafcc6cbf3f08f81083c6166))
- **(template)** allow '${key}' placeholders for multiline templates - ([d25eef1](https://github.com/es-ude/elastic-ai.creator/commit/d25eef1369c754911e56ed5aa4a92f62b2716325))
- **(tests)** linear layer system test with elastic node works now - ([238964a](https://github.com/es-ude/elastic-ai.creator/commit/238964a119b31db57c44085c336c8605e10c8e9a))
- **(tests)** echo server works now - ([a4359a0](https://github.com/es-ude/elastic-ai.creator/commit/a4359a0f08fa5a620ce414c8de6e133613427a65))
- **(vhdl)** add automatic deterministic skeleton id generation - ([eb7e59f](https://github.com/es-ude/elastic-ai.creator/commit/eb7e59f7aa7506206e6807ef6a649e8b458930b4))
- **(vhdl)** added an example for the echoserver with skeleton v2 - ([3f46780](https://github.com/es-ude/elastic-ai.creator/commit/3f46780d44dc6f0d220b3a3d82f71e33ae38fdac))
- **(vhdl)** added a generator for echo server with skeleton #378 - ([2c3faf5](https://github.com/es-ude/elastic-ai.creator/commit/2c3faf575df21f1aca236138257097ddd2320bff))
- **(vhdl)** added skeleton version 2 to project - ([6ed2c94](https://github.com/es-ude/elastic-ai.creator/commit/6ed2c94abbcb0d090eac3844fb59e27983f7ed11))

- remove ir2vhdl (shouldnt have been committed) - ([20fb891](https://github.com/es-ude/elastic-ai.creator/commit/20fb8916f78ea3a78ea7eeef9af1d3f071168ca2))

- add basic but flexible templating component - ([2ae0506](https://github.com/es-ude/elastic-ai.creator/commit/2ae050611b9bc2cc93624e99bad7c1244dd2b6c4))

- add plugin loader and improve function registry - ([0a8ac61](https://github.com/es-ude/elastic-ai.creator/commit/0a8ac61fef8792ab177f7635d86d4f9ae23029b1))

- abstract class DesignCreator inherits from torch.nn.Module - ([d6e70ed](https://github.com/es-ude/elastic-ai.creator/commit/d6e70ed1c84d025bb2eebbfda45c67ad9ba1f987))

- allow python3.12 - ([46d6cfb](https://github.com/es-ude/elastic-ai.creator/commit/46d6cfb52eb2c6de471d9dd0310a9152200ec0db))

- convert negative numbers to bit patterns using two's complement - ([c94dc3b](https://github.com/es-ude/elastic-ai.creator/commit/c94dc3ba59698e87eac2efe702a43dfc925401bd))

- add support for less than 8 bit in skeleton - ([231f0ca](https://github.com/es-ude/elastic-ai.creator/commit/231f0ca808248b740421e5bb516b71e5f0c434ce))

- add skeleton for sequential layer - ([34e8202](https://github.com/es-ude/elastic-ai.creator/commit/34e8202281e0be4d79d78df66cbcffc9b4db3878))

- added a bash script to automatically build the vivado file with the help of vivado 2021.1 on a server - ([eb8c835](https://github.com/es-ude/elastic-ai.creator/commit/eb8c835529d736037b085f5ede0490ca342bac3e))
#### Bug Fixes
- **(MiddlewareSpec)** correct counter in example code - ([1860657](https://github.com/es-ude/elastic-ai.creator/commit/1860657968f0828502eecfb892c6e58fab93bf10))
- **(MiddlewareSpec)** transmit high byte first instead of low - ([f2bd5af](https://github.com/es-ude/elastic-ai.creator/commit/f2bd5af1cf6d9e4f40da8c89a89c61663cf12086))
- **(contribution,docs)** fix tables and language - ([63a1b9d](https://github.com/es-ude/elastic-ai.creator/commit/63a1b9d42aa7a8f3866b978516a7269cec10e61b))
- **(dependencies)** fixed the poetry lock file - ([31868ca](https://github.com/es-ude/elastic-ai.creator/commit/31868caefb3959d0966c93d01a45c234a9041b55))
- **(dependencies)** fixed the dependency for the runtime utils - ([cfaf318](https://github.com/es-ude/elastic-ai.creator/commit/cfaf318915a046f0b5707a56c2fcbdb9e312f1dc))
- **(firmwareEnv5)** save testbench to separate folder - ([9937431](https://github.com/es-ude/elastic-ai.creator/commit/99374317966a0db05c147bf99d322da5b14b0f5a))
- **(imports)** remove toplevel __init__.py - ([c7f0a78](https://github.com/es-ude/elastic-ai.creator/commit/c7f0a7820c094789d8ae7e4bc9076c5cda167f8d))
- **(ir)** make graph iterators deterministic - ([2c3b27a](https://github.com/es-ude/elastic-ai.creator/commit/2c3b27a0e8afbf7bdbea3ce8e45abbbc65408184))
- **(ir)** remove dead code and fix typing - ([609eb51](https://github.com/es-ude/elastic-ai.creator/commit/609eb51c45c298e190a1e6f2133623b456e9ee2c))
- **(ir)** fix conceptual problems with abstract ir data type - ([1e6210d](https://github.com/es-ude/elastic-ai.creator/commit/1e6210db742f3e1b9b2613126cc48262e6eddee4))
- **(ir)** fix bug where iterator was not remembering visited nodes - ([3da0dbc](https://github.com/es-ude/elastic-ai.creator/commit/3da0dbc5e84748fdd7db5ee78b9cd40636f19e7e))
- **(lstm)** skeleton naming - ([21d057d](https://github.com/es-ude/elastic-ai.creator/commit/21d057d2497d817c09c12e90f43047aeed71e6d8))
- **(lstm_skeleton)** xil to work lib - ([005ed36](https://github.com/es-ude/elastic-ai.creator/commit/005ed36a4ff8bac6bb1ba1ed29e5e9cfe0be6c73))
- **(nn)** fixed code generation test for linear layer - ([7f8f445](https://github.com/es-ude/elastic-ai.creator/commit/7f8f4455ea1352d0c30fea05e22fb7ce561d654c))
- **(nn)** fixed error in convolution - ([390656c](https://github.com/es-ude/elastic-ai.creator/commit/390656cc00cd4f827c27badae232ac8073f480a2))
- **(nn)** revert changes in linear.tpl.vhd - ([f11b606](https://github.com/es-ude/elastic-ai.creator/commit/f11b6061ff88c4b16e70131508b9f10758c9b90d))
- **(nn)** linear layer uses signals now. So simulation works - ([7f5d30c](https://github.com/es-ude/elastic-ai.creator/commit/7f5d30c6c066506b66112c9ba15fe367ce33f9a8))
- **(nn)** fixed fxp_mac reset. This modules reset is now working at any point in time independent of the other inputs - ([054b8ab](https://github.com/es-ude/elastic-ai.creator/commit/054b8ab4d3569b3ae105b791ea0e8f116a8ddfd6))
- **(nn)** fixed fxp_mac reset. This modules reset is now working at any point in time independent of the other inputs - ([6969839](https://github.com/es-ude/elastic-ai.creator/commit/696983974ac1fee6ef25dc32072b052fb98fdc1f))
- **(precomp)** fix incorrectly sorted inputs, add input/output widths and assert them - ([74f5e26](https://github.com/es-ude/elastic-ai.creator/commit/74f5e265fb388249a3d46c1743bc3b3e38366a78))
- **(project)** removed init in elasticAi - ([be65e7c](https://github.com/es-ude/elastic-ai.creator/commit/be65e7c223e339b5ec03fc3b54ec3e4782a58d98))
- **(skeleton)** fix skeleton for mlp use case - ([e4b67cc](https://github.com/es-ude/elastic-ai.creator/commit/e4b67ccbc4178629f35f3f3a89259d2bfae3aba0))
- **(skeleton)** fix wrong signal name in integration test - ([74ebc32](https://github.com/es-ude/elastic-ai.creator/commit/74ebc32e938936d2d60c3da09773439c5675106d))
- **(template)** implement function only supported in python3.11 and higher - ([5223adf](https://github.com/es-ude/elastic-ai.creator/commit/5223adfdf6551f9758ee4dbdef9df1f2aed36377))
- **(test)** add expected newline to end of skeleton - ([dcb20b7](https://github.com/es-ude/elastic-ai.creator/commit/dcb20b712945439b1c3db799404beb33d8587e4f))
- **(tests)** fix wrong path that leads to temporary files created in the project tree - ([0ff9c0b](https://github.com/es-ude/elastic-ai.creator/commit/0ff9c0b04442e6da0d76f3991254cae63bf260e8))
- **(vhdl)** fixed test for changes in sensitivity list and for rising/falling edge clock - ([088bc1f](https://github.com/es-ude/elastic-ai.creator/commit/088bc1f7a206739701d6bf9735b3974add0262c0))
- **(vhdl)** fixed error in test - ([2d7f140](https://github.com/es-ude/elastic-ai.creator/commit/2d7f140bc9382be47074c1cbda3015f10ecdfaab))
- **(vhdl)** fixed the test for the firmware with skelton v2 - ([75ef96a](https://github.com/es-ude/elastic-ai.creator/commit/75ef96a21236c9fc3a6e830aedc5978a9e033c9e))
- **(vhdl)** fixed an error in the skeleton v2 template - ([dc2ee96](https://github.com/es-ude/elastic-ai.creator/commit/dc2ee964c8898a359fafe75b2bcb216ab39ebf2a))
- **(vhdl)** fixed error in template - ([948af39](https://github.com/es-ude/elastic-ai.creator/commit/948af39669a9070518a96bd3611cb6d43b405986))
- **(vhdl)** #374 remove unnecessary data_buf from top module. It is not used anywhere so should have no effect - ([e8cd5a7](https://github.com/es-ude/elastic-ai.creator/commit/e8cd5a797c92b4f29b94dad1a9b7de4d090a98ae))
- **(vhdl)** warn when using skeleton v1 - ([5a46331](https://github.com/es-ude/elastic-ai.creator/commit/5a4633193cf2c6db699fa19c47ddfbc53599c1fe))
- **(vhdl)** added an exception raise for the skeleton for not supported configurations - ([fc129fe](https://github.com/es-ude/elastic-ai.creator/commit/fc129fe8d83b7cacb54c615f698f42d633c73e5c))
- **(vhdl)** added an exception raise for the skeleton for not supported configurations - ([11d006c](https://github.com/es-ude/elastic-ai.creator/commit/11d006c146b9eaa809460729132222a0595e6793))
- **(vhdl)** fixed the test for the old skeleton and added another one for skeleton v2 - ([87b11a4](https://github.com/es-ude/elastic-ai.creator/commit/87b11a4455ec059ed8e3fdb3a405a976435facd6))

- remove outdated srcs in skeleton plugin - ([57ae044](https://github.com/es-ude/elastic-ai.creator/commit/57ae0442099fe36a2e8c31fe100d2eba59779093))

- make type hints 3.10 compatible - ([db8a0f8](https://github.com/es-ude/elastic-ai.creator/commit/db8a0f8dc836e09bc4cc978e574b4d22be798954))

- update deps to resolve security issues - ([6568d28](https://github.com/es-ude/elastic-ai.creator/commit/6568d2830120922f77d2e183aa5764369143135f))

- added skeleton_1.vhd needs to be changed - ([632bf89](https://github.com/es-ude/elastic-ai.creator/commit/632bf8974ace775c8289351d03c026e587c237ed))

- use linear layer name - ([9d48f09](https://github.com/es-ude/elastic-ai.creator/commit/9d48f098e60ead2854af61a6c1394e824e762538))

- remove unnecessary instance name templ variables - ([7da4b5a](https://github.com/es-ude/elastic-ai.creator/commit/7da4b5a473e9976392eda1d4e686cc4ff9b12d0d))

- fix skeleton test - ([12b7c27](https://github.com/es-ude/elastic-ai.creator/commit/12b7c274d1d9116d06be711066f2d5ee1cf5725e))

- fix fxp mac test - ([baad73b](https://github.com/es-ude/elastic-ai.creator/commit/baad73b8e97f5c7b65e52a1c9755eb2086de02aa))

- add skeleton, etc. to generated files - ([2bbd588](https://github.com/es-ude/elastic-ai.creator/commit/2bbd588ceaec6408f61f48c2f61289c90afffef9))
#### Miscellaneous Chores
- **(commitlint)** add test/build/style as types - ([7d00767](https://github.com/es-ude/elastic-ai.creator/commit/7d0076794361fc99aeaaf7b80474679e8cd6d257))
- **(dependency)** fixed dependency of elasticai-runtime-env5 from develop branch to specific commit - ([a5ac0df](https://github.com/es-ude/elastic-ai.creator/commit/a5ac0dfb7e0db20f21daaa75d4dd1e162f298cea))
- **(fn-registry)** remove redundant tests - ([3f0c243](https://github.com/es-ude/elastic-ai.creator/commit/3f0c243ad1f35172e199e94d3fe060b86b943661))
- **(gitignore)** add devenv/direnv/uvlock - ([476cefa](https://github.com/es-ude/elastic-ai.creator/commit/476cefa326eb3084a74717c872981dbbf97feff0))
- **(pipeline)** added package.json to fix the verison of commitlint - ([f8a7a0f](https://github.com/es-ude/elastic-ai.creator/commit/f8a7a0f3f888e4937baa8d0c7423637facaf443d))
- **(poetry)** synchronize pyproject.toml and poetry.lock - ([76a8baa](https://github.com/es-ude/elastic-ai.creator/commit/76a8baa01b52daf59248c33994e747f7b3dc4cb8))
- **(pre-commit)** run black with python3.12 to support new syntax - ([ff51308](https://github.com/es-ude/elastic-ai.creator/commit/ff5130872f82206e68075f9b6c99b53dfa746a39))
- **(pyproject.toml)** clean up external deps - ([2f83d46](https://github.com/es-ude/elastic-ai.creator/commit/2f83d46e0a39a079f9f3884ee71eb37a87d972c0))

- allow 'bump' for commit msg type - ([8b341ca](https://github.com/es-ude/elastic-ai.creator/commit/8b341caf19271235caaab3f161b1b561b6a8fbf5))

- create coverage report for develop as well - ([8c11c01](https://github.com/es-ude/elastic-ai.creator/commit/8c11c01ae66485d70538f07effbb637533de085f))

- update pre-commit (necessary to fix broken black deps) - ([9102843](https://github.com/es-ude/elastic-ai.creator/commit/9102843a898829a0dea0001f009a41265b4cf919))

- use python3.10 to run the tests - ([3151ba2](https://github.com/es-ude/elastic-ai.creator/commit/3151ba269b6265c1f90bf12b125f0c62e5e969f0))

- only throw a warning if commit message exceeds char limit - ([3e1d509](https://github.com/es-ude/elastic-ai.creator/commit/3e1d509e23d8aa5302e708bb309514294b9d7984))
#### Documentation
- **(middleware)** add register documentation - ([59f7ed4](https://github.com/es-ude/elastic-ai.creator/commit/59f7ed4c044b6062d37d66c7dc97cb31a056939b))
- **(nn)** added more context for the parse reported content functions - ([70c8b4b](https://github.com/es-ude/elastic-ai.creator/commit/70c8b4bbd01aaf272ccf6a91af4d91a333dce41f))
- **(nn)** added comments to parsing functions in testbenches - ([55c9f4d](https://github.com/es-ude/elastic-ai.creator/commit/55c9f4de6ce269f60f25edd91592aea1debe8701))
- **(nn)** removed unnecessary comments - ([6b1256f](https://github.com/es-ude/elastic-ai.creator/commit/6b1256f0af5bb7e857d14560d800d6455b95003a))
- **(nn)** removed unnecessary comments - ([25cdf90](https://github.com/es-ude/elastic-ai.creator/commit/25cdf904571da8d4f60418ce52447c8959b4c87b))
- **(skeleton)** explain we need to read each result byte two times - ([96572fb](https://github.com/es-ude/elastic-ai.creator/commit/96572fb21958c7b79505e1ea004cdf9681e8097d))
- **(skeleton)** add timing diagram to skeleton/middleware spec - ([574116b](https://github.com/es-ude/elastic-ai.creator/commit/574116b529b20334c6646de1fd20f3e95dc47218))

- Listed deprecated modules and those in development - ([9c7c12c](https://github.com/es-ude/elastic-ai.creator/commit/9c7c12c59ed6d84eee5caf70fb5ca8722964a139))

- Added preliminary documentation for creating new modules - ([1d47f05](https://github.com/es-ude/elastic-ai.creator/commit/1d47f05290811af95c62e22dab7605b608209a8d))

- started documenting supported features - ([91c5167](https://github.com/es-ude/elastic-ai.creator/commit/91c5167056b88402190b5f51b59e49d8db7090d3))

- specified required versions for dev dependencies - ([3b39f0e](https://github.com/es-ude/elastic-ai.creator/commit/3b39f0e8c6a8e7a68a748df9217ad81f638705fc))

- fix hw function id length - ([3539c9f](https://github.com/es-ude/elastic-ai.creator/commit/3539c9f65a6bccf72c4cfb0312a4b2408e0b4fb9))

- add more middleware/skeleton specification - ([b62d982](https://github.com/es-ude/elastic-ai.creator/commit/b62d982f13360adffdfbd6a041ae30aaf83f7571))
#### Refactoring
- **(examples)** added __init__.py - ([ad897c2](https://github.com/es-ude/elastic-ai.creator/commit/ad897c20df13c550eb8110299c2c85f3ba960eeb))
- **(ir)** use new descriptor for registering fns in lowerable - ([2bd382d](https://github.com/es-ude/elastic-ai.creator/commit/2bd382ddd78212aaf925592b9c7f7838c85e89cb))
- **(ir)** decouple fn registering and calling - ([ab737b9](https://github.com/es-ude/elastic-ai.creator/commit/ab737b9bc4f86781e45d5d0b2d804ab5892a495d))
- **(lowering)** avoid keeping two registries - ([21a951e](https://github.com/es-ude/elastic-ai.creator/commit/21a951ed97229f2451d4ddf28c52db794e6f86be))
- **(lstm)** rename _integration_test to example - ([6828739](https://github.com/es-ude/elastic-ai.creator/commit/6828739ea5ee27cbe077b5c31ffbf14d66d5f480))
- **(nn)** moved simulated layer. MAC operator design simulations do not work - ([3b927b6](https://github.com/es-ude/elastic-ai.creator/commit/3b927b693e989a4e82f97b925df28641b7b33fab))
- **(nn)** moved mac operators to vhdl shared design - ([4925d67](https://github.com/es-ude/elastic-ai.creator/commit/4925d673e8086fceb5af31d1e577c56a003f1dd2))
- **(nn)** add new replacement variable in log2 calculation of linear layer - ([082b8fd](https://github.com/es-ude/elastic-ai.creator/commit/082b8fd23ba37f7428b49ab7dcbfabb056c37544))
- **(tests)** moved the opening of the serial port to context manager - ([8431bc7](https://github.com/es-ude/elastic-ai.creator/commit/8431bc74cd1f8444df5a0b6f5b184e52979ffc95))
- **(tests)** making the test a bit more convinient - ([6c778bc](https://github.com/es-ude/elastic-ai.creator/commit/6c778bca80ed20c85f2ed2d00701e3b0f2152486))
- **(vhdl)** removed unnecessary print statements and added type hint - ([c3615bd](https://github.com/es-ude/elastic-ai.creator/commit/c3615bddd1a0fab13001dc2457b5f9094c7a91e7))
- **(vhdl)** changed sensitivity list to clock only - ([01dd3c5](https://github.com/es-ude/elastic-ai.creator/commit/01dd3c5095d2d515cd1516b3a7ca56fe370bee6d))
- **(vhdl)** changing the wake_up signal to best practice method - ([04221ec](https://github.com/es-ude/elastic-ai.creator/commit/04221ec93724939dcc3bc4d3e28428ca1afffe28))
- **(vhdl)** made the name a property so the name is already set correctly and still accessible - ([593682b](https://github.com/es-ude/elastic-ai.creator/commit/593682b933642a809496dd2a9b00fdce0e9ba19d))
- **(vhdl)** better ghdl simulation class - ([873fd42](https://github.com/es-ude/elastic-ai.creator/commit/873fd421db4f0bfb179479c43ee459b71dbeee01))
- **(vhdl)** made the name a property so the name is already set correctly and still accessible - ([b422ccb](https://github.com/es-ude/elastic-ai.creator/commit/b422ccbdc6458fcab2e99b934d53886e08f1a5be))
- **(vhdl)** better ghdl simulation class - ([75c1cbe](https://github.com/es-ude/elastic-ai.creator/commit/75c1cbeaebf373561317af48db4c5081265e2452))

- rename the module design_creator to design_creator_module - ([49d3ab4](https://github.com/es-ude/elastic-ai.creator/commit/49d3ab4e1d244c13d4994df65dda2f47a846aaad))

- move design_creator module to nn package - ([f17a6da](https://github.com/es-ude/elastic-ai.creator/commit/f17a6dac5e612ba99bfc906862de3e3048aa7a17))

- renamed tests folder to test_utils and moved to elasticai/creator - ([e2b86a8](https://github.com/es-ude/elastic-ai.creator/commit/e2b86a8f8b556cd3d285f59052f11a9c173592a8))

- remove unnecessary files - ([26b8afa](https://github.com/es-ude/elastic-ai.creator/commit/26b8afaa457969777b03f04e537470f1f6917055))
#### Build system

- omit *_test in main folder for coverage - ([da8104d](https://github.com/es-ude/elastic-ai.creator/commit/da8104da6da54438104cd6bdecd68cc06d08cadd))

- clean up pyproject - ([fa27806](https://github.com/es-ude/elastic-ai.creator/commit/fa27806b53b5d27f84c4ccee57e7abefbe242cf6))

- add missing pytest-cov - ([1707f07](https://github.com/es-ude/elastic-ai.creator/commit/1707f07eb91693a03e3e33e1f3a4cc1ab12037c2))
#### Style

- beautify b61c6180a56de314e569338eebbbdbe45a889f42 - ([ce92f2b](https://github.com/es-ude/elastic-ai.creator/commit/ce92f2b593d3b705e34867a8b87d90e3f4a7d9a9))

- beautify 903501083d6acaee8b472f22e7bf24cddb3647b8 - ([cc6fda5](https://github.com/es-ude/elastic-ai.creator/commit/cc6fda534289fdf4359c46e9e08ba984c7638a07))

- - -

## [v0.59.2](https://github.com/es-ude/elastic-ai.creator/compare/v0.59.1..v0.59.2) - 2025-05-08
#### Bug Fixes

- copy model to cpu for quantized inference - ([0c5d88e](https://github.com/es-ude/elastic-ai.creator/commit/0c5d88e26e55eb11d2a729c5a7bf6b865927b61f))

- - -

## [v0.59.1](https://github.com/es-ude/elastic-ai.creator/compare/v0.59.0..v0.59.1) - 2025-05-08
#### Features

- inject network to FirmwareENv5 - ([bf2c53f](https://github.com/es-ude/elastic-ai.creator/commit/bf2c53f554bfc312875f08cb99d95e028364667b))

- add lstm_network testbench - ([37d7921](https://github.com/es-ude/elastic-ai.creator/commit/37d79212f52b949634f7af321e7b7bc56306ffeb))

- reintegrate lstm implementation - ([9440fbb](https://github.com/es-ude/elastic-ai.creator/commit/9440fbb1ed9c81218e62f4e60917127e128d3856))

- added skeleton, middleware, top module and Vivado constraints fopr env5 LSTM example - ([67117a4](https://github.com/es-ude/elastic-ai.creator/commit/67117a443032cbafd3bcf8abcab7177a801fd659))

- lstm reintegration - ([c23aa53](https://github.com/es-ude/elastic-ai.creator/commit/c23aa532d02e096f3ca7a83c38a46b6fb6d295d7))

- lstm reintegration - ([0ffa9b0](https://github.com/es-ude/elastic-ai.creator/commit/0ffa9b029636ab36570019e9f99fd5c788281e26))
#### Bug Fixes
- **(precomputed)** do not copy input to cpu - ([586b774](https://github.com/es-ude/elastic-ai.creator/commit/586b77458e9529dc8f12023fbefb6a3747fd222e))
- **(precomputed)** set step_lut as non trainable parameter - ([95954c2](https://github.com/es-ude/elastic-ai.creator/commit/95954c2cbd52a85f762128ce3a88259085431536))
- **(tests)** add step_lut to state_dict - ([e18f46f](https://github.com/es-ude/elastic-ai.creator/commit/e18f46f0a70b1000e8e6d0ea3ecdddce2ad325d5))

- don't save uut in testbench - ([7f09a2a](https://github.com/es-ude/elastic-ai.creator/commit/7f09a2ab3549d309759e531dd5b6ec4051a9d3e7))

- names and templates for lstm - ([3ad358c](https://github.com/es-ude/elastic-ai.creator/commit/3ad358c698bca447376b910bdd275bc806eb6db6))

- move `create_testbench` to correct class - ([53bc568](https://github.com/es-ude/elastic-ai.creator/commit/53bc5684682abbc721969f357b0810175a89a25f))

- correct `create_testbench` for lstm - ([e82af52](https://github.com/es-ude/elastic-ai.creator/commit/e82af52217f0ef1874abfcd0b43f1d905ed3e4bb))

- fix lstm test bench file name - ([f879b4d](https://github.com/es-ude/elastic-ai.creator/commit/f879b4de526bc7c7cd93371b5874c1eac2f465f5))

- fix lstm names - ([da279f7](https://github.com/es-ude/elastic-ai.creator/commit/da279f742409b9c31d690d385c4d04896e3afbb4))

- parametrize names - ([c13576d](https://github.com/es-ude/elastic-ai.creator/commit/c13576d380c58f658c4054a645f8407041a95faf))

- added missing file for network skeleton tpl - ([38fe8a7](https://github.com/es-ude/elastic-ai.creator/commit/38fe8a7337479e5b7c02d66d682431e95b03e190))

- add saving constraints and sources to plug&play ENV5 - ([41aa4f1](https://github.com/es-ude/elastic-ai.creator/commit/41aa4f105fb59bcdc76b031800af916eb0c76f35))

- turn `vhdl.top` into a package - ([2a272ea](https://github.com/es-ude/elastic-ai.creator/commit/2a272ea4e0b470bf8684098cf48e8121ee92d27f))
#### Documentation

- explain relationship between LSTM, LSTMNetwork and their sw/hw impl - ([7db8974](https://github.com/es-ude/elastic-ai.creator/commit/7db8974e11a586e60c7f154e7cbbbf27b75a9c41))

- - -

## [v0.59.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.58.0..v0.59.0) - 2025-05-08
#### Features

- add xnor-popcount based mac bin impl - ([6a63eb3](https://github.com/es-ude/elastic-ai.creator/commit/6a63eb358ce3279bcdbca468aed25445ab0be13e))
#### Refactoring

- add create_simulation method to layer - ([d3f8746](https://github.com/es-ude/elastic-ai.creator/commit/d3f874636e2c78d584e7b39e31f10d0ba6ab9e9b))

- add create_simulation method to layer - ([6c93c81](https://github.com/es-ude/elastic-ai.creator/commit/6c93c81b5910dfcc2eb71d64c137be5f9b8d0fad))
#### Style

- beautify f94e16fd03d289124dd20dd844776d517fb91e4a - ([db35406](https://github.com/es-ude/elastic-ai.creator/commit/db354066168885e5ae91d18efd705d239319b81a))

- - -

## [v0.58.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.57.1..v0.58.0) - 2025-05-08
#### Features
- **(linear)** use bias as default - ([14b01be](https://github.com/es-ude/elastic-ai.creator/commit/14b01be0e3aad2315daa1864588701ce2fd8dff7))
- **(tests)** add small integration test to verify that conv1d layer generates correct design - ([da45cc3](https://github.com/es-ude/elastic-ai.creator/commit/da45cc30b481ee27e9bd9d5d172e29a0bd0b519f))
- **(tests)** add small integration test to verify that linear layer creates correct design - ([f71e43f](https://github.com/es-ude/elastic-ai.creator/commit/f71e43f92cbfbf2c7a735a610c3dbca790ea8299))
- **(tests)** add tests for linear design - ([55a366a](https://github.com/es-ude/elastic-ai.creator/commit/55a366a34049708ee62d0c440f67640435716900))
- **(tests)** add tests for fixed point linear layer - ([2e959b6](https://github.com/es-ude/elastic-ai.creator/commit/2e959b619ea9ef66dfede042131f7116b18f3532))
- **(tests)** add tests for the conv1d design - ([c2c94bd](https://github.com/es-ude/elastic-ai.creator/commit/c2c94bd91e09e507120cfcf71e28b0797dca419c))
- **(tests)** add tests for conv1d layer - ([c645297](https://github.com/es-ude/elastic-ai.creator/commit/c6452975ef02d6c2a46ca4550311238a46917636))

- basic fxp mac + hw/sw simulation tests - ([f34a1ed](https://github.com/es-ude/elastic-ai.creator/commit/f34a1edc70c34d456daa222bed537d793ee0c29e))

- add parameter getters - ([3c184c0](https://github.com/es-ude/elastic-ai.creator/commit/3c184c075cdb94ffae05ea7424e33dd98a4c09f9))

- handle colons in ghdl sim parsing - ([f485531](https://github.com/es-ude/elastic-ai.creator/commit/f485531881b0874eb14df58ccb3873889cb1cac6))

- parse ghdl output - ([3f7e05f](https://github.com/es-ude/elastic-ai.creator/commit/3f7e05f44f86776ea01b28b176116730d94a9354))

- add number_conversion - ([1fde323](https://github.com/es-ude/elastic-ai.creator/commit/1fde32308a83a3de71af736eda47a81bb5bc468b))
#### Bug Fixes
- **(chore)** exclude simulations from coverage - ([de6a293](https://github.com/es-ude/elastic-ai.creator/commit/de6a29324c6e688139a82bea0b08afda1f0388bc))
- **(chore)** exclude simulations from coverage - ([3b31a45](https://github.com/es-ude/elastic-ai.creator/commit/3b31a45bccb262d60a1d920618457514a2bd8a95))

- make mac impl use round to zero logic - ([f8c674f](https://github.com/es-ude/elastic-ai.creator/commit/f8c674f5a2382a1d8b9b1630a8af35977dc0c9c9))

- remove already dropped padding, stride and dilation - ([79494fe](https://github.com/es-ude/elastic-ai.creator/commit/79494fe6116921ff7b4d1287bd67451ba7498ecd))

- remove stride and padding as supported parameters for conv1d - ([05b57d1](https://github.com/es-ude/elastic-ai.creator/commit/05b57d1d2e112a21259fa2221f2204b3f6d87bfe))

- ignore one more line for ghdl out parsing - ([c48ec8f](https://github.com/es-ude/elastic-ai.creator/commit/c48ec8f42f906222223b7282c677ffdfe5cd06ec))

- remove need for csv in testbench - ([f905bbf](https://github.com/es-ude/elastic-ai.creator/commit/f905bbf6e1f9119677b7f73caf58b35343e0d7bb))

- remove unnecessary output quantization of the SiLU base module - ([350faa5](https://github.com/es-ude/elastic-ai.creator/commit/350faa52d93994738a68cc0222dfb907b5174f12))
#### Miscellaneous Chores

- add simulation test tag - ([f7fda58](https://github.com/es-ude/elastic-ai.creator/commit/f7fda587861e83c729eaa90791b34efc4f9b433d))
#### Documentation
- **(readme)** use create_design function instead of translate in minimal example - ([b9351ca](https://github.com/es-ude/elastic-ai.creator/commit/b9351ca28b37a0ee7f34187b01986c3ba11c6827))

- add documentation for number conversion - ([617e2c6](https://github.com/es-ude/elastic-ai.creator/commit/617e2c61fab2b21459a1a62c65c184cdbcc35e09))
#### Refactoring
- **(conv1d)** remove failing test - ([e56a07f](https://github.com/es-ude/elastic-ai.creator/commit/e56a07feb90b8853f3b2901503bbe034fa7b4f16))
- **(linear)** remove unused import - ([24164b0](https://github.com/es-ude/elastic-ai.creator/commit/24164b0ff52f9e26a21a2342c9be2d5079a270e6))
- **(linear)** split layer.py into multiple files to improve readability - ([37128c9](https://github.com/es-ude/elastic-ai.creator/commit/37128c95d0a28268b1889783b31e536074d01ab9))
- **(silu)** add test for the fixed point silu - ([2b0da94](https://github.com/es-ude/elastic-ai.creator/commit/2b0da947ff8cddf282294f4139a74b4b723cc4cb))
- **(tests)** remove not necessary fixed weights - ([d9482ce](https://github.com/es-ude/elastic-ai.creator/commit/d9482cec23e20d4b370be8037fb528552b92c2cf))

- move number conversion modules - ([ed3086c](https://github.com/es-ude/elastic-ai.creator/commit/ed3086c149388876e8cd243cd535199bded8f9f5))

- rename hw_integ_test.py - ([8324d1a](https://github.com/es-ude/elastic-ai.creator/commit/8324d1abaad92a12d5fcdee9925c58b9c9743aff))

- simplify simulating test benches - ([b607211](https://github.com/es-ude/elastic-ai.creator/commit/b6072118e1d84e59b1faac73ec3c75c6aca88ee9))

- rename translatable module to design_creator - ([0aa4d72](https://github.com/es-ude/elastic-ai.creator/commit/0aa4d72458aef4a91c87334048c357a276627d3d))

- rename Translatable protocol to DesignCreator - ([e60bf7f](https://github.com/es-ude/elastic-ai.creator/commit/e60bf7f0688ba6e875d821c8cab4c34f4054cdec))

- split batch normed conv1d and conv1d layers in seperate files and add parameter getters - ([924e5f3](https://github.com/es-ude/elastic-ai.creator/commit/924e5f3646c151667c881360ed68aad890dc5a67))

- simplify number conversion implementations - ([715d6c7](https://github.com/es-ude/elastic-ai.creator/commit/715d6c7540748fb53e93f78782f653ee45e6bdb4))

- rename SiLUWithTrainableScaleBeta to AdaptableSiLU - ([fdf34b0](https://github.com/es-ude/elastic-ai.creator/commit/fdf34b0c04eae35af3867c8f9109cf24400b4d33))
#### Style

- beautify 4b05c4a847ee33048b9552a93f816d6fec3c404f - ([c10028b](https://github.com/es-ude/elastic-ai.creator/commit/c10028bd5a2019c8de67f95bf1c25c930c5ec8d0))

- - -

## [v0.57.1](https://github.com/es-ude/elastic-ai.creator/compare/v0.57.0..v0.57.1) - 2025-05-08
#### Bug Fixes

- try to exclude test files from build - ([f282ac0](https://github.com/es-ude/elastic-ai.creator/commit/f282ac06aae45451d3787d74cda54b51e7f28200))

- - -

## [v0.57.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.56.0..v0.57.0) - 2025-05-08
#### Features

- exclude tests from build - ([72b8e0a](https://github.com/es-ude/elastic-ai.creator/commit/72b8e0af0e3fddc154be5763b26f5174cc49d7f4))

- global math operations depend on math operations of supported layers - ([127ffdb](https://github.com/es-ude/elastic-ai.creator/commit/127ffdb29853587e4f819d75077e524e7a168bc5))
#### Refactoring

- rename test files from test_*.py to *_test.py to improve readabilitly - ([b7d3557](https://github.com/es-ude/elastic-ai.creator/commit/b7d3557338617c90b9b70459c4eeff12cc1c4623))

- move unit tests to the elasticai package - ([bb0ab8b](https://github.com/es-ude/elastic-ai.creator/commit/bb0ab8b8b8636c07318bae5e662836a07b5f33ec))

- - -

## [v0.56.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.55.2..v0.56.0) - 2025-05-08
#### Features

- add a public quantize function to allow initial quantization of model inputs - ([c8a170c](https://github.com/es-ude/elastic-ai.creator/commit/c8a170cb1ba529855790dcaf2dad97c38171174e))

- add layers to __init__ file in fixed_point package to improve usability - ([93917d8](https://github.com/es-ude/elastic-ai.creator/commit/93917d8654794b7f5baa005e989f2984d6c846e3))
#### Bug Fixes

- fix broken imports and other errors that leads to failing tests - ([6eac561](https://github.com/es-ude/elastic-ai.creator/commit/6eac56189e1e449d378867a4b4d8003967f48689))

- remove deprecated layers - ([59cdaae](https://github.com/es-ude/elastic-ai.creator/commit/59cdaaea412f2d7aa6a8fd63a8ba931688ddc855))

- remove deprecated layers - ([d55c041](https://github.com/es-ude/elastic-ai.creator/commit/d55c041e0141fe30bf3038074c061704ed057682))

- outdated imports - ([650a71e](https://github.com/es-ude/elastic-ai.creator/commit/650a71e36f9c12db1160b86f82ad4d37715b19d7))
#### Miscellaneous Chores

- remove fixed commit scopes - ([9f6b2e7](https://github.com/es-ude/elastic-ai.creator/commit/9f6b2e793e7a952f5ddff7efc89677a2f00c935e))
#### Documentation

- add glossary entry - ([8855e02](https://github.com/es-ude/elastic-ai.creator/commit/8855e0248e97a2628c1ff73e69538c969f52d685))

- update minimal example to reflect the most recent changes - ([8f122a0](https://github.com/es-ude/elastic-ai.creator/commit/8f122a04699c42ff7abb7179d3cb7412cf94c0ef))

- start glossary - ([b2d82cd](https://github.com/es-ude/elastic-ai.creator/commit/b2d82cdcc663d13492b611a700321c4bbcf452be))
#### Refactoring

- reformat code - ([3c728a0](https://github.com/es-ude/elastic-ai.creator/commit/3c728a020ffc29c052d60b8917b7721399a05766))

- reformat code - ([1fb5ed6](https://github.com/es-ude/elastic-ai.creator/commit/1fb5ed6663424a98a36db3ad6ef899d62c74b75c))

- adapt the structure of the tests directory to the latest changes - ([b3cd5cc](https://github.com/es-ude/elastic-ai.creator/commit/b3cd5cc5c3cb68e7b5adb8136322426830d6db40))

- rename inputs parameter to x in forward parameter lists - ([314b747](https://github.com/es-ude/elastic-ai.creator/commit/314b747e72509a2d34e72cc1d7738e2e26c18bd3))

- remove arithmetics and autograd_functions from base_modules - ([1e23c74](https://github.com/es-ude/elastic-ai.creator/commit/1e23c7406ffb0c0ab0a62aa31f8d61a502a4886f))

- rename and move modules to fit our new scheme - ([8effe1a](https://github.com/es-ude/elastic-ai.creator/commit/8effe1ac03fceebd324e4ad07f9d305c8e7d0c08))

- separate interface for conv1d - ([76ba0ac](https://github.com/es-ude/elastic-ai.creator/commit/76ba0ac054d7acd07ace2eb9875e9bd3473eeca3))

- move batchnormed layers to their base versions - ([cf4f1d5](https://github.com/es-ude/elastic-ai.creator/commit/cf4f1d5974e6d2201320bc6d2e017745870a14e0))

- move batchnormed layers to their base versions - ([b1e0feb](https://github.com/es-ude/elastic-ai.creator/commit/b1e0feb6e45fd4f300065397ab2698382135c4b5))

- restructure packages - ([2fa7a4f](https://github.com/es-ude/elastic-ai.creator/commit/2fa7a4f58a481868edca1bdd3a568686130873dd))

- improve separation of core packages - ([b5f469f](https://github.com/es-ude/elastic-ai.creator/commit/b5f469f1fbb354547560c9fefcd88e271382ae91))

- remove empty folders, start docs improvements - ([28a9a2d](https://github.com/es-ude/elastic-ai.creator/commit/28a9a2d96521fbcb53ca889b746470bb99aef20f))

- remove mlframework/typing.py - ([6858537](https://github.com/es-ude/elastic-ai.creator/commit/6858537f7906b75495376828c4713690b14cb461))
#### Style

- beautify fcb153ea3aa32a73e07dd1f71d148634698a6cda - ([6515ab0](https://github.com/es-ude/elastic-ai.creator/commit/6515ab0225bd4b55b4e8a7ad1a5e4acb2d397ea3))

- - -

## [v0.55.2](https://github.com/es-ude/elastic-ai.creator/compare/v0.55.1..v0.55.2) - 2025-05-08
#### Bug Fixes

- add dummy batch dimension to meet the requirements of the batch norm - ([0f499f0](https://github.com/es-ude/elastic-ai.creator/commit/0f499f048c0606de3e14163f16e8bf049708e6f1))

- - -

## [v0.55.1](https://github.com/es-ude/elastic-ai.creator/compare/v0.55.0..v0.55.1) - 2025-05-08
#### Bug Fixes

- fix non existing in_channels variable and remove unused import - ([0e73c2d](https://github.com/es-ude/elastic-ai.creator/commit/0e73c2dc6876772d0caba46639af77bd5ac53b62))

- - -

## [v0.55.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.54.0..v0.55.0) - 2025-05-08
#### Features

- implemented batch normed conv1d layer - ([cd6836c](https://github.com/es-ude/elastic-ai.creator/commit/cd6836cc72b3fecee1b522f9b8934fabefd46d63))
#### Bug Fixes

- typing and errors - ([af6f859](https://github.com/es-ude/elastic-ai.creator/commit/af6f85913fd6111bcc7164a106a9cbb8d4b7b9a0))
#### Refactoring

- set alias for Design/FPLinear - ([559395c](https://github.com/es-ude/elastic-ai.creator/commit/559395cde0dca73344cd162df04fea510a621b49))

- - -

## [v0.54.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.53.0..v0.54.0) - 2025-05-08
#### Bug Fixes

- use same bit width for all rom values - ([cd609e6](https://github.com/es-ude/elastic-ai.creator/commit/cd609e65f306e62110fbdc4113f4bb330f960f19))
#### Refactoring

- rename precomputed monotonic increasing module - ([ab8dfdf](https://github.com/es-ude/elastic-ai.creator/commit/ab8dfdf4c19646ae14dd787203a380eda47c281d))

- - -

## [v0.53.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.52.0..v0.53.0) - 2025-05-08

- - -

## [v0.52.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.51.0..v0.52.0) - 2025-05-08
#### Features

- implement fixed point one dimensional convolution - ([2ea9389](https://github.com/es-ude/elastic-ai.creator/commit/2ea9389a37eac7be62e26a9727b8824b47fc2085))

- added Swish activation function precomputed - ([26d292e](https://github.com/es-ude/elastic-ai.creator/commit/26d292e83183ac0b6bee7afa70f3d616e42b2438))
#### Bug Fixes

- fix missing parameter in tests for conv1d - ([d8f8d4c](https://github.com/es-ude/elastic-ai.creator/commit/d8f8d4c40ec1576c5dc58a38b2b80d9d4130b4fd))
#### Refactoring

- simplify string - ([de8d3ec](https://github.com/es-ude/elastic-ai.creator/commit/de8d3ec98a6105d48630a0b2e6d82f15c3e75a9e))

- - -

## [v0.51.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.50.0..v0.51.0) - 2025-05-08
#### Features

- revert changes and explicitly set semantic release version to v7 instead of v8 - ([2ecf0db](https://github.com/es-ude/elastic-ai.creator/commit/2ecf0db3c22ce034c4f36a26c96027f0229a4bf0))

- increase semantic release version to v8.0.4 - ([bb29612](https://github.com/es-ude/elastic-ai.creator/commit/bb2961243f20ade3e7c4a142601f58fca6e9b5ad))

- apply proposed semantic release migration procedure - ([d5ea981](https://github.com/es-ude/elastic-ai.creator/commit/d5ea981cd8852e5790c77d9667187168c34c81e3))

- seperate semantic release run into multiple steps - ([475b425](https://github.com/es-ude/elastic-ai.creator/commit/475b425910c4a124c34ca9a68fd5c49b4789541b))

- enable debug messages - ([36ca597](https://github.com/es-ude/elastic-ai.creator/commit/36ca597ded38bb3c5343e872ed7cf9cb09065a6f))

- add debug messages - ([ee09864](https://github.com/es-ude/elastic-ai.creator/commit/ee09864686d87471617aae4ae65118096d31a6ff))

- rename custom float to float to improve readability - ([794bfe7](https://github.com/es-ude/elastic-ai.creator/commit/794bfe79a6a821050b33cc246e9e1cad09e7e682))

- added Swish activation function precomputed - ([fd487b5](https://github.com/es-ude/elastic-ai.creator/commit/fd487b57bb7d3e7525f935f1533f815e58f1dc0d))

- added nn module for Swish activation function - ([b7579c9](https://github.com/es-ude/elastic-ai.creator/commit/b7579c9f1111521064a7fd0366647da7a45e2d7a))

- implemeted base module for SiLU aka Swish activation function - ([93b5954](https://github.com/es-ude/elastic-ai.creator/commit/93b59544c1d164de2a9f9362f0aefe1aaae8d7d8))
#### Bug Fixes

- try to fix semantic release - ([0eab187](https://github.com/es-ude/elastic-ai.creator/commit/0eab187389b3d435be473671d4a593ead8586e78))

- remove not implemented jvp function - ([0ea4834](https://github.com/es-ude/elastic-ai.creator/commit/0ea48341c02d116dd3ef2a94e0997ce8e0641b60))
#### Refactoring

- remove newline - ([33fa0a9](https://github.com/es-ude/elastic-ai.creator/commit/33fa0a932b5c2126a004b429702bcda72e696069))

- remove noise comment - ([f6be240](https://github.com/es-ude/elastic-ai.creator/commit/f6be240b03484876627f5f7de5198fd1332d6ba7))

- deleted unnecessary File test_silu.py from branch - ([80e8919](https://github.com/es-ude/elastic-ai.creator/commit/80e8919078ba32dd9af0146a94bd38b63bc761b1))

- removed unnecessary file - ([fe58c0f](https://github.com/es-ude/elastic-ai.creator/commit/fe58c0f22d38284e289542b6f3e58fbff60963f9))

- delete files that aren´t necessary - ([53aebf3](https://github.com/es-ude/elastic-ai.creator/commit/53aebf3702c9c3511ef81e8fd9a1fcca018bf26d))

- changed names of learnable parameters in the swish function - ([bb2b7a8](https://github.com/es-ude/elastic-ai.creator/commit/bb2b7a81bf6365a54590575f39a447b6cd769cd9))

- - -

## [v0.50.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.49.0..v0.50.0) - 2025-05-08
#### Features

- implement custom float arithmetics - ([b72713e](https://github.com/es-ude/elastic-ai.creator/commit/b72713e3db1e15957e865ed95216a2f180523114))

- implement RoundToCustomFloat autograd function - ([0794a8e](https://github.com/es-ude/elastic-ai.creator/commit/0794a8e900d6f87edc03dbd71162e7300e13b5ae))
#### Bug Fixes

- return wrong number of values in the backward pass - ([6bcfa4e](https://github.com/es-ude/elastic-ai.creator/commit/6bcfa4eff8d9b7c0c0461a61800ef68ef6b0cb62))
#### Refactoring

- rename CustomFloatArithmetics to FloatArithmetics - ([824b029](https://github.com/es-ude/elastic-ai.creator/commit/824b029971789c951a243937d942d7597225e829))

- rename FloatArithmetics to TorchArithmetics - ([5cd7a3b](https://github.com/es-ude/elastic-ai.creator/commit/5cd7a3b6913b14456f524a9486bac2c42dc72412))

- - -

## [v0.49.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.48.1..v0.49.0) - 2025-05-08
#### Features

- update readme and add small improvements - ([8f2bbd0](https://github.com/es-ude/elastic-ai.creator/commit/8f2bbd093e18c15421abab20ecb0f9afbc6d12a1))
#### Documentation

- add minimal example that demonstrates the usage of the creator - ([64030f2](https://github.com/es-ude/elastic-ai.creator/commit/64030f2eb129ff8275022ab0b8bf4945d42626a8))

- complete table of contents - ([cf0ef63](https://github.com/es-ude/elastic-ai.creator/commit/cf0ef63eb628521f14406fb7d59cee53c71c8d60))

- - -

## [v0.48.1](https://github.com/es-ude/elastic-ai.creator/compare/v0.48.0..v0.48.1) - 2025-05-08
#### Bug Fixes

- only create coverage reports in PR - ([1bd728f](https://github.com/es-ude/elastic-ai.creator/commit/1bd728f4e8edb6595a35dafd71c5d68263a7358f))

- - -

## [v0.48.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.47.2..v0.48.0) - 2025-05-08
#### Features

- only trigger coverage report when pushing to main - ([b4b23c9](https://github.com/es-ude/elastic-ai.creator/commit/b4b23c988803165895c14a8357427a3069f09233))

- add coverage workflow to create reports - ([3f6caca](https://github.com/es-ude/elastic-ai.creator/commit/3f6caca6a626923ec3d8078320fa9b70092495ee))

- add pytest-cov dependency - ([a737729](https://github.com/es-ude/elastic-ai.creator/commit/a7377290ffee7359f6f8c0392960d7038fe2a73b))

- use binary values instead of hex values to fill the rom template - ([af56c02](https://github.com/es-ude/elastic-ai.creator/commit/af56c02da42433c2db1a9a2a6ddb3705d213d765))
#### Bug Fixes

- use poetry run to run pytest - ([7058e42](https://github.com/es-ude/elastic-ai.creator/commit/7058e42cc7fa0849841578f2bafd6a3fc6155f2a))
#### Refactoring

- remove unused to_vhdl_hex_string function - ([24ccbf1](https://github.com/es-ude/elastic-ai.creator/commit/24ccbf1a1d9ff3d270faba19581a6f72eadb751e))

- improve readability - ([e4de568](https://github.com/es-ude/elastic-ai.creator/commit/e4de5682419829675a92ff95f8e853dc28cf181e))

- - -

## [v0.47.2](https://github.com/es-ude/elastic-ai.creator/compare/v0.47.1..v0.47.2) - 2025-05-08
#### Bug Fixes

- fix error when passing a cuda tensor to the IdentityStepFunction - ([7f49617](https://github.com/es-ude/elastic-ai.creator/commit/7f496171a547bae17c69976c35d437428022447f))

- - -

## [v0.47.1](https://github.com/es-ude/elastic-ai.creator/compare/v0.47.0..v0.47.1) - 2025-05-08
#### Bug Fixes

- remove wrongly committed files - ([4fdea0c](https://github.com/es-ude/elastic-ai.creator/commit/4fdea0c9ff2db5e8af3f208bbd83d995332d5b85))
#### Miscellaneous Chores

- add do_not_commit path to prevent files from being committed by mistake - ([af13e16](https://github.com/es-ude/elastic-ai.creator/commit/af13e1687f57fc3545d0c114263ed439b78973cd))
#### Refactoring

- merge fp quant and fp dequant into a roundtofixedpoint autograd function - ([b986a62](https://github.com/es-ude/elastic-ai.creator/commit/b986a62ea7a0a58e6479aa5082ddd2de11ed27d7))

- - -

## [v0.47.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.46.1..v0.47.0) - 2025-05-08
#### Features

- simplify project structure - ([81cbcb3](https://github.com/es-ude/elastic-ai.creator/commit/81cbcb343b26473290609c7715051059127a924b))
#### Refactoring

- remove unused manifest module - ([55f8e6d](https://github.com/es-ude/elastic-ai.creator/commit/55f8e6deac74d953b97b031a22e0dd9a73ecf20c))
#### Style

- beautify 7beebdbc67074dc6f8e8a0320563385ee49a7915 - ([1c7fead](https://github.com/es-ude/elastic-ai.creator/commit/1c7feadd0825be6648702c7ecffcdb1c2ce974f5))

- - -

## [v0.46.1](https://github.com/es-ude/elastic-ai.creator/compare/v0.46.0..v0.46.1) - 2025-05-08
#### Bug Fixes

- fix wrong port definitions - ([9a4c8af](https://github.com/es-ude/elastic-ai.creator/commit/9a4c8af6f8f8be2bf6fff49c25fc0ca12cbea45a))

- - -

## [v0.46.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.45.0..v0.46.0) - 2025-05-08
#### Bug Fixes

- quantize weights before inference - ([61153e6](https://github.com/es-ude/elastic-ai.creator/commit/61153e60d6854bacf0bd2501d96efc3f6e62714e))

- fix some syntax errors - ([3997bbd](https://github.com/es-ude/elastic-ai.creator/commit/3997bbdb134a94defd4e32ad1a2eb3aa236d6b96))
#### Refactoring

- remove debug print call - ([f85172e](https://github.com/es-ude/elastic-ai.creator/commit/f85172ebddfceb98a7c661cd3f57db60b19b61c0))

- - -

## [v0.45.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.44.0..v0.45.0) - 2025-05-08
#### Features

- simplify usage for the elasticai.creator.nn.vhdl package by adding layers to __init__ - ([2c7c968](https://github.com/es-ude/elastic-ai.creator/commit/2c7c96858ec9d935389a960baee46e8c506f9b5c))
#### Bug Fixes

- fix broken import in base template generator and move it with its template to own folder - ([9eb1f70](https://github.com/es-ude/elastic-ai.creator/commit/9eb1f70cff10e075712d5bf7e3fc9fcfed2aae19))
#### Miscellaneous Chores

- update dependencies - ([a2558b5](https://github.com/es-ude/elastic-ai.creator/commit/a2558b5649d5416c730cfdfebdd4d38ce48a6a88))
#### Refactoring

- use Identity class from base_modules instead of torch - ([8f179f0](https://github.com/es-ude/elastic-ai.creator/commit/8f179f0bbbb2510b294665e0502715b6b69346c8))

- remove unused imports - ([e8881d3](https://github.com/es-ude/elastic-ai.creator/commit/e8881d31e11d3e27489322deabb3c29d420e568b))

- use container types from collections.abc instead of typing because they are deprecated - ([7a45e67](https://github.com/es-ude/elastic-ai.creator/commit/7a45e672cdcc47b426a57a8297febc8aa9664744))

- rename ports module to port_definitions - ([b5a64b8](https://github.com/es-ude/elastic-ai.creator/commit/b5a64b812145a34dd1dd0d20cb2ca31f18804a1f))

- remove unused base signal definition - ([14dc275](https://github.com/es-ude/elastic-ai.creator/commit/14dc275beea7f3c757433eb9b3872c895fc6fca3))

- remove deprecated documentation - ([349a9f8](https://github.com/es-ude/elastic-ai.creator/commit/349a9f866e001ce0494f9876d894ef0c5833817d))

- create rom design folder - ([9e40f5f](https://github.com/es-ude/elastic-ai.creator/commit/9e40f5fa9b40c2542e4ef99cf02d1b6004ad2a60))

- better separation of designs and modules - ([44f22ae](https://github.com/es-ude/elastic-ai.creator/commit/44f22ae25a02c0c4810e64c970cdc5dd28135c89))

- rename monotonously increasing scalar function - ([baff8b2](https://github.com/es-ude/elastic-ai.creator/commit/baff8b2fd8569c60b51906458c2d541e1371f111))

- remove unused pytest-bdd dependency - ([e9203a0](https://github.com/es-ude/elastic-ai.creator/commit/e9203a0223ef3adfcbd40af841e569438684e1c8))

- transform bdd test to pytest test - ([475ec7b](https://github.com/es-ude/elastic-ai.creator/commit/475ec7bd12ed0f43b65438a2ef62aa97d3ca8b14))

- remove some newlines, use create_port function and fix wrong template - ([1bc4a70](https://github.com/es-ude/elastic-ai.creator/commit/1bc4a70f173c9f380a76438842ba6708d1659aad))

- rename template and remove some newlines - ([707310b](https://github.com/es-ude/elastic-ai.creator/commit/707310b3202ec1b48f847a228455f8cd77436219))

- remove unused and redundant port definition - ([b376b75](https://github.com/es-ude/elastic-ai.creator/commit/b376b757f6dd0e6400813688a2dfdf6ca392a6f9))

- rename sequential layer module according to our convention - ([ae1da5e](https://github.com/es-ude/elastic-ai.creator/commit/ae1da5e5aced255e38f0c13691a1d42f90dd5cb3))

- remove unused template resources - ([d58f267](https://github.com/es-ude/elastic-ai.creator/commit/d58f267772839df6c254b9d749b8e5653b9a20e1))

- - -

## [v0.44.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.43.0..v0.44.0) - 2025-05-08
#### Bug Fixes

- use new Sequential constructor - ([6bb111b](https://github.com/es-ude/elastic-ai.creator/commit/6bb111b748567502c23a48a52d7e477645969996))

- port def and impl of monotonous function design - ([2d423d4](https://github.com/es-ude/elastic-ai.creator/commit/2d423d46faa86fbf43cb8ba1d01aafe92c5bfa23))

- children of sequential layer determine signal widths - ([3dd5c0c](https://github.com/es-ude/elastic-ai.creator/commit/3dd5c0cc4f7a52c7b3a86cec437005b86aa0a509))
#### Refactoring

- cleanup imports - ([c402a03](https://github.com/es-ude/elastic-ai.creator/commit/c402a031f5996c6f7a1b3a5199e1cf9697e7dc5a))
#### Style

- beautify 95ca25571e9757d932a45749e9cf92531c13ab36 - ([cdf44ce](https://github.com/es-ude/elastic-ai.creator/commit/cdf44cec1a9a656ce6b3a9d19a717a9e7163d1b6))

- - -

## [v0.43.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.42.0..v0.43.0) - 2025-05-08
#### Features

- add tests for the FPMonotonouslyIncreasingModule - ([9ba64ae](https://github.com/es-ude/elastic-ai.creator/commit/9ba64ae3d253db76a6368c5e561ce28bcec2aab5))

- introduce FPMonotonouslyIncreasingModule to easily add new activations - ([b78c922](https://github.com/es-ude/elastic-ai.creator/commit/b78c9225f7f70ec329bee5705c11d9e7b1392c41))
#### Bug Fixes

- set correct signal names for x and y address - ([5354a2a](https://github.com/es-ude/elastic-ai.creator/commit/5354a2a0e85bc0788f5d74377c1a685e9d0e0de7))

- use elsif in lookup table - ([f375ba3](https://github.com/es-ude/elastic-ai.creator/commit/f375ba3784bf92887e689f77f592dfc2fa2c7e2c))

- increase default sampling intervall - ([07620d3](https://github.com/es-ude/elastic-ai.creator/commit/07620d3e2ee9db1bc6aa081a15274cb79b5ee4b0))
#### Refactoring

- remove unnecessary tests - ([c0756b3](https://github.com/es-ude/elastic-ai.creator/commit/c0756b3d7a7468aa0e3d7c55e126170790bae076))

- move all arithmetics to arithmetics folder in base_modules - ([de0fd46](https://github.com/es-ude/elastic-ai.creator/commit/de0fd460eae7d7d155188d2e73dd4cc82b913718))

- - -

## [v0.42.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.41.0..v0.42.0) - 2025-05-08
#### Features

- reimplement hard tanh activation function - ([9b86f9d](https://github.com/es-ude/elastic-ai.creator/commit/9b86f9d440cc991d624a6f3492a3caf7419bdbf3))

- add working hardsigmoid implementation - ([db03ff0](https://github.com/es-ude/elastic-ai.creator/commit/db03ff080f878c9b9fe54303ead97c673022f3a1))

- make sure that inplace parameter is fixed defined - ([79b7a1e](https://github.com/es-ude/elastic-ai.creator/commit/79b7a1eea0cb71f5a838cfebf02970927410f594))

- - -

## [v0.41.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.40.0..v0.41.0) - 2025-05-08
#### Features

- add fixed point ReLU module - ([62c1555](https://github.com/es-ude/elastic-ai.creator/commit/62c15557fc515c89644c674aef9fc39d22ab672f))
#### Bug Fixes

- remove obsolete parsing functionality - ([7f85d05](https://github.com/es-ude/elastic-ai.creator/commit/7f85d05aa3da2e0fd7c266bfc9c1aad573adecc4))

- - -

## [v0.40.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.39.0..v0.40.0) - 2025-05-08
#### Features

- simplify the use of the sequential layer (same as in torch) - ([9fad15d](https://github.com/es-ude/elastic-ai.creator/commit/9fad15d774f3573fb26f168295f9bd2ae5cdd046))

- improve performance of the identity step autograd function - ([46f036c](https://github.com/es-ude/elastic-ai.creator/commit/46f036c8fb2d007d21e32214ac92d4d9aa2fe9d1))

- add quantized tanh implementation with lookup tables - ([3a1fb10](https://github.com/es-ude/elastic-ai.creator/commit/3a1fb10944e566ca33e3e745b939b6700421fdb9))

- implement bufferless component interface for precomputed scalar function - ([f701a57](https://github.com/es-ude/elastic-ai.creator/commit/f701a57db54e0d5f3e5e43047725b28646cb5f15))

- pass step lut to identity step function and improve readablility - ([c1b6747](https://github.com/es-ude/elastic-ai.creator/commit/c1b67473c33ddc27590068472dcff6969f9e7135))

- rename autograd function and pass step lut to autograd function - ([d607e98](https://github.com/es-ude/elastic-ai.creator/commit/d607e98bd14dfa1ae23e9726b2046baaede21361))

- check for autowiring protocol violation - ([3f17e00](https://github.com/es-ude/elastic-ai.creator/commit/3f17e002e050dc92516e4ff5468041f06ebd6760))

- add AutoWirer - ([f4159c8](https://github.com/es-ude/elastic-ai.creator/commit/f4159c800fe54cc0fe73fbebdf2ac0410ddac635))

- implement autograd fn to map inputs to a subset of inputs - ([26c6ec7](https://github.com/es-ude/elastic-ai.creator/commit/26c6ec7a203eea4fed4c3eb3d5c3e4893acb545f))

- add a function to easily compare tensors with pytest - ([24e737e](https://github.com/es-ude/elastic-ai.creator/commit/24e737eaea48044df3e8addaca0d1cc804a3b6f4))

- use conv1d arithmetics function to implement conv1d module - ([69778be](https://github.com/es-ude/elastic-ai.creator/commit/69778be7fd1becab2ad5099ebb8d64d4a0db0de5))

- add conv1d function to arithmetics - ([1cab190](https://github.com/es-ude/elastic-ai.creator/commit/1cab1901e324eb100f1cbccf6d54fae429210b33))

- add intermediate symbols to rule definitions - ([624b310](https://github.com/es-ude/elastic-ai.creator/commit/624b310fc9beb130902fdf3269e3f30714fe0c3f))

- support parsing partial files - ([f2c2eb6](https://github.com/es-ude/elastic-ai.creator/commit/f2c2eb69ceb8a0b02c1c4617511ccb1528931e23))

- support parsing partial files - ([8170012](https://github.com/es-ude/elastic-ai.creator/commit/817001208b774e57cfb27fb4d4ee9d704541c9f8))

- test that conv1d uses different arithmetics - ([7eb01db](https://github.com/es-ude/elastic-ai.creator/commit/7eb01dbaa2afbbb02410e6fc6272ba02fec7878a))

- add standalone parser module - ([5a9b141](https://github.com/es-ude/elastic-ai.creator/commit/5a9b141285fefecf61f581417061428cda382ad5))

- add basic vhdl parsing - ([5df2a3f](https://github.com/es-ude/elastic-ai.creator/commit/5df2a3ff4e9ba7ec33398a267cd983ad886d1fe7))

- add the ability to sum over dimension - ([c45c0e6](https://github.com/es-ude/elastic-ai.creator/commit/c45c0e676e1df70bf99c4c943874168781ef2a93))
#### Bug Fixes

- fix that last io pair was dropped when calling save_to function - ([2bc46ac](https://github.com/es-ude/elastic-ai.creator/commit/2bc46ac9c535b65ef7a3dc5cbe12b27d253c3b37))

- fix missing creation of a subpath in the save_to function - ([2a4dbdf](https://github.com/es-ude/elastic-ai.creator/commit/2a4dbdf2f6fce4de567281002dd4640ff3ae54ed))
#### Refactoring

- remove the leading underscore of the class name - ([6643bf1](https://github.com/es-ude/elastic-ai.creator/commit/6643bf13dfbe50f7b98c0a49a238041c49fa8b89))

- change indentations - ([d5f5bf0](https://github.com/es-ude/elastic-ai.creator/commit/d5f5bf07b85d7b1902d474975da58d29bc615f6d))

- remove unused import - ([4de2055](https://github.com/es-ude/elastic-ai.creator/commit/4de205551938c7a284af78b5c2c418fdf95358f6))

- remove default sampling intervall - ([9d7caea](https://github.com/es-ude/elastic-ai.creator/commit/9d7caeae98408d2eaf0c97032dae0b5b4b312429))

- remove unnecessary tests - ([23f78db](https://github.com/es-ude/elastic-ai.creator/commit/23f78db7aec7efeef669a32ebe76ea3ebcb6b133))

- move sequential layer to nn.vhdl - ([caea325](https://github.com/es-ude/elastic-ai.creator/commit/caea325588f8c87cc28d5df248129b0e73111e3d))

- small change of the folder structure - ([58783a8](https://github.com/es-ude/elastic-ai.creator/commit/58783a83a891d85c50c43a6af2ac3efa3e634657))

- remove unused base modules - ([97d1e7d](https://github.com/es-ude/elastic-ai.creator/commit/97d1e7dbc181fc03562ccbcde976eb9e661c381e))

- move torch dependency to base_moduels - ([06d1aca](https://github.com/es-ude/elastic-ai.creator/commit/06d1aca6e3ca95a1e371253aa97dee831119250c))

- remove redundant tests - ([c828d53](https://github.com/es-ude/elastic-ai.creator/commit/c828d536110205b3e00f61a33e31d0cae1eaee6f))

- pull up parse function - ([1b8f187](https://github.com/es-ude/elastic-ai.creator/commit/1b8f1874eff63130e71c1754257d5bb3d05bb827))

- pull up tokenize functions - ([ace6f1e](https://github.com/es-ude/elastic-ai.creator/commit/ace6f1eb5d0162d7454d56a5baf6f3fb59f3dc06))

- remove obsolete test helper code - ([17e4e12](https://github.com/es-ude/elastic-ai.creator/commit/17e4e1250c1b94b3f72ac9dba57f7ee66825f381))

- improve readablility - ([004c736](https://github.com/es-ude/elastic-ai.creator/commit/004c736cab22b4e8eed5eb867c203b4b62e7e235))

- - -

## [v0.39.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.38.0..v0.39.0) - 2025-05-08
#### Features
- **(template)** make precomputed scalar functions bufferless - ([89986fa](https://github.com/es-ude/elastic-ai.creator/commit/89986fad041c89d0543fe9a22946e5f5f49e2b61))

- port expansion/template based on autowiring protocol - ([0d14618](https://github.com/es-ude/elastic-ai.creator/commit/0d146181c8b789b09871af43654ca2d83ea55ddb))
#### Bug Fixes

- allow to set affine and bias equals false in translate function - ([b351284](https://github.com/es-ude/elastic-ai.creator/commit/b351284335a77caec838a8f4ea57684e429cc35b))

- adjust tests to follow previous change - ([c328bd5](https://github.com/es-ude/elastic-ai.creator/commit/c328bd565d6ba84a9d1fab788051c3e884ea2094))

- correct tuple type annotation - ([f0e7da0](https://github.com/es-ude/elastic-ai.creator/commit/f0e7da0cf186015004970102f2b9b57a9f839585))
#### Refactoring

- remove redundant quantize function - ([02094cf](https://github.com/es-ude/elastic-ai.creator/commit/02094cf412f2846821c9c2925bedcdc585fe8a8d))

- make identity layer/design names more specific - ([0aed47e](https://github.com/es-ude/elastic-ai.creator/commit/0aed47ebd3dbd784156a949822b8fc7c117e07c0))

- remove obsolete module - ([5adc999](https://github.com/es-ude/elastic-ai.creator/commit/5adc999c3f4fb5a45e569680fa466694127688da))

- - -

## [v0.38.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.37.2..v0.38.0) - 2025-05-08
#### Bug Fixes

- remove broken lstm implementation - ([c524ca2](https://github.com/es-ude/elastic-ai.creator/commit/c524ca20cc49333007c4e0bbfa167912580e5c01))

- add variable - ([229d452](https://github.com/es-ude/elastic-ai.creator/commit/229d452d0c2f798ee1dd0124f50be8f01d69ede4))

- remove dequantize - ([c111022](https://github.com/es-ude/elastic-ai.creator/commit/c111022854ce6965b705b3a3de296e032d7ff107))
#### Miscellaneous Chores

- remove unused workflow - ([dd08e08](https://github.com/es-ude/elastic-ai.creator/commit/dd08e08b0af74c4d7ba927c892de6081717657db))

- - -

## [v0.37.2](https://github.com/es-ude/elastic-ai.creator/compare/v0.37.1..v0.37.2) - 2025-05-08
#### Bug Fixes

- try manual publishing - ([c8b6c35](https://github.com/es-ude/elastic-ai.creator/commit/c8b6c355896c1f3b0630c227af8414f281b5d3ff))

- - -

## [v0.37.1](https://github.com/es-ude/elastic-ai.creator/compare/v0.37.0..v0.37.1) - 2025-05-08
#### Features

- add check that all variables are filled when saving a template - ([c988d2b](https://github.com/es-ude/elastic-ai.creator/commit/c988d2bc203790ba8ab900e8a2de6996b22d6fcb))

- add function to get all unfilled variables of a template - ([d635cb6](https://github.com/es-ude/elastic-ai.creator/commit/d635cb6098735b451aea259a8a6f15619bfcd64f))

- write function of InMemoryFile and OnDiskFile now takes Template object - ([a867ea1](https://github.com/es-ude/elastic-ai.creator/commit/a867ea15980b8ca1390327f2999c4d7b91ef3041))
#### Bug Fixes

- try to fix semantic release - ([2625e89](https://github.com/es-ude/elastic-ai.creator/commit/2625e8982c021cbf5b778e95194cc53170ab0afb))

- fix not inserted process name - ([dbabea0](https://github.com/es-ude/elastic-ai.creator/commit/dbabea07c888a5309d9ca55cd2c01ae0debea57d))
#### Refactoring

- remove deprecated and broken relu and tanh implementations - ([286686c](https://github.com/es-ude/elastic-ai.creator/commit/286686cd6a2a185a94c03585f41d15dea794b1a2))

- remove RawTemplate class - ([eb91cd8](https://github.com/es-ude/elastic-ai.creator/commit/eb91cd81475a6a9aa94fc8ab4ccf3457cef55d01))

- remove InProjectVHDLTemplate and InMemoryVHDLTemplate - ([e625399](https://github.com/es-ude/elastic-ai.creator/commit/e6253997447b0976de4ed60ec671de80ec6740a6))

- rename TemplateConfig protocol to Template - ([33d01ee](https://github.com/es-ude/elastic-ai.creator/commit/33d01eef31e7c9cb919a9684150dfba8ce1c60a5))

- temporarily rename template class - ([6fb83a2](https://github.com/es-ude/elastic-ai.creator/commit/6fb83a2d773bb474bf96f4c248de8537f91673aa))

- - -

## [v0.37.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.36.0..v0.37.0) - 2025-05-08
#### Features

- implement batch normed linear layer - ([9322f6f](https://github.com/es-ude/elastic-ai.creator/commit/9322f6f699f9884273c3f9815b9a026c9f7840ae))

- assert that all inserted variables exists in template and remove AbstractBaseTemplate - ([51f1a08](https://github.com/es-ude/elastic-ai.creator/commit/51f1a0883a8d0a54caee66080ef85f84049ad806))

- add experimental precomputed tanh in fixed point - ([0e76d03](https://github.com/es-ude/elastic-ai.creator/commit/0e76d03b6d0f23d8932b94bb7728cbeea2de0289))
#### Miscellaneous Chores

- update force-publish workflow - ([a56d2a9](https://github.com/es-ude/elastic-ai.creator/commit/a56d2a986102c26b925f20e982dd6af1e5b2fdfc))

- update force-publish workflow - ([c7b011c](https://github.com/es-ude/elastic-ai.creator/commit/c7b011cd289baa1615cde11224f2a0ec25221e15))

- update force publish workflow - ([9a0a7ac](https://github.com/es-ude/elastic-ai.creator/commit/9a0a7aca438f92e728c0310ec16adb0ded902f29))

- add  force-publish workflow - ([b59268d](https://github.com/es-ude/elastic-ai.creator/commit/b59268d15b8ef605c6dbb48e606f5b1ad746548f))
#### Refactoring
- **(unit)** remove duplicated test - ([cfd304e](https://github.com/es-ude/elastic-ai.creator/commit/cfd304e630ba4f13ee87fc074c7d05fd99b1c98a))

- rename FPLinear1d design to FPLinear - ([238f167](https://github.com/es-ude/elastic-ai.creator/commit/238f1671a28b9b5735ca7e01360d4dda7122a2a7))

- move binarize autograd function to autograd_functions folder - ([03d5bc8](https://github.com/es-ude/elastic-ai.creator/commit/03d5bc86462b36be30c2887593360ec48a908ab1))

- remove unused parameter - ([89ca654](https://github.com/es-ude/elastic-ai.creator/commit/89ca65467a983230a1dc54d8b1502e82185f2acc))

- - -

## [v0.36.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.35.0..v0.36.0) - 2025-05-08
#### Features
- **(unit)** add tests for sequential model with two layer - ([df73a4f](https://github.com/es-ude/elastic-ai.creator/commit/df73a4fb27a8867a4b633c4ffdd737ead34d2f16))
- **(unit)** test all subdesigns generated by sequential layer gets a unique name - ([009405b](https://github.com/es-ude/elastic-ai.creator/commit/009405bc64cd5e8a86909330bb450ee58ee98289))

- autogenerate sequential signal connections - ([6dfca07](https://github.com/es-ude/elastic-ai.creator/commit/6dfca078b735a3387b65c20de601426ea27371c6))

- test signal definitions, layer connections and instantiations separately - ([65201c8](https://github.com/es-ude/elastic-ai.creator/commit/65201c83bae07c62efcd705f67f34d9ff88da557))

- sequential layer can have a name - ([9e46938](https://github.com/es-ude/elastic-ai.creator/commit/9e46938e9e5fc6960e70bef26aa72ec51566a007))

- introduce abstract Translatable class - ([5d9fa2d](https://github.com/es-ude/elastic-ai.creator/commit/5d9fa2d167a8c46c301bb4a0da25718b1fcf0dee))

- add indentations to template - ([aa254d1](https://github.com/es-ude/elastic-ai.creator/commit/aa254d12f38712e798db9b31a5a58e197a44121a))
#### Bug Fixes
- **(unit)** fix that test ignores parameter - ([f448919](https://github.com/es-ude/elastic-ai.creator/commit/f448919ff4882696c0991d6aec3608616e258596))

- correct expected connections - ([2fb0f8e](https://github.com/es-ude/elastic-ai.creator/commit/2fb0f8edc45a7a38e2a9b7433dee90f139b10006))

- fix syntax error - ([396f5c4](https://github.com/es-ude/elastic-ai.creator/commit/396f5c45b382454d6cc97e4be573fcfe45a4592a))

- fix syntax errors - ([f9b57e4](https://github.com/es-ude/elastic-ai.creator/commit/f9b57e4f8173dc0bd52c21b1da351304ceb5a122))

- add missing save_to function - ([ef24ee2](https://github.com/es-ude/elastic-ai.creator/commit/ef24ee21672099359867bc4a74f5804af0c10158))
#### Miscellaneous Chores

- adjust main.yml - ([2680a3a](https://github.com/es-ude/elastic-ai.creator/commit/2680a3a142e0df535fd07b716fdd6f5d7b0c1c14))

- adjust main.yml - ([359889c](https://github.com/es-ude/elastic-ai.creator/commit/359889c28d4ff4776eec3ff5e6d22dfab450cb4e))

- adjust main.yml - ([93550cc](https://github.com/es-ude/elastic-ai.creator/commit/93550cccd7eda401dc7f759da8efe048661c2573))
#### Refactoring

- reduce code duplication - ([ae65808](https://github.com/es-ude/elastic-ai.creator/commit/ae65808bc66ebd2982a80ec3b6c5d70f749723d8))

- rename FPLinear1d to FPLinear - ([5550dd9](https://github.com/es-ude/elastic-ai.creator/commit/5550dd97956171f53edc59e534dd02161c463133))

- use identity instead of linear layer to simplify test - ([28a75c3](https://github.com/es-ude/elastic-ai.creator/commit/28a75c337734b6bed887b1a3f9fc0369d92d330b))

- fix/add missing type annotations - ([d47a8c1](https://github.com/es-ude/elastic-ai.creator/commit/d47a8c1c8919066e557a702f3bccc3928f35fa69))

- remove unused import - ([602c137](https://github.com/es-ude/elastic-ai.creator/commit/602c1376cefe7dc4a95ef7cf04b9f67b0e2cf1e3))

- remove unused translatable protocol and rename module - ([9d59f8c](https://github.com/es-ude/elastic-ai.creator/commit/9d59f8cd533b32baf6f90365e0db5a8b18d1c5a7))

- remove unused imports - ([735dcfa](https://github.com/es-ude/elastic-ai.creator/commit/735dcfaaba2ed4cace1b30d328fdaaf5433c5c42))

- remove unused imports - ([d9592ec](https://github.com/es-ude/elastic-ai.creator/commit/d9592ecb3677ba8050cb737bbc112987e72f25b5))

- - -

## [v0.35.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.34.0..v0.35.0) - 2025-05-08
#### Features

- use fixed base template - ([432dfd9](https://github.com/es-ude/elastic-ai.creator/commit/432dfd9518a0a33a7ba08cf95436f9472b274b52))

- implement translatable identity module - ([54327fa](https://github.com/es-ude/elastic-ai.creator/commit/54327fa3e45ca3617d642134ca8d842e7d2afc4c))

- generate template from manifest.toml - ([51276a0](https://github.com/es-ude/elastic-ai.creator/commit/51276a01de5ff37bedc598f5c758e3dc681aa49c))

- add translate_to_vhdl function - ([ba0edc2](https://github.com/es-ude/elastic-ai.creator/commit/ba0edc25b93075cbb2d104c2216dcc15df36c13c))

- generate first base template - ([a65d72e](https://github.com/es-ude/elastic-ai.creator/commit/a65d72ea1ad2dd87a0443b56711d11ce321d14b6))
#### Bug Fixes
- **(unit)** fix tests and remove hard sigmoid test in sequential test case - ([a1ada6f](https://github.com/es-ude/elastic-ai.creator/commit/a1ada6f0ceec750bb80abf866d28f96719f2f1f9))

- set correct resource options in rom and fix signal definitions - ([2c2964c](https://github.com/es-ude/elastic-ai.creator/commit/2c2964ceaa746163ebbeaef09181e09c06ecb4f2))
#### Miscellaneous Chores

- upgrade to python3.11 - ([f39c779](https://github.com/es-ude/elastic-ai.creator/commit/f39c7798f4ccc3799c707c8dcefbd176f9b6813b))

- move to python3.11 - ([389e4ec](https://github.com/es-ude/elastic-ai.creator/commit/389e4ec6d60dbf594026993bf8f7d94d4bea1da8))
#### Refactoring

- remove superfluous module protocols - ([4e25dc6](https://github.com/es-ude/elastic-ai.creator/commit/4e25dc65dfa0c226c298f5e589a6c887d72a3c19))

- - -

## [v0.34.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.33.3..v0.34.0) - 2025-05-08
#### Features

- make precomputed scalar functions use unified interface - ([6b59da5](https://github.com/es-ude/elastic-ai.creator/commit/6b59da53a896db7676119de2f74129bcc47287ed))

- binary_arithmetics - ([54e38d5](https://github.com/es-ude/elastic-ai.creator/commit/54e38d57f27db2d8d0baff5fee3c35a91e26ecd9))
#### Bug Fixes

- correct import paths - ([169f868](https://github.com/es-ude/elastic-ai.creator/commit/169f8686108845702f01482170df53e3fabbfe8b))
#### Miscellaneous Chores

- add mypy and pylint to pyproject.toml - ([aad5549](https://github.com/es-ude/elastic-ai.creator/commit/aad5549c7bbfbaf648fc3bbab0f77cd6c0ad49ca))

- remove unneeded import - ([e3df52a](https://github.com/es-ude/elastic-ai.creator/commit/e3df52a091e4673460f7b1ad733d766bad4afd02))

- - -

## [v0.33.3](https://github.com/es-ude/elastic-ai.creator/compare/v0.33.2..v0.33.3) - 2025-05-08
#### Bug Fixes

- remove DualPort2ClockRam design - ([f9224c6](https://github.com/es-ude/elastic-ai.creator/commit/f9224c6809b3a6f72bfe0405419de494b099b17c))

- set correct rom names - ([9570826](https://github.com/es-ude/elastic-ai.creator/commit/95708269900ca99b79da9ba37078f593724e5d17))
#### Documentation

- remove deprecated documentation - ([11b9945](https://github.com/es-ude/elastic-ai.creator/commit/11b9945bf3b6bf96899a09751963a93eb98d846d))
#### Refactoring

- move lstm designs in designs directory - ([36a807b](https://github.com/es-ude/elastic-ai.creator/commit/36a807b00794bac42a5018759e2ec09238bf043e))

- reorder class definitions to avoid the usage of quotes - ([780c1fe](https://github.com/es-ude/elastic-ai.creator/commit/780c1fe67d18893400226e8acc6e77504da6a6ad))

- move hardware specific lstm parts to nn package - ([bfe575c](https://github.com/es-ude/elastic-ai.creator/commit/bfe575c50291388eb2f8b243d3411ff9e847490c))

- rename translatable_modules to nn - ([333ac57](https://github.com/es-ude/elastic-ai.creator/commit/333ac5776788367ed3a8c17632fa20e11556f43e))

- rename nn to base_modules - ([44207a8](https://github.com/es-ude/elastic-ai.creator/commit/44207a8f72e426fcd1cb4acc5b3c53c4ac8fa2f2))

- - -

## [v0.33.2](https://github.com/es-ude/elastic-ai.creator/compare/v0.33.1..v0.33.2) - 2025-05-08
#### Bug Fixes
- **(translation)** add missing rom files and calculate correct twos complement - ([f700409](https://github.com/es-ude/elastic-ai.creator/commit/f70040956b7637844a471a5eff171d9cc6ba4c72))
- **(translation)** add missing ROMs and set correct names in fp_linear1d template - ([ad4c6f0](https://github.com/es-ude/elastic-ai.creator/commit/ad4c6f095102965ff1dffa83dab4f2cb9749ce49))
- **(unit)** fix failing unittests that are using the linear1d layer and design - ([ff582e1](https://github.com/es-ude/elastic-ai.creator/commit/ff582e185ea01cc6282cb4553e14701e88a9d8f8))

- small import fix - ([07d2e29](https://github.com/es-ude/elastic-ai.creator/commit/07d2e29c36e60d35066d2145782223aa42d64519))

- fix type annotation - ([8da1107](https://github.com/es-ude/elastic-ai.creator/commit/8da1107b2640d695816c71dd3980c0783b522122))
#### Miscellaneous Chores

- allow all torch versions >= 1.11 and < 2.0 - ([7321d7c](https://github.com/es-ude/elastic-ai.creator/commit/7321d7cf5694588a607975d13958edbfa5a3b331))
#### Refactoring

- small file and folder renames - ([9602a86](https://github.com/es-ude/elastic-ai.creator/commit/9602a868e6067889e2386c764e173c36f33e304c))

- - -

## [v0.33.1](https://github.com/es-ude/elastic-ai.creator/compare/v0.33.0..v0.33.1) - 2025-05-08
#### Bug Fixes

- usage of lstm output in lstm_network impl - ([2e16141](https://github.com/es-ude/elastic-ai.creator/commit/2e1614184cdaa073fdcc686b891748861fe5c7cc))

- wrong fixed point config object used for linear layers - ([3626113](https://github.com/es-ude/elastic-ai.creator/commit/36261136add4b4d378598dc8c9e858240f6557c5))

- - -

## [v0.33.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.32.1..v0.33.0) - 2025-05-08
#### Features

- add rom design for saving weights - ([75862b7](https://github.com/es-ude/elastic-ai.creator/commit/75862b7db4e64173daf7e6cdcb8413b0f510d396))
#### Bug Fixes

- correctly pad rom memory - ([fe768d5](https://github.com/es-ude/elastic-ai.creator/commit/fe768d5f93c34ade65c24479c70f3528c66b0408))
#### Refactoring

- move conversions to twos complement from designs to translatable modules - ([50ada18](https://github.com/es-ude/elastic-ai.creator/commit/50ada185de5a081295515e16773b7fefdaa107eb))

- use rom design in implementation - ([a8bfe4a](https://github.com/es-ude/elastic-ai.creator/commit/a8bfe4a2395a9bd81aa33f1989154f84a21bf001))

- rom design - ([975ad7e](https://github.com/es-ude/elastic-ai.creator/commit/975ad7e139a15466338cff72cfedeedf0c532f75))

- - -

## [v0.32.1](https://github.com/es-ude/elastic-ai.creator/compare/v0.32.0..v0.32.1) - 2025-05-08
#### Bug Fixes

- typo in test for lstm cell designs - ([2ffeaec](https://github.com/es-ude/elastic-ai.creator/commit/2ffeaecf3ba7c3c0946c57ab3bee92af55746887))

- set library for lstm_cell - ([2b3a565](https://github.com/es-ude/elastic-ai.creator/commit/2b3a565039672ca89a1c5f593db5a5f32742f771))

- - -

## [v0.32.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.31.0..v0.32.0) - 2025-05-08
#### Features
- **(translation)** add support for single buffered module to sequential - ([5402782](https://github.com/es-ude/elastic-ai.creator/commit/5402782c0c37a6838b77b19d8040d256217d72ba))
- **(translation)** sequential layer with bufferless layers - ([d7cea69](https://github.com/es-ude/elastic-ai.creator/commit/d7cea69ad0696f63e00762991e7407ad09d8a94c))

- add linear layer to lstm network - ([bccb50c](https://github.com/es-ude/elastic-ai.creator/commit/bccb50cd6e3bc4e3e3115a41e051a1b962f6be52))

- add linear layer to lstm network - ([48982f0](https://github.com/es-ude/elastic-ai.creator/commit/48982f0aca675098b77edb2c8419b09ebc388835))
#### Bug Fixes
- **(translation)** correct values for x/y_address_width - ([c7af1af](https://github.com/es-ude/elastic-ai.creator/commit/c7af1af71ef9319ed2ee7fffd7afcbaa5ffda580))

- tests and remove type annotations leading to deps - ([75ed6cc](https://github.com/es-ude/elastic-ai.creator/commit/75ed6cc4f3a92b80656433b8209c0c932595900e))
#### Miscellaneous Chores

- update gh-workflow - ([7418a7b](https://github.com/es-ude/elastic-ai.creator/commit/7418a7b46764c808649a78f7e132a8fe51880376))

- update gh-workflow - ([b1d714d](https://github.com/es-ude/elastic-ai.creator/commit/b1d714d4d408917ddd389db7fa29eed6c0230684))

- update gh workflow to match new tests location - ([58b7151](https://github.com/es-ude/elastic-ai.creator/commit/58b71513d05aa0bbf34533dc72b070ceaee34e83))
#### Refactoring
- **(nn)** replace fixed point factory by fixed point config - ([b5a08ac](https://github.com/es-ude/elastic-ai.creator/commit/b5a08acc11453ad550e2457836f1f4a2f5cbbae1))
- **(translation)** refactor autowiring for sequential network module - ([431862f](https://github.com/es-ude/elastic-ai.creator/commit/431862f21b6f074021973a88789a654461ae269e))
- **(translation)** move modules - ([24e522f](https://github.com/es-ude/elastic-ai.creator/commit/24e522fb10224bbd4065d841b2df97fa0f561021))

- lstm roms - ([a2e08ec](https://github.com/es-ude/elastic-ai.creator/commit/a2e08ec2f1492cd0efc9f4e60b76b4a42c0d093f))

- remove code generation dependency on fixed point data types - ([4d83d1b](https://github.com/es-ude/elastic-ai.creator/commit/4d83d1bc8f1a91de6dfd8995373155151d74fc25))

- tweak module hierarchy - ([40bc371](https://github.com/es-ude/elastic-ai.creator/commit/40bc371d6602c504ed6e69542ef3a51d525fda70))

- start moving relevant tests to top-level tests dir - ([577f43d](https://github.com/es-ude/elastic-ai.creator/commit/577f43d16a30fb1e6cc73c7dca7a4d6391559f79))

- - -

## [v0.31.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.30.4..v0.31.0) - 2025-05-08
#### Features
- **(translation)** add missing suffixes - ([cb05d0f](https://github.com/es-ude/elastic-ai.creator/commit/cb05d0f3f8665ac98c0cff70cbb2dbd8d2a5b2f2))

- add data flow node, sink node and source node - ([9a511de](https://github.com/es-ude/elastic-ai.creator/commit/9a511de4d2618c3131abcd3c481b918ffa96545e))

- introduce vhdl_design class - ([2431ba4](https://github.com/es-ude/elastic-ai.creator/commit/2431ba40b71c19dff161ea9b78d7b5277970a6f9))

- add logic and logic vector signals - ([551f241](https://github.com/es-ude/elastic-ai.creator/commit/551f24113be03b45ab1811cb734e521671620d89))

- add connectable base in/out signals - ([fea05ed](https://github.com/es-ude/elastic-ai.creator/commit/fea05ed0507550c23701bf6f5e3a562b68af73d4))

- introduce vhdl_design class - ([20566f6](https://github.com/es-ude/elastic-ai.creator/commit/20566f600383ccb68fed60483bede9db5436913f))

- add logic and logic vector signals - ([1947baa](https://github.com/es-ude/elastic-ai.creator/commit/1947baac032e1b3958344779a00b84615b5581a1))

- add connectable base in/out signals - ([7ad67f9](https://github.com/es-ude/elastic-ai.creator/commit/7ad67f916815b692daddae98d4c93b9a5eb21641))
#### Bug Fixes

- typing - ([b0bfa39](https://github.com/es-ude/elastic-ai.creator/commit/b0bfa39b98555b37f0d2626a235ac74987e2c9ad))

- fix unit tests after major rebase - ([3b596e9](https://github.com/es-ude/elastic-ai.creator/commit/3b596e9c20e302bbf42efda7577e01498c05bc6c))

- type annotations for tracing module - ([da598a9](https://github.com/es-ude/elastic-ai.creator/commit/da598a92fc8f76b3c19d0b960d77122b82d171ac))

- fix incorrect vector signal initialization - ([3b23f7a](https://github.com/es-ude/elastic-ai.creator/commit/3b23f7a64afda8cd2ee320f6af7dc372f9daf5e2))

- fix incorrect vector signal initialization - ([3c68255](https://github.com/es-ude/elastic-ai.creator/commit/3c68255057dad325ab4ba89601f6f1e2384f0d95))
#### Miscellaneous Chores

- clean up external deps - ([d1be65a](https://github.com/es-ude/elastic-ai.creator/commit/d1be65aee7144be24c79a280c93537115acd2e31))

- add class diagram for vhdldesign - ([01c63e0](https://github.com/es-ude/elastic-ai.creator/commit/01c63e02759ca71c93dc3f985d416d3ffa2c31af))

- update deps - ([00700fe](https://github.com/es-ude/elastic-ai.creator/commit/00700fe92b86442cc7e0db29794fa78d20ba48f9))

- tweak import contract - ([306de20](https://github.com/es-ude/elastic-ai.creator/commit/306de20163ad6e751b5e8d5e66601e90d1856b50))

- introduce private package import lint rule - ([b497e1c](https://github.com/es-ude/elastic-ai.creator/commit/b497e1ca3c512d2414cc0736305e19a867251741))
#### Refactoring

- remove superfluous protocol - ([741c53b](https://github.com/es-ude/elastic-ai.creator/commit/741c53baf3ca0ee9ccb27d5cf5a64d172eac7781))

- simplify ports - ([4bdf84a](https://github.com/es-ude/elastic-ai.creator/commit/4bdf84a4f72f1b99d89afa84de234c74a637fcd0))

- simplify signals - ([884ad64](https://github.com/es-ude/elastic-ai.creator/commit/884ad648fde4381a4dd892542bf576a7cd2d090b))

- adjust architecture in design.md and move modules accordingly - ([236e6c3](https://github.com/es-ude/elastic-ai.creator/commit/236e6c3457cbbb413b8fd79015bfe1e97c49563d))

- move more modules/classes to fix dependency issues - ([0e25d94](https://github.com/es-ude/elastic-ai.creator/commit/0e25d949c94a6efa6e0ffe6f0530f09e72c2f5b5))

- move more modules/classes to fix dependency issues - ([ae82c14](https://github.com/es-ude/elastic-ai.creator/commit/ae82c143100ddb9a49a7cfae36d8ea5289789fa4))

- move modules/classes to fix dependency issues - ([22564d7](https://github.com/es-ude/elastic-ai.creator/commit/22564d7ce4b05770d49078c0d5ce13fe3ace231d))

- remove deprecated vhdl.language module - ([e29f6da](https://github.com/es-ude/elastic-ai.creator/commit/e29f6da7e76018dce7d32f9698a7973de6e5e832))

- separate template from file - ([73f00e0](https://github.com/es-ude/elastic-ai.creator/commit/73f00e0e2e1e6302f2d8325fe9075d9bd51c25a3))

- only return file object from package without opening it - ([2c57287](https://github.com/es-ude/elastic-ai.creator/commit/2c572879a98a4af72978bbd471704395606b96fc))

- remove/move/merge protocols - ([8391a1c](https://github.com/es-ude/elastic-ai.creator/commit/8391a1c7e459bbf176840976a741317a28f3abd6))

- simplify data flow node - ([82c8ba8](https://github.com/es-ude/elastic-ai.creator/commit/82c8ba825bfa3b5d367bc3d6f473d2055ef217d6))

- use relative imports inside packages - ([ef8d588](https://github.com/es-ude/elastic-ai.creator/commit/ef8d58878058b2eb6ef5f177171350c6759132f7))

- simplify signals and move classes - ([aacb702](https://github.com/es-ude/elastic-ai.creator/commit/aacb7021bcb83cb96053092640a7b7cdc6e2077d))

- remove obsolete vhdl_design module - ([d4e61bd](https://github.com/es-ude/elastic-ai.creator/commit/d4e61bd7440d42a878f7539af7c256d637c2b7ba))

- remove obsolete graph package - ([ac53d76](https://github.com/es-ude/elastic-ai.creator/commit/ac53d7684135e3bab4d940d1c80951b297d19d77))

- rename test_logic_signals - ([9d16019](https://github.com/es-ude/elastic-ai.creator/commit/9d160195fd3c03ddaffb0b5609e9b1d5dcc56d02))

- move code test utility files - ([8efc21e](https://github.com/es-ude/elastic-ai.creator/commit/8efc21e0df17149a4f36f134d940f2bc98cf1c44))

- remove unintended print statement - ([5f2891e](https://github.com/es-ude/elastic-ai.creator/commit/5f2891e5c02d0448b357a6aa8b6433d2da25f4bf))

- move file reading to CodeTestCase - ([2c0ecb4](https://github.com/es-ude/elastic-ai.creator/commit/2c0ecb44ab3663e26e178bbb650b8c4f5298b195))

- merge utilities for testing code - ([6704e87](https://github.com/es-ude/elastic-ai.creator/commit/6704e87c1964615aa8b5d24042703a29b0b9ca1f))

- simplify architecture - ([1f5f1f1](https://github.com/es-ude/elastic-ai.creator/commit/1f5f1f19510f6dd9282e5bdda5beab904b2328b3))

- rename test_logic_signals - ([f817425](https://github.com/es-ude/elastic-ai.creator/commit/f817425f96895cdf52ff184f7cc32473e3c85fe9))

- move code test utility files - ([d390af1](https://github.com/es-ude/elastic-ai.creator/commit/d390af12f9658952fd08b4493b467ee820c45f5f))

- remove unintended print statement - ([b43befd](https://github.com/es-ude/elastic-ai.creator/commit/b43befdb529389a8cc8c08d087631ca45163f51c))

- move file reading to CodeTestCase - ([3cc9c5e](https://github.com/es-ude/elastic-ai.creator/commit/3cc9c5e4c67fea3e8bea566eeb1a30feea7c1b56))

- merge utilities for testing code - ([333c09a](https://github.com/es-ude/elastic-ai.creator/commit/333c09a9b396f450e24d7d2390daa8b502b5cdac))

- - -

## [v0.30.4](https://github.com/es-ude/elastic-ai.creator/compare/v0.30.3..v0.30.4) - 2025-05-08
#### Bug Fixes
- **(translation)** get rid of the duplicated suffix on rom component - ([9cd0e0b](https://github.com/es-ude/elastic-ai.creator/commit/9cd0e0be9481a286820eea5c8d5bdc9d28fcc0d8))

- - -

## [v0.30.3](https://github.com/es-ude/elastic-ai.creator/compare/v0.30.2..v0.30.3) - 2025-05-08
#### Bug Fixes
- **(template)** linear layer template - ([96bdf03](https://github.com/es-ude/elastic-ai.creator/commit/96bdf030ca4c27d67a4978e3b8609ef57c40a01e))
- **(unit)** add rounding to prevent tests from failing due to floating point loss - ([b7314b7](https://github.com/es-ude/elastic-ai.creator/commit/b7314b797ef39c2f693554821ec7bb3d96689661))

- - -

## [v0.30.2](https://github.com/es-ude/elastic-ai.creator/compare/v0.30.1..v0.30.2) - 2025-05-08
#### Bug Fixes

- use non-static path to example folder - ([613a152](https://github.com/es-ude/elastic-ai.creator/commit/613a152e65fbe0f7116a1f772fea8a3836d888af))

- ignore single import mypy error - ([dd85159](https://github.com/es-ude/elastic-ai.creator/commit/dd851590719ec76ab66dc9d908493991fc235e7e))
#### Miscellaneous Chores

- remove unused dependencies and update poetry lock - ([7b4b658](https://github.com/es-ude/elastic-ai.creator/commit/7b4b658c2649500809ade7efd716e8dca4153576))
#### Refactoring
- **(integration)** remove non-deterministic test - ([ebed2a7](https://github.com/es-ude/elastic-ai.creator/commit/ebed2a73beaba1f9e6abdc843eb5771cc1d34061))
- **(nn)** remove unused module - ([d2e643b](https://github.com/es-ude/elastic-ai.creator/commit/d2e643b1368a5776829a0353730afa5039c19590))
- **(unit)** move tensor_test_case in tests directory - ([3cf635b](https://github.com/es-ude/elastic-ai.creator/commit/3cf635b2d5ecbad524cfed75d4d4b7543c2dbcc2))
- **(unit)** move test in the unit folder - ([89df933](https://github.com/es-ude/elastic-ai.creator/commit/89df933b50eb35e0528042f81a37a59ba8630ff5))

- rename example - ([84d4792](https://github.com/es-ude/elastic-ai.creator/commit/84d479296c1930f4e7f334ae1d2fd89ba84b595a))

- delete not relevant example - ([3c0fce9](https://github.com/es-ude/elastic-ai.creator/commit/3c0fce95db8c078b8e37e34d0018872164402c4f))

- remove deprecated example - ([008241c](https://github.com/es-ude/elastic-ai.creator/commit/008241c8d5414cbe9478e1cdb226c22c48b2c663))

- create integration test from POS tagger example - ([cb73343](https://github.com/es-ude/elastic-ai.creator/commit/cb73343957c6b75df2a741b08c66c11545b86f2d))

- remove deprecated examples - ([eec3f0e](https://github.com/es-ude/elastic-ai.creator/commit/eec3f0e75a7875a8a2d1da9c2ffe586a4a18ebf9))

- - -

## [v0.30.1](https://github.com/es-ude/elastic-ai.creator/compare/v0.30.0..v0.30.1) - 2025-05-08
#### Bug Fixes
- **(unit)** make test more deterministic - ([97fd410](https://github.com/es-ude/elastic-ai.creator/commit/97fd4101af93cf17d446cb0cb38a419080d5bee6))

- - -

## [v0.30.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.29.0..v0.30.0) - 2025-05-08
#### Features
- **(integration)** convert example translate_linear_model to automated integration test - ([5d92d0b](https://github.com/es-ude/elastic-ai.creator/commit/5d92d0b15d8c0a1d76f842fd7a8bbc591bd1cf18))
- **(nn)** remove quantized_forward function and adopt tests - ([c865c73](https://github.com/es-ude/elastic-ai.creator/commit/c865c73a53e89c40ecebc9c4b49ba6d5c14256c1))
- **(nn)** implement concept of arithmetics - ([e7ad504](https://github.com/es-ude/elastic-ai.creator/commit/e7ad50471e2ac7300e0db781bd37cbba1364a5e6))
- **(nn)** remove input_quant and param_quant and add quantize function to arithmetics - ([ee91e42](https://github.com/es-ude/elastic-ai.creator/commit/ee91e42801b0d1163a0d52130fc578477da60c74))
- **(nn)** integrate arithmetics for the linear layer - ([a961558](https://github.com/es-ude/elastic-ai.creator/commit/a9615581159ba4b962fac8458d9b76de0a61d98f))
- **(nn)** rename quant_typings module to quantization and implement FakeQuant - ([0e5f24a](https://github.com/es-ude/elastic-ai.creator/commit/0e5f24aeb9f43258f9e971ffa777c585faff05f0))
- **(translation)** integrate hard tanh layer - ([eb74d3a](https://github.com/es-ude/elastic-ai.creator/commit/eb74d3a3671616db37ba8f554332ca1ddc33dffe))
- **(translation)** lstm uses fp hard sigmoid - ([fd265ac](https://github.com/es-ude/elastic-ai.creator/commit/fd265ac3e1ef7f11e28236705e4a38760462bddc))
- **(unit)** improve TensorTestCase class - ([d4273a6](https://github.com/es-ude/elastic-ai.creator/commit/d4273a60c169669ddba5f80636d1430b69c77d90))
- **(unit)** add unit tests for the fixed point quant/dequant autograd functions - ([f82431c](https://github.com/es-ude/elastic-ai.creator/commit/f82431c164b9536899d0cca9b391a057add8187a))
- **(unit)** add unit tests for the LSTMBase layer - ([589f803](https://github.com/es-ude/elastic-ai.creator/commit/589f803fd858b22985485d795f4441a9abf97742))

- small example for translating combination of lstm and linear layer - ([12e7101](https://github.com/es-ude/elastic-ai.creator/commit/12e7101e8c62e8424bc2ed580cfbe645e8d33510))

- add example to demonstrate that the new kinds of layers are trainable - ([231e325](https://github.com/es-ude/elastic-ai.creator/commit/231e325815c469596c63259c5f345dc9afb0f3b7))

- convert example parametrize_convolution to automated integration test - ([3dde1c2](https://github.com/es-ude/elastic-ai.creator/commit/3dde1c250fa4ebb617bbd543c9b26cb320d430f7))
#### Bug Fixes
- **(nn)** fix LSTMCell raises Error for unbatched input data and add a test for this case - ([5ce3e21](https://github.com/es-ude/elastic-ai.creator/commit/5ce3e2125b4bcd1115d77ebe5c833e52d58bad77))
- **(nn)** fix imports and use new FixedPointFactory features - ([e8c74c3](https://github.com/es-ude/elastic-ai.creator/commit/e8c74c34ec1c5a4b5189d74f2a19a993a5ae9779))
- **(translation)** fix errors in the lstm template and remove lstm_common component - ([c4a28ce](https://github.com/es-ude/elastic-ai.creator/commit/c4a28ce2f40dc84e7a5e4470c62a40911b73901f))
- **(translation)** add layer_name to all vhdl templates and components - ([2d9c47d](https://github.com/es-ude/elastic-ai.creator/commit/2d9c47dc60642d94efeb58cc3014f6a7790a6f26))
- **(translation)** change not existing layer_id field to layer_name - ([f7425c5](https://github.com/es-ude/elastic-ai.creator/commit/f7425c515395243962db1517116b9961b1668cd7))
- **(translation)** use model.children() instead of model.modules() to avoid recursion - ([a3c349b](https://github.com/es-ude/elastic-ai.creator/commit/a3c349b13af0fef383b494850973d8ff9ac2dd68))
- **(translation)** remove sigmoid_resolution - ([dd4f033](https://github.com/es-ude/elastic-ai.creator/commit/dd4f03366920f1a3774772a16a49efaa8756d249))
- **(translation)** rename to .tpl.vhd - ([fe3c85c](https://github.com/es-ude/elastic-ai.creator/commit/fe3c85cd77d0f2fefb90f2d3ff6eadde8570d000))
- **(translation)** infer fixed_point_factory of linear and lstm in build functions - ([81df686](https://github.com/es-ude/elastic-ai.creator/commit/81df686fe13db5f85c91b65c73713b7da8e6c64f))
- **(translation)** change torch LSTM layer to our FixedPointLSTM layer - ([5e7a39a](https://github.com/es-ude/elastic-ai.creator/commit/5e7a39a78684c09a1d374476f8fb611019ae994f))
- **(translation)** add similar concept of translation arguments to fix the translation process - ([e387ae2](https://github.com/es-ude/elastic-ai.creator/commit/e387ae26918fbe8e4a0ee01ccc4361849746bd66))
- **(unit)** remove unused OperationType type and FakeQuant class - ([596dbd8](https://github.com/es-ude/elastic-ai.creator/commit/596dbd8cdf3cde67eedea2779a35ff682c9ac9f7))
- **(unit)** fix unit and integration tests to use the new layers correctly - ([0553017](https://github.com/es-ude/elastic-ai.creator/commit/05530178cf7fb64dc88cab82b89c24b2a1406e8d))

- fix some mypy errors and remove unused imports - ([08e2362](https://github.com/es-ude/elastic-ai.creator/commit/08e2362fa32efd13e388140ad58c93b0e79229b3))

- adapt basic qtorch example to recent changes of the creator - ([a17d900](https://github.com/es-ude/elastic-ai.creator/commit/a17d9006240a67da97b8a539620aa1974e07e942))
#### Miscellaneous Chores

- remove vhdl scope - ([5c9571b](https://github.com/es-ude/elastic-ai.creator/commit/5c9571b384588551c7439f3e45ad63d8f718b79f))

- relax commitlint rules - ([108e361](https://github.com/es-ude/elastic-ai.creator/commit/108e361f763f23843b72c5620cbebd0c171a9433))
#### Documentation

- add commit types and scopes - ([e759fd3](https://github.com/es-ude/elastic-ai.creator/commit/e759fd38fb41d413ccf03617f84f87f6df9aeb12))
#### Refactoring
- **(integration)** move integration test to more specific location - ([0115399](https://github.com/es-ude/elastic-ai.creator/commit/01153996ac556eb9a96f404e8efed2af5bbdf1dd))
- **(nn)** add more precise type annotation - ([0c47fe0](https://github.com/es-ude/elastic-ai.creator/commit/0c47fe0b485cb71662ef017b7c454b848baa0b4f))
- **(nn)** remove default bias value from linear layer - ([8d55471](https://github.com/es-ude/elastic-ai.creator/commit/8d5547180a50f07ee259f37cd8cd89ffe496e421))
- **(translation)** add fixed_point_factory property to fp layers and remove FixedPointLSTMCell - ([9f0a5d3](https://github.com/es-ude/elastic-ai.creator/commit/9f0a5d3505dc05d53aaf9fa9fb1c607049c661fd))
- **(translation)** remove outdated evaluators - ([8c0009a](https://github.com/es-ude/elastic-ai.creator/commit/8c0009ae54dfed9f24223ca01a6b146ee0c06f04))
- **(translation)** remove unnecessary print statement - ([2f8a0a7](https://github.com/es-ude/elastic-ai.creator/commit/2f8a0a75b602d6d7621f310e33ccf0bf0d5c1e28))
- **(unit)** move unit test to correct location - ([c03c362](https://github.com/es-ude/elastic-ai.creator/commit/c03c3621c6cdef58a44e1c3e279d025ebdf34aa6))

- remove examples belonging to the removed precomputation package - ([4dc681b](https://github.com/es-ude/elastic-ai.creator/commit/4dc681b18207dd92d767c97df2c70e2fd3e6cd2e))
#### Style

- beautify 6209df2bbc3c693f1829ce8b93822fc84152f69b - ([423b081](https://github.com/es-ude/elastic-ai.creator/commit/423b081476868df0a7f90fbcaeec16203670551f))

- - -

## [v0.29.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.28.0..v0.29.0) - 2025-05-08
#### Features

- set pypi project api token - ([37ba8c9](https://github.com/es-ude/elastic-ai.creator/commit/37ba8c9794acc6b4bdf64087c98c61172446fcb6))
#### Miscellaneous Chores

- tighten commitlint rules - ([47a35da](https://github.com/es-ude/elastic-ai.creator/commit/47a35da220ba1c6081af11b0a6e7945978f2fe77))

- - -

## [v0.28.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.27.0..v0.28.0) - 2025-05-08
#### Miscellaneous Chores

- temporary relax commitlint rules - ([7b007dc](https://github.com/es-ude/elastic-ai.creator/commit/7b007dc81ac7f1420b5d08e3a77f51d087a17dcf))

- temporary relax commitlint rules - ([437c3d7](https://github.com/es-ude/elastic-ai.creator/commit/437c3d7cec0487f5754ec357fb4d313343fd2cbc))

- enable semantic release for main again - ([6c93920](https://github.com/es-ude/elastic-ai.creator/commit/6c939203995883b390a20bc98b098a252563c669))
#### Refactoring
- **(nn)** remove unused import - ([14d1d60](https://github.com/es-ude/elastic-ai.creator/commit/14d1d60bb7b56c2c6bdd00feb767a8248a09699c))
- **(unit)** rename _init_quantizable_convolution function - ([2b57dbc](https://github.com/es-ude/elastic-ai.creator/commit/2b57dbcaa02f202c7654d8d15b53c84a0210ee1f))
- **(unit)** rename package from qat to nn - ([e211ae6](https://github.com/es-ude/elastic-ai.creator/commit/e211ae63d9ee7fdc2c0fad15a40730399fac7654))

- - -

## [v0.27.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.26.1..v0.27.0) - 2025-05-08
#### Features
- **(examples)** update qlstm sine wave example to the correctly implemented QLSTM layer - ([dc62cd2](https://github.com/es-ude/elastic-ai.creator/commit/dc62cd2aa05067b164009301ab7c5e110797c503))
- **(qat)** remove constraints - ([6b7b483](https://github.com/es-ude/elastic-ai.creator/commit/6b7b4835dc9f9f6b6fc83bc619727aa948c19161))
- **(qat)** add constraint type - ([dc4c4e5](https://github.com/es-ude/elastic-ai.creator/commit/dc4c4e57a9615a9be6941ecc750d3838458ff919))
- **(vhdl)** distinguish x/y width - ([2f52100](https://github.com/es-ude/elastic-ai.creator/commit/2f52100d32502520ce66a240bae90dd48e070ebd))
- **(vhdl)** introduce HWBlockCollection - ([a80bda2](https://github.com/es-ude/elastic-ai.creator/commit/a80bda2d705992030b18649ff99f3a6ce75d7ef3))
- **(vhdl)** introduce HWEquivalentGraph - ([844bb84](https://github.com/es-ude/elastic-ai.creator/commit/844bb84a2d36e50f3de7ae4b713d370011d3240e))
- **(vhdl)** add module_nodes to graph decorator - ([6d0a612](https://github.com/es-ude/elastic-ai.creator/commit/6d0a61217b36b9db8e9df19210e5f0d3aeed4ef2))
- **(vhdl)** implement HWBlocks interface for sigmoid,linear - ([0177373](https://github.com/es-ude/elastic-ai.creator/commit/0177373eeddfa9c32100777bbcd7a94765dc1122))
- **(vhdl)** extend code file with parameters - ([4833f8b](https://github.com/es-ude/elastic-ai.creator/commit/4833f8b2d5553cf02d322b8485587612cd67a9e8))
- **(vhdl)** introduce HWBlocks - ([ab03eaf](https://github.com/es-ude/elastic-ai.creator/commit/ab03eaf28c74483fcd9dbd78d247d39e248bdea1))
- **(vhdl)** generate layer instantiations - ([7a75fc3](https://github.com/es-ude/elastic-ai.creator/commit/7a75fc31780a6173424ffdcf3129bc60d5a83e59))
- **(vhdl)** generate vhdl signal definitions - ([c593d3d](https://github.com/es-ude/elastic-ai.creator/commit/c593d3d501082595d4918be3c3425b6d9c636332))
- **(vhdl)** generate vhdl signal definitions - ([53408f6](https://github.com/es-ude/elastic-ai.creator/commit/53408f6cb9daa5c44931e880fda0712c2924b822))
- **(vhdl)** tracer records reference to module for call_module nodes - ([20ed7da](https://github.com/es-ude/elastic-ai.creator/commit/20ed7dab9677e476925a8b1250cbbc2004d43246))
- **(vhdl)** add hw equivalent module tracer - ([3f2c2c7](https://github.com/es-ude/elastic-ai.creator/commit/3f2c2c7acc5046131d420d513a4bb3d3981ac0c5))
- **(vhdl)** generate portmap output_address - ([c6a26a6](https://github.com/es-ude/elastic-ai.creator/commit/c6a26a61d98c90fa29b02e6619116e67a4a67ac5))
- **(vhdl)** support generation of layer connections - ([1d43c42](https://github.com/es-ude/elastic-ai.creator/commit/1d43c4212ef54c5488df7e7dc3829df31a7e8484))
- **(vhdl)** distinguish x/y width - ([73549f9](https://github.com/es-ude/elastic-ai.creator/commit/73549f94a0c582170e2f43baea4afcb4c9c20124))
- **(vhdl)** introduce HWBlockCollection - ([cdcb324](https://github.com/es-ude/elastic-ai.creator/commit/cdcb324abe3c69893a782df075b24d734f244a6c))
- **(vhdl)** introduce HWEquivalentGraph - ([f0bdd73](https://github.com/es-ude/elastic-ai.creator/commit/f0bdd73d6e6e6ed9c8306a7771443e4d13e874ce))
- **(vhdl)** add module_nodes to graph decorator - ([bee0438](https://github.com/es-ude/elastic-ai.creator/commit/bee0438fb9b35d666998f4f516a1469c729b5829))
- **(vhdl)** implement HWBlocks interface for sigmoid,linear - ([53e05c7](https://github.com/es-ude/elastic-ai.creator/commit/53e05c7b772f8576b4f221e610360dc52601d852))
- **(vhdl)** extend code file with parameters - ([2bdfca3](https://github.com/es-ude/elastic-ai.creator/commit/2bdfca352b05756bb911eafb2b702f6536561b26))
- **(vhdl)** introduce HWBlocks - ([141148f](https://github.com/es-ude/elastic-ai.creator/commit/141148f13c40725755a1b02b24d8899e01ae9ced))
- **(vhdl)** generate layer instantiations - ([925b837](https://github.com/es-ude/elastic-ai.creator/commit/925b837d33120d4bd1abdd8cae812d89d4979a9a))
- **(vhdl)** generate vhdl signal definitions - ([c76d03d](https://github.com/es-ude/elastic-ai.creator/commit/c76d03db443cffd831abee60a8546aa3547c5fe6))
- **(vhdl)** generate vhdl signal definitions - ([5da3986](https://github.com/es-ude/elastic-ai.creator/commit/5da3986472a65e7f15cbedd3cba473ad4d67dde9))
- **(vhdl)** tracer records reference to module for call_module nodes - ([ea1f0ee](https://github.com/es-ude/elastic-ai.creator/commit/ea1f0ee893c11065bdf17086badd248b998d29de))
- **(vhdl)** add hw equivalent module tracer - ([fcb2e10](https://github.com/es-ude/elastic-ai.creator/commit/fcb2e102f5409a2e1dc358ce26e4cba6110a7e24))
- **(vhdl)** generate portmap output_address - ([33e66d9](https://github.com/es-ude/elastic-ai.creator/commit/33e66d99b5b8c0801c93e520463ea92c6392e2b8))
- **(vhdl)** support generation of layer connections - ([fdd3176](https://github.com/es-ude/elastic-ai.creator/commit/fdd3176ba5d4652718e76dfd74dc92167f86b4f4))
#### Bug Fixes
- **(onnx)** remove unmaintained onnx support - ([dc773d3](https://github.com/es-ude/elastic-ai.creator/commit/dc773d39fe2c0ea5785e3fb0bf7a43f3bf83495f))
- **(onnx)** remove unmaintained onnx support - ([c200394](https://github.com/es-ude/elastic-ai.creator/commit/c200394239ff58ee31e5273d5999d731fbe5daca))
- **(qat)** fix error when passing flat input data to _QLSTMBase and batch_first set to True - ([29918d1](https://github.com/es-ude/elastic-ai.creator/commit/29918d11c508e3e91fe00a0e07988be0ed198b35))
- **(qat)** fix the problem of wrong shapes for the QLSTM layer - ([b75f478](https://github.com/es-ude/elastic-ai.creator/commit/b75f47804016a3dfdad3f8d2dd575f4252cac5ff))
- **(qat)** fix circular dependency - ([1d5615b](https://github.com/es-ude/elastic-ai.creator/commit/1d5615bf81757bf16904eb75c33fead69a68dd43))
- **(vhdl)** remove obsolete vhdl formatter - ([83d81e3](https://github.com/es-ude/elastic-ai.creator/commit/83d81e348152e047482ccc45a2ccaf6173f772d9))
- **(vhdl)** remove obsolete vhdl formatter - ([128ba6b](https://github.com/es-ude/elastic-ai.creator/commit/128ba6bdbecd8763f77cc6862373446f5418201e))
#### Miscellaneous Chores
- **(gh-workflow)** remove superfluous line - ([71edbc4](https://github.com/es-ude/elastic-ai.creator/commit/71edbc4369a59c90c561ff3e8b335bd85ecbba7e))
- **(gh-workflow)** set correct path to unit and integration tests - ([538eb2f](https://github.com/es-ude/elastic-ai.creator/commit/538eb2f036f24ea99135f0e66ad59c3738e60231))

- add missing argument to poetry configuration - ([1567b0c](https://github.com/es-ude/elastic-ai.creator/commit/1567b0c7269f14a1454b206b959c2c33862fe239))

- reorder poetry calls and cache setup for action - ([2a0fb0d](https://github.com/es-ude/elastic-ai.creator/commit/2a0fb0d65d5bf80746b65d1d5f29f63cc59f36f1))

- create cache-dir in action - ([f0ecc17](https://github.com/es-ude/elastic-ai.creator/commit/f0ecc17eedd1e9acdc6c0d4baa713eee6a5e2495))

- specify shell in gh-action - ([a5fb59e](https://github.com/es-ude/elastic-ai.creator/commit/a5fb59e35e8b557011559ba5d55b68a452574710))

- fetch repo in job instead of action - ([05d8bd1](https://github.com/es-ude/elastic-ai.creator/commit/05d8bd14a7c287c90755ffb68f2c899d3d182ad2))

- add commit hash to action reference - ([459a4cc](https://github.com/es-ude/elastic-ai.creator/commit/459a4ccd2487762c67a1be86f2ae071dc89396e8))

- rename actions yml - ([7882524](https://github.com/es-ude/elastic-ai.creator/commit/78825240f3cd78863110f516574d915781f3a4c5))

- add github action for test environment setup - ([8a38722](https://github.com/es-ude/elastic-ai.creator/commit/8a3872210155601a900b8ac59757808974961999))

- add noop to semantic-release and trigger on workflow call - ([ecdb463](https://github.com/es-ude/elastic-ai.creator/commit/ecdb463514c0e5b8b0d0d22818071c728e6997e2))

- use gh-action provided by python-semantic-release - ([0d0321e](https://github.com/es-ude/elastic-ai.creator/commit/0d0321e44455d40c3b04929df13cccfe7056c35c))

- add style again to pyproject and commitlint - ([d7aaf28](https://github.com/es-ude/elastic-ai.creator/commit/d7aaf28042881c272f851e5402135d15a149ec42))

- tweak pyproject and commitlint - ([addc521](https://github.com/es-ude/elastic-ai.creator/commit/addc521744804fb8a6deeadde8510bd9fe37d87b))

- don't install extras prior publishing - ([effa8c0](https://github.com/es-ude/elastic-ai.creator/commit/effa8c004a2d8356a96e3869763e85e58ee92924))

- update poetry.lock - ([0f78c4b](https://github.com/es-ude/elastic-ai.creator/commit/0f78c4bfdddad038bd69b1a92f3b1fba4c5ab9f8))

- more specific commitlint rules - ([bbb88e9](https://github.com/es-ude/elastic-ai.creator/commit/bbb88e9080ecd873209f99aa01473b9d57bd2012))
#### Documentation
- **(readme)** move tests and remove deprecated lines - ([4a074a8](https://github.com/es-ude/elastic-ai.creator/commit/4a074a87fb31df535d415c2ab6aede7e4d7d8949))
#### Refactoring
- **(qat)** remove noise comments - ([ccc3979](https://github.com/es-ude/elastic-ai.creator/commit/ccc397911d899bdc49b917d0438336e17e37d100))
- **(qat)** create a _QLSTMCellBase and _QLSTMBase class to avoid default parameters - ([98ba0b7](https://github.com/es-ude/elastic-ai.creator/commit/98ba0b78090c120070e38bf9b0502b3027e0fa33))
- **(qat)** remove default quantizer from QLSTM and QLSTMCell layer - ([cce2f8f](https://github.com/es-ude/elastic-ai.creator/commit/cce2f8f1d22c384f136583f74d3a2b396500b0e0))
- **(qat)** remove noise comments and remove default quantizer from QLSTM and QLSTMCell layer - ([4a57ca9](https://github.com/es-ude/elastic-ai.creator/commit/4a57ca900a6c5dad1710f6d558c1ade17527d2b4))
- **(qat)** remove unused code - ([d49ca79](https://github.com/es-ude/elastic-ai.creator/commit/d49ca79cc4ec5834222a6253d96ff5402f905151))
- **(qat)** remove unused code and make Identity quantizer public - ([dcd726e](https://github.com/es-ude/elastic-ai.creator/commit/dcd726e183c5b74b05c27155ec64cc08f395802e))
- **(qat)** remove unused code - ([43e5992](https://github.com/es-ude/elastic-ai.creator/commit/43e5992f1e48d078113bba7863c0ac5e3e967ada))
- **(qat)** remove unused import - ([b6cf349](https://github.com/es-ude/elastic-ai.creator/commit/b6cf3494b36cca9d2fd732a24952423b68ad6c46))
- **(qat)** split LayersTest class into classes for each layer - ([55c12b3](https://github.com/es-ude/elastic-ai.creator/commit/55c12b36ce0b9807ffa4f5dd8344e3b8143f1212))
- **(qat)** move BatchNormedActivatedConv1d from layers module to blocks module - ([6269522](https://github.com/es-ude/elastic-ai.creator/commit/6269522bcdb978a905c84693e6c9fa4bdc32bfa7))
- **(templates)** use streams for reading templates - ([2af5d2a](https://github.com/es-ude/elastic-ai.creator/commit/2af5d2a41d72406a4abcbb2c55f3c0c01150cad4))
- **(templates)** reading text from resources returns list[str] - ([361c5a5](https://github.com/es-ude/elastic-ai.creator/commit/361c5a571432f24b8b2be0327fdc1b4edea1c6fe))
- **(templates)** use streams for reading templates - ([13422aa](https://github.com/es-ude/elastic-ai.creator/commit/13422aaae48eeb22fae01001ee31e3a71d85c337))
- **(templates)** reading text from resources returns list[str] - ([d10178f](https://github.com/es-ude/elastic-ai.creator/commit/d10178f89f4d5e1b24b8860846bd17b72af93ec0))
- **(tests)** move last tests out of old test folder structure - ([a3e12c1](https://github.com/es-ude/elastic-ai.creator/commit/a3e12c11df45b4de8babb2f1862ea92a1778c92a))
- **(tests)** move all unit and integration tests in a tests folder - ([8afb751](https://github.com/es-ude/elastic-ai.creator/commit/8afb751a5dc9f7fd4e2fa4a1dd1167682efe590f))
- **(typing)** add missing type annotations - ([c83a746](https://github.com/es-ude/elastic-ai.creator/commit/c83a7466cecfc043dd95800c69b6ee5df8b5bd4f))
- **(vhdl)** sort imports - ([2c114f1](https://github.com/es-ude/elastic-ai.creator/commit/2c114f1b8c839ef939051f5a1c5b6d40585908cb))
- **(vhdl)** move classes out of hw_equivalent_layers.__init__ - ([0713211](https://github.com/es-ude/elastic-ai.creator/commit/071321164dee3e9a9db3d113848c6ce3dd960b1c))
- **(vhdl)** rename BaseHWBlockInterface to BaseHWBlock - ([4c69682](https://github.com/es-ude/elastic-ai.creator/commit/4c696826e559603ac54681eae7ff50e34d22a1ac))
- **(vhdl)** move files - ([01999c4](https://github.com/es-ude/elastic-ai.creator/commit/01999c4049f21e357390c2fc88a09bfc987d0cb6))
- **(vhdl)** use VHDLFile for root module generation - ([e2b8423](https://github.com/es-ude/elastic-ai.creator/commit/e2b842310f237ff8802bd92ac4f7f537d9ede707))
- **(vhdl)** refactor FPHardSigmoidFile - ([3c54418](https://github.com/es-ude/elastic-ai.creator/commit/3c544184877d3416445f064d33bee3e95d78ac31))
- **(vhdl)** remove CodeModule/CodeComponent - ([89e27dd](https://github.com/es-ude/elastic-ai.creator/commit/89e27dd4e1bd77502d69054c15a3277f0a4b0826))
- **(vhdl)** move files - ([1006de2](https://github.com/es-ude/elastic-ai.creator/commit/1006de2c4b126e467b91b05236cebdcd40be48df))
- **(vhdl)** use VHDLFile for root module generation - ([e386141](https://github.com/es-ude/elastic-ai.creator/commit/e386141d3a318782f5b7793a1672c388d74b5563))
- **(vhdl)** refactor FPHardSigmoidFile - ([461ff71](https://github.com/es-ude/elastic-ai.creator/commit/461ff71ad50f22c92dcf430a7b98b708a32a86a8))
- **(vhdl)** fix some mypy errors - ([cd9899f](https://github.com/es-ude/elastic-ai.creator/commit/cd9899f55476cd91993f7276cdda02fc7e3d7b26))
- **(vhdl)** rename call function to code in the precomputed scalar functions and test benches - ([a40553e](https://github.com/es-ude/elastic-ai.creator/commit/a40553e05f64d9bd57473fed4d40b269858ef65f))
- **(vhdl)** rename call function to code in the language module - ([4cf795e](https://github.com/es-ude/elastic-ai.creator/commit/4cf795ee1cc679ea6b4b7cf51198cb536a5d9af5))
- **(vhdl)** add missing type annotations and remove unused parts - ([6fb622b](https://github.com/es-ude/elastic-ai.creator/commit/6fb622b3cff4caf8d849c8df8696275fa38fa9bb))
- **(vhdl)** using code function to generate code instead of call - ([843ad64](https://github.com/es-ude/elastic-ai.creator/commit/843ad64d33e2018da9c88fd487ccd46fd598c58f))
- **(vhdl)** remove CodeModule/CodeComponent - ([f014414](https://github.com/es-ude/elastic-ai.creator/commit/f014414d9e513303b8f128b4cf87550a04a863d7))

- move files, simplify, correct types, merge - ([57d3754](https://github.com/es-ude/elastic-ai.creator/commit/57d37541fed53c29ad9e6a665aa72b99fe5a2df0))

- use new vhdl file class for hard_sigmoid - ([a9e1f6c](https://github.com/es-ude/elastic-ai.creator/commit/a9e1f6ccba2f978b8290be94c754872432d3c311))

- use new vhdl file class for hard_sigmoid - ([36cae74](https://github.com/es-ude/elastic-ai.creator/commit/36cae74c068219e96bc3d3eeabbfc87c7ed2e9e8))
#### Style
- **(vhdl)** introduce template code file interface - ([69fb2b6](https://github.com/es-ude/elastic-ai.creator/commit/69fb2b681497289d230f8d203f5f430c91a3ff54))
- **(vhdl)** introduce template code file interface - ([4b233c3](https://github.com/es-ude/elastic-ai.creator/commit/4b233c37c41f231e94a9b6cb800146eb5d0ecb62))

- beautify 5fcfc23c342983a98efc1d527648ef17644c472c - ([a228cc0](https://github.com/es-ude/elastic-ai.creator/commit/a228cc00bf0a44327a858835e9a73531af56e59e))

- remove deprecated code and move/rename - ([f6b8020](https://github.com/es-ude/elastic-ai.creator/commit/f6b8020f8a5dfc5d9226578efa5f2512b84223e5))

- remove deprecated code and move/rename - ([ea9125a](https://github.com/es-ude/elastic-ai.creator/commit/ea9125a649611f260e2c7600fd8619d74f3c5fba))

- - -

## [v0.26.1](https://github.com/es-ude/elastic-ai.creator/compare/v0.26.0..v0.26.1) - 2025-05-08
#### Features
- **(vhdl)** start implementing lstm base module - ([b154ca5](https://github.com/es-ude/elastic-ai.creator/commit/b154ca5525c00f735150c21f64324da87328ba5e))
- **(vhdl)** implement quantized forward function of the fixed point lstm cell - ([7818e15](https://github.com/es-ude/elastic-ai.creator/commit/7818e15bc6c41454090b77fe5df7a8e7930ab570))
- **(vhdl)** implement and test lstm cell base class and start implementing fp lstm cell - ([f458fb6](https://github.com/es-ude/elastic-ai.creator/commit/f458fb6c216385a119774a3f98788941e13ed5c9))
#### Bug Fixes
- **(vhdl)** remove layer_name parameter - ([1bb40cd](https://github.com/es-ude/elastic-ai.creator/commit/1bb40cd0e44f7f207f60ffbb33e8c59f00b64e82))
- **(vhdl)** remove layer_name parameter - ([7a83b1e](https://github.com/es-ude/elastic-ai.creator/commit/7a83b1eed3095a8b7f90438c78ba24bba6e44958))
#### Refactoring
- **(examples)** remove examples that are not relevant anymore - ([3b241e2](https://github.com/es-ude/elastic-ai.creator/commit/3b241e2ddfe14a248e411f0b8da9ec6cf85cc8bc))
- **(vhdl)** remove vhdl formatter that is not used anymore - ([007a8c4](https://github.com/es-ude/elastic-ai.creator/commit/007a8c4ec4c42382390b0af034a2f5f3226fea86))
- **(vhdl)** rename lstm module to lstm_cell - ([97bf791](https://github.com/es-ude/elastic-ai.creator/commit/97bf791a8a3fbab68179f5a9a20e9410c3bcccf7))
- **(vhdl)** move OperationType type in the typing module - ([bf0d3fe](https://github.com/es-ude/elastic-ai.creator/commit/bf0d3feacdfdf461527056b8a24aec63907a2578))

- - -

## [v0.26.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.25.0..v0.26.0) - 2025-05-08
#### Features
- **(vhdl)** start implementing lstm base layer - ([39ce891](https://github.com/es-ude/elastic-ai.creator/commit/39ce891d56be59d5a20a36889b0e9c2f13e00bd1))
- **(vhdl)** clean the code - ([d737d02](https://github.com/es-ude/elastic-ai.creator/commit/d737d02122207bcd24f4b7c960b71db095d34a26))
- **(vhdl)** make linear layers better timing - ([1c6a3ae](https://github.com/es-ude/elastic-ai.creator/commit/1c6a3aeeeaee929affbb092eb485c1cf7a323355))
- **(vhdl)** merge from main - ([fefd3ba](https://github.com/es-ude/elastic-ai.creator/commit/fefd3ba4ab1fa8ae9d09bfc6185f906175f7a6ff))
#### Bug Fixes
- **(vhdl)** fix error during integrating to a MLP model - ([0e2b89c](https://github.com/es-ude/elastic-ai.creator/commit/0e2b89c898497f35a2ad840bd3065429799bdf61))
- **(vhdl)** fix small error in the template file - ([fe94518](https://github.com/es-ude/elastic-ai.creator/commit/fe94518ff2e5e44f7c1ff8f9bf8b4ff8f0b5cf41))
- **(vhdl)** remove the layer name in the example file - ([767b5f9](https://github.com/es-ude/elastic-ai.creator/commit/767b5f9c62d493d35e5a294b1363c861d5438fa5))
#### Miscellaneous Chores
- **(gh-workflow)** remove minor python versions from gh workflow - ([26b8035](https://github.com/es-ude/elastic-ai.creator/commit/26b803589da4d8f60cc7c52dc2a27b97cca88ab9))
- **(gh-workflow)** remove minor python versions from gh workflow - ([fc517a6](https://github.com/es-ude/elastic-ai.creator/commit/fc517a6bb81fb037f2b9d3466d32506aa0573020))

- - -

## [v0.25.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.24.0..v0.25.0) - 2025-05-08
#### Bug Fixes
- **(vhdl)** fix the error from merging braches - ([c386766](https://github.com/es-ude/elastic-ai.creator/commit/c386766ea654852c5ad5254cefc1fab28f544c66))

- - -

## [v0.24.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.23.0..v0.24.0) - 2025-05-08
#### Features
- **(vhdl)** implement FixedPointHardTanh layer - ([ed72810](https://github.com/es-ude/elastic-ai.creator/commit/ed728101fb596a08e1a76d936d04306a066c50b5))
- **(vhdl)** apply the expand_template function to the already existing templates - ([c958f54](https://github.com/es-ude/elastic-ai.creator/commit/c958f545f4c2cf2414a007753b416ec73c410458))
- **(vhdl)** add expand_template function that fills string templates instead of format strings - ([eb9ee98](https://github.com/es-ude/elastic-ai.creator/commit/eb9ee987f73ffb26e8280ec3c32b32e38896d3c1))
- **(vhdl)** add layer_id parameter to build function and set it to a unique value during translation - ([cfdf949](https://github.com/es-ude/elastic-ai.creator/commit/cfdf9492190e24230293e3b0b1b312bfc9710952))
#### Bug Fixes
- **(vhdl)** fix wrong return type - ([eb53ed9](https://github.com/es-ude/elastic-ai.creator/commit/eb53ed972ec9078f6c405ecd7c92043eaf8ed419))
- **(vhdl)** remove duplicated key - ([5a4bcd6](https://github.com/es-ude/elastic-ai.creator/commit/5a4bcd6fb6de9cff6c639866db1dd50918f3039b))
#### Miscellaneous Chores
- **(gh-workflow)** set correct parameters for the commit action - ([0193074](https://github.com/es-ude/elastic-ai.creator/commit/019307473d639c14f3949b5e7d42be4cc14f655f))
- **(gh-workflow)** set correct parameters for the commit action - ([373ffd2](https://github.com/es-ude/elastic-ai.creator/commit/373ffd20d331152cedde6083d72fdb40d83c741d))
- **(gh-workflow)** set correct commit action - ([a7a2439](https://github.com/es-ude/elastic-ai.creator/commit/a7a2439d8b8a1358983d18c22de6e57820757d82))
- **(gh-workflow)** update action versions and remove usage of set-output - ([d106116](https://github.com/es-ude/elastic-ai.creator/commit/d1061167f9da09f7aa2a191340652ae56e3335e0))
#### Refactoring
- **(vhdl)** move common helper functions to a separate utils.py module - ([b459f0a](https://github.com/es-ude/elastic-ai.creator/commit/b459f0af4b8b9884c03277928d7a0437b68f9716))

- - -

## [v0.23.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.22.0..v0.23.0) - 2025-05-08
#### Features
- **(vhdl)** remove the previous linear_1d implementation - ([0f1b9aa](https://github.com/es-ude/elastic-ai.creator/commit/0f1b9aa2f1c12f5c0fc1fe6a3db482f40041c057))
- **(vhdl)** enable multiple linear layers in the same model, by adding layer_name - ([3a99a30](https://github.com/es-ude/elastic-ai.creator/commit/3a99a3059dd53b913e7d619cbce28014007bf854))
- **(vhdl)** merge main to current working branch - ([35db3c5](https://github.com/es-ude/elastic-ai.creator/commit/35db3c56608493c6b33d05e0c2250cedb0374c8e))
- **(vhdl)** check the component interface - ([53791c5](https://github.com/es-ude/elastic-ai.creator/commit/53791c5eb9a72793b16a0a41eb79ed8932b8e32d))
#### Documentation
- **(vhdl)** change documentation - ([d3fb540](https://github.com/es-ude/elastic-ai.creator/commit/d3fb5402c7acb09cee3df535671f22d5011f2f47))
#### Refactoring
- **(vhdl)** change the interface name of the template - ([a693041](https://github.com/es-ude/elastic-ai.creator/commit/a693041a050ef77828a4f4dba791b0a38a845184))

- - -

## [v0.22.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.21.0..v0.22.0) - 2025-05-08
#### Features
- **(vhdl)** raise an exception if the build folder already exists - ([d09bfa1](https://github.com/es-ude/elastic-ai.creator/commit/d09bfa105d909b58432cf8883ee55a6b11639add))
#### Documentation
- **(vhdl)** add missing parameter in docstring of the translate_model function - ([458a02c](https://github.com/es-ude/elastic-ai.creator/commit/458a02c38402a0860500d5821b68890fcc78c01a))

- - -

## [v0.21.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.20.1..v0.21.0) - 2025-05-08
#### Features
- **(vhdl)** add default build function mapping and small changes - ([b1d6f2a](https://github.com/es-ude/elastic-ai.creator/commit/b1d6f2ac1040e63781d5f4af7ee29e486d9b6d69))
- **(vhdl)** add fp_linear build function, and test passed - ([ffcbb1d](https://github.com/es-ude/elastic-ai.creator/commit/ffcbb1d57408ad03e91bd1228bc6d3289f1d0c66))
- **(vhdl)** add fp_linear_module and test passed - ([241fd65](https://github.com/es-ude/elastic-ai.creator/commit/241fd652495d6ce582873f1bcc297302f3d61764))
- **(vhdl)** add fp_linear_component and its template unittest is passed - ([6e97316](https://github.com/es-ude/elastic-ai.creator/commit/6e973168ca244e4cf407c48b31406d2eed73b4b0))
- **(vhdl)** add default build mapping for fp_hard_sigmoid and fp_relu - ([c9c4d9f](https://github.com/es-ude/elastic-ai.creator/commit/c9c4d9f329ed2c56d47f2b698dbe1d3b34c1c8a5))

- - -

## [v0.20.1](https://github.com/es-ude/elastic-ai.creator/compare/v0.20.0..v0.20.1) - 2025-05-08
#### Features
- **(vhdl)** add fixed point relu to translator - ([80935ce](https://github.com/es-ude/elastic-ai.creator/commit/80935ce550a2e99267a55b41ad272906faf211a5))
#### Bug Fixes
- **(vhdl)** fix incompatible signature of the forward function - ([ff6c165](https://github.com/es-ude/elastic-ai.creator/commit/ff6c165cd0bf17477051548018b791809fff33c9))
#### Refactoring
- **(vhdl)** remove usage of deprecated assertEquals function - ([6a6f4f3](https://github.com/es-ude/elastic-ai.creator/commit/6a6f4f3af28735e27fc70b51f857324cd1ead7ef))
- **(vhdl)** small change of the FixedPointFactory type - ([c7629bd](https://github.com/es-ude/elastic-ai.creator/commit/c7629bd05764de09f03fd8445437dee671518d38))
- **(vhdl)** get rid of some comments - ([ced1b12](https://github.com/es-ude/elastic-ai.creator/commit/ced1b127031c02d8576ccc35fdd9f143017c3368))

- - -

## [v0.20.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.19.0..v0.20.0) - 2025-05-08
#### Features
- **(examples)** add example using quantized modules to verify the current state of the translator - ([0c55e00](https://github.com/es-ude/elastic-ai.creator/commit/0c55e00657c0d260766155995b75f25bff642e24))
- **(vhdl)** integrate fixed point hard sigmoid to the translator - ([0a07cee](https://github.com/es-ude/elastic-ai.creator/commit/0a07ceeb3d238456dad08448b543f4a075873322))
#### Documentation
- **(vhdl)** add documentation for the quantized_modules package - ([9da4a0d](https://github.com/es-ude/elastic-ai.creator/commit/9da4a0d380304a7ab8834049ad93bed547816ddb))
#### Refactoring
- **(examples)** remove unused import - ([fbb684d](https://github.com/es-ude/elastic-ai.creator/commit/fbb684daaeb8376ec7a56b413959cb9e9f2dc600))
- **(examples)** rename example to fit its actual content - ([9ac5a83](https://github.com/es-ude/elastic-ai.creator/commit/9ac5a83a558887d0cf4830a6f7ba94ede92de594))
- **(vhdl)** change name of the output_quant parameter to output_dequant to be more precise - ([3186b5b](https://github.com/es-ude/elastic-ai.creator/commit/3186b5b848e4b7be8e0bc0d94a1897722d2e2397))

- - -

## [v0.19.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.18.0..v0.19.0) - 2025-05-08
#### Features
- **(vhdl)** merge translate_model and generate_code functions - ([c12562e](https://github.com/es-ude/elastic-ai.creator/commit/c12562ee4a55c61b5ef82b5ef37568fe32e8f525))

- - -

## [v0.18.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.17.0..v0.18.0) - 2025-05-08
#### Features
- **(examples)** add simulated fixed point inference to the example - ([4f81d8d](https://github.com/es-ude/elastic-ai.creator/commit/4f81d8d3d44f1c677fc1a12edf94b7b614d72efb))
- **(vhdl)** add clamp to min or max fixed point integer for overflowing values - ([ca3fc19](https://github.com/es-ude/elastic-ai.creator/commit/ca3fc19aec062d4de34a4698c9e0a9351b41c761))
- **(vhdl)** implement evaluator that evaluates a model according to a given metric - ([a0b089a](https://github.com/es-ude/elastic-ai.creator/commit/a0b089ad1f7c32acc0c4522bf830080442e8414d))
- **(vhdl)** implement evaluator for simulation of a quantized inference - ([353e82e](https://github.com/es-ude/elastic-ai.creator/commit/353e82e798359c3b15a42a02dcdc63e071b2d34e))
- **(vhdl)** use fixed point hard sigmoid and relu in the example - ([90350b9](https://github.com/es-ude/elastic-ai.creator/commit/90350b91b9ac917c8c1f0ab50c2744fb09671947))
- **(vhdl)** implement a version of relu for qat and quantized inference - ([ddd9607](https://github.com/es-ude/elastic-ai.creator/commit/ddd9607e8dbf333817112dfe24f795ac717f609e))
- **(vhdl)** fix wrong calculation of fixed point values and add quantized forward functions - ([93046d3](https://github.com/es-ude/elastic-ai.creator/commit/93046d3b93d1a977c4106cf56e7f98847a47aa00))
- **(vhdl)** refactoring and start implementing hard sigmoid activation function - ([ff94c9d](https://github.com/es-ude/elastic-ai.creator/commit/ff94c9dd1d1297f02e82a0d1f7f203f80c8d2732))
#### Refactoring
- **(examples)** rename fixed point linear layer example - ([59216da](https://github.com/es-ude/elastic-ai.creator/commit/59216da6973daca87e80c105513586df1c682ba6))
- **(vhdl)** remove unused line of code - ([f020554](https://github.com/es-ude/elastic-ai.creator/commit/f020554dc3f1ae6ed4ac025711e6ec1025ba8964))
- **(vhdl)** make floating point values more explicit - ([231b903](https://github.com/es-ude/elastic-ai.creator/commit/231b903127dcdc0b90bc6a4e29ccd29543033935))
- **(vhdl)** small changes to make the code easier to understand - ([7168ba1](https://github.com/es-ude/elastic-ai.creator/commit/7168ba145243616d247006f56d99de3c21e91401))
- **(vhdl)** removed unfinished fixed point configuration finder - ([fa6dc44](https://github.com/es-ude/elastic-ai.creator/commit/fa6dc44e0a02f57993f08b381b62297c2682b167))
- **(vhdl)** create a better module structure - ([b2dfeee](https://github.com/es-ude/elastic-ai.creator/commit/b2dfeee795d980aabc2822e3b1470f2e41d63416))

- - -

## [v0.17.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.16.0..v0.17.0) - 2025-05-08
#### Features
- **(examples)** visualize model parameters - ([5e1b4fc](https://github.com/es-ude/elastic-ai.creator/commit/5e1b4fc4c827c55d19cb9bc4206f706bcc737fba))

- - -

## [v0.16.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.15.0..v0.16.0) - 2025-05-08
#### Features
- **(example)** add the ability to plot the model parameters - ([b1b0b5e](https://github.com/es-ude/elastic-ai.creator/commit/b1b0b5e7697992c4c53825c739e2fb2dcc903dac))
- **(examples)** commit current state of the fixed point linear example - ([9b8ecae](https://github.com/es-ude/elastic-ai.creator/commit/9b8ecae971bc1dedabf17e79272008a3cbfb5123))
- **(examples)** start implementing an example for learning a simple logic function - ([6cff6de](https://github.com/es-ude/elastic-ai.creator/commit/6cff6deccd5c2080e930d93f5e145e4d7ea6a41e))
- **(vhdl)** implement qat for linear layer - ([d3ba49e](https://github.com/es-ude/elastic-ai.creator/commit/d3ba49e266b2931c1b16677dd91f17a75f091501))
- **(vhdl)** move the input, weight and output quantization to the linear layer - ([0c8b259](https://github.com/es-ude/elastic-ai.creator/commit/0c8b259ef688c606ebd4f8486ef7b6f48e0f8713))
- **(vhdl)** add feature to automatically derive fixed point parameters from a factory - ([70618d5](https://github.com/es-ude/elastic-ai.creator/commit/70618d512718efd7e718491af52e1acbc6c86622))
- **(vhdl)** move tracing example to example folder - ([b942155](https://github.com/es-ude/elastic-ai.creator/commit/b942155a240a6f34f0f02361b6631b431a448443))
- **(vhdl)** make base linear package private - ([b3cfa55](https://github.com/es-ude/elastic-ai.creator/commit/b3cfa55daff5c401bc036ffe2bba8b0c6b2f2554))
- **(vhdl)** add a type for fixed point factories - ([53b8499](https://github.com/es-ude/elastic-ai.creator/commit/53b84991671832c2e7fa24e61d927b7c039832d9))
- **(vhdl)** implement custom linear layers that allows to do fixed point calculations - ([c2364f6](https://github.com/es-ude/elastic-ai.creator/commit/c2364f6182bb8406e90a78d632bc868537705fd2))
- **(vhdl)** add function to get attribute names of an object matching a regex - ([acc8e29](https://github.com/es-ude/elastic-ai.creator/commit/acc8e29e2771d5642e1371af6fb3c44f83b5ebc7))
#### Bug Fixes
- **(vhdl)** fix bug in the linear matix multiplication and rename _BaseLinear layer to _LinearBase - ([da11356](https://github.com/es-ude/elastic-ai.creator/commit/da113561d69158ccc2a9266adb1eddcc79b1cb7d))
#### Miscellaneous Chores
- **(vhdl)** skip some tests that are not relevant at the moment - ([2dbfd55](https://github.com/es-ude/elastic-ai.creator/commit/2dbfd55a805166e97b640fafd5ebc1214288a863))
#### Refactoring
- **(creator)** remove unused typevars and change typing - ([2770991](https://github.com/es-ude/elastic-ai.creator/commit/2770991c00a9395697180fcec98e733164efde24))

- - -

## [v0.15.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.14.0..v0.15.0) - 2025-05-08
#### Features
- **(vhdl)** implement clipped fixed point representation - ([8e53506](https://github.com/es-ude/elastic-ai.creator/commit/8e53506fce0ba5adaa124ccd61de3b340bf1c95f))

- - -

## [v0.14.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.13.0..v0.14.0) - 2025-05-08
#### Features
- **(vhdl)** implement from_unsigned_int and from_signed_int function and remove unused function - ([aca77f5](https://github.com/es-ude/elastic-ai.creator/commit/aca77f5eac396f21821b07706ff250b2589dd037))
- **(vhdl)** working further on the fixed point configuration finder - ([beb9da0](https://github.com/es-ude/elastic-ai.creator/commit/beb9da0ec8c3fbc6bb4ff65a97e7424e4da6dd0d))
- **(vhdl)** start implementing fixed point evaluator - ([0f9c62a](https://github.com/es-ude/elastic-ai.creator/commit/0f9c62a38f9df6ee4e84f1a3b5524df03511b438))
- **(vhdl)** implement signed fixed point integer to FixedPoint object - ([0a2fc79](https://github.com/es-ude/elastic-ai.creator/commit/0a2fc7952dc13ea48c749856bf809a5540166598))
- **(vhdl)** implement automatic derivation of fixed point parameters in the lstm example - ([504008d](https://github.com/es-ude/elastic-ai.creator/commit/504008d7ef3f402f8476bb77f02a4a37176d229e))
#### Bug Fixes
- **(vhdl)** reimplement unsigned_int_values_to_fixed_point function - ([cdd069e](https://github.com/es-ude/elastic-ai.creator/commit/cdd069e6adffa882bb34fea2b7179891c282045b))

- - -

## [v0.13.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.12.1..v0.13.0) - 2025-05-08
#### Features
- **(gh-workflow)** explicitly set poetry version - ([82d202f](https://github.com/es-ude/elastic-ai.creator/commit/82d202f0229e7931fc7371f69abe0d1fe3a58134))
- **(vhdl)** remove translatable protocol - ([ca52a92](https://github.com/es-ude/elastic-ai.creator/commit/ca52a92d1bdc017773b872eaa5011b5117394472))
- **(vhdl)** remove translatable protocol - ([37412e8](https://github.com/es-ude/elastic-ai.creator/commit/37412e87d89d16c9159cf12ef00032343119100c))
#### Refactoring
- **(vhdl)** remove translatable protocol - ([ef5f8fd](https://github.com/es-ude/elastic-ai.creator/commit/ef5f8fd4b914cbaf9ff4369aaa72437a1b68f5d3))
#### Style

- beautify 2024c7e9a8aa9ed2487f58586aa41beabd6f63d2 - ([7a58041](https://github.com/es-ude/elastic-ai.creator/commit/7a58041b71ccd258d9fcb16b1ac1a15be32e212d))

- - -

## [v0.12.1](https://github.com/es-ude/elastic-ai.creator/compare/v0.12.0..v0.12.1) - 2025-05-08
#### Bug Fixes
- **(qat)** reimplement binarize - ([9bbccdd](https://github.com/es-ude/elastic-ai.creator/commit/9bbccddfc6ce6c2b928166cdfaf1112b294dba17))
#### Style

- beautify 7a60a043e83aedcdf281ec9357ee9f274aca59dd - ([0723cb1](https://github.com/es-ude/elastic-ai.creator/commit/0723cb12e8fc6290403efd68b6d552a81ad69a99))

- - -

## [v0.12.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.11.1..v0.12.0) - 2025-05-08
#### Features
- **(vhdl)** make work library name customizable - ([95fd8aa](https://github.com/es-ude/elastic-ai.creator/commit/95fd8aa0d7e512aeb04893de2df2e58cc4b3e641))
- **(vhdl)** insert values in the updated lstm.vhd template - ([4d9dccb](https://github.com/es-ude/elastic-ai.creator/commit/4d9dccbdb11afebb466f476c22539828bf5458b1))
#### Bug Fixes
- **(vhdl)** remove some comments - ([13cc1a1](https://github.com/es-ude/elastic-ai.creator/commit/13cc1a1ade14ccc7aa686523270dec20936ed14d))
- **(vhdl)** remove unused work library - ([c68fd9d](https://github.com/es-ude/elastic-ai.creator/commit/c68fd9d00b152c5bdb70d2d2c90ca8d3e9f381d0))
- **(vhdl)** fix test - ([c125bf1](https://github.com/es-ude/elastic-ai.creator/commit/c125bf16297ee9e39660ee904ab54268e8901d48))
- **(vhdl)** add changes from Chao after testing the translator - ([5a5d532](https://github.com/es-ude/elastic-ai.creator/commit/5a5d5325a3f598e0163d4eac0601b5961c2f5780))
- **(vhdl)** pre-add input-hidden and hidden-hidden bias - ([750941c](https://github.com/es-ude/elastic-ai.creator/commit/750941c3150cabefa2f393f6b12105a358a70f7f))
- **(vhdl)** fix calculation of the addr_width of the linear1d layer - ([6fa2b2a](https://github.com/es-ude/elastic-ai.creator/commit/6fa2b2a3bc83d3a51eb955d1464501662f6676a8))
#### Miscellaneous Chores
- **(docker)** add a dockerfile - ([e2f54b3](https://github.com/es-ude/elastic-ai.creator/commit/e2f54b373bf26ee6c94b0c0c448b6e34affb2e64))
#### Documentation
- **(readme)** move translator documentation to the vhdl package - ([9a90949](https://github.com/es-ude/elastic-ai.creator/commit/9a90949528978ff4732f585986a71cedd44e82a5))
- **(readme)** small changes of the documentation - ([9e7699c](https://github.com/es-ude/elastic-ai.creator/commit/9e7699ce617581f67f85cf4ef7d945d99df241be))
- **(readme)** update documentation according the newly added linear1d layer - ([41e2486](https://github.com/es-ude/elastic-ai.creator/commit/41e24868aecbf310ee4c9ad815f6ccc0da3f9f9b))
- **(vhdl)** adapt diagrams to the latest changes - ([c1750eb](https://github.com/es-ude/elastic-ai.creator/commit/c1750eb19f92a705f8f36ccefc9729d3545f0743))
#### Refactoring
- **(vhdl)** simplify code and reuse components - ([37cffd7](https://github.com/es-ude/elastic-ai.creator/commit/37cffd76953e1f7756de8ec7ebc5b356fb89f1ad))

- - -

## [v0.11.1](https://github.com/es-ude/elastic-ai.creator/compare/v0.11.0..v0.11.1) - 2025-05-08
#### Features
- **(examples)** add linear layer to the translation example - ([5f1e1db](https://github.com/es-ude/elastic-ai.creator/commit/5f1e1db8da7ce533cb592d56ca97e25ca563a60e))
- **(vhdl)** add an easy way to get a fixed point factory - ([d98ff03](https://github.com/es-ude/elastic-ai.creator/commit/d98ff0351f739859ed668a2ec295421e29fd24ec))
- **(vhdl)** implement the translation of a linear1d layer - ([b627e78](https://github.com/es-ude/elastic-ai.creator/commit/b627e780d054adcdd89009d87aa33fa31c913504))
#### Documentation
- **(readme)** add documentation on how the translator works - ([91ebea3](https://github.com/es-ude/elastic-ai.creator/commit/91ebea3fb7e7883f56b2cd9152769d151449a49a))
#### Refactoring
- **(vhdl)** small naming changes - ([fd5c9b4](https://github.com/es-ude/elastic-ai.creator/commit/fd5c9b4f9fccb95b9fd4a0223e87a791fd02224c))
- **(vhdl)** change naming for better understanding - ([17d8a3d](https://github.com/es-ude/elastic-ai.creator/commit/17d8a3d89dcbbb4882c62953ddb928d268945852))
- **(vhdl)** change naming of the translator components - ([fdf5586](https://github.com/es-ude/elastic-ai.creator/commit/fdf5586da727542be6bfab57fba4a98d8ec482d7))
#### Build system
- **(gh-workflow)** perform tests with more verbose output - ([8d7b50b](https://github.com/es-ude/elastic-ai.creator/commit/8d7b50b7ae2c6d513f67027e5f209cf3115e0964))
#### Style

- beautify 52e7e3e55053a9e95e786bf899056148753cddfc - ([a5c17b4](https://github.com/es-ude/elastic-ai.creator/commit/a5c17b428c20f8c55b7c9350e5d9a33ef8b76822))

- - -

## [v0.11.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.10.1..v0.11.0) - 2025-05-08
#### Features
- **(examples)** add an example using the vhdl translator for pytorch - ([395adcd](https://github.com/es-ude/elastic-ai.creator/commit/395adcd3e843b7f55f6156ba183dc8800055ef51))
- **(vhdl)** adapt the example to the changes of the translation - ([6a5644e](https://github.com/es-ude/elastic-ai.creator/commit/6a5644e30a7cd00ed1be1c2cb6fa2e0b4b114c1e))
- **(vhdl)** change translation from LSTMCell to LSTM - ([5e4f1cf](https://github.com/es-ude/elastic-ai.creator/commit/5e4f1cff380fabd3685660a0c279b9098c4ef278))
- **(vhdl)** removed the possibility to get a build function from a type - ([dbc2e8f](https://github.com/es-ude/elastic-ai.creator/commit/dbc2e8ffd95f5ddc2476fede9d170c9d4eb020c2))
- **(vhdl)** make build function mapping more general so that it can be reused for other frameworks - ([3369d7f](https://github.com/es-ude/elastic-ai.creator/commit/3369d7fb6a7d08930514a7c0553c9efe65fc54b9))
- **(vhdl)** change build function mapping to a different approach - ([b1b79b2](https://github.com/es-ude/elastic-ai.creator/commit/b1b79b2e5e9ea0cf627b16a41f1f75bf434b795e))
- **(vhdl)** add LSTMCellTranslationArguments to __init__.py file - ([061ead4](https://github.com/es-ude/elastic-ai.creator/commit/061ead404dc82ddc79ac75c155328ad5733eb04a))
- **(vhdl)** pass an DTO to a translatable instead of raw arguments to fix typing errors - ([2c33869](https://github.com/es-ude/elastic-ai.creator/commit/2c33869cce5bed725a90ea3a4980bc026aec1ac4))
- **(vhdl)** pass an DTO to a translatable instead of raw arguments to fix typing errors - ([4738725](https://github.com/es-ude/elastic-ai.creator/commit/4738725d09ca9114064c4c42dd2818fc6d5c973b))
- **(vhdl)** implement a more functional build function mapping - ([1425e03](https://github.com/es-ude/elastic-ai.creator/commit/1425e0304cf35617106199936d3b014c0d8ca483))
- **(vhdl)** add the ability to infer the build function from a given layer object or type - ([306df14](https://github.com/es-ude/elastic-ai.creator/commit/306df1427177d15c1b1e2c59b2e774a2a6e2c471))
- **(vhdl)** first untested draft for the pytorch translator - ([7e59462](https://github.com/es-ude/elastic-ai.creator/commit/7e5946259381af397e1ccd25006815af8256026f))
- **(vhdl)** implementation of the mapping of a torch module to the corresponding build function - ([b076fa3](https://github.com/es-ude/elastic-ai.creator/commit/b076fa32cef3c64f8fcc45df24814f4333c90b5c))
- **(vhdl)** use __init__ files to simplify the usage - ([3cc07ee](https://github.com/es-ude/elastic-ai.creator/commit/3cc07ee048a349ef5a6a5383dcd829d64b48de2d))
- **(vhdl)** add a build function to create an abstract LSTMCell object from a PyTorch LSTMCell - ([baca5bb](https://github.com/es-ude/elastic-ai.creator/commit/baca5bb6c22692cf9bfc02a9147711b8869930fd))
- **(vhdl)** abstract LSTM cell takes float weights instead of FixedPoint weights - ([a5818cc](https://github.com/es-ude/elastic-ai.creator/commit/a5818cc0edd918ef3ca49e843738823e988bfd79))
- **(vhdl)** introduce translation arguments - ([2c3a8c7](https://github.com/es-ude/elastic-ai.creator/commit/2c3a8c72cfe8df70fd960e692d4fe037e2e86b6f))
- **(vhdl)** add a protocol specify a translatable layer - ([0fa966e](https://github.com/es-ude/elastic-ai.creator/commit/0fa966e7f99ef2adb19321b3ca92202616b4c0a2))
- **(vhdl)** add ability to pass kwargs to the translate function of a translatable layer - ([196812e](https://github.com/es-ude/elastic-ai.creator/commit/196812eecd0dc49a1b8c2d6675b9018ca07e003e))
- **(vhdl)** implementation of a LSTMCell class that can be translated to VHDL - ([ace37fe](https://github.com/es-ude/elastic-ai.creator/commit/ace37fe4b215327bc5b43344ffcd0c44a4822dda))
#### Bug Fixes
- **(examples)** use LSTMTranslationArguments object instead of a dictionary - ([98a4d97](https://github.com/es-ude/elastic-ai.creator/commit/98a4d97f8fbd217f67ed4009ab63ccc4705f720d))
- **(qat)** remove deprecated threshold and codomain properties - ([5db9669](https://github.com/es-ude/elastic-ai.creator/commit/5db9669fc3942851e65607a869bb822430df7836))
- **(vhdl)** remove print call - ([55164b7](https://github.com/es-ude/elastic-ai.creator/commit/55164b78c61f37f4cdadde0385965ee540e4f555))
- **(vhdl)** rename LSTMCell translatable to LSTM - ([e05cd04](https://github.com/es-ude/elastic-ai.creator/commit/e05cd042daf0420b2046607e00eeef3606a6defb))
- **(vhdl)** fix mypy typing errors - ([e1dba31](https://github.com/es-ude/elastic-ai.creator/commit/e1dba317585c269ad58719184fb4764cc66485ae))
- **(vhdl)** fix wrong pytorch lstm cell class path - ([85a733c](https://github.com/es-ude/elastic-ai.creator/commit/85a733cb5ff821bb602b5021f6438b7d5909382e))
- **(vhdl)** fix test - ([528910c](https://github.com/es-ude/elastic-ai.creator/commit/528910cf3fe28958ebb7b246104e83df77bbf3f4))
#### Documentation
- **(readme)** fix commands of install dev dependencies - ([870e2de](https://github.com/es-ude/elastic-ai.creator/commit/870e2de30f48223d8005bcf1240b624ebb314ad7))
- **(vhdl)** add some docstrings to the functions of the translator - ([6f9215e](https://github.com/es-ude/elastic-ai.creator/commit/6f9215e5fc35287517d884a702bf887d7a09aa7f))
#### Refactoring
- **(examples)** change the name of an example - ([606c0a3](https://github.com/es-ude/elastic-ai.creator/commit/606c0a30e37e5bd7d7ddc5529c770594debd7605))
- **(typing)** vhdl module type is now an iterable instead of an iterator to be more flexible - ([1b471ca](https://github.com/es-ude/elastic-ai.creator/commit/1b471ca3a8f5b3ff3c7c28e105ae3f7f2419367d))
- **(typing)** use better typing - ([86a019d](https://github.com/es-ude/elastic-ai.creator/commit/86a019d6d3db8696850b65047481e9566da66cd8))
- **(vhdl)** remove empty module - ([03edaca](https://github.com/es-ude/elastic-ai.creator/commit/03edaca097759ff381b012f757631662c4b5fe3a))
- **(vhdl)** change some names to make the code more understandable - ([a8d8b0c](https://github.com/es-ude/elastic-ai.creator/commit/a8d8b0c2fd3a27911a530db20dc3596113fc80e8))
- **(vhdl)** correct name of a test - ([73d360f](https://github.com/es-ude/elastic-ai.creator/commit/73d360f9f3c9fc6fdf5380ff45c947b49f475199))
- **(vhdl)** change typings and Translatable yield VHDLComponent - ([eedacb1](https://github.com/es-ude/elastic-ai.creator/commit/eedacb16afaf805eb6a990aa1ad40273722e02a3))
- **(vhdl)** remove custom template mapping, fixed point args and use protocols instead of abc - ([fed2658](https://github.com/es-ude/elastic-ai.creator/commit/fed26585c8ccd123e590476b8e0a8ec4df8891f6))

- - -

## [v0.10.1](https://github.com/es-ude/elastic-ai.creator/compare/v0.10.0..v0.10.1) - 2025-05-08
#### Bug Fixes
- **(gh-workflow)** fix error in the main.yml - ([4a6ff5e](https://github.com/es-ude/elastic-ai.creator/commit/4a6ff5e61f35661a3ef83ce4335c109333834d6d))
- **(gh-workflow)** try to fix the error with the semantic release tool - ([bc115f8](https://github.com/es-ude/elastic-ai.creator/commit/bc115f899bd85e720448bfa67fe9964bb56c594b))
#### Miscellaneous Chores
- **(gh-workflow)** try to fix/investigate the error with the semantic release tool - ([7697877](https://github.com/es-ude/elastic-ai.creator/commit/7697877c44fa382bf0fd3838077078d61b5117dc))

- - -

## [v0.10.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.9.0..v0.10.0) - 2025-05-08
#### Features
- **(vhdl)** format_vhdl function blocks the process until formatting is complete - ([a8a1bd0](https://github.com/es-ude/elastic-ai.creator/commit/a8a1bd0e7a4db075d0cef4a9eb125a860a697719))
#### Miscellaneous Chores
- **(gh-workflow)** removing the compilation steps for onnx as they are no longer needed - ([1118ee4](https://github.com/es-ude/elastic-ai.creator/commit/1118ee4ad89713a56d8a36fb93be46f0a2a33a32))
- **(pyproject)** add matplotlib dependency for the qlstm example - ([dadbc20](https://github.com/es-ude/elastic-ai.creator/commit/dadbc20f5e4328d6475c418277b08059b9ba1391))
- **(pyproject)** update numpy, onnx and add pre-commit to dev dependencies - ([a23c00a](https://github.com/es-ude/elastic-ai.creator/commit/a23c00ad3faef0ed5e2318f83553ad243749c920))
#### Documentation
- **(readme)** remove compile instructions for onnx as they are no longer needed - ([3bee70a](https://github.com/es-ude/elastic-ai.creator/commit/3bee70abe4185a0a6708ffc998bdc74004f90b8a))
- **(vhdl)** add a docstring with an example to the FixedPoint class - ([961d766](https://github.com/es-ude/elastic-ai.creator/commit/961d76678d366730f57bbd69b43c38124c003bf7))
#### Refactoring
- **(creator)** remove unused file - ([ca2b04f](https://github.com/es-ude/elastic-ai.creator/commit/ca2b04f8a3b4204ff18619a89f9fdc44b291b20a))
- **(precomputation)** apply python3.10 typing, renaming functions/classes, remove unused imports - ([2c4212f](https://github.com/es-ude/elastic-ai.creator/commit/2c4212ffa10df0550f5ac0924eee55e46a1ece5d))
- **(qat)** apply python3.10 typing, renaming functions/classes, remove unused imports - ([15f4b8a](https://github.com/es-ude/elastic-ai.creator/commit/15f4b8a52c78a680e3ad95fc70dbd85864282606))
- **(typing)** add missing typing - ([1e58596](https://github.com/es-ude/elastic-ai.creator/commit/1e58596b12eef51de75fe01f60529271f4caaa6b))
- **(typing)** set correct typings - ([600f6fb](https://github.com/es-ude/elastic-ai.creator/commit/600f6fb9db4e908e7c6eda4652af858258c903aa))
- **(typing)** apply python3.10 typing, remove unused imports - ([f2a31c6](https://github.com/es-ude/elastic-ai.creator/commit/f2a31c6d7d75e1f545ea63cd1ed6f19dc7be7249))
- **(vhdl)** move _int_to_bin_str to ToLogicEncoder class and refactor the class - ([c6495a0](https://github.com/es-ude/elastic-ai.creator/commit/c6495a05c77962ce4cfb4a4110bf0add74d11869))
- **(vhdl)** apply python3.10 typing, renaming functions/classes, remove unused imports - ([1554885](https://github.com/es-ude/elastic-ai.creator/commit/1554885edfe073f5066d82c47182704fdf14d415))
- **(vhdl)** remove unused file - ([577c91e](https://github.com/es-ude/elastic-ai.creator/commit/577c91ed4279ed7dbcdae71b5f4e8f868f6092ab))

- - -

## [v0.9.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.8.0..v0.9.0) - 2025-05-08
#### Features
- **(vhdl)** integrate FixedPoint type - ([b67a609](https://github.com/es-ude/elastic-ai.creator/commit/b67a6096023a51ff4882a8cdd03a7765884c8d93))
- **(vhdl)** add function to convert FixedPoint to signed int representation - ([03001ed](https://github.com/es-ude/elastic-ai.creator/commit/03001ed608ac934e8bbdcdfa1acb2fc7c163a89a))
- **(vhdl)** verify total bits and frac bits for multiple lists - ([360d318](https://github.com/es-ude/elastic-ai.creator/commit/360d318db0076d9077ceb94f3f7904d95e2b12f6))
- **(vhdl)** integrate FixedPoint datatype in the LSTM test bench classes - ([7cbb88a](https://github.com/es-ude/elastic-ai.creator/commit/7cbb88a7f77728776e0e976dcc68505b4162f0cc))
- **(vhdl)** add function to convert a list of ints to a list of FixedPoint objects - ([abece1f](https://github.com/es-ude/elastic-ai.creator/commit/abece1fd38af607c5f5734aeacd77a1743ff3411))
- **(vhdl)** add function to convert list of float values to a list of FixedPoint objects - ([02b26d8](https://github.com/es-ude/elastic-ai.creator/commit/02b26d868cad2a5a5bed2350a2929cf362ccdca8))
- **(vhdl)** change Rom that it uses the FixedPoint datatype - ([876cdb8](https://github.com/es-ude/elastic-ai.creator/commit/876cdb821ff0ac67ae2345c8a36e4a742cce0949))
- **(vhdl)** add a function to infer total and frac bits from a sequence of FixedPoint values - ([9cc2b72](https://github.com/es-ude/elastic-ai.creator/commit/9cc2b721b147628b2abf524129eeaac8f68520d5))
- **(vhdl)** separate hex/bin representation from vhdl hex/bin representation - ([eb8fe60](https://github.com/es-ude/elastic-ai.creator/commit/eb8fe60300ee7572500f9f9d11b62a9c5abff802))
#### Bug Fixes
- **(vhdl)** correct usage of the lookup_table_generator_function according to the type hints - ([9812ee8](https://github.com/es-ude/elastic-ai.creator/commit/9812ee85cd467e261af942b30493ac0e970ea5e4))
- **(vhdl)** remove old brevitas code - ([86a8104](https://github.com/es-ude/elastic-ai.creator/commit/86a8104cb6049dc016a5c8da08a7d2abc011935b))
- **(vhdl)** change value so that it fits into the value range of a fixed point value - ([b4e973e](https://github.com/es-ude/elastic-ai.creator/commit/b4e973ebb8a087351e07966821229f69dc345d79))
#### Refactoring
- **(vhdl)** remove no longer needed fixed point converter classes - ([4fcf0d1](https://github.com/es-ude/elastic-ai.creator/commit/4fcf0d16cdcac082c19dd654210b5d37991f9139))
- **(vhdl)** merge gen_func_for_one_lstm_cell and gen_func_for_lstm_layer in one module - ([06158a9](https://github.com/es-ude/elastic-ai.creator/commit/06158a92feca1023dd1b781da691a6965529c842))
- **(vhdl)** use resource_utils instead of importlib directly - ([a7598b4](https://github.com/es-ude/elastic-ai.creator/commit/a7598b4779d98dc0a843f6a90d459f70c6d632f3))
- **(vhdl)** small code quality improvement by using the chain function - ([6517cdd](https://github.com/es-ude/elastic-ai.creator/commit/6517cdd4090574d2be5bbbdf6ae68571d6679f05))
- **(vhdl)** apply python 3.10 typing - ([1c73b26](https://github.com/es-ude/elastic-ai.creator/commit/1c73b265bb8c935d8618f0355c50ca42d1b47168))

- - -

## [v0.8.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.7.0..v0.8.0) - 2025-05-08
#### Features
- **(gh-workflow)** increase python version - ([02403e6](https://github.com/es-ude/elastic-ai.creator/commit/02403e6cb7d8c9acc4357d9649fd2ae0834030a0))
- **(pyproject)** drop brevitas support - ([103f188](https://github.com/es-ude/elastic-ai.creator/commit/103f1882c8da81cdf114f10b1b76c2ce89a07cba))

- bump python version to 3.10 - ([47f5f07](https://github.com/es-ude/elastic-ai.creator/commit/47f5f0718460a966faaa937b2c6b016720434082))
#### Bug Fixes
- **(gh-workflow)** set more explicit python version - ([9c44093](https://github.com/es-ude/elastic-ai.creator/commit/9c44093c6cd41d05a2d178e6e113bd10f7b86016))
- **(precomputation)** change ModuleProto to Module - ([cfe418e](https://github.com/es-ude/elastic-ai.creator/commit/cfe418e41889708a53c255a8a7abcd6f1648f8f2))
- **(pyproject)** update poetry lock file - ([0116934](https://github.com/es-ude/elastic-ai.creator/commit/0116934b994c4e743b1be009172de0e07acd9182))
- **(pyproject)** update poetry lock file - ([9230672](https://github.com/es-ude/elastic-ai.creator/commit/92306722dabe5c4196e79a7cbbebab1e75ac3e6d))
- **(pyproject)** correct deps - ([7935ba1](https://github.com/es-ude/elastic-ai.creator/commit/7935ba19bcbda7e47ddbc358c12af3aa2a01df0a))
- **(pyproject)** set correct version numbers and add protobuf dependency - ([260e5fb](https://github.com/es-ude/elastic-ai.creator/commit/260e5fb31c425ad9ba2ec31f2fa292961fd28ffa))
- **(vhdl)** fix import of Sequence type - ([2c463ac](https://github.com/es-ude/elastic-ai.creator/commit/2c463acdbdae0ed7dc9fa99730f53db94deb7142))

- specify exact python version in github workflow - ([f3ffb18](https://github.com/es-ude/elastic-ai.creator/commit/f3ffb183e86b722cec5efb31e0937c4810542aef))

- fix dependencies + onnx integration tests - ([f06d0f8](https://github.com/es-ude/elastic-ai.creator/commit/f06d0f8436ca2a7ed3410aee4ad36df1cdad45c0))

- resolve dependency version conflicts - ([32bd544](https://github.com/es-ude/elastic-ai.creator/commit/32bd544b2e74b8b57497f3fd604deb5ed86ebb42))
#### Style

- beautify 6772407f9929e398f7e03858e91b02c52bc8e3ec - ([ecb21e2](https://github.com/es-ude/elastic-ai.creator/commit/ecb21e271271e52d63c268b311d598fb8c86af15))

- - -

## [v0.7.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.6.1..v0.7.0) - 2025-05-08
#### Features
- **(vhdl)** add rich comparison methods, multiple operators and a bit iterator to FixedPoint - ([116b006](https://github.com/es-ude/elastic-ai.creator/commit/116b00647c05ef6854d3cbd1ab0f79c58f0c450d))
- **(vhdl)** start implementing FixedPoint datatype - ([8c4f420](https://github.com/es-ude/elastic-ai.creator/commit/8c4f42097ff416f8e9056af430bda01a5bd42df5))

- - -

## [v0.6.1](https://github.com/es-ude/elastic-ai.creator/compare/v0.6.0..v0.6.1) - 2025-05-08
#### Bug Fixes
- **(vhdl)** saving generated examples to a directory instead of a giving an explicit file path - ([eb41d8d](https://github.com/es-ude/elastic-ai.creator/commit/eb41d8db9af5171ac2826f41e98b5d85598b582d))

- - -

## [v0.6.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.5.0..v0.6.0) - 2025-05-08
#### Bug Fixes
- **(vhdl)** move missing files - ([e4ae3c2](https://github.com/es-ude/elastic-ai.creator/commit/e4ae3c2815a33b8f4f33c9578ab5cae0842277aa))
- **(vhdl)** fix previously broken imports - ([bf694f8](https://github.com/es-ude/elastic-ai.creator/commit/bf694f80fbd3a5478d99e8ae6b198a9e363569c9))
  - **BREAKING CHANGE**: move modules out of generator package
#### Refactoring
- **(vhdl)** remove usage of protected functions in tests - ([47ca401](https://github.com/es-ude/elastic-ai.creator/commit/47ca401e9c19f3f80140bc9c06c1a3e162c6849c))

- - -

## [v0.5.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.4.2..v0.5.0) - 2025-05-08
#### Features
- **(vhdl)** add multiline template expansion - ([309ea35](https://github.com/es-ude/elastic-ai.creator/commit/309ea350fae2b4e54bf06101aadc28e227d30cbb))
- **(vhdl)** add multiline template expansion - ([0d7f91f](https://github.com/es-ude/elastic-ai.creator/commit/0d7f91f7347a5501eba02ad40499a5c0fdcce3bc))
#### Miscellaneous Chores
- **(layers)** remove deprecation warning about Q* layers - ([c696596](https://github.com/es-ude/elastic-ai.creator/commit/c6965961f37a5154356a9b299fc1de36888cd184))
#### Documentation
- **(readme)** add git commit message scopes - ([fe8e328](https://github.com/es-ude/elastic-ai.creator/commit/fe8e328eda5a5f9e4cac886fcbfc9388f13d3d0f))
- **(readme)** fix table of contents and section headers - ([ecdef5d](https://github.com/es-ude/elastic-ai.creator/commit/ecdef5da63c2c10e61a159c144c5c3707a5699e8))
- **(readme)** shorten - ([e535ea0](https://github.com/es-ude/elastic-ai.creator/commit/e535ea0fd9d783f29ebb32d756077289d8baa8c9))
#### Refactoring
- **(precomputation)** make IOTable grouping an IOTable method - ([c97ec8c](https://github.com/es-ude/elastic-ai.creator/commit/c97ec8c40e1f525a19cdc6838f73be312c209b10))
- **(precomputation)** make IOTable grouping an IOTable method - ([9f08625](https://github.com/es-ude/elastic-ai.creator/commit/9f08625704d76776bc7d8f09f15e8571bb81d4ba))
- **(typing)** use correct numpy typing - ([3d3ce3f](https://github.com/es-ude/elastic-ai.creator/commit/3d3ce3fe11e96c882e5392cc98ca059addb2b145))

- move implementations to packages corresponding to scopes - ([fa4487b](https://github.com/es-ude/elastic-ai.creator/commit/fa4487b6d491f2f3b089000aca7fe04366b441d0))

- - -

## [v0.4.2](https://github.com/es-ude/elastic-ai.creator/compare/v0.4.1..v0.4.2) - 2025-05-08

- - -

## [v0.4.1](https://github.com/es-ude/elastic-ai.creator/compare/v0.4.0..v0.4.1) - 2025-05-08
#### Features
- **(vhdl)** add multiline template expansion - ([5779708](https://github.com/es-ude/elastic-ai.creator/commit/5779708c2de34d9c32a18af689b4b094d513ef8d))
- **(vhdl)** add multiline template expansion - ([3177fcd](https://github.com/es-ude/elastic-ai.creator/commit/3177fcd4e5f1830608e9f6590b5af312bb74b7a9))
#### Bug Fixes

- minor errors - ([812809e](https://github.com/es-ude/elastic-ai.creator/commit/812809e1d0e706df3a0514b3503dc283ea12d7a4))

- - -

## [v0.4.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.3.10..v0.4.0) - 2025-05-08
#### Features
- **(vhdl)** allow ToLogicEncoder to register symbols in batches - ([9209279](https://github.com/es-ude/elastic-ai.creator/commit/9209279debe651b653d2fee44533ccbdae945b32))
#### Bug Fixes
- **(vhdl)** improve names in scope of ToLogicEncoder - ([67f7312](https://github.com/es-ude/elastic-ai.creator/commit/67f73129faefe343e9fb5e84563d125b1d36bab6))
  - **BREAKING CHANGE**: rename numerics attr to _symbols,
    mapping attribute to _mapping.
    rename add_numeric to register_symbol

- - -

## [v0.3.10](https://github.com/es-ude/elastic-ai.creator/compare/v0.3.9..v0.3.10) - 2025-05-08
#### Bug Fixes
- **(number-repr)** fix a bug and add some parameter checks - ([a78e9e8](https://github.com/es-ude/elastic-ai.creator/commit/a78e9e8f669c477d0629695f5c7c8ad8628f0522))
- **(types)** add missing mlframework types - ([3b5cf5f](https://github.com/es-ude/elastic-ai.creator/commit/3b5cf5f8be829e109db363c25ecff76634f9d94f))
- **(typing)** fix some mypy errors - ([35b8fdf](https://github.com/es-ude/elastic-ai.creator/commit/35b8fdf4cb0736770d9592f86499192e1e84d673))
#### Miscellaneous Chores
- **(gitignore)** add htmlcov produced from coverage - ([5179c0d](https://github.com/es-ude/elastic-ai.creator/commit/5179c0d526dd549fe06101342a33f89117acc022))
- **(gitignore)** add coverage and onnx outputs - ([a034785](https://github.com/es-ude/elastic-ai.creator/commit/a03478528034243c1cbe8358890bb65a2845423c))

- - -

## [v0.3.9](https://github.com/es-ude/elastic-ai.creator/compare/v0.3.8..v0.3.9) - 2025-05-08
#### Miscellaneous Chores
- **(gh-workflow)** remove pre-commit usage - ([9cd3f34](https://github.com/es-ude/elastic-ai.creator/commit/9cd3f34c8e8b6ef1dc0904f071b1d2e3a2c0e684))
- **(gh-workflow)** correct token for  test pypi - ([112eb37](https://github.com/es-ude/elastic-ai.creator/commit/112eb374c0f4b43b61b4988e8425a82881bd6802))
- **(gh-workflow)** trigger precommit on push - ([391cc8e](https://github.com/es-ude/elastic-ai.creator/commit/391cc8ef81c2d92bf432adb7814cbe95e9961c38))
- **(gh-workflow)** manually trigger precommit workflow - ([f6611d9](https://github.com/es-ude/elastic-ai.creator/commit/f6611d9360f2b8a9ece7ace714050c85884fd6ce))
- **(gitignore)** add npm files - ([352c38f](https://github.com/es-ude/elastic-ai.creator/commit/352c38f3c83982b3abd52eb0d2bb1a654ff9bb57))
- **(gitignore)** add mypy cache - ([0a8c31e](https://github.com/es-ude/elastic-ai.creator/commit/0a8c31e0045ad244192bdb9fc91803a5d6470de1))
- **(pre-commit)** put mypy+dead into manual stage - ([98d9620](https://github.com/es-ude/elastic-ai.creator/commit/98d9620f3a33a42a14d3dae04841660e82187609))
- **(pre-commit)** default install commit-msg stage - ([28c4716](https://github.com/es-ude/elastic-ai.creator/commit/28c4716672a446623ad957b5f7f090f1eff211af))
- **(precommit)** configure hook stages - ([1d9ffc5](https://github.com/es-ude/elastic-ai.creator/commit/1d9ffc57bdad9832c286d70412a2dcccce866f29))
- **(pyproject)** update version number - ([94bdcab](https://github.com/es-ude/elastic-ai.creator/commit/94bdcabfa74fe4e634932eac6e4b5e36a02df236))
#### Documentation
- **(readme)** add brief explanation of pre-commit - ([3626bb0](https://github.com/es-ude/elastic-ai.creator/commit/3626bb07cc1c8600b193bc380ae8275116ebaba8))

- - -

## [v0.3.8](https://github.com/es-ude/elastic-ai.creator/compare/v0.3.5-alpha.0..v0.3.8) - 2025-05-08
#### Bug Fixes
- **(gh-workflow)** updat changelog correctly - ([e76a41c](https://github.com/es-ude/elastic-ai.creator/commit/e76a41cf55463cbc2a4ffa5b2b233d49695302b9))
- **(gh-workflow)** install latest isort fixing broken imports - ([a61ef44](https://github.com/es-ude/elastic-ai.creator/commit/a61ef445672f913ec4ebc4cc8b46c2ef9099bec7))
- **(gh-workflow)** set git user+mail to gh-actions - ([174ed47](https://github.com/es-ude/elastic-ai.creator/commit/174ed478b04b846912d6b0315f1143f24bc94524))
- **(gh-workflow)** fix job dependencies - ([6a7d3ee](https://github.com/es-ude/elastic-ai.creator/commit/6a7d3eeb975ca303aa30fce21bd29d14cf9982d3))
- **(gh-workflow)** fix typo - ([7f86205](https://github.com/es-ude/elastic-ai.creator/commit/7f8620502ee544917db42ea12c7cb2eadbaef8cc))
- **(gh-workflow)** typo unit-test -> unit-tests - ([1dbd71f](https://github.com/es-ude/elastic-ai.creator/commit/1dbd71f5f3dae489b4752a5f6fdf9d10e4251a73))
- **(gh-workflow)** close brace - ([a6e4b99](https://github.com/es-ude/elastic-ai.creator/commit/a6e4b999dadd163881fa96d03977c9c392a9267b))
- **(gh-workflow)** bump version - ([2cb3a72](https://github.com/es-ude/elastic-ai.creator/commit/2cb3a72b2aa9a86c0b4da71e3d7bff962a5728f6))
- **(gh-workflow)** fix syntax error - ([895326d](https://github.com/es-ude/elastic-ai.creator/commit/895326d67eb7ba1bb866a45c8b149778c93dc043))
- **(input_domains)** add missing import of itertools - ([a6b0344](https://github.com/es-ude/elastic-ai.creator/commit/a6b0344ac4b933112b19b8603358a0adc7274533))
- **(pre-commit)** add missing commitlint.config.js - ([2251de8](https://github.com/es-ude/elastic-ai.creator/commit/2251de83f60823d21346aedcc2b2e9aac4c27458))
- **(precomputation)** correct numpy namespace for type alias - ([a6c5842](https://github.com/es-ude/elastic-ai.creator/commit/a6c5842920c00ae6e53e226650e0fbfe48aac44a))
- **(pyproject)** fix duplicate field - ([6616cab](https://github.com/es-ude/elastic-ai.creator/commit/6616cab3b0342f0b5d0b8bbdbbdf719de56d5631))

- add missing tags_utils again - ([910c611](https://github.com/es-ude/elastic-ai.creator/commit/910c6116600b82e2c52c7d46896d92b63954d7c7))
#### Miscellaneous Chores
- **(gh-workflow)** deploy to pypi instead of testpypi - ([18aee87](https://github.com/es-ude/elastic-ai.creator/commit/18aee872212ba9f066d579e4c2a5edd11e5b4a59))
- **(gh-workflow)** automatically update changelog - ([45bfef3](https://github.com/es-ude/elastic-ai.creator/commit/45bfef38bd0dc3e86a9a291553b1f3ea5570dc9e))
- **(gh-workflow)** remove tag_utils.py - ([a8baca4](https://github.com/es-ude/elastic-ai.creator/commit/a8baca48073d6efa1330f82f87f23ec205ac02e9))
- **(gh-workflow)** use emojis for automatic semver - ([93a60cc](https://github.com/es-ude/elastic-ai.creator/commit/93a60cc2755c08098b9c1a1f8ff5dfecee289c76))
- **(gh-workflow)** setup semantic versioning - ([f714503](https://github.com/es-ude/elastic-ai.creator/commit/f714503474da4695a396ca784f110378b2147591))
- **(gh-workflow)** setup semantic versioning - ([ac06cf2](https://github.com/es-ude/elastic-ai.creator/commit/ac06cf26b4cfd66a3b49b82106ace1f236c01eb4))
- **(gh-workflow,pyproject)** build via semantic-release - ([e78e882](https://github.com/es-ude/elastic-ai.creator/commit/e78e882bb02b0a7adc9ff10c437a37bf6cc08dbc))
- **(gh-workflows)** try semantic release - ([8a23dbf](https://github.com/es-ude/elastic-ai.creator/commit/8a23dbfeafeae82f1332c3cd28c5cbf72215a9c8))
- **(gitignore)** add node_modules - ([23e1234](https://github.com/es-ude/elastic-ai.creator/commit/23e12348b598edea69cf0a79e4bee26c45f62f43))
- **(semver)** revert to angular style commit messages - ([55f99dd](https://github.com/es-ude/elastic-ai.creator/commit/55f99ddd6f809169f91d707a51f29477523f26b0))
#### Documentation

- update changelog - ([e1aa8c9](https://github.com/es-ude/elastic-ai.creator/commit/e1aa8c93554fc15c25a586b8e89eecda6dc03514))
#### Style
- **(imports)** sort imports - ([de31b33](https://github.com/es-ude/elastic-ai.creator/commit/de31b335ed9ee8cf04d3823d0b9058e54df07eb9))

- run pre-commit tools on all files - ([c22eecf](https://github.com/es-ude/elastic-ai.creator/commit/c22eecf97792e104596e6575d692c6f4564e66c2))

- beautify 1d617cd289068f3c6552da1bd6e9468759cb5747 - ([0bb5d39](https://github.com/es-ude/elastic-ai.creator/commit/0bb5d39e73c6b4e746f1fb0308b863273d86b7f3))

- beautify 6fe04eccb8dc55714b78e1a7222113c93a0b258c - ([919ac6e](https://github.com/es-ude/elastic-ai.creator/commit/919ac6ecfc5702c9a705f3da181916c2b9265366))

- - -

## [v0.3.5-alpha.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.3.4..v0.3.5-alpha.0) - 2025-05-08

- - -

## [v0.3.4](https://github.com/es-ude/elastic-ai.creator/compare/v0.3.0..v0.3.4) - 2025-05-08

- - -

## [v0.3.0](https://github.com/es-ude/elastic-ai.creator/compare/ff34a4e9a49830e5c160f47b132efe0d51d764ce..v0.3.0) - 2025-05-08


